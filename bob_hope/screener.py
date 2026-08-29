"""
Intraday ticker screener with ntfy.sh alerts — NOTIFICATION ONLY.

This tool reads market data from Twelve Data and sends push notifications.
It contains no order-placement, brokerage, or trading-API code of any kind
and never will; every signal is informational only.

Data flow per poll cycle (every POLL_INTERVAL_MINUTES during market hours):
  1. Daily bar history (time_series, interval=1day) is fetched once per
     trading day per symbol and cached in memory. EMA50 / SMA200 / Bollinger
     Bands are computed from CLOSED daily bars only.
  2. A batched /quote call (chunked to respect the 8-credit/min limit) gives
     each symbol's live price and previous close.
  3. Conditions are evaluated with the LIVE price: red/green day vs. prior
     close (stock and SPY), price vs. EMA50/SMA200/Bollinger levels, and RSI
     recomputed with the live price standing in for today's unfinished bar.
  4. For value-zone tickers, the real ~30-delta/~30-day put strike comes from
     Cboe's free delayed options chain (cached hourly per symbol), with a
     Black-Scholes estimate as fallback.

Run:  python screener.py        (see README_screener.md for details)
"""

import json
import logging
import math
import os
import re
import sys
import time
from collections import deque
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import config

ET = ZoneInfo("America/New_York")
SCRIPT_DIR = Path(__file__).resolve().parent
TD_BASE = "https://api.twelvedata.com"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("screener")


# ---------------------------------------------------------------------------
# Environment / secrets
# ---------------------------------------------------------------------------
def load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=value lines) so we avoid an extra dependency."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv(SCRIPT_DIR / ".env")

API_KEY = os.environ.get("TWELVEDATA_API_KEY", "")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", config.NTFY_TOPIC_DEFAULT)
NTFY_URL = f"{config.NTFY_SERVER}/{NTFY_TOPIC}"


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def now_et() -> datetime:
    return datetime.now(tz=ET)


def et_date_str(dt: datetime | None = None) -> str:
    return (dt or now_et()).strftime("%Y-%m-%d")


def is_market_hours(dt: datetime) -> bool:
    """Weekdays 9:30–16:00 ET. Holidays are caught via the quote endpoint's
    is_market_open flag (see run_poll_cycle)."""
    if dt.weekday() >= 5:
        return False
    minutes = dt.hour * 60 + dt.minute
    return (9 * 60 + 30) <= minutes < (16 * 60)


def seconds_until_next_open(dt: datetime) -> float:
    """Seconds from dt until the next weekday 9:30 ET."""
    candidate = dt.replace(hour=9, minute=30, second=0, microsecond=0)
    while candidate <= dt or candidate.weekday() >= 5:
        candidate += timedelta(days=1)
        candidate = candidate.replace(hour=9, minute=30, second=0, microsecond=0)
    return (candidate - dt).total_seconds()


# ---------------------------------------------------------------------------
# Persisted state
# ---------------------------------------------------------------------------
def load_state() -> dict:
    path = SCRIPT_DIR / config.STATE_FILE
    state = {"signal_active": {}, "value_zone_in_range": {},
             "alert_cooldowns": {}, "credits": {"date": "", "used": 0}}
    if path.exists():
        try:
            state.update(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read state file (%s); starting fresh", exc)
    # Legacy layout: technical-signal history lived under "signals" (per-day
    # date strings, later timestamps). Those signals are edge-triggered now, so
    # only the Cboe-warning cooldowns are still meaningful; carry them over and
    # drop the rest. Worst case after upgrading: one extra alert per active
    # setup on the first poll, because nothing is known to be "on" yet.
    legacy = state.pop("signals", {})
    state["alert_cooldowns"].update(
        {k: v for k, v in legacy.items()
         if k.endswith(":CBOE_FAIL") and isinstance(v, (int, float))})
    return state


def save_state(state: dict) -> None:
    path = SCRIPT_DIR / config.STATE_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def prune_alert_cooldowns(state: dict) -> None:
    """Drop cooldown timestamps older than any cooldown could care about."""
    cutoff = time.time() - max(7 * 86400, config.CBOE_FAIL_COOLDOWN_MINUTES * 60)
    state["alert_cooldowns"] = {k: v for k, v in state["alert_cooldowns"].items()
                                if v >= cutoff}


def should_fire(state: dict, key: str, cooldown_minutes: float) -> bool:
    """True if `key` has not alerted within cooldown_minutes (0 = always fire).

    Records the fire time as a side effect, so only call it at the moment of
    actually sending — a True return consumes the slot.
    """
    now = time.time()
    last = state["alert_cooldowns"].get(key)
    if last is not None and now - last < cooldown_minutes * 60:
        return False
    state["alert_cooldowns"][key] = now
    return True


def signal_transition(state: dict, key: str, active: bool) -> str | None:
    """Record whether `key` is currently in action territory and report the
    edge: 'on' (just became true), 'off' (just stopped being true), or None
    while it holds steady either way. Call once per poll for every signal —
    skipping a poll (e.g. missing data) leaves the last known state intact
    rather than faking an 'off'.
    """
    # bool() is load-bearing: the conditions are built from pandas/numpy
    # comparisons and come back as numpy.bool, which json.dumps refuses —
    # storing one raw would blow up save_state and take the loop down with it.
    active = bool(active)
    was = state["signal_active"].get(key, False)
    state["signal_active"][key] = active
    if active and not was:
        return "on"
    if was and not active:
        return "off"
    return None


# ---------------------------------------------------------------------------
# Credit budgeting (free tier: 8 credits/min, 800/day)
# ---------------------------------------------------------------------------
class CreditTracker:
    """Blocks until the per-minute budget allows a call; refuses calls that
    would blow past the daily soft cap. Daily usage persists in the state file
    so restarts don't reset the count."""

    def __init__(self, state: dict):
        self.state = state
        self.minute_events: deque[tuple[float, int]] = deque()
        self._roll_date()

    def _roll_date(self) -> None:
        today = et_date_str()
        if self.state["credits"].get("date") != today:
            self.state["credits"] = {"date": today, "used": 0}

    def daily_used(self) -> int:
        self._roll_date()
        return self.state["credits"]["used"]

    def acquire(self, n: int) -> bool:
        """Reserve n credits. Sleeps as needed for the minute window; returns
        False (without sleeping) if the daily soft cap would be exceeded."""
        self._roll_date()
        if self.daily_used() + n > config.DAILY_CREDIT_SOFT_CAP:
            log.warning(
                "Daily credit soft cap reached (%d used, %d requested, cap %d) — skipping call",
                self.daily_used(), n, config.DAILY_CREDIT_SOFT_CAP,
            )
            return False
        while True:
            now = time.monotonic()
            while self.minute_events and now - self.minute_events[0][0] > 60:
                self.minute_events.popleft()
            in_window = sum(c for _, c in self.minute_events)
            if in_window + n <= config.CREDITS_PER_MINUTE:
                break
            wait = 61 - (now - self.minute_events[0][0])
            log.info("Rate limit: waiting %.0fs for minute credit window", wait)
            time.sleep(max(wait, 1))
        self.minute_events.append((time.monotonic(), n))
        self.state["credits"]["used"] += n
        return True


# ---------------------------------------------------------------------------
# HTTP session, shared by Twelve Data, Cboe, and ntfy calls.
#
# Transport-level retries handle stale keep-alive connections: the loop sleeps
# 15 min between polls, servers drop idle connections in the meantime, and the
# first request of the next cycle then dies with WinError 10054 ("connection
# forcibly closed by the remote host") unless retried on a fresh socket.
# POST retries are deliberate: a rare duplicate ntfy push beats a lost alert.
# ---------------------------------------------------------------------------
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=3, connect=3, read=2, backoff_factor=1,
    status_forcelist=(502, 503, 504),
    allowed_methods=("GET", "POST"),
)))


def td_get(endpoint: str, params: dict) -> dict | None:
    """GET with retries; returns parsed JSON or None on failure."""
    params = {**params, "apikey": API_KEY}
    for attempt in range(3):
        try:
            resp = session.get(f"{TD_BASE}/{endpoint}", params=params, timeout=20)
            data = resp.json()
            if isinstance(data, dict) and data.get("status") == "error":
                # 429 = rate limited; worth backing off and retrying
                if data.get("code") == 429 and attempt < 2:
                    log.warning("Twelve Data rate-limit response; backing off 65s")
                    time.sleep(65)
                    continue
                log.error("Twelve Data error on %s: %s", endpoint, data.get("message"))
                return None
            return data
        except (requests.RequestException, ValueError) as exc:
            log.warning("Request to %s failed (attempt %d/3): %s", endpoint, attempt + 1, exc)
            time.sleep(5 * (attempt + 1))
    return None


def fetch_daily_closes(symbol: str, credits: CreditTracker) -> pd.Series | None:
    """Closed daily bars, oldest→newest, today's in-progress bar excluded."""
    if not credits.acquire(1):
        return None
    data = td_get("time_series", {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": config.DAILY_BARS,
    })
    if not data or "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    df["close"] = pd.to_numeric(df["close"])
    df = df.iloc[::-1].reset_index(drop=True)          # API returns newest first
    df = df[df["datetime"] < et_date_str()]            # drop today's partial bar
    return pd.Series(df["close"].values, index=df["datetime"].values)


def fetch_quotes(symbols: list[str], credits: CreditTracker) -> dict[str, dict]:
    """Batched /quote, chunked so one chunk never exceeds the per-minute credit
    budget. Returns {symbol: {price, prev_close, is_market_open}}."""
    out: dict[str, dict] = {}
    chunk_size = config.CREDITS_PER_MINUTE  # 1 credit per symbol in a batch quote
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        if not credits.acquire(len(chunk)):
            break
        data = td_get("quote", {"symbol": ",".join(chunk)})
        if data is None:
            continue
        if len(chunk) == 1:                 # single-symbol responses aren't keyed
            data = {chunk[0]: data}
        for sym in chunk:
            q = data.get(sym)
            if not isinstance(q, dict) or q.get("status") == "error" or "close" not in q:
                log.warning("No quote for %s (bad symbol or no data)", sym)
                continue
            try:
                out[sym] = {
                    "price": float(q["close"]),
                    "prev_close": float(q["previous_close"]),
                    "is_market_open": bool(q.get("is_market_open", True)),
                }
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("Malformed quote for %s: %s", sym, exc)
    return out


# ---------------------------------------------------------------------------
# Indicators — manual pandas implementations, on daily closes
# ---------------------------------------------------------------------------
def wilder_rsi(closes: pd.Series, period: int) -> float:
    """RSI with Wilder's smoothing: ewm(alpha=1/period, adjust=False) on
    gains/losses. Returns the final value of the series."""
    delta = closes.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    last_gain, last_loss = avg_gain.iloc[-1], avg_loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return 100.0 - 100.0 / (1.0 + rs)


def compute_indicators(closes: pd.Series) -> dict:
    """Levels from CLOSED daily bars: EMA50, SMA200, Bollinger(20, 2)."""
    ind = {"ema50": math.nan, "sma200": math.nan,
           "bb_upper": math.nan, "bb_lower": math.nan}
    if len(closes) >= config.EMA_PERIOD:
        ind["ema50"] = closes.ewm(span=config.EMA_PERIOD, adjust=False).mean().iloc[-1]
    if len(closes) >= config.SMA_PERIOD:
        ind["sma200"] = closes.rolling(config.SMA_PERIOD).mean().iloc[-1]
    if len(closes) >= config.BB_PERIOD:
        window = closes.tail(config.BB_PERIOD)
        mid = window.mean()
        sd = window.std(ddof=0)  # population std, the classic Bollinger convention
        ind["bb_upper"] = mid + config.BB_STD * sd
        ind["bb_lower"] = mid - config.BB_STD * sd
    return ind


def live_rsi(closes: pd.Series, live_price: float) -> float:
    """RSI(14) with the live price standing in for today's not-yet-closed bar."""
    extended = pd.concat([closes, pd.Series([live_price])], ignore_index=True)
    return wilder_rsi(extended, config.RSI_PERIOD)


def approx_csp_strike(closes: pd.Series, live_price: float) -> float:
    """FALLBACK strike estimate for a ~CSP_TARGET_DELTA put, ~CSP_TARGET_DTE_DAYS
    out, used only when the Cboe chain (get_csp_strike) is unavailable.

    Inverts Black-Scholes for the strike where put delta = -CSP_TARGET_DELTA,
    using annualized realized volatility of the last REALIZED_VOL_LOOKBACK
    daily returns as an IV proxy:  put delta = N(d1) - 1  =>
    d1 = N^-1(1 - target_delta),
    K = S * exp((r + sigma^2/2) * T - d1 * sigma * sqrt(T)).
    """
    if len(closes) < config.REALIZED_VOL_LOOKBACK + 1:
        return math.nan
    rets = (closes / closes.shift(1)).apply(math.log).dropna()
    sigma = rets.tail(config.REALIZED_VOL_LOOKBACK).std(ddof=1) * math.sqrt(252)
    if not sigma or math.isnan(sigma):
        return math.nan
    T = config.CSP_TARGET_DTE_DAYS / 365.0
    d1 = NormalDist().inv_cdf(1.0 - config.CSP_TARGET_DELTA)
    return live_price * math.exp(
        (config.RISK_FREE_RATE + sigma ** 2 / 2) * T - d1 * sigma * math.sqrt(T)
    )


# ---------------------------------------------------------------------------
# Real put strikes from Cboe's free delayed options chain (READ-ONLY market
# data; no key, no account, and no impact on the Twelve Data credit budget).
# Refreshed at most every CBOE_REFRESH_MINUTES per symbol; falls back to
# approx_csp_strike when Cboe is unreachable or has no chain for a symbol.
# ---------------------------------------------------------------------------
# OCC option symbol, e.g. AAPL260918P00220000 = root + YYMMDD + C/P + strike*1000
OCC_SYMBOL_RE = re.compile(r"^(.+?)(\d{6})([CP])(\d{8})$")

cboe_cache: dict[str, dict] = {}


def select_csp_put(options: list[dict], today: date) -> dict | None:
    """From a full Cboe chain, pick the put on the expiration nearest
    CSP_TARGET_DTE_DAYS whose delta is nearest -CSP_TARGET_DELTA."""
    by_exp: dict[date, list[tuple[float, float]]] = {}
    for opt in options:
        m = OCC_SYMBOL_RE.match(opt.get("option", ""))
        if not m or m.group(3) != "P":
            continue
        delta = opt.get("delta")
        # delta >= 0 also drops stale/unpriced rows Cboe reports as 0.0
        if not isinstance(delta, (int, float)) or delta >= 0:
            continue
        try:
            expiry = datetime.strptime(m.group(2), "%y%m%d").date()
        except ValueError:
            continue
        if expiry <= today:
            continue
        strike = int(m.group(4)) / 1000.0
        by_exp.setdefault(expiry, []).append((strike, float(delta)))
    if not by_exp:
        return None
    expiry = min(by_exp, key=lambda e: abs((e - today).days - config.CSP_TARGET_DTE_DAYS))
    strike, delta = min(by_exp[expiry], key=lambda sd: abs(sd[1] + config.CSP_TARGET_DELTA))
    return {"strike": strike, "delta": delta,
            "expiry": expiry.isoformat(), "dte": (expiry - today).days}


def fetch_cboe_csp_put(symbol: str) -> dict | None:
    """Fetch the delayed chain for one symbol and select the target put."""
    url = config.CBOE_OPTIONS_URL.format(symbol=symbol)
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        options = resp.json()["data"]["options"]
    except (requests.RequestException, ValueError, KeyError) as exc:
        log.warning("Cboe chain fetch failed for %s: %s", symbol, exc)
        return None
    selected = select_csp_put(options, now_et().date())
    if selected is None:
        log.warning("Cboe chain for %s contained no usable puts", symbol)
    return selected


def notify_cboe_failure(symbol: str, state: dict, detail: str) -> None:
    """Push a warning about a failed Cboe refresh, rate-limited per symbol by
    CBOE_FAIL_COOLDOWN_MINUTES (deliberately much longer than the technical
    signal cooldown — an outage is worth knowing about once, not every poll)."""
    if not should_fire(state, f"{symbol}:CBOE_FAIL", config.CBOE_FAIL_COOLDOWN_MINUTES):
        return
    notify(f"Screener warning: Cboe fetch failed ({symbol})",
           f"Could not refresh the Cboe options chain for {symbol}; {detail}. "
           f"Will keep retrying each poll (this warning repeats at most every "
           f"{config.CBOE_FAIL_COOLDOWN_MINUTES // 60}h).",
           "warning")


def get_csp_strike(symbol: str, closes: pd.Series | None, live_price: float,
                   state: dict) -> tuple[float, str]:
    """(strike, source description) for the ~30-delta/~30-day put. Prefers the
    real Cboe strike (cached CBOE_REFRESH_MINUTES; on a failed refresh the last
    good value is reused and retried next poll), else the BS approximation."""
    cached = cboe_cache.get(symbol)
    if cached is None or time.monotonic() - cached["ts"] >= config.CBOE_REFRESH_MINUTES * 60:
        selected = fetch_cboe_csp_put(symbol)
        if selected is not None:
            cached = {"ts": time.monotonic(), **selected}
            cboe_cache[symbol] = cached
        elif cached is not None:
            log.warning("Reusing previous Cboe strike for %s", symbol)
            notify_cboe_failure(symbol, state,
                                f"reusing the last good strike ({cached['strike']:.2f}, "
                                f"exp {cached['expiry']})")
        else:
            notify_cboe_failure(symbol, state,
                                "no previous strike available, so the value-zone check "
                                "is using the Black-Scholes estimate")
    if cached is not None:
        desc = (f"{abs(cached['delta']):.2f}-delta put, exp {cached['expiry']} "
                f"({cached['dte']}d, Cboe delayed)")
        return cached["strike"], desc
    strike = approx_csp_strike(closes, live_price) if closes is not None else math.nan
    return strike, (f"approx {config.CSP_TARGET_DELTA:.0%}-delta "
                    f"{config.CSP_TARGET_DTE_DAYS}d put (BS estimate)")


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------
def notify(title: str, message: str, tags: str) -> None:
    try:
        session.post(
            NTFY_URL,
            data=message.encode("utf-8"),
            headers={"Title": title, "Tags": tags},
            timeout=15,
        )
        log.info("ALERT [%s] %s", title, message)
    except requests.RequestException as exc:
        log.error("ntfy notification failed: %s", exc)


# ---------------------------------------------------------------------------
# Signal evaluation (NOTIFICATION ONLY — no trading of any kind)
# ---------------------------------------------------------------------------
def evaluate_technical(sym: str, quote: dict, closes: pd.Series | None,
                       spy_pct: float, state: dict) -> None:
    """CSP / buy-or-ATM-put / covered-call signals, EDGE-TRIGGERED: one push
    when a setup enters action territory, one when it leaves, and nothing in
    between while it holds. The three signals are tracked independently, so an
    already-active CSP that strengthens into a buy/ATM-put keeps its own state
    and is not re-announced."""
    if closes is None or len(closes) < 2:
        return
    live, prev_close = quote["price"], quote["prev_close"]
    stock_pct = (live - prev_close) / prev_close * 100
    ind = compute_indicators(closes)
    rsi = live_rsi(closes, live)

    red = stock_pct < 0 and spy_pct < 0
    green = stock_pct > 0 and spy_pct > 0

    rsi_lo, rsi_hi = config.CSP_RSI_RANGE
    csp = (red and not math.isnan(ind["ema50"]) and live <= ind["ema50"]
           and rsi_lo <= rsi <= rsi_hi)

    near_sma200 = (not math.isnan(ind["sma200"])
                   and abs(live - ind["sma200"]) / ind["sma200"] <= config.SMA200_TOLERANCE)
    buy_or_put = csp and near_sma200

    pct_b = math.nan
    if not math.isnan(ind["bb_upper"]) and ind["bb_upper"] != ind["bb_lower"]:
        pct_b = (live - ind["bb_lower"]) / (ind["bb_upper"] - ind["bb_lower"])
    cc = (green and not math.isnan(pct_b) and pct_b >= config.CC_PERCENT_B_MIN
          and rsi >= config.CC_RSI_MIN)

    day_ctx = f"{sym} {stock_pct:+.2f}%, SPY {spy_pct:+.2f}%"

    # Record all three edges first, so every signal's state stays current even
    # when another one is the thing being announced this poll.
    csp_edge = signal_transition(state, f"{sym}:CSP", csp)
    bop_edge = signal_transition(state, f"{sym}:BUY_OR_PUT", buy_or_put)
    cc_edge = signal_transition(state, f"{sym}:CC", cc)

    if bop_edge == "on":
        notify(
            f"Signal ON: {sym} BUY or ATM PUT",
            f"{sym} @ {live:.2f} — CSP conditions met AND within "
            f"{config.SMA200_TOLERANCE:.0%} of SMA200 ({ind['sma200']:.2f}). "
            f"Red day ({day_ctx}), price <= EMA50 ({ind['ema50']:.2f}), RSI {rsi:.1f}.",
            "moneybag",
        )
    elif bop_edge == "off":
        notify(
            f"Signal OFF: {sym} BUY or ATM PUT",
            f"{sym} @ {live:.2f} — buy/ATM-put setup no longer holds "
            f"({day_ctx}, RSI {rsi:.1f}).",
            "heavy_minus_sign",
        )

    if csp_edge == "on":
        notify(
            f"Signal ON: {sym} CSP",
            f"{sym} @ {live:.2f} — sell-CSP setup. Red day ({day_ctx}), "
            f"price <= EMA50 ({ind['ema50']:.2f}), RSI {rsi:.1f} in "
            f"[{rsi_lo}, {rsi_hi}].",
            "chart_with_downwards_trend",
        )
    elif csp_edge == "off":
        notify(
            f"Signal OFF: {sym} CSP",
            f"{sym} @ {live:.2f} — sell-CSP setup no longer holds "
            f"({day_ctx}, RSI {rsi:.1f}).",
            "heavy_minus_sign",
        )

    if cc_edge == "on":
        notify(
            f"Signal ON: {sym} Covered Call",
            f"{sym} @ {live:.2f} — covered-call setup. Green day ({day_ctx}), "
            f"%B {pct_b:.2f} >= {config.CC_PERCENT_B_MIN} "
            f"(band {ind['bb_lower']:.2f}–{ind['bb_upper']:.2f}), RSI {rsi:.1f}.",
            "chart_with_upwards_trend",
        )
    elif cc_edge == "off":
        notify(
            f"Signal OFF: {sym} Covered Call",
            f"{sym} @ {live:.2f} — covered-call setup no longer holds "
            f"({day_ctx}, RSI {rsi:.1f}).",
            "heavy_minus_sign",
        )


def evaluate_value_zone(sym: str, quote: dict, closes: pd.Series | None,
                        state: dict) -> None:
    """Fires once on the transition from outside to inside the zone, where
    'inside' means the live price OR the strike of a ~30-delta/~30-day CSP
    (real Cboe chain data, BS approximation as fallback) falls in the range."""
    zone = config.VALUE_ZONES.get(sym)
    if zone is None:
        return
    lo, hi = zone
    live = quote["price"]
    strike, strike_src = get_csp_strike(sym, closes, live, state)

    price_in = lo <= live <= hi
    strike_in = not math.isnan(strike) and lo <= strike <= hi
    in_zone = bool(price_in or strike_in)   # bool(): keep numpy out of the state file

    was_in = state["value_zone_in_range"].get(sym, False)
    state["value_zone_in_range"][sym] = in_zone

    if in_zone and not was_in:
        if price_in:
            detail = f"live price {live:.2f} entered {lo}-{hi}"
        else:
            detail = (f"CSP strike {strike:.2f} ({strike_src}) "
                      f"entered {lo}-{hi} (spot {live:.2f})")
        notify(f"Value Zone: {sym}", f"{sym}: {detail}", "dart")


# ---------------------------------------------------------------------------
# Poll cycle
# ---------------------------------------------------------------------------
def refresh_daily_cache(daily_cache: dict, credits: CreditTracker) -> None:
    """Fetch daily-bar history for any symbol whose cache is not from today.
    One time_series call per symbol per trading day."""
    today = et_date_str()
    for sym in config.WATCHLIST:
        cached = daily_cache.get(sym)
        if cached and cached["date"] == today:
            continue
        closes = fetch_daily_closes(sym, credits)
        if closes is not None and len(closes) > 0:
            daily_cache[sym] = {"date": today, "closes": closes}
            log.info("Daily history refreshed for %s (%d bars)", sym, len(closes))
        elif cached:
            log.warning("Daily refresh failed for %s; reusing %s bars", sym, cached["date"])
        else:
            log.warning("No daily history for %s; technical signals skipped", sym)


def run_poll_cycle(state: dict, daily_cache: dict, credits: CreditTracker) -> str:
    """One full evaluation pass. Returns 'ok', 'closed', or 'skipped'."""
    prune_alert_cooldowns(state)
    refresh_daily_cache(daily_cache, credits)

    symbols = list(dict.fromkeys(config.WATCHLIST + [config.MARKET_PROXY]))
    quotes = fetch_quotes(symbols, credits)

    spy = quotes.get(config.MARKET_PROXY)
    if spy is None:
        log.error("No quote for %s — cannot determine red/green day; skipping cycle",
                  config.MARKET_PROXY)
        return "skipped"
    if not spy["is_market_open"]:
        log.info("Twelve Data reports market closed (holiday or off-hours)")
        return "closed"
    spy_pct = (spy["price"] - spy["prev_close"]) / spy["prev_close"] * 100

    for sym in config.WATCHLIST:
        quote = quotes.get(sym)
        if quote is None:
            continue  # missing quote: leave zone state untouched, no evaluation
        cached = daily_cache.get(sym)
        closes = cached["closes"] if cached else None
        try:
            evaluate_technical(sym, quote, closes, spy_pct, state)
            evaluate_value_zone(sym, quote, closes, state)
        except Exception:
            log.exception("Evaluation failed for %s", sym)

    log.info("Cycle done — credits used today: %d/%d",
             credits.daily_used(), config.CREDITS_PER_DAY)
    return "ok"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    if "--test-notify" in sys.argv:
        notify("Screener test", "Test notification — ntfy pipeline works.", "white_check_mark")
        return
    if not API_KEY:
        sys.exit("TWELVEDATA_API_KEY is not set. Put it in a .env file next to "
                 "this script (TWELVEDATA_API_KEY=...) or export it as an "
                 "environment variable.")
    log.info("Screener starting — %d tickers + %s, polling every %d min, "
             "ntfy topic %r", len(config.WATCHLIST), config.MARKET_PROXY,
             config.POLL_INTERVAL_MINUTES, NTFY_TOPIC)

    state = load_state()
    credits = CreditTracker(state)
    daily_cache: dict[str, dict] = {}
    poll_seconds = config.POLL_INTERVAL_MINUTES * 60

    while True:
        now = now_et()
        if not is_market_hours(now):
            wait = seconds_until_next_open(now)
            log.info("Market closed — sleeping %.1f hours until next open", wait / 3600)
            time.sleep(wait + 30)  # small buffer past 9:30
            continue

        started = time.monotonic()
        try:
            result = run_poll_cycle(state, daily_cache, credits)
        except Exception:
            log.exception("Poll cycle crashed; will retry next interval")
            result = "skipped"
        finally:
            try:
                save_state(state)
            except Exception:
                # Deliberately broad: a bad value in the state dict must never
                # be able to kill the polling loop from inside this `finally`.
                log.exception("Could not save state; continuing")
            # Drop idle keep-alive sockets before the long sleep; servers close
            # them anyway, and reusing a dead one is what triggers WinError 10054.
            session.close()

        if result == "closed":
            time.sleep(3600)  # likely a holiday; re-check hourly, cheap on credits
            continue
        elapsed = time.monotonic() - started
        time.sleep(max(poll_seconds - elapsed, 30))


if __name__ == "__main__":
    main()
