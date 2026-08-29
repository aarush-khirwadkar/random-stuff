"""
Configuration for the intraday ticker screener.

Everything you'd want to tweak lives in this file — tickers, value zones,
thresholds, polling interval. No logic here; edit freely.

The Twelve Data API key and the ntfy topic are read from the environment
(or a local .env file next to the script):
    TWELVEDATA_API_KEY=...   (required)
    NTFY_TOPIC=...           (optional, defaults to NTFY_TOPIC_DEFAULT below)
"""

# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------
WATCHLIST = [
    "AMZN", "MSFT", "GOOG", "AAPL", "NVDA", "TSLA",
    "SPCX", "HOOD", "PLTR", "SOXL", "CRWV", "SMCI", "INTC", "META"
]

# Market proxy used for the red-day / green-day comparison
MARKET_PROXY = "SPY"

# ---------------------------------------------------------------------------
# Value zones — alert once when the live price (or the approximate ~30-delta,
# ~30-day CSP strike) ENTERS one of these [low, high] price ranges.
# Add/remove/edit lines freely; tickers not listed here are simply skipped.
# ---------------------------------------------------------------------------
VALUE_ZONES = {
    "AAPL": (250, 260),
    "AMZN": (210, 220),
    "TSLA": (370, 390),
    "NVDA": (170, 190),
    "SOXL": (70, 80),
    "HOOD": (70, 75),
    "PLTR": (120, 125),
    "GOOG": (320, 340),
    "MSFT": (380, 400),
}

# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------
POLL_INTERVAL_MINUTES = 60   # lower this if you move to a paid Twelve Data tier

# ---------------------------------------------------------------------------
# Indicator parameters
# ---------------------------------------------------------------------------
EMA_PERIOD = 50
SMA_PERIOD = 200
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0

# ---------------------------------------------------------------------------
# Signal thresholds
# ---------------------------------------------------------------------------
CSP_RSI_RANGE = (30, 50)     # RSI must be inside [lo, hi] for the CSP signal
SMA200_TOLERANCE = 0.02      # "within 2% of the 200-day SMA" for buy/ATM-put
CC_PERCENT_B_MIN = 0.80      # %B threshold for the covered-call signal
CC_RSI_MIN = 60              # RSI threshold for the covered-call signal

# Technical signals are EDGE-TRIGGERED: one push when a setup becomes true and
# one when it stops being true, nothing while it simply stays true. There is no
# repeat/cooldown knob for them by design.

# Cooldown for the "Cboe fetch failed" warning. Kept long on purpose: a data
# source outage is worth knowing about once, not every poll.
CBOE_FAIL_COOLDOWN_MINUTES = 1440

# ---------------------------------------------------------------------------
# ~30-delta / ~30-day put strike (used only for the value-zone check).
# Primary source: Cboe's free delayed-quotes chain (real strikes + greeks,
# 15-min delayed, no API key, separate from the Twelve Data credit budget).
# Fallback when Cboe is unavailable: Black-Scholes inversion using realized
# volatility as an IV proxy.
# ---------------------------------------------------------------------------
CSP_TARGET_DELTA = 0.30      # absolute put delta
CSP_TARGET_DTE_DAYS = 30     # days to expiration
CBOE_OPTIONS_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
CBOE_REFRESH_MINUTES = 60    # refresh each ticker's chain at most this often
RISK_FREE_RATE = 0.04        # annualized, for the fallback approximation
REALIZED_VOL_LOOKBACK = 20   # trading days of returns for fallback realized vol

# ---------------------------------------------------------------------------
# Twelve Data limits (free tier: 8 credits/min, 800 credits/day)
# ---------------------------------------------------------------------------
DAILY_BARS = 250             # daily bars to request (need >= 200 for SMA200)
CREDITS_PER_MINUTE = 8
CREDITS_PER_DAY = 800
DAILY_CREDIT_SOFT_CAP = 760  # stop making calls once this many are spent

# ---------------------------------------------------------------------------
# Notifications (ntfy.sh)
# ---------------------------------------------------------------------------
NTFY_SERVER = "https://ntfy.sh"
NTFY_TOPIC_DEFAULT = "aarush-ticker-dKRa43w1"   # override with NTFY_TOPIC env var

# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
STATE_FILE = "screener_state.json"
