# Intraday Ticker Screener (notification-only)

Polls Twelve Data during market hours (9:30am–4:00pm ET, weekdays), evaluates
technical + value-zone conditions per ticker, and pushes alerts to ntfy.sh.
**It never places, submits, or executes trades — it only reads data and notifies.**

## Setup

```
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and put your Twelve Data API key in it:

```
TWELVEDATA_API_KEY=your_key_here
```

(Alternatively set `TWELVEDATA_API_KEY` as an environment variable. The ntfy
topic defaults to the one in `config.py`; override with `NTFY_TOPIC` in `.env`.)

Subscribe to the topic in the ntfy app (or https://ntfy.sh/aarush-ticker-dKRa43w1
in a browser) to receive alerts.

## Run

```
python screener.py
```

It's a long-running loop: sleeps outside market hours, wakes at the next open,
and polls every `POLL_INTERVAL_MINUTES` (default 15) while the market is open.
On holidays the quote endpoint reports the market closed and the script
re-checks hourly instead of polling at full rate.

### Keeping it running

- **Simplest:** leave `python screener.py` running in a terminal (or on Windows,
  a background start via Task Scheduler at logon / a `pythonw` shortcut).
- **Windows Task Scheduler:** trigger daily at ~9:25 AM *your local time
  adjusted to ET* and start `python C:\...\screener.py`. The script itself
  checks ET market hours, so an early start is harmless — it just sleeps.
- **cron (Linux/macOS):** e.g. `25 9 * * 1-5` to launch it each weekday.
  **Caveat: cron uses the machine's local timezone, not ET.** Either set
  `CRON_TZ=America/New_York` at the top of the crontab (where supported) or
  shift the hour to your local equivalent of 9:25 ET. Since the script sleeps
  until market open on its own, starting it once and leaving it running is the
  least error-prone option.

## Editing the watchlist / zones / thresholds

Everything lives in `config.py` — watchlist tickers, SPY proxy, value-zone
price ranges, RSI bands, %B cutoff, SMA200 tolerance, polling interval, and
credit caps. No logic in that file; edit and restart.

## Credit budget (free tier: 8 credits/min, 800/day)

- Daily-bar refresh: 1 `time_series` credit per watchlist symbol per trading
  day = 12.
- Each poll: 1 `quote` credit per symbol (batched, chunked into groups of 8 so
  a single chunk never exceeds the 8/min limit) = 13 per poll.
- 26 polls/day × 13 + 12 ≈ **350 credits/day**, well under 800.

The script tracks daily usage in `screener_state.json` (survives restarts),
blocks within the per-minute window, and stops calling the API entirely once
`DAILY_CREDIT_SOFT_CAP` (760) is reached.

## State

`screener_state.json` (auto-created, gitignored) stores:

- last-fired timestamps per ticker+signal. Technical signals **re-alert every
  poll while their condition stays true**; raise
  `TECHNICAL_SIGNAL_COOLDOWN_MINUTES` in `config.py` to throttle (0 = every
  poll, 60 = hourly, 1440 = once per rolling 24h),
- per-ticker "currently in value zone" booleans (zone alerts fire only on the
  outside→inside transition, across polls and across days),
- today's credit usage.

Delete the file to reset all alert state.

## Notes / approximations

- Indicators (EMA50, SMA200, RSI-14 Wilder, Bollinger 20/±2σ) are computed
  manually in pandas from **closed** daily bars; the live quote is compared
  against those levels, and RSI's final value is recomputed with the live
  price standing in for today's unfinished bar.
- The "~30-delta, ~30-day CSP strike" for value-zone alerts comes from
  **Cboe's free delayed options chain**
  (`cdn.cboe.com/api/global/delayed_quotes/options/{SYMBOL}.json` — real
  strikes and greeks, 15-min delayed, no API key, and it doesn't consume
  Twelve Data credits). The script picks the expiration nearest 30 DTE and the
  put whose delta is nearest −0.30, caching the result per symbol for
  `CBOE_REFRESH_MINUTES` (default 60). Only tickers with a value zone are
  fetched. The 15-minute delay is immaterial here — the target strike drifts
  slowly. Caveat: this is a public data feed, not a documented API; if it ever
  fails, the script logs a warning, sends an ntfy warning push (at most once
  per symbol per 24h, `CBOE_FAIL_COOLDOWN_MINUTES`), reuses the last good
  strike, and otherwise falls back to a
  Black-Scholes estimate using 20-day realized volatility as an IV proxy
  (`approx_csp_strike` in `screener.py`).
- `SPCX` and `CRWV` may not resolve on Twelve Data (SPCX in particular is not
  an active US listing). Unresolvable symbols are logged and skipped each
  cycle without crashing anything — remove them from `config.py` if the log
  noise bothers you.
