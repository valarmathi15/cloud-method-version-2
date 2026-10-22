"""
live_m2.py  -  LIVE PAPER TRADING : Method 2 (Cloud Scale-In, Google-sheet logic)   [single file]
=====================================================================================
BUILDER   : pick folder, exchange, symbols, candle timeframe, capital, qty, Max lots (many) and
            Target % (many)  ->  press Start  ->  ONE FILE is saved for every symbol x lots x target
            (e.g. folder "Nifty test":  Nifty 50 4H 10lots 1%.json,  ... 10lots 2%.json, ...)
MY STRATEGIES : open a folder -> summary table of all its files, all running live side by side;
            or open any file(s) to see that strategy's live position, chart, bar log and trades.

How "live paper" works: Method 2 is fully determined by the candles from the strategy START time,
so every time you open / refresh, the state is rebuilt from Angel One candles (closed candles only).
No background process is needed for the numbers. (ntfy push alerts need the app to be open.)

Run : python -m streamlit run live_m2.py
Login details (API key, Client ID, MPIN, TOTP secret) are typed on the first screen.
Strategy files are saved in the  strategies/<ClientID>/<folder>/  folder next to this file.
NOTE: on Streamlit Cloud that folder is wiped when the app reboots - run it on your own PC to keep files.
"""
import datetime as dt
import json
import os
import re
import shutil
import threading
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
STRAT_ROOT = os.path.join(HERE, 'strategies')
STATE_FILE = os.path.join(HERE, 'last_signal.json')

# timeframe -> (Angel interval, raw bar length, warm-up calendar days needed for the cloud)
def now_ist():
    """current India time as a naive Timestamp - the server (e.g. Streamlit Cloud) may run in UTC, Angel candles are IST."""
    return pd.Timestamp.now(tz='Asia/Kolkata').tz_localize(None)


def ts_ist(epoch):
    return pd.Timestamp(epoch, unit='s', tz='UTC').tz_convert('Asia/Kolkata').tz_localize(None)


TF = {
    '1D':  ('ONE_DAY',        pd.Timedelta(days=1),      260),
    '4H':  ('ONE_HOUR',       pd.Timedelta(hours=1),     160),   # 4H = 4 hourly bars, built per day
    '1H':  ('ONE_HOUR',       pd.Timedelta(hours=1),     100),
    '30m': ('THIRTY_MINUTE',  pd.Timedelta(minutes=30),  25),
    '15m': ('FIFTEEN_MINUTE', pd.Timedelta(minutes=15),  14),
    '10m': ('TEN_MINUTE',     pd.Timedelta(minutes=10),  10),
    '5m':  ('FIVE_MINUTE',    pd.Timedelta(minutes=5),   7),
    '3m':  ('THREE_MINUTE',   pd.Timedelta(minutes=3),   5),
    '1m':  ('ONE_MINUTE',     pd.Timedelta(minutes=1),   5),
}
# max days Angel returns per getCandleData request (kept a bit under the real limit)
INTERVAL_MAX_DAYS = {'ONE_DAY': 1900, 'ONE_HOUR': 380, 'THIRTY_MINUTE': 190, 'FIFTEEN_MINUTE': 190,
                     'TEN_MINUTE': 95, 'FIVE_MINUTE': 95, 'THREE_MINUTE': 55, 'ONE_MINUTE': 28}
LOTS_OPTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
TARGET_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SPAN_OPTIONS = {'SPAN A': 'A', 'SPAN B': 'B', 'SPAN A & B': 'A&B', 'SPAN A OR B': 'AorB'}
EXIT_OPTIONS = TARGET_OPTIONS + list(SPAN_OPTIONS)               # the 14 options of "Profit target / Span exit"
AMOUNT_OPTIONS = [1000, 5000, 10000, 25000, 50000, 100000, 200000, 500000]
QTY_OPTIONS = [1, 2, 5, 10, 25, 50, 75, 100, 150, 200, 500, 1000]
CANDLE_OPTIONS = {'Every candle': 'every', 'Daily': 'daily', 'Weekly': 'weekly'}


def exit_label(x):
    return f'{x}%' if isinstance(x, (int, float)) else x


def exit_text(s):
    return s.get('exit_label') or f"{s['profit']}%"


def candle_text(s):
    return {v: k for k, v in CANDLE_OPTIONS.items()}.get(s.get('candle', 'every'), 'Every candle')


# =============================================================== ENGINE (M2 only, copied from m.py - same logic)
TENKAN_P, KIJUN_P, SENKOU_B_P, DISPLACEMENT = 9, 26, 52, 26


def add_ichimoku_cloud(df):
    df = df.copy()
    high, low = df['High'], df['Low']
    tenkan = (high.rolling(TENKAN_P).max() + low.rolling(TENKAN_P).min()) / 2
    kijun = (high.rolling(KIJUN_P).max() + low.rolling(KIJUN_P).min()) / 2
    df['SenkouA'] = ((tenkan + kijun) / 2).shift(DISPLACEMENT)
    df['SenkouB'] = ((high.rolling(SENKOU_B_P).max() + low.rolling(SENKOU_B_P).min()) / 2).shift(DISPLACEMENT)
    df['CloudTop'] = df[['SenkouA', 'SenkouB']].max(axis=1, skipna=False)
    df['CloudBottom'] = df[['SenkouA', 'SenkouB']].min(axis=1, skipna=False)
    df['AboveCloud'] = df['Close'] > df['CloudTop']
    return df


SPAN_RULES = {                       # exit rule for the "Span" options (price = bar Close)
    'A':    lambda p, a, b: p < a,
    'B':    lambda p, a, b: p < b,
    'A&B':  lambda p, a, b: p < a and p < b,      # below BOTH spans  (= below the cloud bottom)
    'AorB': lambda p, a, b: p < a or p < b,       # below EITHER span (= below the cloud top)
}
SPAN_TEXT = {'A': 'Span A', 'B': 'Span B', 'A&B': 'Span A & B', 'AorB': 'Span A or B'}


def run_m2(df, profit_target_pct, max_lots, start_date=None, lot_qty=1.0, trace=None, span=None, candle='every', lot_amount=None):
    """Method 2 (Google-sheet logic).
    span   : None = exit on Profit target %;  'A' / 'B' / 'A&B' / 'AorB' = exit when Close falls below that span
             (whole position closed; span exits go flat, no repeat exits).
    lot_amount : None = every lot buys `lot_qty` units;  number = every lot invests this AMOUNT (units = amount / lot price).
    candle : 'every' = buy 1 lot on every candle above the cloud;  'daily' = at most 1 lot per day;
             'weekly' = at most 1 lot per week (first candle of the period that closes above the cloud).
    Returns trades DataFrame; trace (list) gets one dict per bar."""
    full = add_ichimoku_cloud(df)
    if start_date is not None:   # cloud uses full history, simulation starts flat from start_date
        full = full[full['Date'] >= pd.Timestamp(start_date)].reset_index(drop=True)
    rows = full.to_dict('records')
    span_hit = SPAN_RULES.get(span)

    def pkey(d):
        if candle == 'daily':
            return d.date()
        if candle == 'weekly':
            y, w, _ = d.isocalendar()
            return (y, w)
        return None

    exit_reason = (f'Closed below {SPAN_TEXT[span]} (Method 2)' if span_hit
                   else f'Profit target {profit_target_pct}% hit (Method 2 - sheet logic)')
    trades = []
    trade_id = 0
    m2_first, m2_lots, m2_avg, m2_prev_exit, m2_seg_start = True, 0, None, False, None
    m2_qty, m2_cost = 0.0, 0.0                     # units held and money spent (avg price = cost / units)
    last_buy_key = None

    for row in rows:
        if pd.isna(row['CloudTop']) or pd.isna(row['CloudBottom']):
            continue
        price, date = row['Close'], row['Date']
        above = bool(row['AboveCloud'])
        bought, prev_lots = False, 0
        if m2_first:
            m2_first = False
            m2_lots = 1 if above else 0
            bought = m2_lots == 1
            if m2_lots == 1:
                last_buy_key = pkey(date)
        else:
            prev_lots, bought = m2_lots, False
            can_buy = candle == 'every' or pkey(date) != last_buy_key
            if not above or not can_buy:
                pass
            elif m2_prev_exit:
                m2_lots, bought = 1, True
            elif m2_lots < max_lots:
                m2_lots, bought = m2_lots + 1, True
            if bought:
                last_buy_key = pkey(date)
        if bought:                                         # avg price changes ONLY when a lot is bought
            q = (lot_amount / price) if lot_amount else lot_qty
            if prev_lots == 0:
                m2_qty, m2_cost = q, q * price
            else:
                m2_qty, m2_cost = m2_qty + q, m2_cost + q * price
            m2_avg = m2_cost / m2_qty
        m2_prev_exit = False
        if m2_seg_start is None and m2_lots > 0:
            m2_seg_start = date
        if m2_lots > 0 and m2_avg:
            if span_hit:
                hit = span_hit(price, row['SenkouA'], row['SenkouB'])
            else:
                hit = (price - m2_avg) / m2_avg >= profit_target_pct / 100.0
            if hit:
                trade_id += 1
                trades.append({
                    'Trade ID': trade_id, 'Entry Date (1st lot)': m2_seg_start, 'Exit Date': date,
                    'Holding Days': (date - m2_seg_start).days, 'Lots Filled': m2_lots,
                    'Max Lots Allowed': max_lots, 'Avg Entry Price': round(m2_avg, 4),
                    'Exit Price': round(price, 4),
                    'Profit % (on avg entry)': round((price - m2_avg) / m2_avg * 100, 2),
                    'PnL (pts x qty)': round((price - m2_avg) * m2_qty, 2),
                    'Exit Reason': exit_reason})
                m2_prev_exit = True
                m2_seg_start = None
                m2_lots, m2_avg, m2_qty, m2_cost = 0, None, 0.0, 0.0   # exit = flat; fresh trade on the next close above cloud
        if trace is not None:
            trace.append({'Date': date, 'Close': price, 'CloudTop': row['CloudTop'],
                          'CloudBottom': row['CloudBottom'], 'Above': above, 'Lots': m2_lots, 'Bought': bought, 'Qty': m2_qty,
                          'AvgPrice': None if not m2_avg else round(m2_avg, 4),
                          'Action': 'EXIT' if m2_prev_exit else ''})

    if m2_lots > 0 and m2_avg and not m2_prev_exit and m2_seg_start is not None and rows:
        price, date = full.iloc[-1]['Close'], full.iloc[-1]['Date']
        trade_id += 1
        trades.append({
            'Trade ID': trade_id, 'Entry Date (1st lot)': m2_seg_start, 'Exit Date': date,
            'Holding Days': (date - m2_seg_start).days, 'Lots Filled': m2_lots,
            'Max Lots Allowed': max_lots, 'Avg Entry Price': round(m2_avg, 4),
            'Exit Price': round(price, 4),
            'Profit % (on avg entry)': round((price - m2_avg) / m2_avg * 100, 2),
            'PnL (pts x qty)': round((price - m2_avg) * m2_qty, 2),
            'Exit Reason': 'Open at end of data (marked to last close)'})
    return pd.DataFrame(trades)


# =============================================================== DATA FEED (Angel One SmartAPI)
SCRIP_URL = 'https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json'


def angel_login(client_id, api_key, mpin, totp_secret):
    """Login with Client ID + MPIN + TOTP (the 6-digit code is generated from your TOTP secret)."""
    import pyotp
    from SmartApi import SmartConnect
    smart = SmartConnect(api_key=api_key)
    res = smart.generateSession(client_id, mpin, pyotp.TOTP(totp_secret.replace(' ', '')).now())
    if not res or not res.get('status'):
        raise ValueError((res or {}).get('message', 'login failed'))
    return smart


@st.cache_resource(ttl=86400, show_spinner='Loading Angel One instrument list ...')
def _scrip_master():
    return requests.get(SCRIP_URL, timeout=120).json()


def _instrument_token(exchange, symbol):
    sym = symbol.strip().upper()
    hits = [r for r in _scrip_master() if r['exch_seg'] == exchange
            and r['symbol'].upper() in (sym, sym + '-EQ')]
    if not hits:
        raise ValueError(f'{symbol} not found on {exchange}  (equity: SBIN-EQ / index: Nifty 50 / MCX: GOLDM05DEC25FUT)')
    hits.sort(key=lambda r: r['symbol'].upper() != sym)          # exact match first
    return hits[0]['token']


_RATE_LOCK = threading.Lock()
_LAST_CALL = [0.0]
MIN_GAP = 0.7            # seconds between two Angel candle requests (Angel allows ~3/sec; we stay well under)
RECHECK_SECS = 45        # an already-loaded symbol is only topped up with new candles this often


def _is_rate_msg(msg):
    m = str(msg).lower()
    return 'access rate' in m or 'exceeding' in m or 'too many' in m or 'rate limit' in m


def _call_candle_api(smart, params):
    """one getCandleData call: spaced out + retried with back-off when Angel says 'exceeding access rate'."""
    last = 'candle request failed'
    for i in range(5):
        with _RATE_LOCK:
            wait = MIN_GAP - (time.time() - _LAST_CALL[0])
            if wait > 0:
                time.sleep(wait)
            _LAST_CALL[0] = time.time()
        try:
            res = smart.getCandleData(params)
        except Exception as e:
            last = str(e)
            if _is_rate_msg(last):
                time.sleep(2 * (i + 1))
                continue
            raise
        if res and res.get('status'):
            return res
        last = (res or {}).get('message', 'candle request failed')
        if _is_rate_msg(last):
            time.sleep(2 * (i + 1))
            continue
        raise ValueError(last)
    raise ValueError(f'Angel rate limit: {last}')


def _fetch_range(smart, exchange, token, interval, frm, to):
    step = pd.Timedelta(days=INTERVAL_MAX_DAYS.get(interval, 28))      # Angel per-request limits
    rows, cur = [], frm
    while cur < to:
        end = min(cur + step, to)
        res = _call_candle_api(smart, {'exchange': exchange, 'symboltoken': token, 'interval': interval,
                                       'fromdate': cur.strftime('%Y-%m-%d %H:%M'),
                                       'todate': end.strftime('%Y-%m-%d %H:%M')})
        rows += res.get('data') or []
        cur = end + pd.Timedelta(minutes=1)
    if not rows:
        return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close'])
    df = pd.DataFrame(rows, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'])
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)
    return df[['Date', 'Open', 'High', 'Low', 'Close']]


def fetch_candles(smart, exchange, symbol, tf, start_date):
    """-> raw candles (Date, Open, High, Low, Close), tz-naive IST, oldest first."""
    interval, _, warm_days = TF[tf]
    token = _instrument_token(exchange, symbol)
    df = _fetch_range(smart, exchange, token, interval, pd.Timestamp(start_date) - pd.Timedelta(days=warm_days), now_ist())
    if df.empty:
        raise ValueError('no candles returned')
    return df.drop_duplicates('Date', keep='last').sort_values('Date').reset_index(drop=True)


@st.cache_resource
def _candle_store():
    return {'lock': threading.Lock(), 'data': {}}


def load_candles(smart, exchange, symbol, tf, start_day):
    """History is downloaded ONCE per symbol/timeframe; after that only the newest candles are fetched
    (1 small request, at most every RECHECK_SECS) -> no more 'exceeding access rate'."""
    S = _candle_store()
    key = (exchange, symbol, tf, start_day)
    with S['lock']:
        e = S['data'].get(key)
        if e is None:
            e = S['data'][key] = {'df': fetch_candles(smart, exchange, symbol, tf, start_day),
                                  'checked': time.time(), 'error': None}
        elif time.time() - e['checked'] >= RECHECK_SECS:
            try:
                interval = TF[tf][0]
                back = pd.Timedelta(days=5 if interval == 'ONE_DAY' else 2)
                new = _fetch_range(smart, exchange, _instrument_token(exchange, symbol), interval,
                                   e['df']['Date'].iloc[-1] - back, now_ist())
                if not new.empty:
                    df = pd.concat([e['df'], new]).drop_duplicates('Date', keep='last').sort_values('Date')
                    e['df'] = df.reset_index(drop=True)
                e['error'] = None
            except Exception as ex:                         # keep showing the last good data, retry next refresh
                e['error'] = str(ex)
            e['checked'] = time.time()
        return e['df'].copy(), e['checked'], e['error']


# =============================================================== candle preparation
def closed_only(df, delta, close_t, now):
    """drop the candle that has not finished yet (its end = start+length, capped at market close)."""
    sess_end = df['Date'].dt.normalize() + pd.Timedelta(hours=close_t.hour, minutes=close_t.minute)
    end = pd.concat([df['Date'] + delta, sess_end], axis=1).min(axis=1)
    return df[end <= now].reset_index(drop=True)


def to_4h(h, close_t):
    """hourly -> 4H: per day, bars 1-4 = first 4H candle, the rest = second (e.g. NSE 09:15 + 13:15)."""
    h = h.copy()
    h['day'] = h['Date'].dt.normalize()
    h['n'] = h.groupby('day').cumcount() // 4
    out = (h.groupby(['day', 'n'])
             .agg(Date=('Date', 'first'), Open=('Open', 'first'), High=('High', 'max'),
                  Low=('Low', 'min'), Close=('Close', 'last'), cnt=('Close', 'size'), last=('Date', 'last'))
             .reset_index(drop=True))
    day_close = out['Date'].dt.normalize() + pd.Timedelta(hours=close_t.hour, minutes=close_t.minute)
    ok = (out['cnt'] == 4) | ((out['last'] + pd.Timedelta(hours=1)) >= day_close)
    return out[ok][['Date', 'Open', 'High', 'Low', 'Close']].reset_index(drop=True)


def prepare(raw, tf, close_t, now):
    _, delta, _ = TF[tf]
    df = closed_only(raw, delta, close_t, now)
    return to_4h(df, close_t) if tf == '4H' else df


# =============================================================== M2 live state
def compute_m2(df, profit_pct, max_lots, start, qty=1.0, span=None, candle='every', amount=None):
    trace = []
    trades = run_m2(df, profit_pct, max_lots, start_date=start, lot_qty=qty, trace=trace, span=span, candle=candle, lot_amount=amount)
    tr = pd.DataFrame(trace)
    if tr.empty:
        return tr, trades
    prev_lots = tr['Lots'].shift(fill_value=0)
    prev_exit = tr['Action'].shift(fill_value='') == 'EXIT'
    buy = tr['Bought'].astype(bool)
    exit_ = tr['Action'] == 'EXIT'
    tr['Signal'] = np.select([buy & exit_, exit_, buy], ['BUY+EXIT', 'EXIT', 'BUY'], 'HOLD')
    tr['Unrealised %'] = np.where(tr['AvgPrice'].notna(),
                                  (tr['Close'] - tr['AvgPrice']) / tr['AvgPrice'] * 100, np.nan).round(2)
    return tr, trades


# =============================================================== ntfy (only on a NEW signal)
def notify_new_signal(topic, key, last_row):
    state = {}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
    bar = str(last_row['Date'])
    first_time = key not in state
    if state.get(key) != bar:
        state[key] = bar
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        if not first_time and topic and last_row['Signal'] != 'HOLD':     # first run = baseline only
            msg = (f"{key}: {last_row['Signal']} @ {last_row['Close']:.2f}  |  lots {int(last_row['Lots'])}"
                   f"  avg {last_row['AvgPrice']}  |  bar {bar}")
            try:
                requests.post(f'https://ntfy.sh/{topic}', data=msg.encode('utf-8'),
                              headers={'Title': 'Method 2 signal'}, timeout=10)
            except Exception as e:
                st.warning(f'ntfy failed: {e}')


def get_smart():
    """Logged-in SmartConnect for THIS browser session (re-logs in automatically if it expired)."""
    c = st.session_state['creds']
    if 'smart' not in st.session_state:
        st.session_state['smart'] = angel_login(c['client_id'], c['api_key'], c['mpin'], c['totp'])
    return st.session_state['smart']


def relogin():
    st.session_state.pop('smart', None)
    return get_smart()


# =============================================================== strategy files (folder / file)
def safe(s):
    return re.sub(r'[^A-Za-z0-9 _.%\-]', '_', str(s)).strip() or 'x'


def user_dir():
    return os.path.join(STRAT_ROOT, safe(st.session_state['creds']['client_id']))


def list_folders():
    d = user_dir()
    return sorted(f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))) if os.path.isdir(d) else []


def delete_folder(folder):
    d = os.path.join(user_dir(), safe(folder))
    if os.path.isdir(d):
        shutil.rmtree(d)


def list_files(folder):
    d = os.path.join(user_dir(), safe(folder))
    return sorted(f for f in os.listdir(d) if f.endswith('.json')) if os.path.isdir(d) else []


def load_strategy(folder, fname):
    with open(os.path.join(user_dir(), safe(folder), fname), encoding='utf-8') as f:
        return json.load(f)


def delete_strategy(folder, fname):
    os.remove(os.path.join(user_dir(), safe(folder), fname))


def create_strategies(folder, prefix, exchange, symbols, tfs, capital, sizes, close, lots_list, exits, candles, start):
    """one file per symbol x timeframe x lots x size x exit option x candle mode. Existing file is NOT overwritten (keeps its start time)."""
    d = os.path.join(user_dir(), safe(folder))
    os.makedirs(d, exist_ok=True)
    made, skipped = [], []
    for sym in symbols:
      for tf in tfs:
        for lots in lots_list:
          for kind, val in sizes:
            for ex in exits:
                for cd in candles:
                    cdc = CANDLE_OPTIONS[cd]
                    name = (f"{prefix + ' ' if prefix else ''}{sym} {tf} {lots}lots {exit_label(ex)}"
                            + ('' if cdc == 'every' else f' {cdc}')
                            + (f' amt{val:g}' if kind == 'amt' else (f' q{val:g}' if len(sizes) > 1 else '')))
                    path = os.path.join(d, safe(name) + '.json')
                    if os.path.exists(path):
                        skipped.append(name)
                        continue
                    is_span = ex in SPAN_OPTIONS
                    s = {'name': name, 'folder': folder, 'mode': 'paper', 'exchange': exchange, 'symbol': sym,
                         'tf': tf, 'profit': None if is_span else ex, 'span': SPAN_OPTIONS.get(ex),
                         'exit_label': exit_label(ex), 'candle': cdc, 'lots': lots, 'capital': capital,
                         'qty': val if kind == 'qty' else 1.0, 'amount': val if kind == 'amt' else None, 'close': close, 'start': start,
                         'created': now_ist().to_pydatetime().strftime('%Y-%m-%d %H:%M')}
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(s, f, indent=1)
                    made.append(name)
    return made, skipped


# =============================================================== evaluate one saved strategy
def get_candles(s):
    """-> (raw candles, time of last data check, error text or None)"""
    day = pd.Timestamp(s['start']).normalize().strftime('%Y-%m-%d')
    try:
        return load_candles(get_smart(), s['exchange'], s['symbol'], s['tf'], day)
    except Exception as e:
        if _is_rate_msg(e):
            raise
        return load_candles(relogin(), s['exchange'], s['symbol'], s['tf'], day)   # session expired?


def evaluate(s):
    """-> dict with tr (bar log), trades and the numbers shown in the summary."""
    h, m = [int(x) for x in s['close'].split(':')]
    raw, checked, data_err = get_candles(s)
    df = prepare(raw, s['tf'], dt.time(h, m), now_ist())
    tr, trades = compute_m2(df, s.get('profit') or 0.0, int(s['lots']), s['start'], s['qty'],
                            span=s.get('span'), candle=s.get('candle', 'every'), amount=s.get('amount'))
    out = {'s': s, 'tr': tr, 'trades': trades, 'status': 'Waiting for 1st candle', 'last': None,
           'checked': checked, 'data_err': data_err,
           'closed_n': 0, 'closed_pnl': 0.0, 'open_pnl': 0.0, 'invested': 0.0}
    if tr.empty:
        return out
    last = tr.iloc[-1]
    flat = last['Signal'] in ('EXIT', 'BUY+EXIT') or int(last['Lots']) == 0 or pd.isna(last['AvgPrice'])
    openrows = trades[trades['Exit Reason'].str.startswith('Open at end')] if not trades.empty else trades
    closed = trades.drop(openrows.index) if not trades.empty else trades
    out.update(last=last, flat=flat, closed_n=len(closed),
               closed_pnl=float(closed['PnL (pts x qty)'].sum()) if len(closed) else 0.0,
               open_pnl=0.0 if flat or openrows.empty else float(openrows['PnL (pts x qty)'].sum()),
               invested=0.0 if flat else float(last['Qty']) * float(last['AvgPrice']),
               status='Flat' if flat else f"Holding {int(last['Lots'])} lots")
    return out


def money(x):
    return f"{x:,.0f}"


# =============================================================== pages
def checkbox_dropdown(title, options, default, key, fmt=str):
    """Drop-down with a check box beside every option + Select all. Returns the ticked options."""
    base = f'cb_{key}'
    for o in options:
        if f'{base}_{o}' not in st.session_state:
            st.session_state[f'{base}_{o}'] = o in default

    def _toggle_all():
        v = st.session_state[f'{base}__all']
        for o in options:
            st.session_state[f'{base}_{o}'] = v

    chosen = [o for o in options if st.session_state[f'{base}_{o}']]
    st.session_state[f'{base}__all'] = len(chosen) == len(options)
    with st.popover(f'{title} · {len(chosen)} selected', width='stretch'):
        st.checkbox('Select all', key=f'{base}__all', on_change=_toggle_all)
        for o in options:
            st.checkbox(fmt(o), key=f'{base}_{o}')
    return chosen


NIFTY50_STOCKS = ['ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE',
                  'BAJAJFINSV', 'BEL', 'BHARTIARTL', 'CIPLA', 'COALINDIA', 'DRREDDY', 'EICHERMOT', 'ETERNAL',
                  'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'INDIGO',
                  'INFY', 'ITC', 'JIOFIN', 'JSWSTEEL', 'KOTAKBANK', 'LT', 'M&M', 'MARUTI', 'NESTLEIND', 'NTPC',
                  'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN', 'SHRIRAMFIN', 'SUNPHARMA', 'TATACONSUMER',
                  'TATAMOTORS', 'TATASTEEL', 'TCS', 'TECHM', 'TITAN', 'TRENT', 'ULTRACEMCO', 'WIPRO']


@st.cache_data(ttl=86400, show_spinner='Loading instrument list ...')
def instrument_choices(exchange):
    """{'Index': [...], 'Stocks': [...], 'Commodity': [...]} for the drop-downs (built from Angel's instrument list)."""
    try:
        master = _scrip_master()
    except Exception:
        return {}
    rows = [r for r in master if r.get('exch_seg') == exchange]
    out = {}
    idx = sorted({r['symbol'] for r in rows if r.get('instrumenttype') == 'AMXIDX'})
    if idx:
        out['Index'] = idx
    if exchange == 'NSE':
        eq = {r['symbol'] for r in rows if r['symbol'].endswith('-EQ')}
        st_ = [x for x in NIFTY50_STOCKS if x + '-EQ' in eq]
        if st_:
            out['Stocks'] = st_
    if exchange == 'MCX':
        today, best = now_ist().to_pydatetime(), {}
        for r in rows:
            if r.get('instrumenttype') != 'FUTCOM':
                continue
            try:
                exp = dt.datetime.strptime(r['expiry'], '%d%b%Y')
            except Exception:
                continue
            if exp.date() >= today.date() and (r['name'] not in best or exp < best[r['name']][0]):
                best[r['name']] = (exp, r['symbol'])
        if best:
            out['Commodity'] = [v[1] for k, v in sorted(best.items())]      # nearest-expiry future of each commodity
    return out


def page_builder():
    st.subheader('Strategy builder - Method 2')
    st.caption('Paper trading. Every Symbol x Max lots x Target % you pick is saved as its own file inside the folder.')
    st.markdown("""<style>
    div[data-testid='stVerticalBlock']{gap:.6rem}
    /* check-box drop-downs look exactly like the other inputs (same grey, same height) */
    div[data-testid='stPopover'] button{background:rgba(151,166,195,.15);border:1px solid transparent;
        min-height:2.5rem;border-radius:.5rem}
    div[data-testid='stPopover'] button:hover{border-color:rgba(255,75,75,.6)}
    </style>""", unsafe_allow_html=True)

    def lbl(text):                                            # same label look for EVERY field -> rows line up
        st.markdown(f"<div style='font-size:14px;line-height:1.5;padding:0 0 10px 0;opacity:.85'>{text}</div>", unsafe_allow_html=True)

    folders = list_folders()
    c1, c2, c3, c4 = st.columns(4)                            # ---- row 1
    with c1:
        lbl('Folder')
        pick = st.selectbox('Folder', ['+ New folder'] + folders, label_visibility='collapsed')
    with c2:
        lbl('New folder name')
        new_name = st.text_input('New folder name', label_visibility='collapsed', placeholder='e.g. Nifty 4H test',
                                 disabled=(pick != '+ New folder'))
    with c3:
        lbl('File name prefix')
        prefix = st.text_input('File name prefix', label_visibility='collapsed', placeholder='optional, e.g. morning')
    with c4:
        lbl('Exchange')
        exchange = st.selectbox('Exchange', ['NSE', 'MCX', 'NFO', 'BSE'], label_visibility='collapsed')
    folder = new_name if pick == '+ New folder' else pick

    c1, c2, c3, c4 = st.columns(4)                            # ---- row 2
    with c1:
        lbl('Capital')
        capital = st.number_input('Capital', min_value=0.0, value=1000000.0, step=10000.0, label_visibility='collapsed')
    with c2:
        lbl('Timeframe (one file each)')
        tfs = checkbox_dropdown('Select timeframe', list(TF), ['1D'], 'tf')
    with c3:
        lbl('Market close')
        close_t = st.time_input('Market close', dt.time(23, 30) if exchange == 'MCX' else dt.time(15, 30),
                                key=f'close_{exchange}', label_visibility='collapsed')
    with c4:
        lbl('Buy on (one file each)')
        candles = checkbox_dropdown('Select candle', list(CANDLE_OPTIONS), ['Every candle'], 'candle')

    st.markdown('**Entry condition**  - Ichimoku cloud, Method 2 (live: starts from now)')
    picked = []
    groups = list(instrument_choices(exchange).items())
    cols = st.columns(len(groups) + 1)                        # ---- row 3: Index | Stocks | type names
    for col, (label, opts) in zip(cols, groups):
        with col:
            lbl(f'{label} ({len(opts)})')
            picked += checkbox_dropdown(f'Select {label.lower()}', opts, [], f'sym_{exchange}_{label}')
    with cols[-1]:
        lbl('Or type names (comma separated)')
        typed_txt = st.text_input('Or type names', label_visibility='collapsed', placeholder='Nifty 50, RELIANCE, GOLDM05DEC25FUT',
                                  help='Index: Nifty 50  |  Stock: RELIANCE (or RELIANCE-EQ)  |  MCX: exact future symbol. '
                                       'Typed names are added to the ticked ones.')

    c1, c2, c3, c4 = st.columns(4)                            # ---- row 4: lots | size by | exit
    with c1:
        lbl('Max lots (one file each)')
        lots_list = checkbox_dropdown('Select lots', LOTS_OPTIONS, [10], 'lots')
    with c2:
        lbl('Size each lot by')
        modes = checkbox_dropdown('Select size type', ['Quantity', 'Amount'], ['Quantity'], 'mode')
    with c3:
        lbl(f'Profit target / Span exit ({len(EXIT_OPTIONS)})')
        targets = checkbox_dropdown('Select exit', EXIT_OPTIONS, [1, 2], 'exits', fmt=exit_label)

    def typed_values(txt):
        out = []
        for x in txt.split(','):
            try:
                v = float(x.strip())
            except ValueError:
                continue
            if v > 0 and v not in out:
                out.append(v)
        return out

    sizes = []
    slots = st.columns(4)                                     # ---- row 5: values for the chosen size type(s)
    i = 0
    if 'Quantity' in modes:
        with slots[i]:
            lbl('Quantity per lot')
            qv = [float(q) for q in checkbox_dropdown('Select quantity', QTY_OPTIONS, [1], 'qty')]
        with slots[i + 1]:
            lbl('Or type quantity')
            qt = st.text_input('Or type quantity', label_visibility='collapsed', placeholder='3, 15, 40', key='size_txt_qty')
        sizes += [('qty', v) for v in dict.fromkeys(qv + typed_values(qt))]
        i += 2
    if 'Amount' in modes:
        with slots[i]:
            lbl('Amount per lot')
            av = [float(q) for q in checkbox_dropdown('Select amount', AMOUNT_OPTIONS, [10000], 'amt')]
        with slots[i + 1]:
            lbl('Or type amount')
            at = st.text_input('Or type amount', label_visibility='collapsed', placeholder='15000, 75000', key='size_txt_amt',
                               help='Money put in every lot (units = amount / price of that lot).')
        sizes += [('amt', v) for v in dict.fromkeys(av + typed_values(at))]
    start = now_ist().strftime('%Y-%m-%d %H:%M')    # live only: always start from NOW (no past backfill)

    symbols, seen = [], set()
    for x in picked + [t.strip() for t in typed_txt.split(',') if t.strip()]:
        if x.upper() not in seen:
            seen.add(x.upper())
            symbols.append(x)
    n = len(symbols) * len(tfs) * len(lots_list) * len(sizes) * len(targets) * len(candles)
    st.caption(f'{len(symbols)} symbol(s) x {len(tfs)} timeframe(s) x {len(lots_list)} lots x {len(sizes)} size(s) x {len(targets)} exits x '
               f'{len(candles)} candle mode(s) = {n} file(s) will be saved.  Start = {start}  (live - first candle that closes after this)')
    st.caption('Live trading (real orders) is not included - run Paper first and check the numbers.')
    if st.button('Start', type='primary'):
        if not folder or not folder.strip():
            st.error('Give a folder name.')
        elif n == 0:
            st.error('Pick (or type) at least one symbol, one timeframe, one Max lots, one quantity/amount, one candle option and one exit option.')
        elif n > 150:
            st.error('Too many files (max 150 at a time).')
        else:
            try:
                for sym in symbols:
                    _instrument_token(exchange, sym)          # fail early if a symbol is wrong
            except Exception as e:
                st.error(str(e))
                return
            made, skipped = create_strategies(folder.strip(), prefix.strip(), exchange, symbols, tfs, capital, sizes,
                                              close_t.strftime('%H:%M'), lots_list, targets, candles, start)
            if made:
                st.success(f'Saved {len(made)} file(s) in folder "{folder.strip()}". Open "My strategies" to watch them live.')
            if skipped:
                st.warning(f'{len(skipped)} file(s) already existed and were kept as they are (their start time is unchanged).')


def detail(ev):
    s, tr, trades, last = ev['s'], ev['tr'], ev['trades'], ev['last']
    st.markdown(f"#### {s['name']}")
    st.caption(f"{s['exchange']}:{s['symbol']}  |  {s['tf']}  |  {s['lots']} lots  |  exit {exit_text(s)}  |  buy {candle_text(s).lower()}  |  {('amount/lot ' + format(s['amount'], ',.0f')) if s.get('amount') else ('qty/lot ' + format(s['qty'], 'g'))}"
               f"  |  capital {money(s['capital'])}  |  started {s['start']}")
    if last is None:
        st.info('Waiting for the first closed candle after the start time (cloud needs 78+ bars of history).')
        return
    flat = ev['flat']
    c = st.columns(3)
    c[0].metric('Last closed bar', str(last['Date'])[:16])
    c[1].metric('Close', f"{last['Close']:.2f}")
    c[2].metric('Signal', last['Signal'])
    c = st.columns(3)
    c[0].metric('Position', ev['status'])
    c[1].metric('Avg entry', '-' if flat else f"{last['AvgPrice']:.2f}",
                None if flat or pd.isna(last['Unrealised %']) else f"{last['Unrealised %']:.2f}%")
    c[2].metric('Cloud top / bottom', f"{last['CloudTop']:.2f} / {last['CloudBottom']:.2f}")
    total = ev['closed_pnl'] + ev['open_pnl']
    c = st.columns(3)
    c[0].metric('Closed P&L', money(ev['closed_pnl']), f"{ev['closed_n']} trades", delta_color='off')
    c[1].metric('Open P&L', money(ev['open_pnl']))
    c[2].metric('Total P&L', money(total), f"{total / s['capital'] * 100:.2f}% of capital" if s['capital'] else None)
    st.line_chart(tr.tail(150).set_index('Date')[['Close', 'CloudTop', 'CloudBottom']])
    t1, t2 = st.tabs(['Bar log (latest first)', 'Trades'])
    with t1:
        st.dataframe(tr.sort_values('Date', ascending=False).head(100), width='stretch', hide_index=True)
    with t2:
        if trades.empty:
            st.write('No trades yet.')
        else:
            st.dataframe(trades.sort_values('Exit Date', ascending=False), width='stretch', hide_index=True)


def page_view(topic, refresh):
    folders = list_folders()
    if not folders:
        st.info('No strategies yet. Create some in the Builder.')
        return
    folder = st.session_state.get('open_folder')
    if folder not in folders:
        st.session_state.pop('open_folder', None)
        folder = None

    if folder is None:                                            # ---- folder list: click a folder to open it
        st.markdown('#### Your folders')
        st.caption('Click a folder to watch its strategies run live.')
        for fol in folders:
            n = len(list_files(fol))
            row = st.columns([14, 1])
            if row[0].button(f'📁  {fol}      |      {n} strateg{"y" if n == 1 else "ies"}', key=f'open_{fol}', width='stretch'):
                st.session_state['open_folder'] = fol
                st.rerun()
            with row[1].popover('🗑️', width='stretch'):          # icon only; one more click to confirm
                st.write(f'{fol} ({n})')
                if st.button('✔ Yes', key=f'delfbtn_{fol}', type='primary'):
                    delete_folder(fol)
                    st.rerun()
        return

    top = st.columns([1.3, 8, 0.7])                               # ---- inside one folder
    if top[0].button('← All folders'):
        st.session_state.pop('open_folder', None)
        st.rerun()
    top[1].markdown(f'#### 📁 {folder}')
    with top[2].popover('🗑️', width='stretch'):
        st.write(f'{folder} ({len(list_files(folder))})')
        if st.button('✔ Yes', key=f'delfbtn_in_{folder}', type='primary'):
            delete_folder(folder)
            st.session_state.pop('open_folder', None)
            st.rerun()
    _folder_live(topic, refresh, folder)


def _folder_live(topic, refresh, folder):
    def body():
        files = list_files(folder)
        if not files:
            st.info('This folder is empty.')
            return
        strategies = [load_strategy(folder, f) for f in files]
        evs, errs = {}, {}
        for f, s in zip(files, strategies):
            try:
                evs[f] = evaluate(s)
            except Exception as e:
                errs[s['name']] = str(e)
        for f, ev in evs.items():                                  # push alert on a NEW closed-candle signal
            if ev['last'] is not None:
                notify_new_signal(topic, f"{st.session_state['creds']['client_id']}/{folder}/{f}", ev['last'])

        # ---- live status banner: shows the app is really running
        now = now_ist().to_pydatetime()
        close_txt = strategies[0].get('close', '15:30')
        open_txt = '09:00' if strategies[0]['exchange'] == 'MCX' else '09:15'
        is_open = now.weekday() < 5 and open_txt <= now.strftime('%H:%M') <= close_txt
        stale = [ev for ev in evs.values() if ev.get('data_err')]
        ok = not (errs or stale)
        b1, b2, b3, b4 = st.columns(4)
        b1.metric('Status', 'LIVE',
                  ('market open' if is_open else 'market closed') + (' - running' if ok else ' - check warnings'), delta_color='off' if ok else 'inverse')
        b2.metric('Last refresh', now.strftime('%H:%M:%S'), f'next in {int(refresh)}s', delta_color='off')
        checks = [ev['checked'] for ev in evs.values() if ev.get('checked')]
        b3.metric('Angel data checked', ts_ist(max(checks)).strftime('%H:%M:%S') if checks else '-',
                  f'{len(evs)} / {len(files)} files OK', delta_color='off')
        n_sig = sum(int((ev['tr']['Signal'] != 'HOLD').sum()) for ev in evs.values() if not ev['tr'].empty)
        b4.metric('Signals since start', n_sig)
        if stale:
            st.warning('Angel rate limit / data problem - showing the last good candles, will retry on the next refresh: '
                       + (stale[0]['data_err'] or '')[:150])
        if errs:
            with st.expander(f'{len(errs)} file(s) could not be evaluated', expanded=False):
                for n_, m_ in errs.items():
                    st.write(f'**{n_}**: {m_[:200]}')

        # ---- live activity feed (every BUY / EXIT since the strategies started)
        acts = []
        for f, ev in evs.items():
            tr = ev['tr']
            if tr.empty:
                continue
            for _, r in tr[tr['Signal'] != 'HOLD'].iterrows():
                acts.append({'Bar time': str(r['Date'])[:16], 'File': ev['s']['name'], 'Signal': r['Signal'],
                             'Price': round(float(r['Close']), 2), 'Lots after': int(r['Lots'])})
        with st.expander(f'Live activity - {len(acts)} signal(s) since start', expanded=True):
            if acts:
                st.dataframe(pd.DataFrame(acts).sort_values('Bar time', ascending=False).head(100),
                             width='stretch', hide_index=True)
            else:
                st.info('No signal yet. Strategies started at ' + min(s['start'] for s in strategies) +
                        ' - they act only when a candle CLOSES after that time (cloud history is loaded automatically).')
        st.caption(f"Refreshed {now:%H:%M:%S}  |  closed candles only  |  every {refresh}s")

        tot = sum(ev['closed_pnl'] + ev['open_pnl'] for ev in evs.values())
        st.metric('Folder total P&L (all files)', money(tot))
        st.markdown(f'**Strategies in "{folder}"** ({len(files)}) - click "Live details" under a strategy to watch it run')

        for f in files:                                            # one card per file, one below another
            ev = evs.get(f)
            if ev is None:
                continue
            s, last = ev['s'], ev['last']
            total = ev['closed_pnl'] + ev['open_pnl']
            icon = '⏳' if last is None else ('⚪' if ev['flat'] else '🟢')
            with st.container(border=True):
                c = st.columns([3.2, 1.5, 1.2, 1.2, 1.3])
                c[0].markdown(f"**{icon} {s['name']}**")
                c[0].caption(f"{s['exchange']}:{s['symbol']} | {s['tf']} | exit {exit_text(s)} | {candle_text(s).lower()} | started {s['start']}")
                c[1].caption('Status')
                c[1].write(ev['status'])
                c[2].caption('Avg entry')
                c[2].write('-' if last is None or ev['flat'] else f"{last['AvgPrice']:.2f}")
                c[3].caption('Close')
                c[3].write('-' if last is None else f"{last['Close']:.2f}")
                c[4].caption('Total P&L')
                c[4].write(money(total))
                with st.expander('Live details'):                  # constant label -> stays open while it auto-refreshes
                    detail(ev)
                    with st.popover('🗑️'):
                        if st.button('✔ Yes', key=f'delbtn_{folder}_{f}', type='primary'):
                            delete_strategy(folder, f)
                            st.rerun()

    st.fragment(run_every=int(refresh))(body)()


# =============================================================== app
def main():
    st.set_page_config(page_title='Live Paper Trading', layout='wide')
    st.title('Live Paper Trading - Method 2')

    # ---- login screen (nothing is stored on the server)
    if 'creds' not in st.session_state:
        st.subheader('Enter your Angel One details')
        api_key = st.text_input('API key', type='password')
        client_id = st.text_input('Client ID')
        mpin = st.text_input('MPIN', type='password')
        totp = st.text_input('TOTP secret key', type='password',
                             help='The long text shown when you enable TOTP (not the 6-digit code)')
        if st.button('Login') and api_key and client_id and mpin and totp:
            creds = {'api_key': api_key.strip(), 'client_id': client_id.strip().upper(),
                     'mpin': mpin.strip(), 'totp': totp.strip()}
            try:
                st.session_state['creds'] = creds
                get_smart()
                st.rerun()
            except Exception as e:
                st.session_state.pop('creds', None)
                st.session_state.pop('smart', None)
                st.error(f'Login failed: {e}')
        return

    with st.sidebar:
        with st.expander('Account', expanded=False):
            st.write(f"Client ID: {st.session_state['creds']['client_id']}")
            topic = st.text_input('ntfy topic (optional)')
            refresh = st.number_input('Refresh every (seconds)', 15, 3600, 60, 15)
            if st.button('Logout / change details'):
                for k_ in ('creds', 'smart'):
                    st.session_state.pop(k_, None)
                st.rerun()

    page = st.radio('Page', ['Builder', 'My strategies'], horizontal=True, label_visibility='collapsed')
    if page == 'Builder':
        page_builder()
    else:
        page_view(topic, refresh)


if __name__ == '__main__':
    main()
