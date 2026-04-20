### Growth Rate Model by Larbi Moukhlis
### Panel XGBoost Growth Rate Predictor — called by DCF_Hybrid_Valuation.py

import requests
import pandas as pd
import numpy as np
import yfinance as yf
import simfin as sf
import time
import warnings
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from config import FRED_API_KEY, SIMFIN_API_KEY
warnings.filterwarnings('ignore')

#  Simfin Setup 
sf.set_api_key(SIMFIN_API_KEY)
sf.set_data_dir('~/simfin_data')  

SECTOR_ETF_MAP = {
    "Technology":             "XLK",
    "Healthcare":             "XLV",
    "Financials":             "XLF",
    "Energy":                 "XLE",
    "Industrials":            "XLI",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Utilities":              "XLU",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Communication Services": "XLC",
}

# ── Simfin Bulk Loader ────────────────────────────────────────────────────────
# Loaded once at module level — avoids re-downloading for every peer ticker.
_SF_CASHFLOW  = None
_SF_INCOME    = None
_SF_COMPANIES = None

def load_simfin_data():
    global _SF_CASHFLOW, _SF_INCOME, _SF_COMPANIES
    if _SF_CASHFLOW is not None:
        return  # already loaded this session
    print("Loading Simfin bulk data (cached to disk after first run)...")
    try:
        _SF_CASHFLOW  = sf.load_cashflow(variant='annual', market='us')
        _SF_INCOME    = sf.load_income(variant='annual',   market='us')
        _SF_COMPANIES = sf.load_companies(market='us')
        print(f"  Simfin loaded: {len(_SF_COMPANIES)} companies, "
              f"{len(_SF_CASHFLOW)} cashflow rows, {len(_SF_INCOME)} income rows.")
    except Exception as e:
        print(f"  WARNING: Simfin load failed — {e}")
        print("  Falling back to yfinance for all fundamental data.")
        _SF_CASHFLOW  = pd.DataFrame()
        _SF_INCOME    = pd.DataFrame()
        _SF_COMPANIES = pd.DataFrame()

#  FRED Helper 
FRED_FALLBACKS = {"DGS10": 4.25}

def fetch_fred(series_id, retries=3, backoff=2):
    last_error = None
    for attempt in range(retries):
        try:
            r = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id":         series_id,
                    "api_key":           FRED_API_KEY,
                    "file_type":         "json",
                    "observation_start": "2010-01-01"
                },
                timeout=15
            )
            payload = r.json()
            if 'observations' in payload:
                df = pd.DataFrame(payload['observations'])
                df['date']  = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                return df.set_index('date')['value'].dropna()
            last_error = payload
            print(f"  FRED attempt {attempt+1}/{retries} failed for {series_id}: {payload}")
        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"  FRED attempt {attempt+1}/{retries} network error: {e}")
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))

    fallback = FRED_FALLBACKS.get(series_id)
    if fallback is not None:
        print(f"  WARNING: FRED unavailable for {series_id}. Using fallback {fallback}.")
        idx = pd.date_range("2010-01-01", periods=1, freq="YE")
        return pd.Series([fallback], index=idx)

    raise ValueError(
        f"FRED API failed for '{series_id}' after {retries} attempts: {last_error}\n"
        f"Check FRED_API_KEY in config.py — free key at https://fred.stlouisfed.org/docs/api/api_key.html"
    )

#  Regime Detection 
def detect_macro_regime(year):
    recession_years  = {2020, 2009, 2008}
    tightening_years = {2022, 2023, 2018, 2015, 2013}
    recovery_years   = {2021, 2010, 2011}
    if year in recession_years:   return 2
    elif year in tightening_years: return 1
    elif year in recovery_years:   return 3
    else:                          return 0

#  Peer Discovery 
def get_sector_peers(stock_ticker, company_info):
    sector   = company_info.get('sector',   'Unknown')
    industry = company_info.get('industry', 'Unknown')
    print(f"Sector: {sector} | Industry: {industry}")
    print(f"Finding sector peers for {stock_ticker} via ETF holdings...")
    time.sleep(2)

    etf_ticker = SECTOR_ETF_MAP.get(sector, 'XLK')
    etf        = yf.Ticker(etf_ticker)
    holdings   = etf.funds_data.top_holdings

    if holdings is not None and not holdings.empty:
        peers = [t for t in holdings.index.tolist() if t != stock_ticker][:9]
        peers = [stock_ticker] + peers
        print(f"Found {len(peers)} peers from {etf_ticker}: {peers}")
        return peers, sector, industry

    raise ValueError(f"Could not fetch ETF holdings from {etf_ticker} — check yfinance connection.")

#  Simfin FCF Extractor 
def _simfin_slice(df, ticker):
    """Return the rows for a given ticker from a Simfin bulk DataFrame."""
    if isinstance(df.index, pd.MultiIndex):
        level_names = [n for n in df.index.names if n is not None]
        ticker_level = next((n for n in level_names if 'ticker' in n.lower()), None)
        if ticker_level and ticker in df.index.get_level_values(ticker_level):
            return df.xs(ticker, level=ticker_level)
    # Fallback: column-based filter
    if 'Ticker' in df.columns:
        return df[df['Ticker'] == ticker]
    return pd.DataFrame()

def _to_year_series(series):
    """Collapse a DatetimeIndex series to integer-year keys."""
    if not isinstance(series.index, pd.DatetimeIndex):
        series = series.copy()
        series.index = pd.to_datetime(series.index)
    return pd.Series({ts.year: val for ts, val in series.items()}).sort_index()

def fetch_simfin_fcf(ticker):
    try:
        if _SF_CASHFLOW is None or _SF_CASHFLOW.empty:
            raise ValueError("Simfin not loaded")
        df = _simfin_slice(_SF_CASHFLOW, ticker)
        if df.empty:
            raise ValueError(f"{ticker} not in Simfin cashflow")

        if 'Free Cash Flow' in df.columns:
            raw = df['Free Cash Flow'].dropna()
        elif 'Net Cash from Operating Activities' in df.columns and \
             'Purchase of Property, Plant & Equipment' in df.columns:
            raw = (df['Net Cash from Operating Activities'] +
                   df['Purchase of Property, Plant & Equipment']).dropna()
        else:
            raise ValueError("Cannot construct FCF from Simfin columns")

        result = _to_year_series(raw)
        return result if len(result) >= 3 else None
    except Exception:
        return _fetch_yf_fcf(ticker)

def _fetch_yf_fcf(ticker):
    try:
        cf = yf.Ticker(ticker).cashflow
        if cf is None or cf.empty:
            return None
        if 'Free Cash Flow' in cf.index:
            s = cf.loc['Free Cash Flow']
        elif 'Operating Cash Flow' in cf.index and 'Capital Expenditure' in cf.index:
            s = cf.loc['Operating Cash Flow'] + cf.loc['Capital Expenditure']
        else:
            return None
        s = s.dropna()
        result = pd.Series({ts.year: v for ts, v in s.items()}).sort_index()
        return result if len(result) >= 3 else None
    except Exception:
        return None

#  Simfin Revenue Extractor 
def fetch_simfin_revenue(ticker):
    try:
        if _SF_INCOME is None or _SF_INCOME.empty:
            raise ValueError("Simfin not loaded")
        df = _simfin_slice(_SF_INCOME, ticker)
        if df.empty:
            raise ValueError(f"{ticker} not in Simfin income")

        rev_col = next((c for c in ['Revenue', 'Total Revenue', 'Revenues'] if c in df.columns), None)
        if rev_col is None:
            raise ValueError("No revenue column in Simfin income")

        result = _to_year_series(df[rev_col].dropna())
        return result if len(result) >= 2 else None
    except Exception:
        return _fetch_yf_revenue(ticker)

def _fetch_yf_revenue(ticker):
    try:
        fin = yf.Ticker(ticker).financials
        if fin is None or fin.empty:
            return None
        col = next((r for r in ['Total Revenue', 'Revenue'] if r in fin.index), None)
        if col is None:
            return None
        s = fin.loc[col].dropna()
        return pd.Series({ts.year: v for ts, v in s.items()}).sort_index()
    except Exception:
        return None

#  Panel Feature Builder 
def build_panel_features(stock_ticker, peers, sector, industry, stages, target_fcf=None):
    print(f"Building {stages}-stage panel feature matrix across {len(peers)} peers...")
    load_simfin_data()
    time.sleep(1)

    _gdp_raw = fetch_fred("GDP").resample('YE').last().pct_change().dropna()
    _cpi_raw = fetch_fred("CPIAUCSL").resample('YE').last().pct_change().dropna()
    rates    = fetch_fred("DGS10").resample('YE').last().dropna()

    if _gdp_raw.empty:
        print("  WARNING: GDP growth empty — using 2.5% flat fallback.")
        _gdp_raw = pd.Series([0.025]*14, index=pd.date_range("2010-12-31", periods=14, freq="YE"))
    if _cpi_raw.empty:
        print("  WARNING: CPI growth empty — using 2.5% flat fallback.")
        _cpi_raw = pd.Series([0.025]*14, index=pd.date_range("2010-12-31", periods=14, freq="YE"))

    gdp = _gdp_raw
    cpi = _cpi_raw

    # Fetch SP500 returns once — reused for all beta calculations
    try:
        sp500_ret = yf.Ticker("^GSPC").history(period="max")['Close'].pct_change().dropna()
    except Exception:
        sp500_ret = pd.Series(dtype=float)

    all_rows = []

    for ticker in peers:
        print(f"  Loading data for {ticker}...")

        if ticker == stock_ticker and target_fcf is not None:
            fcf_by_year = {}
            for timestamp, value in target_fcf.items():
                year = timestamp.year if isinstance(timestamp, pd.Timestamp) else int(timestamp)
                fcf_by_year[year] = value
            fcf_series = pd.Series(fcf_by_year).sort_index(ascending=True)
        else:
            fcf_series = fetch_simfin_fcf(ticker)

        revenue_series = fetch_simfin_revenue(ticker)

        if fcf_series is None or len(fcf_series) < 3:
            print(f"  Skipping {ticker} — insufficient FCF data")
            continue

        fcf_growth     = fcf_series.pct_change().dropna()
        revenue_growth = revenue_series.pct_change().dropna() if revenue_series is not None else pd.Series(dtype=float)

        try:
            stock_ret = yf.Ticker(ticker).history(period="max")['Close'].pct_change().dropna()
            combined  = pd.concat([stock_ret, sp500_ret], axis=1, sort=False).dropna()
            combined.columns = ['Stock', 'Market']
            roll_cov  = combined['Stock'].rolling(252).cov(combined['Market'])
            roll_var  = combined['Market'].rolling(252).var()
            roll_beta = (roll_cov / roll_var).dropna().resample('YE').last().ffill()
        except Exception:
            roll_beta = pd.Series(dtype=float)

        is_target = 1 if ticker == stock_ticker else 0

        for year_idx in fcf_growth.index:
            year    = int(year_idx)
            year_dt = pd.Timestamp(f'{year}-12-31')

            gdp_val  = gdp.get(year_dt,  np.nan)
            cpi_val  = cpi.get(year_dt,  np.nan)
            rate_val = rates.get(year_dt, np.nan)
            beta_val = roll_beta.get(year_dt, np.nan)
            rev_val  = revenue_growth.get(year_idx, np.nan)
            regime   = detect_macro_regime(year)

            row = {
                'ticker':        ticker,
                'year':          year,
                'fcf_growth':    fcf_growth[year_idx],
                'revenue_growth': rev_val,
                'gdp_growth':    gdp_val,
                'cpi_growth':    cpi_val,
                'interest_rate': rate_val,
                'beta':          beta_val,
                'regime':        regime,
                'is_target':     is_target,
                'beta_x_gdp':    beta_val * gdp_val  if not np.isnan(beta_val) and not np.isnan(gdp_val)  else np.nan,
                'beta_x_rate':   beta_val * rate_val if not np.isnan(beta_val) and not np.isnan(rate_val) else np.nan,
                'beta_x_regime': beta_val * regime   if not np.isnan(beta_val)                            else np.nan,
                'stages':        stages,
            }
            all_rows.append(row)

    panel = pd.DataFrame(all_rows).dropna(subset=['fcf_growth', 'gdp_growth', 'cpi_growth'])
    print(f"Panel built: {len(panel)} observations across {panel['ticker'].nunique()} stocks")
    return panel

#  XGBoost Panel Trainer 
def train_xgboost_panel(panel, stages):
    print(f"Training XGBoost on panel data ({stages}-stage model)...")
    time.sleep(2)

    feature_cols = [
        'fcf_growth', 'revenue_growth', 'gdp_growth', 'cpi_growth',
        'interest_rate', 'beta', 'regime', 'is_target',
        'beta_x_gdp', 'beta_x_rate', 'beta_x_regime', 'stages'
    ]

    panel = panel.sort_values(['ticker', 'year']).copy()

    panel['short_term_target'] = panel.groupby('ticker')['fcf_growth'].shift(-1)

    if stages == 2:
        panel['long_term_target'] = panel.groupby('ticker')['fcf_growth'].transform(
            lambda x: x.shift(-2).rolling(3, min_periods=1).mean()
        )
        stage_labels = ['Short-Term (1-3yr)', 'Terminal']
    elif stages == 3:
        panel['long_term_target'] = panel.groupby('ticker')['fcf_growth'].transform(
            lambda x: x.shift(-2).rolling(3, min_periods=1).mean()
        )
        panel['transition_target'] = panel.groupby('ticker')['fcf_growth'].transform(
            lambda x: x.shift(-4).rolling(4, min_periods=2).mean()
        )
        stage_labels = ['Short-Term (1-3yr)', 'Transition (4-7yr)', 'Terminal']

    core_cols  = ['fcf_growth', 'gdp_growth', 'cpi_growth']
    panel_base = panel.dropna(subset=core_cols).copy()

    for col in feature_cols:
        if panel_base[col].isna().any():
            panel_base[col] = panel_base[col].fillna(panel_base[col].median())

    if len(panel_base) == 0:
        raise ValueError("Panel has 0 clean rows — not enough data to train.")

    scaler = StandardScaler()
    scaler.fit(panel_base[feature_cols])

    def _train(df, target_col):
        sub = df.dropna(subset=[target_col])
        if len(sub) == 0:
            raise ValueError(f"No rows with valid {target_col}.")
        m = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42)
        m.fit(scaler.transform(sub[feature_cols]), sub[target_col])
        print(f"  {target_col} model trained on {len(sub)} rows.")
        return m

    xgb_short      = _train(panel_base, 'short_term_target')
    xgb_long       = _train(panel_base, 'long_term_target')
    xgb_transition = _train(panel_base, 'transition_target') if stages == 3 else None

    print(f"Models trained for stages: {stage_labels}")
    return xgb_short, xgb_long, xgb_transition, scaler, feature_cols

#  Growth Rate Predictor 
def predict_growth_rates(stock_ticker, panel, xgb_short, xgb_long, xgb_transition, scaler, feature_cols, stages):
    print(f"Predicting growth rates for {stock_ticker}...")
    time.sleep(2)

    target_rows = panel[panel['ticker'] == stock_ticker].sort_values('year')
    if target_rows.empty:
        print("No panel data found for target stock — using most recent panel row.")
        target_rows = panel.sort_values('year')

    latest = target_rows.iloc[[-1]][feature_cols].copy()
    for col in feature_cols:
        if latest[col].isna().any():
            latest[col] = latest[col].fillna(panel[col].median())

    latest_scaled = scaler.transform(latest)

    short_term_growth = float(np.clip(xgb_short.predict(latest_scaled)[0], -0.20, 0.20))
    long_term_growth  = float(np.clip(xgb_long.predict(latest_scaled)[0],  -0.15, 0.15))
    transition_growth = None

    if stages == 3 and xgb_transition is not None:
        transition_growth = float(np.clip(xgb_transition.predict(latest_scaled)[0], -0.15, 0.15))
        print(f"Short-Term Growth  (1-3yr):         {short_term_growth:.2%}")
        print(f"Transition Growth  (4-7yr):         {transition_growth:.2%}")
        print(f"Long-Term Growth   (Terminal base): {long_term_growth:.2%}")
    else:
        print(f"Short-Term Growth  (1-3yr): {short_term_growth:.2%}")
        print(f"Long-Term Growth   (4-5yr): {long_term_growth:.2%}")

    return short_term_growth, long_term_growth, transition_growth

#  Main Entry Point 
def get_growth_rates(stock_ticker, company_info, fcf, stages):
    peers, sector, industry                                = get_sector_peers(stock_ticker, company_info)
    panel                                                  = build_panel_features(stock_ticker, peers, sector, industry, stages, target_fcf=fcf)
    xgb_short, xgb_long, xgb_trans, scaler, feature_cols  = train_xgboost_panel(panel, stages)
    short_term_growth, long_term_growth, transition_growth = predict_growth_rates(
        stock_ticker, panel, xgb_short, xgb_long, xgb_trans, scaler, feature_cols, stages
    )
    return short_term_growth, long_term_growth, transition_growth
### Growth Rate Model by Larbi Moukhlis
### Panel XGBoost Growth Rate Predictor — called by DCF_Hybrid_Valuation.py

import requests
import pandas as pd
import numpy as np
import yfinance as yf
import simfin as sf
import time
import warnings
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from config import FRED_API_KEY, SIMFIN_API_KEY
warnings.filterwarnings('ignore')

#  Simfin Setup 
sf.set_api_key(SIMFIN_API_KEY)
sf.set_data_dir('~/simfin_data')

SECTOR_ETF_MAP = {
    "Technology":             "XLK",
    "Healthcare":             "XLV",
    "Financials":             "XLF",
    "Energy":                 "XLE",
    "Industrials":            "XLI",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Utilities":              "XLU",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Communication Services": "XLC",
}

# Broader static peer universe per sector — used to pad beyond ETF top-10
SECTOR_PEER_UNIVERSE = {
    "Technology": [
        "AAPL","MSFT","NVDA","AVGO","AMD","QCOM","TXN","INTC","MU","AMAT",
        "LRCX","KLAC","ADI","MRVL","CSCO","ORCL","CRM","NOW","SNPS","CDNS",
        "PLTR","PANW","FTNT","ZBRA","KEYS"
    ],
    "Healthcare": [
        "JNJ","UNH","LLY","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN",
        "GILD","ISRG","SYK","BDX","ZTS","REGN","VRTX","HCA","CI","CVS",
        "MDT","ELV","HUM","IQV","A"
    ],
    "Financials": [
        "BRK-B","JPM","BAC","WFC","GS","MS","C","AXP","BLK","SCHW",
        "CB","MMC","AON","TRV","PGR","MET","PRU","AFL","ALL","L",
        "USB","PNC","TFC","STT","FITB"
    ],
    "Energy": [
        "XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","PXD",
        "HAL","DVN","HES","FANG","APA","MRO","KMI","WMB","OKE","ET",
        "BKR","NOV","FTI","HP","RRC"
    ],
    "Industrials": [
        "GE","HON","UPS","RTX","CAT","DE","LMT","NOC","GD","BA",
        "MMM","EMR","ETN","ITW","PH","ROK","FDX","CSX","UNP","NSC",
        "CTAS","RSG","WM","VRSK","FAST"
    ],
    "Consumer Cyclical": [
        "AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","TJX","BKNG","MAR",
        "GM","F","ORLY","AZO","BBY","DRI","YUM","CMG","ROST","ULTA",
        "DHI","LEN","PHM","NVR","TOL"
    ],
    "Consumer Defensive": [
        "PG","KO","PEP","WMT","COST","PM","MO","CL","KMB","GIS",
        "K","CPB","SJM","HRL","MKC","CAG","LW","MNST","STZ","BF-B",
        "EL","CHD","CLX","COTY","ENR"
    ],
    "Communication Services": [
        "GOOGL","META","NFLX","DIS","CMCSA","T","VZ","TMUS","CHTR","EA",
        "TTWO","ATVI","MTCH","SNAP","PINS","RDDT","WBD","PARA","FOX","NYT",
        "IAC","ZG","ANGI","YELP","TRIP"
    ],
    "Utilities": [
        "NEE","DUK","SO","D","AEP","EXC","SRE","PCG","ED","WEC",
        "XEL","ES","AWK","DTE","ETR","FE","PPL","CMS","NI","ATO",
        "LNT","EVRG","OGE","POR","AVA"
    ],
    "Materials": [
        "LIN","APD","SHW","ECL","DD","NEM","FCX","NUE","VMC","MLM",
        "ALB","CE","EMN","IFF","PPG","RPM","SEE","SON","AVY","PKG",
        "IP","WRK","ATI","CMC","STLD"
    ],
    "Real Estate": [
        "AMT","PLD","CCI","EQIX","PSA","O","SPG","WELL","AVB","EQR",
        "DLR","VTR","BXP","ARE","KIM","REG","FRT","NNN","WPC","STOR",
        "EXR","CUBE","LSI","NSA","REXR"
    ],
}

#  Simfin Bulk Loader 
_SF_CASHFLOW  = None
_SF_INCOME    = None
_SF_COMPANIES = None

def load_simfin_data():
    global _SF_CASHFLOW, _SF_INCOME, _SF_COMPANIES
    if _SF_CASHFLOW is not None:
        return
    print("Loading Simfin bulk data (cached to disk after first run)...")
    try:
        _SF_CASHFLOW  = sf.load_cashflow(variant='annual', market='us')
        _SF_INCOME    = sf.load_income(variant='annual',   market='us')
        _SF_COMPANIES = sf.load_companies(market='us')
        print(f"  Simfin loaded: {len(_SF_COMPANIES)} companies, "
              f"{len(_SF_CASHFLOW)} cashflow rows, {len(_SF_INCOME)} income rows.")
    except Exception as e:
        print(f"  WARNING: Simfin load failed — {e}")
        print("  Falling back to yfinance for all fundamental data.")
        _SF_CASHFLOW  = pd.DataFrame()
        _SF_INCOME    = pd.DataFrame()
        _SF_COMPANIES = pd.DataFrame()

#  FRED Helper 
FRED_FALLBACKS = {"DGS10": 4.25}

def fetch_fred(series_id, retries=3, backoff=2):
    last_error = None
    for attempt in range(retries):
        try:
            r = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id":         series_id,
                    "api_key":           FRED_API_KEY,
                    "file_type":         "json",
                    "observation_start": "2010-01-01"
                },
                timeout=15
            )
            payload = r.json()
            if 'observations' in payload:
                df = pd.DataFrame(payload['observations'])
                df['date']  = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                return df.set_index('date')['value'].dropna()
            last_error = payload
            print(f"  FRED attempt {attempt+1}/{retries} failed for {series_id}: {payload}")
        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"  FRED attempt {attempt+1}/{retries} network error: {e}")
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))

    fallback = FRED_FALLBACKS.get(series_id)
    if fallback is not None:
        print(f"  WARNING: FRED unavailable for {series_id}. Using fallback {fallback}.")
        idx = pd.date_range("2010-01-01", periods=1, freq="YE")
        return pd.Series([fallback], index=idx)

    raise ValueError(
        f"FRED API failed for '{series_id}' after {retries} attempts: {last_error}\n"
        f"Check FRED_API_KEY in config.py — free key at https://fred.stlouisfed.org/docs/api/api_key.html"
    )

#  Regime Detection 
def detect_macro_regime(year):
    recession_years  = {2020, 2009, 2008}
    tightening_years = {2022, 2023, 2018, 2015, 2013}
    recovery_years   = {2021, 2010, 2011}
    if year in recession_years:    return 2
    elif year in tightening_years: return 1
    elif year in recovery_years:   return 3
    else:                          return 0

#  Peer Discovery 
def get_sector_peers(stock_ticker, company_info):
    sector   = company_info.get('sector',   'Unknown')
    industry = company_info.get('industry', 'Unknown')
    print(f"Sector: {sector} | Industry: {industry}")
    print(f"Finding sector peers for {stock_ticker} via ETF holdings...")
    time.sleep(2)

    etf_ticker = SECTOR_ETF_MAP.get(sector, 'XLK')
    etf_peers  = []
    try:
        etf      = yf.Ticker(etf_ticker)
        holdings = etf.funds_data.top_holdings
        if holdings is not None and not holdings.empty:
            etf_peers = [t for t in holdings.index.tolist() if t != stock_ticker][:9]
    except Exception:
        pass

    static_universe = SECTOR_PEER_UNIVERSE.get(sector, [])
    extra = [t for t in static_universe if t != stock_ticker and t not in etf_peers]

    combined = [stock_ticker] + etf_peers + extra
    seen, peers = set(), []
    for t in combined:
        if t not in seen:
            seen.add(t)
            peers.append(t)
    peers = peers[:25]

    print(f"Using {len(peers)} peers (ETF: {len(etf_peers)}, static universe padded): {peers}")
    return peers, sector, industry

#  Simfin Slicer 
def _simfin_slice(df, ticker):
    """Return rows for a given ticker from a Simfin bulk DataFrame."""
    if isinstance(df.index, pd.MultiIndex):
        ticker_level = next(
            (n for n in df.index.names if n is not None and 'ticker' in n.lower()),
            None
        )
        if ticker_level:
            try:
                if ticker in df.index.get_level_values(ticker_level):
                    return df.xs(ticker, level=ticker_level)
            except Exception:
                pass
    if 'Ticker' in df.columns:
        return df[df['Ticker'] == ticker]
    return pd.DataFrame()

def _to_year_series(series):
    """Collapse a DatetimeIndex series to integer-year keys."""
    if not isinstance(series.index, pd.DatetimeIndex):
        series = series.copy()
        series.index = pd.to_datetime(series.index)
    return pd.Series({ts.year: val for ts, val in series.items()}).sort_index()

#  Simfin FCF Extractor 
def fetch_simfin_fcf(ticker):
    """
    Construct FCF from Simfin bulk cashflow data.
    Simfin standard tier uses 'Change in Fixed Assets & Intangibles' for CapEx
    (already negative), not 'Purchase of Property, Plant & Equipment'.
    """
    try:
        if _SF_CASHFLOW is None or _SF_CASHFLOW.empty:
            raise ValueError("Simfin not loaded")
        df = _simfin_slice(_SF_CASHFLOW, ticker)
        if df.empty:
            raise ValueError(f"{ticker} not in Simfin cashflow")

        if 'Free Cash Flow' in df.columns:
            raw = df['Free Cash Flow'].dropna()
        elif ('Net Cash from Operating Activities' in df.columns and
              'Change in Fixed Assets & Intangibles' in df.columns):
            raw = (df['Net Cash from Operating Activities'] +
                   df['Change in Fixed Assets & Intangibles']).dropna()
        elif 'Net Cash from Operating Activities' in df.columns:
            raw = df['Net Cash from Operating Activities'].dropna()
        else:
            raise ValueError("Cannot construct FCF from Simfin columns")

        result = _to_year_series(raw)
        return result if len(result) >= 3 else None
    except Exception:
        return _fetch_yf_fcf(ticker)

def _fetch_yf_fcf(ticker):
    try:
        cf = yf.Ticker(ticker).cashflow
        if cf is None or cf.empty:
            return None
        if 'Free Cash Flow' in cf.index:
            s = cf.loc['Free Cash Flow']
        elif 'Operating Cash Flow' in cf.index and 'Capital Expenditure' in cf.index:
            s = cf.loc['Operating Cash Flow'] + cf.loc['Capital Expenditure']
        else:
            return None
        s = s.dropna()
        result = pd.Series({ts.year: v for ts, v in s.items()}).sort_index()
        return result if len(result) >= 3 else None
    except Exception:
        return None

#  Simfin Revenue Extractor 
def fetch_simfin_revenue(ticker):
    try:
        if _SF_INCOME is None or _SF_INCOME.empty:
            raise ValueError("Simfin not loaded")
        df = _simfin_slice(_SF_INCOME, ticker)
        if df.empty:
            raise ValueError(f"{ticker} not in Simfin income")

        rev_col = next((c for c in ['Revenue', 'Total Revenue', 'Revenues'] if c in df.columns), None)
        if rev_col is None:
            raise ValueError("No revenue column in Simfin income")

        result = _to_year_series(df[rev_col].dropna())
        return result if len(result) >= 2 else None
    except Exception:
        return _fetch_yf_revenue(ticker)

def _fetch_yf_revenue(ticker):
    try:
        fin = yf.Ticker(ticker).financials
        if fin is None or fin.empty:
            return None
        col = next((r for r in ['Total Revenue', 'Revenue'] if r in fin.index), None)
        if col is None:
            return None
        s = fin.loc[col].dropna()
        return pd.Series({ts.year: v for ts, v in s.items()}).sort_index()
    except Exception:
        return None

#  Panel Feature Builder 
def build_panel_features(stock_ticker, peers, sector, industry, stages, target_fcf=None):
    print(f"Building {stages}-stage panel feature matrix across {len(peers)} peers...")
    load_simfin_data()
    time.sleep(1)

    _gdp_raw = fetch_fred("GDP").resample('YE').last().pct_change().dropna()
    _cpi_raw = fetch_fred("CPIAUCSL").resample('YE').last().pct_change().dropna()
    rates    = fetch_fred("DGS10").resample('YE').last().dropna()

    if _gdp_raw.empty:
        print("  WARNING: GDP growth empty — using 2.5% flat fallback.")
        _gdp_raw = pd.Series([0.025]*14, index=pd.date_range("2010-12-31", periods=14, freq="YE"))
    if _cpi_raw.empty:
        print("  WARNING: CPI growth empty — using 2.5% flat fallback.")
        _cpi_raw = pd.Series([0.025]*14, index=pd.date_range("2010-12-31", periods=14, freq="YE"))

    gdp = _gdp_raw
    cpi = _cpi_raw

    try:
        sp500_ret = yf.Ticker("^GSPC").history(period="max")['Close'].pct_change().dropna()
    except Exception:
        sp500_ret = pd.Series(dtype=float)

    all_rows = []

    for ticker in peers:
        print(f"  Loading data for {ticker}...")

        if ticker == stock_ticker and target_fcf is not None:
            fcf_by_year = {}
            for timestamp, value in target_fcf.items():
                year = timestamp.year if isinstance(timestamp, pd.Timestamp) else int(timestamp)
                fcf_by_year[year] = value
            fcf_series = pd.Series(fcf_by_year).sort_index(ascending=True)
        else:
            fcf_series = fetch_simfin_fcf(ticker)

        revenue_series = fetch_simfin_revenue(ticker)

        if fcf_series is None or len(fcf_series) < 3:
            print(f"  Skipping {ticker} — insufficient FCF data")
            continue

        fcf_growth     = fcf_series.pct_change().dropna()
        # ── Winsorize FCF growth at ±100% to remove distorting outliers ──────
        # e.g. FCF going from near-zero to positive creates 1000%+ spikes
        # that dominate training signal without adding predictive value.
        fcf_growth     = fcf_growth.clip(-1.0, 1.0)
        revenue_growth = revenue_series.pct_change().dropna() if revenue_series is not None else pd.Series(dtype=float)

        try:
            stock_ret = yf.Ticker(ticker).history(period="max")['Close'].pct_change().dropna()
            combined  = pd.concat([stock_ret, sp500_ret], axis=1, sort=False).dropna()
            combined.columns = ['Stock', 'Market']
            roll_cov  = combined['Stock'].rolling(252).cov(combined['Market'])
            roll_var  = combined['Market'].rolling(252).var()
            roll_beta = (roll_cov / roll_var).dropna().resample('YE').last().ffill()
        except Exception:
            roll_beta = pd.Series(dtype=float)

        is_target = 1 if ticker == stock_ticker else 0

        for year_idx in fcf_growth.index:
            year    = int(year_idx)
            year_dt = pd.Timestamp(f'{year}-12-31')

            gdp_val  = gdp.get(year_dt,  np.nan)
            cpi_val  = cpi.get(year_dt,  np.nan)
            rate_val = rates.get(year_dt, np.nan)
            beta_val = roll_beta.get(year_dt, np.nan)
            rev_val  = revenue_growth.get(year_idx, np.nan)
            regime   = detect_macro_regime(year)

            row = {
                'ticker':         ticker,
                'year':           year,
                'fcf_growth':     fcf_growth[year_idx],
                'revenue_growth': rev_val,
                'gdp_growth':     gdp_val,
                'cpi_growth':     cpi_val,
                'interest_rate':  rate_val,
                'beta':           beta_val,
                'regime':         regime,
                'is_target':      is_target,
                'beta_x_gdp':     beta_val * gdp_val  if not np.isnan(beta_val) and not np.isnan(gdp_val)  else np.nan,
                'beta_x_rate':    beta_val * rate_val if not np.isnan(beta_val) and not np.isnan(rate_val) else np.nan,
                'beta_x_regime':  beta_val * regime   if not np.isnan(beta_val)                            else np.nan,
                'stages':         stages,
            }
            all_rows.append(row)

    panel = pd.DataFrame(all_rows).dropna(subset=['fcf_growth', 'gdp_growth', 'cpi_growth'])
    print(f"Panel built: {len(panel)} observations across {panel['ticker'].nunique()} stocks")

    if len(panel) < 30:
        print(f"  WARNING: Panel has only {len(panel)} rows — predictions may be unreliable.")

    return panel

#  XGBoost Panel Trainer 
def train_xgboost_panel(panel, stages):
    n_rows = len(panel)
    print(f"Training XGBoost on panel data ({stages}-stage model, {n_rows} rows)...")
    time.sleep(2)

    feature_cols = [
        'fcf_growth', 'revenue_growth', 'gdp_growth', 'cpi_growth',
        'interest_rate', 'beta', 'regime', 'is_target',
        'beta_x_gdp', 'beta_x_rate', 'beta_x_regime', 'stages'
    ]

    if n_rows < 60:
        depth, estimators, subsample = 2, 100, 0.7
        print(f"  Small panel ({n_rows} rows) — using conservative XGBoost params (depth=2, n=100).")
    elif n_rows < 150:
        depth, estimators, subsample = 3, 150, 0.8
        print(f"  Medium panel ({n_rows} rows) — using moderate XGBoost params (depth=3, n=150).")
    else:
        depth, estimators, subsample = 4, 200, 0.8
        print(f"  Large panel ({n_rows} rows) — using full XGBoost params (depth=4, n=200).")

    panel = panel.sort_values(['ticker', 'year']).copy()

    panel['short_term_target'] = panel.groupby('ticker')['fcf_growth'].shift(-1)

    if stages == 2:
        panel['long_term_target'] = panel.groupby('ticker')['fcf_growth'].transform(
            lambda x: x.shift(-2).rolling(3, min_periods=1).mean()
        )
        stage_labels = ['Short-Term (1-3yr)', 'Terminal']
    elif stages == 3:
        panel['long_term_target'] = panel.groupby('ticker')['fcf_growth'].transform(
            lambda x: x.shift(-2).rolling(3, min_periods=1).mean()
        )
        panel['transition_target'] = panel.groupby('ticker')['fcf_growth'].transform(
            lambda x: x.shift(-4).rolling(4, min_periods=2).mean()
        )
        stage_labels = ['Short-Term (1-3yr)', 'Transition (4-7yr)', 'Terminal']

    core_cols  = ['fcf_growth', 'gdp_growth', 'cpi_growth']
    panel_base = panel.dropna(subset=core_cols).copy()

    for col in feature_cols:
        if panel_base[col].isna().any():
            panel_base[col] = panel_base[col].fillna(panel_base[col].median())

    if len(panel_base) == 0:
        raise ValueError("Panel has 0 clean rows — not enough data to train.")

    scaler = StandardScaler()
    scaler.fit(panel_base[feature_cols])

    def _train(df, target_col):
        sub = df.dropna(subset=[target_col])
        if len(sub) == 0:
            raise ValueError(f"No rows with valid {target_col}.")
        m = XGBRegressor(
            n_estimators=estimators,
            max_depth=depth,
            learning_rate=0.05,
            subsample=subsample,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.5,
            random_state=42
        )
        m.fit(scaler.transform(sub[feature_cols]), sub[target_col])
        print(f"  {target_col} model trained on {len(sub)} rows.")
        return m

    xgb_short      = _train(panel_base, 'short_term_target')
    xgb_long       = _train(panel_base, 'long_term_target')
    xgb_transition = _train(panel_base, 'transition_target') if stages == 3 else None

    print(f"Models trained for stages: {stage_labels}")
    return xgb_short, xgb_long, xgb_transition, scaler, feature_cols

#  Growth Rate Predictor 
def predict_growth_rates(stock_ticker, panel, xgb_short, xgb_long, xgb_transition, scaler, feature_cols, stages):
    print(f"Predicting growth rates for {stock_ticker}...")
    time.sleep(2)

    target_rows = panel[panel['ticker'] == stock_ticker].sort_values('year')
    if target_rows.empty:
        print("No panel data found for target stock — using most recent panel row.")
        target_rows = panel.sort_values('year')

    latest = target_rows.iloc[[-1]][feature_cols].copy()
    for col in feature_cols:
        if latest[col].isna().any():
            latest[col] = latest[col].fillna(panel[col].median())

    latest_scaled = scaler.transform(latest)

    # Clip bounds set to ±50% short-term, ±30% long-term.
    # With FCF winsorized at ±100%, XGBoost predictions stay well within
    short_term_growth = float(np.clip(xgb_short.predict(latest_scaled)[0], -0.50, 0.50))
    long_term_growth  = float(np.clip(xgb_long.predict(latest_scaled)[0],  -0.30, 0.30))
    transition_growth = None

    if stages == 3 and xgb_transition is not None:
        transition_growth = float(np.clip(xgb_transition.predict(latest_scaled)[0], -0.30, 0.30))
        print(f"Short-Term Growth  (1-3yr):         {short_term_growth:.2%}")
        print(f"Transition Growth  (4-7yr):         {transition_growth:.2%}")
        print(f"Long-Term Growth   (Terminal base): {long_term_growth:.2%}")
    else:
        print(f"Short-Term Growth  (1-3yr): {short_term_growth:.2%}")
        print(f"Long-Term Growth   (4-5yr): {long_term_growth:.2%}")

    # Warn if still hitting cap after winsorization — likely a data issue
    if abs(short_term_growth) >= 0.49 or abs(long_term_growth) >= 0.29:
        print("  WARNING: Prediction near clip boundary — check panel data quality.")

    return short_term_growth, long_term_growth, transition_growth

#  Main Entry Point 
def get_growth_rates(stock_ticker, company_info, fcf, stages):
    peers, sector, industry                                = get_sector_peers(stock_ticker, company_info)
    panel                                                  = build_panel_features(stock_ticker, peers, sector, industry, stages, target_fcf=fcf)
    xgb_short, xgb_long, xgb_trans, scaler, feature_cols  = train_xgboost_panel(panel, stages)
    short_term_growth, long_term_growth, transition_growth = predict_growth_rates(
        stock_ticker, panel, xgb_short, xgb_long, xgb_trans, scaler, feature_cols, stages
    )
    return short_term_growth, long_term_growth, transition_growth
