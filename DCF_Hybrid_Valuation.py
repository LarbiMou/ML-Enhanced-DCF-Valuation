### DCF Hybrid Valuation Model by Larbi Moukhlis

import datetime
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import time
import sys
import warnings
from growth_model import get_growth_rates
from config import FRED_API_KEY, AV_API_KEY

warnings.filterwarnings('ignore')

print("Welcome to the DCF Hybrid Valuation Model!")
print("This model will help you estimate the intrinsic value of a stock using a combination of panel-based temporal data, combining both market-wide growth dynamics with stock-specific growth trajectory adaptations to help improve the accuracy and stability of DCF valuations.")

def get_stock_ticker():
    print("Hello, What Stock would you like to analyze? Please enter the stock ticker (e.g., AAPL for Apple Inc.): ")
    while True:
        stock_ticker = input("Ticker: ").upper().strip()
        if stock_ticker.isalpha() and 1 <= len(stock_ticker) <= 5:
            return stock_ticker
        else:
            print("Invalid input. Please enter a valid stock ticker (e.g., AAPL for Apple Inc.): ")

stock_ticker = get_stock_ticker()
print(f"Great! You have selected {stock_ticker} for analysis.")

print("Loading Company Information...")
time.sleep(2)
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception as e:
        print(f"Error fetching company information: {e}")
        return None
print("Company information loaded successfully")
time.sleep(1)
company_info = get_company_info(stock_ticker)
if company_info:
    print(f"Company Name: {company_info.get('longName', 'N/A')}")
    print(f"Sector: {company_info.get('sector', 'N/A')}")
    print(f"Industry: {company_info.get('industry', 'N/A')}")
else:
    print("Failed to load company information.")

def get_free_cash_flow_yfinance(stock_ticker):
    """Fetch free cash flow data from yfinance"""
    try:
        stock = yf.Ticker(stock_ticker)
        cash_flow = stock.cashflow
        if cash_flow is None or cash_flow.empty:
            print(f"No cash flow data available for {stock_ticker}")
            return None
        if 'Free Cash Flow' in cash_flow.index:
            fcf_series = cash_flow.loc['Free Cash Flow']
        elif 'Operating Cash Flow' in cash_flow.index and 'Capital Expenditure' in cash_flow.index:
            operating_cf = cash_flow.loc['Operating Cash Flow']
            capex = cash_flow.loc['Capital Expenditure']
            fcf_series = operating_cf + capex
        else:
            print(f"Could not find necessary cash flow data for {stock_ticker}")
            print(f"Available fields: {cash_flow.index.tolist()}")
            return None
        fcf_series = fcf_series.dropna()
        if fcf_series.empty:
            print(f"No valid free cash flow data found for {stock_ticker}")
            return None
        fcf_series = fcf_series.sort_index(ascending=False)
        return fcf_series
    except Exception as e:
        print(f"Error fetching cash flow data from yfinance: {e}")
        return None

print("Fetching free cash flow data...")
fcf = get_free_cash_flow_yfinance(stock_ticker)
if fcf is None:
    print("Failed to load free cash flow data. Exiting.")
    sys.exit(1)

user_input = input("Would you like to preview the free cash flow data? [Y/n]: ").lower()
if user_input in ("y", "yes", ""):
    print("\nFree Cash Flow Data (Last 5 Years, Formatted):")

    
    fcf_by_year = {ts.year: val for ts, val in fcf.items()}
    most_recent_year = max(fcf_by_year.keys())
    for i in range(5):
        yr = most_recent_year - i
        value = fcf_by_year.get(yr)
        if value is None or pd.isna(value):
            print(f"{yr}: Data not available")
        else:
            print(f"{yr}: ${value:,.2f}")

print("\nNow that we have the free cash flow data, we can proceed to the DCF valuation using the WACC and other inputs we've gathered.")

print("Creating Database for DCF Valuation...")
def create_database():
    print("Getting Risk-Free Rate...")
    time.sleep(2)
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "DGS10",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": "2010-01-01"
    }
    for attempt in range(3):
        r = requests.get(url, params=params)
        fred_data = r.json()
        if 'observations' in fred_data:
            break
        print(f"  FRED attempt {attempt+1}/3 failed (risk-free rate): {fred_data}. Retrying...")
        time.sleep(2 * (attempt + 1))
    else:
        print("  WARNING: Could not fetch risk-free rate from FRED. Defaulting to 4.25%.")
        fred_data = {'observations': [{'date': '2024-01-01', 'value': '4.25'}]}
    df_risk_free_rate = pd.DataFrame(fred_data['observations'])
    df_risk_free_rate['date'] = pd.to_datetime(df_risk_free_rate['date'])
    df_risk_free_rate['value'] = pd.to_numeric(df_risk_free_rate['value'], errors='coerce')
    df_risk_free_rate = df_risk_free_rate[['date', 'value']]
    # risk_free_rate is in percent terms (e.g. 4.25), consistent with CoE below
    risk_free_rate = df_risk_free_rate['value'].dropna().iloc[-1]
    print(df_risk_free_rate.tail(10))

    print("Getting Estimated Market Return...")
    time.sleep(2)
    print("Loading Dividend Yield data...")
    time.sleep(1)
    print("Loading Buyback Yield data...")
    time.sleep(1)
    print("Loading GDP Growth data...")
    time.sleep(1)
    print("Calculating Equity Risk Premium...")
    time.sleep(2)

    def get_rolling_beta(stock_ticker, window=252):
        print("Loading Beta")
        time.sleep(2)
        stock = yf.Ticker(stock_ticker)
        stock_returns = stock.history(period="max")['Close'].pct_change().dropna()
        sp500 = yf.Ticker("^GSPC")
        sp500_returns = sp500.history(period="max")['Close'].pct_change().dropna()
        combined = pd.concat([stock_returns, sp500_returns], axis=1).dropna()
        combined.columns = ['Stock_Returns', 'SP500_Returns']
        rolling_cov = combined['Stock_Returns'].rolling(window).cov(combined['SP500_Returns'])
        rolling_var = combined['SP500_Returns'].rolling(window).var()
        rolling_beta = rolling_cov / rolling_var
        print(rolling_beta.tail(10))
        return rolling_beta.iloc[-1]
    beta = get_rolling_beta(stock_ticker)

    def get_cost_of_debt(stock_ticker):
        print("Loading Debt data...")
        time.sleep(2)
        stock = yf.Ticker(stock_ticker)

        
        total_interest_expense = abs(stock.financials.loc['Interest Expense'].sum())
        total_debt = stock.balance_sheet.loc['Total Debt'].sum()
        print(f"Total Interest Expense: ${total_interest_expense:,.2f}")
        print(f"Total Debt: ${total_debt:,.2f}")
        if total_debt > 0:
            cost_of_debt_pre = total_interest_expense / total_debt
            print(f"Cost of Debt (pre-tax): {cost_of_debt_pre:.2%}")
        else:
            print("No debt data available. Defaulting cost of debt to 0.")
            cost_of_debt_pre = 0.0
        time.sleep(2)
        print("Computing Cost of Debt Post Tax...")
        tax_rate = 0.21
        cost_of_debt_post_tax = cost_of_debt_pre * (1 - tax_rate)

        
        cost_of_debt_post_tax_pct = cost_of_debt_post_tax * 100
        print(f"Cost of Debt Post Tax: {cost_of_debt_post_tax_pct:.2f}%")
        return cost_of_debt_post_tax_pct
    cod = get_cost_of_debt(stock_ticker)



    def get_equity_risk_premium():
        print("Loading Equity Risk Premium...")
        time.sleep(2)

        spy = yf.Ticker("SPY")
        one_year_ago = pd.Timestamp.now(tz='America/New_York') - pd.DateOffset(years=1)

        # SPY dividend yield
        try:
            spy_divs = spy.dividends[spy.dividends.index > one_year_ago].sum()
            spy_price = spy.history(period="1d")['Close'].iloc[-1]
            dividend_yield = (spy_divs / spy_price) * 100
        except Exception:
            r_div = requests.get(url, params={
                "series_id": "SP500DIV", "api_key": FRED_API_KEY,
                "file_type": "json", "observation_start": "2020-01-01"
            })
            df_div = pd.DataFrame(r_div.json()['observations'])
            df_div['value'] = pd.to_numeric(df_div['value'], errors='coerce')
            dividend_yield = df_div['value'].dropna().iloc[-1]
            print("SPY data unavailable, using FRED S&P 500 dividend yield instead")

        # Buyback yield
        spy_market_cap = spy.info.get('totalAssets', 1)
        buyback_yield = (spy.info.get('freeCashflow', 0) / spy_market_cap) * 100 if spy.info.get('freeCashflow') else 1.5

        # GDP growth
        gdp_url = "https://api.stlouisfed.org/fred/series/observations"
        gdp_params = {"series_id": "GDP", "api_key": FRED_API_KEY,
                      "file_type": "json", "observation_start": "2010-01-01"}
        r_gdp = requests.get(gdp_url, params=gdp_params)
        df_gdp = pd.DataFrame(r_gdp.json()['observations'])
        df_gdp['value'] = pd.to_numeric(df_gdp['value'], errors='coerce')
        df_gdp['yoy_growth'] = df_gdp['value'].pct_change(4) * 100
        gdp_growthrate = df_gdp['yoy_growth'].dropna().tail(8).mean()

        expected_market_return = dividend_yield + buyback_yield + gdp_growthrate
        equity_risk_premium = expected_market_return - risk_free_rate  # both in % terms

        print(f"SPY Dividend Yield: {dividend_yield:.2f}%")
        time.sleep(1)
        print(f"Buyback Yield: {buyback_yield:.2f}%")
        time.sleep(1)
        print(f"GDP Growth Rate: {gdp_growthrate:.2f}%")
        time.sleep(1)
        print(f"Market Return: {expected_market_return:.2f}%")
        time.sleep(1)
        print(f"Risk Free Rate: {risk_free_rate:.2f}%")
        time.sleep(1)
        print(f"Equity Risk Premium: {equity_risk_premium:.2f}%")
        time.sleep(2)
        return equity_risk_premium
    erp = get_equity_risk_premium()

    def cost_of_equity(erp):
        print("Calculating Cost of Equity...")
        time.sleep(2)
        coe = risk_free_rate + beta * erp  # result is in % terms (e.g. 9.1)
        print(f"Cost of Equity: {coe:.2f}%")
        return coe
    coe = cost_of_equity(erp)

    def get_weighted_average_cost_of_capital(coe, cod):
        print("Calculating Weighted Average Cost of Capital (WACC)...")
        time.sleep(2)
        stock_obj = yf.Ticker(stock_ticker)
        market_cap = stock_obj.info.get('marketCap', 1)
        total_debt = stock_obj.balance_sheet.loc['Total Debt'].sum()
        total_value = market_cap + total_debt
        equity_weight = market_cap / total_value
        debt_weight = total_debt / total_value

        
        wacc = (equity_weight * coe) + (debt_weight * cod)
        print("Loading WACC data...")
        time.sleep(3)
        print(f"WACC: {wacc:.2f}%")
        return wacc

    wacc = get_weighted_average_cost_of_capital(coe, cod)
    return wacc

user_input = input("Would you like to create the database for the DCF valuation? [Y/n]: ").lower()
wacc = None
if user_input in ("y", "yes", ""):
    wacc = create_database()

print("Database creation complete. You can now proceed to the DCF valuation step using the data we've gathered and calculated.")
print("In order to perform the DCF Valuation, we will need to learn the growth trajectory of the stock, which will be used to project future cash flows.")
time.sleep(1)
print("We can use historical growth rates to estimate this trajectory,")
print("We can also use market-wide growth dynamics to help adjust the stock-specific growth trajectory to better reflect the current economic environment and improve the accuracy of our projections.")

### Panel XGBoost Growth Rate Model + DCF Valuation

def get_dcf_stage_input():
    print("Would you like to use a 2-stage or 3-stage DCF model?")
    print("  2-stage: Short-term growth (1-5yr) + Terminal value")
    print("  3-stage: Short-term growth (1-3yr) + Transition (4-7yr) + Terminal value")
    while True:
        user_input = input("Enter 2 or 3: ").strip()
        if user_input in ("2", "3"):
            stages = int(user_input)
            print(f"Using {stages}-stage DCF model.")
            return stages
        else:
            print("Invalid input. Please enter 2 or 3.")

stages = get_dcf_stage_input()

def dcf_valuation(stock_ticker, fcf, wacc, short_term_growth, long_term_growth, transition_growth, stages, silent=False):
    if not silent:
        print("Performing DCF Valuation...")
        time.sleep(5)

    wacc = wacc / 100  # Convert from % to decimal (works correctly now that WACC is always in %)
    base_fcf = fcf.sort_index(ascending=True).iloc[-1]
    projected_fcf = []

    if stages == 2:
        for i in range(1, 6):
            rate = short_term_growth if i <= 3 else long_term_growth
            projected_fcf.append(base_fcf * (1 + rate) ** i)
    elif stages == 3:
        for i in range(1, 8):
            rate = short_term_growth if i <= 3 else transition_growth
            projected_fcf.append(base_fcf * (1 + rate) ** i)

    if not silent:
        print(f"Projected FCF ({stages}-stage): {[f'${f:,.2f}' for f in projected_fcf]}")

    terminal_growth_rate = float(np.clip(long_term_growth, 0.015, 0.030))
    if wacc <= terminal_growth_rate:
        terminal_growth_rate = wacc - 0.01
    terminal_value = projected_fcf[-1] * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)

    if not silent:
        print(f"Terminal Value: ${terminal_value:,.2f}")

    discount_factors = [(1 + wacc) ** i for i in range(1, len(projected_fcf) + 1)]
    discounted_fcf = [projected_fcf[i] / discount_factors[i] for i in range(len(projected_fcf))]
    discounted_terminal_value = terminal_value / discount_factors[-1]
    dcf_value = sum(discounted_fcf) + discounted_terminal_value

    if not silent:
        print(f"\nESTIMATED INTRINSIC VALUE (Enterprise Value): ${dcf_value:,.2f}")
        try:
            stock = yf.Ticker(stock_ticker)
            shares_outstanding = stock.info.get('sharesOutstanding', None)
            current_price = stock.info.get('currentPrice', None)
            if shares_outstanding:
                intrinsic_value_per_share = dcf_value / shares_outstanding
                print(f"Shares Outstanding: {shares_outstanding:,}")
                print(f"INTRINSIC VALUE PER SHARE: ${intrinsic_value_per_share:.2f}")
                if current_price:
                    print(f"\nCurrent Market Price: ${current_price:.2f}")
                    difference = intrinsic_value_per_share - current_price
                    percent_diff = (difference / current_price) * 100
                    if percent_diff > 0:
                        print(f"Stock appears UNDERVALUED by ${difference:.2f} ({percent_diff:.2f}%)")
                    else:
                        print(f"Stock appears OVERVALUED by ${abs(difference):.2f} ({abs(percent_diff):.2f}%)")
            else:
                print("(Shares outstanding data not available)")
        except Exception as e:
            print(f"Could not calculate per-share value: {e}")
        print()

    return dcf_value

def monte_carlo_dcf(stock_ticker, fcf, wacc, short_term_growth, long_term_growth, transition_growth, stages, n_simulations=10000):
    print(f"\nRunning Monte Carlo Simulation ({n_simulations:,} iterations)...")
    time.sleep(2)

    results = []
    wacc_decimal = wacc / 100  # Convert % → decimal once here

    for _ in range(n_simulations):
        s_short      = np.random.normal(short_term_growth, 0.03)
        s_long       = np.random.normal(long_term_growth,  0.02)
        s_transition = np.random.normal(transition_growth, 0.02) if transition_growth is not None else None
        s_wacc       = np.random.normal(wacc_decimal, 0.005)

        s_wacc = max(s_wacc, 0.04)
        s_short = float(np.clip(s_short, -0.25, 0.25))
        s_long  = float(np.clip(s_long,  -0.15, 0.15))
        if s_transition is not None:
            s_transition = float(np.clip(s_transition, -0.15, 0.15))

        
        value = dcf_valuation(
            stock_ticker, fcf, s_wacc * 100, s_short, s_long, s_transition, stages, silent=True
        )
        results.append(value)

    results = np.array(results)

    try:
        stock = yf.Ticker(stock_ticker)
        shares = stock.info.get('sharesOutstanding', None)
        current_price = stock.info.get('currentPrice', None)
    except Exception:
        shares = None
        current_price = None

    print(f"\n{'='*52}")
    print(f"  MONTE CARLO RESULTS  ({n_simulations:,} simulations)")
    print(f"{'='*52}")
    print(f"  Bear Case  (10th percentile): ${np.percentile(results, 10):>20,.2f}")
    print(f"  Base Case  (50th percentile): ${np.percentile(results, 50):>20,.2f}")
    print(f"  Bull Case  (90th percentile): ${np.percentile(results, 90):>20,.2f}")
    print(f"  Mean:                         ${np.mean(results):>20,.2f}")
    print(f"{'='*52}")

    if shares:
        bear_ps = np.percentile(results, 10) / shares
        base_ps = np.percentile(results, 50) / shares
        bull_ps = np.percentile(results, 90) / shares
        print(f"\n  PER SHARE:")
        print(f"  Bear: ${bear_ps:.2f}  |  Base: ${base_ps:.2f}  |  Bull: ${bull_ps:.2f}")
        if current_price:
            print(f"  Current Price: ${current_price:.2f}")
            prob_undervalued = (results / shares > current_price).mean()
            print(f"  Probability stock is undervalued: {prob_undervalued:.1%}")
    print()

    return results

# Execute Growth Rate Model
print("Running Growth Rate Model...")
short_term_growth, long_term_growth, transition_growth = get_growth_rates(
    stock_ticker, company_info, fcf, stages
)

# Run DCF
user_input = input("Would you like to perform the DCF valuation now? [Y/n]: ").lower()
if user_input in ("y", "yes", ""):
    dcf_value = dcf_valuation(
        stock_ticker, fcf, wacc, short_term_growth, long_term_growth, transition_growth, stages
    )

    mc_input = input("Would you like to run a Monte Carlo simulation? [Y/n]: ").lower()
    if mc_input in ("y", "yes", ""):
        mc_results = monte_carlo_dcf(
            stock_ticker, fcf, wacc, short_term_growth, long_term_growth, transition_growth, stages
        )