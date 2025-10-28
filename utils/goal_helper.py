# utils/goal_helper.py
import math
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain_core.messages import SystemMessage, HumanMessage
from utils.search_helper import get_web_search_tool
from utils.finance_helper import get_stock_data

def _simulate_step_up_sip(goal_amount, initial_monthly_sip, annual_rate, step_up_percent):
    """Internal helper to simulate step-up SIP growth year by year."""
    total_corpus = 0
    current_monthly_sip = initial_monthly_sip
    r_monthly = annual_rate / 12
    step_up_rate = step_up_percent / 100.0
    
    try:
        for year in range(1, 61):
            for _ in range(12):
                total_corpus += current_monthly_sip
                total_corpus *= (1 + r_monthly)
            
            if total_corpus >= goal_amount:
                return year 
            
            current_monthly_sip *= (1 + step_up_rate)
            
        return "60+" 
    except Exception:
        return "Error"

def calculate_tenure(goal_amount, investment_type, amount, annual_rate, is_step_up, step_up_percent):
    """
    Calculates the estimated time (in years) to reach the financial goal.
    """
    try:
        r_monthly = annual_rate / 12
        
        if investment_type == "Lumpsum":
            n_years = math.log(goal_amount / amount) / math.log(1 + annual_rate)
            return round(n_years, 1)
        
        if investment_type == "SIP":
            if not is_step_up:
                p_monthly = amount
                n_months = math.log(((goal_amount / p_monthly) * r_monthly) + 1) / math.log(1 + r_monthly)
                return round(n_months / 12, 1)
            else:
                return _simulate_step_up_sip(goal_amount, amount, annual_rate, step_up_percent)
                
    except (ValueError, OverflowError) as e:
        print(f"Error in tenure calculation: {e}")
        return "Error"

def _get_historical_performance(tickers, years=15):
    """
    Fetches historical data for a list of tickers and calculates
    annualized return, volatility (risk), and Sharpe ratio.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365.25)
    
    performance_data = []
    
    try:
        data = yf.download(
            tickers, 
            start=start_date, 
            end=end_date, 
            group_by='ticker', 
            auto_adjust=False
        )
        
        if data.empty:
             print("YFinance returned no data.")
             return []

        for ticker in tickers:
            try:
                ticker_data = data[ticker]['Adj Close']
                
                if ticker_data.isnull().all(): 
                    print(f"Warning: No 'Adj Close' data for {ticker}. Skipping.")
                    continue

                log_returns = np.log(ticker_data / ticker_data.shift(1)).dropna()
                
                if log_returns.empty:
                    print(f"Warning: No return data for {ticker}. Skipping.")
                    continue

                annual_return = log_returns.mean() * 252
                annual_volatility = log_returns.std() * np.sqrt(252)

                sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
                
                performance_data.append({
                    "ticker": ticker,
                    "annual_return_pct": round(annual_return * 100, 2),
                    "annual_volatility_pct": round(annual_volatility * 100, 2),
                    "sharpe_ratio": round(sharpe_ratio, 2)
                })
            except Exception as inner_e:
                print(f"Error processing {ticker}: {inner_e}")
                
        return performance_data
        
    except Exception as e:
        print(f"Error in historical performance calculation: {e}")
        return []

def get_investment_basket(chat_model, goal_amount, risk_profile, investment_type, amount, tenure_years):
    """
    Generates a personalized investment basket recommendation using LLM and data analysis.
    """
    try:
        ASSET_UNIVERSE = [
            # US Equity (Large Cap)
            'VOO',  # S&P 500
            'VUG',  # S&P 500 Growth
            'VTV',  # S&P 500 Value
            
            # US Equity (Mid/Small Cap)
            'VO',   # Mid-Cap
            'VB',   # Small-Cap
            
            # US Tech / Sectors
            'QQQ',  # NASDAQ 100
            'XLK',  # Technology
            'XLF',  # Financials
            'XLV',  # Health Care
            'XLE',  # Energy
            
            # International Equity
            'VEU',  # All-World ex-US
            'EEM',  # Emerging Markets
            'EFA',  # Developed (EAFE)
            'VGK',  # Europe
            
            # Bonds (Government)
            'BND',  # Total US Bond Market
            'TLT',  # 20+ Year Treasury
            'SHY',  # 1-3 Year Treasury
            'BNDX', # Total International Bond
            
            # Bonds (Corporate)
            'LQD',  # Investment Grade Corporate
            'HYG',  # High-Yield Corporate (Junk)
            
            # Real Estate
            'VNQ',  # US Real Estate
            'VNQI', # International Real Estate
            
            # Commodities
            'GLD',  # Gold
            'SLV',  # Silver
            'DBC',  # Broad Commodities (Oil, Gas, Gold, etc.)
            
            # Crypto (Note: yfinance uses -USD)
            'BTC-USD', # Bitcoin
            'ETH-USD'  # Ethereum
        ]

        print("Fetching historical performance data...")
        performance_data = _get_historical_performance(ASSET_UNIVERSE, years=15)
        print(f"Found {len(performance_data)} assets with valid data.")
        
        if not performance_data:
            return "Error: Could not retrieve historical market data to build your plan. Please try again later."

        top_assets = []
        if risk_profile == "Low Risk":
            low_risk_candidates = [p for p in performance_data if p['annual_volatility_pct'] < 20 and 'HYG' not in p['ticker'] and '-USD' not in p['ticker']]
            top_assets = sorted(low_risk_candidates, key=lambda x: x['annual_volatility_pct'])[:5] 
            search_query = "market outlook for low-volatility assets like bonds and stable ETFs"
        elif risk_profile == "Medium Risk":
            medium_risk_candidates = [p for p in performance_data if p['annual_volatility_pct'] < 35]
            top_assets = sorted(medium_risk_candidates, key=lambda x: x['sharpe_ratio'], reverse=True)[:5] 
            search_query = "market outlook for balanced assets like S&P 500 and diversified ETFs"
        else: 
            high_risk_candidates = [p for p in performance_data if p['annual_return_pct'] > 5] 
            top_assets = sorted(high_risk_candidates, key=lambda x: x['sharpe_ratio'], reverse=True)[:5] 
            search_query = "market outlook for high-growth assets like NASDAQ, Bitcoin, and emerging markets"

        dynamic_tickers = [asset['ticker'] for asset in top_assets]
        print(f"Dynamically selected tickers: {dynamic_tickers}")

        context_str = "--- START DATA-DRIVEN CONTEXT ---\n"
        context_str += f"Here is the 15-year performance analysis for assets matching your '{risk_profile}' profile (Return vs. Risk):\n"
        for asset in top_assets:
            context_str += f"- {asset['ticker']}: Return={asset['annual_return_pct']}%, Risk={asset['annual_volatility_pct']}%, Sharpe={asset['sharpe_ratio']}\n"
        
        search_tool = get_web_search_tool()

        for ticker in dynamic_tickers:
            stock_data = get_stock_data.invoke(ticker)
            context_str += f"\nLive Data for {ticker}:\n{stock_data}\n"

        if search_tool:
            news = search_tool.invoke(search_query)
            context_str += f"\nRecent Market News:\n{news}\n"
        context_str += "--- END CONTEXT ---"

        system_prompt = f"""
        You are an expert financial planner named "NeoFin".
        You are building a investment plan for a user with a **{risk_profile}** risk appetite.
        You MUST adhere to this risk profile.
        
        **CRITICAL_RULE**: You must NEVER give a "buy" or "sell" recommendation. 
        Instead of "you should buy AAPL," suggest asset allocations and representative examples.
        ALWAYS provide a disclaimer that this is not financial advice.
        
        You MUST use the provided "DATA-DRIVEN CONTEXT" to build your recommendation.
        This context contains a list of assets *dynamically selected* to match the user's risk profile,
        based on 15 years of performance data (Return vs. Risk vs. Sharpe Ratio).
        
        Your job is to synthesize this data and present it as a coherent plan.
        """
        
        human_message = f"""
        Here is my financial goal:
        - **Goal Amount:** ${goal_amount:,.2f}
        - **Investment Plan:** ${amount:,.2f} as a {investment_type}
        - **My Risk Profile:** {risk_profile}
        - **Calculated Time Horizon:** {tenure_years} years
        
        Based on all of this, and especially the provided data-driven context, please provide a recommended investment basket.
        
        Your response should include:
        1.  A suggested **Asset Allocation** (e.g., X% Equity, Y% Bonds, Z% Alternatives).
        2.  For each asset class, provide 1-2 **representative examples** from the "DATA-DRIVEN CONTEXT", explaining *why* their historical risk/return profile (e.g., "high Sharpe ratio") fits my goal.
        3.  A brief justification for why this basket aligns with my risk profile.
        4.  The mandatory disclaimer.
        """
        
        messages = [
            SystemMessage(content=system_prompt + "\n\n" + context_str),
            HumanMessage(content=human_message)
        ]

        print("Generating LLM recommendation...")
        response = chat_model.invoke(messages)
        return response.content
        
    except Exception as e:
        print(f"Error getting investment basket: {e}")
        return f"An error occurred while generating the recommendation: {e}"