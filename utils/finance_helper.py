# utils/finance_helper.py
import yfinance as yf
from langchain.tools import tool

@tool
def get_stock_data(ticker_symbol: str):
    """
    Gets the latest stock data for a given ticker symbol.
    Includes current price, day's high/low, and market cap.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        data = {
            "symbol": ticker_symbol,
            "company_name": info.get("longName", "N/A"),
            "current_price": info.get("currentPrice", info.get("previousClose", "N/A")),
            "day_high": info.get("dayHigh", "N/A"),
            "day_low": info.get("dayLow", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
            "summary": info.get("longBusinessSummary", "N/A")[:500] + "..." 
        }
        
        return f"Stock Data for {ticker_symbol}: {data}"
        
    except Exception as e:
        return f"Error fetching data for {ticker_symbol}: {str(e)}. Ticker might be invalid."