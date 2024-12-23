!pip install yfinance
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

def classify_cryptocurrencies(tickers, start_date, end_date):  
    """  
    Classify cryptocurrencies based on the slope of their price line (in degrees).  
    
    Args:  
        tickers (list): List of cryptocurrency tickers.  
        start_date (datetime): Start date for the historical data.  
        end_date (datetime): End date for the historical data.  
    
    Returns:  
        dict: Classified cryptocurrencies in different categories.  
    """  
    classifications = {  
        "Strong Uptrend": [],  
        "Moderate Uptrend": [],  
        "Sideways/Consolidation": [],  
        "Moderate Downtrend": [],  
        "Strong Downtrend": []  
    }  
    
    for ticker in tickers:  
        crypto = yf.Ticker(ticker)  
        data = crypto.history(start=start_date, end=end_date)  
        
        # Calculate the slope of the price line  
        # x = np.arange(len(data))  
        # y = data['Close']  
        # slope, _ = np.polyfit(x, y, 1)
        x = np.arange(len(data)).reshape(-1, 1)
        y = data['Close'].values

        # Fit the linear regression model
        model = LinearRegression()
        model.fit(x, y)  
        # Get the slope and intercept of the regression line
        slope = model.coef_[0]
        intercept = model.intercept_
                
        # Plot the Bitcoin price line with the regression line
        plt.figure(figsize=(6, 3))
        plt.plot(data.index, data['Close'])
        plt.plot(data.index, model.predict(x), color='red')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.title(ticker + ' Price over the Past Year')
        plt.grid()
        plt.legend([ticker + ' Price', 'Regression Line'])
        plt.show()
        
        # Convert the slope to degrees  
        angle = np.arctan(slope) * (180 / np.pi)  
        
        # Classify the cryptocurrency based on the angle  
        if angle >= 45:  
            classifications["Strong Uptrend"].append(ticker)  
        elif 30 <= angle < 45:  
            classifications["Moderate Uptrend"].append(ticker)  
        elif 15 <= angle < 30:  
            classifications["Sideways/Consolidation"].append(ticker)  
        elif 0 <= angle < 15:  
            classifications["Moderate Downtrend"].append(ticker)  
        else:  
            classifications["Strong Downtrend"].append(ticker)  
    
    return classifications  

# Example usage  
tickers = ["BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "USDC-USD", "BUSD-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "DOT-USD", "SHIB-USD", "MATIC-USD", "AVAX-USD", "UNI-USD", "LTC-USD", "LINK-USD", "TRX-USD", "CRO-USD", "NEAR-USD", "SOL-USD", "ATOM-USD", "ETC-USD", "ALGO-USD", "ICP-USD", "XLM-USD", "XMR-USD", "FIL-USD", "VET-USD", "FLOW-USD", "QNT-USD", "EGLD-USD", "MANA-USD", "FTM-USD", "SAND-USD", "AXS-USD", "HBAR-USD", "EOS-USD", "CRV-USD", "KCS-USD", "STX-USD", "DAI-USD", "RUNE-USD", "HNT-USD", "MKR-USD", "LRC-USD", "ZEN-USD", "ONE-USD", "ZIL-USD", "AAVE-USD", "GALA-USD"]
start_date = datetime.now() - timedelta(days=365)  
end_date = datetime.now()  

classifications = classify_cryptocurrencies(tickers, start_date, end_date)  
for category, crypto_list in classifications.items():  
    print(f"{category}: {', '.join(crypto_list)}")
