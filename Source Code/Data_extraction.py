# Stock Price Analysis BAC - Data Extraction and Processing


def mount_dr():
  from google.colab import drive
  drive.mount('/content/drive')
  print("Drive mounted")

mount_dr()

! pip install fredapi
! pip install yfinance
! pip install pandas_datareader
! pip install scikit-learn
! pip install xgboost
! pip install matplotlib
! pip install seaborn
! pip install numpy
! pip install pandas


#Import Libraries
def import_lib():
    global pd, np, plt, sns, xgb, train_test_split, mean_absolute_percentage_error
    global PCA, Lasso, yf, StandardScaler, mean_squared_error, r2_score
    global RepeatedKFold, cross_val_score, stats, variance_inflation_factor,mean_absolute_error,XGBRegressor,LSTM, Dense
    global TimeSeriesSplit, GridSearchCV,absolute,Sequential,MinMaxScaler,Fred, plot_importance,Input
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.linear_model import Lasso
    import yfinance as yf
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import RepeatedKFold, cross_val_score
    import scipy.stats as stats
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    import shap
    from keras.layers import Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from numpy import absolute
    from xgboost import XGBRegressor
    from sklearn.metrics import  mean_absolute_error
    from sklearn.preprocessing import MinMaxScaler
    from xgboost import plot_importance

    print("Libraries imported")

import_lib()

import fredapi as fa
from fredapi import Fred

print(xgb.__version__)

"""# Data Extraction"""



"""### Stock Data from Yahoo Finance"""

#Download stock data of Bank Of America
ticker = "BAC"  # Bank of America
start_date = "1995-01-01"

#Download historical data
stock_data = yf.download(ticker, start=start_date)

#For merging, reset the index to have 'Date' as a column
stock_data.reset_index(inplace=True)

# Display the data
print(stock_data.head())

stock_data.info()

"""### S&P index from Yahoo Finance"""

# Define the S&P 500 ticker (^GSPC)
ticker = "^GSPC"  # S&P 500 index symbol
start_date = "1995-01-01"

# Download historical data from Yahoo Finance
sp500_data = yf.download(ticker, start=start_date)

# For merging, reset the index to convert 'Date' from index to column
sp500_data.reset_index(inplace=True)

# Display the data
print(sp500_data.head())

sp500_data.info()

# Download BAC and S&P 500 data from 1995 onwards
#bac_data = yf.download("BAC", start="1995-01-01",end="2025-02-01", interval='1d',auto_adjust=False)
#sp500_data = yf.download("^GSPC", start="1995-01-01",end="2025-02-01", interval='1d',auto_adjust=False)

"""Test stock data and sp500 data"""

stock_data_cp = stock_data.copy()
sp500_data_cp= sp500_data.copy()

stock_data_cp.head()

stock_data.columns

stock_data.columns = stock_data.columns.map(lambda x: x if isinstance(x, str) else " ".join(x))
stock_data.columns

stock_data.info()

sp500_data_cp.head()

sp500_data.columns = sp500_data.columns.map(lambda x: x if isinstance(x, str) else " ".join(x))
sp500_data.columns

sp500_data.info()

#Save data to CSV to avoid download multiple times
stock_data.to_csv("/content/drive/MyDrive/capstone/bac_stock_data.csv", index=False)
sp500_data.to_csv("/content/drive/MyDrive/capstone/sp500_data.csv", index=False)



"""### Macro Economic Data from FRED"""

# Register FRED API key
fred_api_key = "b9d58ef97c3e562f2252dff3483f179f"
fred = Fred(api_key=fred_api_key)

series = fred.search('GDP')
print(series)

series = fred.search('Consumer Spending')
print(series)

series = fred.search('Unemployment Rate')
print(series)

series = fred.search('Interest Rates')
print(series)

series = fred.search('Unemployment Rate')
print(series)

series = fred.search('Housing Units') #New Privately-Owned Housing Units Started: Total Units (HOUST)
print(series)

#INDPRO
series = fred.search('Industrial Production')
print(series)

series = fred.search('Public Policy')
print(series)

series = fred.search('Nonfarm Payrolls')
print(series)

# Search for series related to economic
series = fred.search('Economic Policy')

# Display the first few results
print(series.head())

#Consumer Price Index
# Search for series related to "Consumer Price Index"
series = fred.search('Economic Policy')

# Display the first few results
print(series.head())

# Define macroeconomic indicators with Series IDs
indicators = {
    "GDP": "GDP",
    "Consumer Spending": "PCE",
    "Unemployment Rate": "UNRATE",
    "Interest Rates": "FEDFUNDS",
    "Inflation": "CPIAUCSL",
    "Housing Market": "HOUST",   #New Privately-Owned Housing Units Started
    "Federal Debt": "GFDEGDQ188S",  #(Gov. Debt to GDP)
    "Nonfarm Payrolls": "PAYEMS",  #measure of the number of U.S. workers in the economy
    "Money Supply": "M2SL"

}

# Fetch data from FRED (starting from 1995)
macro_data = {}

for name, series_id in indicators.items():
    try:
        data = fred.get_series(series_id, observation_start="1995-01-01")

        # Ensure data is retrieved and not empty
        if data is not None and not data.empty:
            data = data.to_frame(name)  # Convert Series to DataFrame
            data.index.name = "Date"  # Set index name
            macro_data[name] = data
            print(f"{name} - Data range: {data.index.min()} to {data.index.max()}")
        else:
            print(f"Warning: No data available for {name} ({series_id})")

    except Exception as e:
        print(f"Error retrieving {name} ({series_id}): {e}")

# Combine all available macroeconomic data into a single DataFrame
if macro_data:
    macro_df = pd.concat(macro_data.values(), axis=1)
    print("\nSummary of Retrieved Data:")
    print(macro_df.info())
    print(macro_df.head())

else:
    print("No data was retrieved.")

"""** Non-farm payroll **data is analyzed closely because of its importance in identifying trends related to the rate of economic growth and inflation.The U.S. Department of Labor's Bureau of Labor Statistics releases the monthly jobs report


"""

macro_df.info()

print(macro_df.tail())

print(macro_df.columns)

# Define a new continuous date range that extends at least until the desired end date
#FRED data is available until March 1st 2025 as of March 21st 2025
new_index = pd.date_range(start=macro_df.index.min(),end="2025-03-01", freq='D')
macro_df = macro_df.reindex(new_index)

# Forward fill missing values so that,if GDP hasn't updated since December 2024,
# the January 2025 rows will carry the December 2024 values.
macro_df = macro_df.ffill()

# Set the index name to "Date"
macro_df.index.name = "Date"

print(macro_df.head())

macro_df.info()

macro_df.tail()

macro_df= macro_df.reset_index()
macro_df.info()

macro_df.to_csv("/content/drive/MyDrive/capstone/macro_economic_data_elaborated2.csv", index=False)