

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

"""## Data Cleaning and Merging"""

#Import CSV
bac_data1=pd.read_csv("/content/drive/MyDrive/capstone/bac_stock_data.csv")
sp500_data1=pd.read_csv("/content/drive/MyDrive/capstone/sp500_data.csv")
macro_df=pd.read_csv("/content/drive/MyDrive/capstone/macro_economic_data_elaborated2.csv")
#macro_df=pd.read_csv("/content/drive/MyDrive/capstone/macro_economic_data_revised.csv")

bac_data1.info()

sp500_data1.info()

macro_df.info()

bac_data1.columns

"""#### Clean stock data columns/headers"""

#Transform columns ideal for machine learning modeling
bac_data1.columns = bac_data1.columns.map(lambda x: "_".join(x).strip().lower() if not isinstance(x, str) else x.strip().replace(" ", "_").lower())

# Print updated column names to verify
print(bac_data1.columns)

bac_data1.head()

"""#### Clean GSPC data columns/headers"""

sp500_data1.head()

#Transform columns ideal for machine learning modeling
sp500_data1.columns = sp500_data1.columns.map(lambda x: "_".join(x).strip().lower() if not isinstance(x, str) else x.strip().replace(" ", "_").lower())

# Print updated column names to verify
print(sp500_data1.columns)

#Remove unnecessary special character also from headers
sp500_data1.columns = sp500_data1.columns.map(lambda x: "_".join(x).strip().lower().replace("^", "") if not isinstance(x, str) else x.strip().replace(" ", "_").lower().replace("^", ""))

# Print updated column names to verify
print(sp500_data1.columns)

"""#### Clean MacroEconomic Data columns/headers"""

#Transform columns ideal for machine learning modeling
macro_df.columns = macro_df.columns.map(lambda x: "_".join(x).strip().lower() if not isinstance(x, str) else x.strip().replace(" ", "_").lower())

# Print updated column names to verify
print(macro_df.columns)

bac_data1.info()

sp500_data1.info()



# Convert 'date' column to datetime in both DataFrames
bac_data1['date'] = pd.to_datetime(bac_data1['date'])
sp500_data1['date'] = pd.to_datetime(sp500_data1['date'])

# Merge bac stock adta and GSPC  datasets on date
merged_data = bac_data1.merge(sp500_data1, how="left", on="date")

print(merged_data.head())

merged_data.info()

merged_bac_gspc_data = merged_data.to_csv("/content/drive/MyDrive/capstone/BAC_SP500_merged_data.csv",index=False)

"""Merge macro data"""

macro_df.info()

macro_df['date'] = pd.to_datetime(macro_df['date'])
# Merge 'merged_data' and 'macro_df' on 'Date' column
merged_data = merged_data.merge(macro_df, how='left', on='date')

# Print the first few rows of the merged data
print(merged_data.head())

stock_analysis_data = merged_data.to_csv("/content/drive/MyDrive/capstone/stock_analysis_data.csv",index=False)

merged_data.info()



merged_data.tail(35)

"""Economic indicators have not been updated since January 2025. The dataset may contain null values starting from February 2025. Therefore, we will remove all data from February 2025 onward."""



"""### Dataset may have null values as FRED data is available until Jan 31st, 2025  - As of March 21st"""

merged_data.isna().sum()

merged_data = merged_data.dropna()

merged_data.duplicated().sum()



merged_data.tail()

# Plot box plot for 'close_bac'
plt.figure(figsize=(8, 5))
sns.boxplot(x=merged_data['close_bac'])
plt.title('Box Plot for close_bac to Identify Outliers')
plt.xlabel('close_bac')
plt.show()

cutoff_date = '2000-01-01'
df_cleaned = merged_data[merged_data['date'] >= cutoff_date]

# Plot box plot for 'close_bac'
plt.figure(figsize=(8, 5))
sns.boxplot(x=df_cleaned['close_bac'])
plt.title('Box Plot for close_bac to Identify Outliers')
plt.xlabel('close_bac')
plt.show()

merged_data.to_csv("/content/drive/MyDrive/capstone/merged_bac_gspc_macro_data.csv",index=False)

df_cleaned.to_csv("/content/drive/MyDrive/capstone/merged_bac_gspc_macro_data2.csv",index=False)

df_cleaned.info()

df_cleaned.head()

"""### Dataset contains values from 3rd January 2000 up to 28th February 2025. Further analysis will be based on the period from January 1995 to February2025.

### Dataset is free of null values and duplicates, and all column names have been transformed to make it ideal for ML modeling.
"""



"""# XGBoost Model for regression

XGBoost models represent all problems as a regression predictive modeling problem that only takes numerical values as input.

Ensembles are constructed from decision tree models. Trees are added one at a time to the ensemble and fit to correct the prediction errors made by prior models. This is a type of ensemble machine learning model referred to as boosting.

Models are fit using any arbitrary differentiable loss function and gradient descent optimization algorithm. This gives the technique its name, “gradient boosting,” as the loss gradient is minimized as the model is fit, much like a neural network.
"""



import_lib()
mount_dr()

"""# Load the dataset"""

# Load the dataset
df = pd.read_csv("/content/drive/MyDrive/capstone/merged_bac_gspc_macro_data2.csv", parse_dates=["date"], index_col="date")

stats.probplot(df['close_bac'], dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.show()

df.columns

print("Highest BAC closing price",max(df['close_bac']))
print("Lowest BAC closing price",min(df['close_bac']))

"""# Highest and lowest stock price-BAC"""

#Highest and lowest stock price for BAC
min_close = df.loc[df["close_bac"].idxmin(), "close_bac"]
max_close = df.loc[df["close_bac"].idxmax(), "close_bac"]
constant_start_index1 = df[df['close_bac'] == max_close].index[0]
constant_start_index2 = df[df['close_bac'] ==min_close].index[0]

print(f"Highest Closing Price Date: {constant_start_index1}",max(df['close_bac']))
print(f"Lowest Closing Price Date: {constant_start_index2}",min(df['close_bac']))

"""## The highest closing price for BAC was  47.44  on February 6, 2025, while the lowest closing price was  2.47   on March 6, 2009. This highlights significant price fluctuations over time from 2000 to present, reflecting market trends and economic conditions."""

df.head()

"""Feature correlation"""

df.columns

"""## Multicollinarity check to avoid biases"""

#Correlation check to avoid bias
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Define feature set
features = ['close_bac', 'high_bac', 'low_bac', 'open_bac', 'volume_bac',
       'close_gspc', 'high_gspc', 'low_gspc', 'open_gspc', 'volume_gspc',
       'gdp', 'consumer_spending', 'unemployment_rate', 'interest_rates',
       'inflation', 'housing_market', 'federal_debt', 'nonfarm_payrolls',
       'money_supply']

# Compute VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = features
vif_data["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]

print(vif_data)

"""### Drop features that shows high multicollinearity"""



#First elimation highly correlated features
df.drop(columns=['high_bac', 'low_bac', 'open_bac', 'high_gspc', 'low_gspc', 'open_gspc'], inplace=True)   #,'consumer_spending' (consider if required after first run of row features)

df.drop(columns=['consumer_spending'], inplace=True)

df.columns

# Define feature set
features = ['close_bac', 'volume_bac', 'close_gspc', 'volume_gspc', 'gdp',
       'unemployment_rate', 'interest_rates', 'inflation', 'housing_market',
       'federal_debt', 'nonfarm_payrolls', 'money_supply']

# Compute VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = features
vif_data["VIF"] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]

print(vif_data)



#Correlation check
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

df.head()



"""# Visualization of Features"""

# Convert index to datetime if needed (assuming df has a Date column)
plot_df = df.copy()  # Create a copy of the DataFrame


import matplotlib.dates as mdates

# Convert DatetimeIndex to numeric format
plot_df["date_ordinal"] = plot_df.index.map(lambda x: x.toordinal())

# Create figure
plt.figure(figsize=(16,10))

# Plot actual closing prices as a line plot
sns.lineplot(data=plot_df, x=plot_df.index, y="close_bac", label="Actual BAC Closing Prices ",color='green')



# Labels & Title
plt.xlabel("Date")
plt.ylabel("BAC Closing Price")
plt.title("BAC Closing Prices from Yahoo Finance")
plt.legend()
plt.grid(True)

# Identify min and max points
min_idx = df["close_bac"].idxmin()
max_idx = df["close_bac"].idxmax()
min_price = df["close_bac"].min()
max_price = df["close_bac"].max()

# Set a light grid background
sns.set_style("whitegrid")

# Create figure and plot
plt.figure(figsize=(14, 8))
plt.plot(df.index, df["close_bac"],  linewidth=2,color='black', linestyle="-", label="BAC Closing Price")

# Add markers at min and max points
plt.scatter([min_idx, max_idx], [min_price, max_price], color="red", s=100, zorder=3, label="Min/Max Points")

# Annotate min and max points
#plt.annotate(f"Min: ${min_price:.2f}", xy=(min_idx, min_price), xytext=(min_idx, min_price - 5),
      #       arrowprops=dict(facecolor="red", arrowstyle="->"), fontsize=12, color="black")

#plt.annotate(f"Max: ${max_price:.2f}", xy=(max_idx, max_price), xytext=(max_idx, max_price + 5),
  #           arrowprops=dict(facecolor="red", arrowstyle="->"), fontsize=12, color="black")

# Customize the plot
plt.xlabel("Date", fontsize=14, fontweight="bold")
plt.ylabel("Closing Price ($)", fontsize=14, fontweight="bold")
plt.title("Stock Closing Price Trend (BAC)", fontsize=16, fontweight="bold")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)  # Lighter grid for aesthetics

# Show the plot
plt.show()

# Create a figure and plot
plt.figure(figsize=(14, 8))
plt.plot(df.index, df["close_bac"], color="darkblue", linewidth=1, linestyle="-")  # Specify x and y directly

# Customize the plot
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.title("Stock Closing Price Trend")
plt.grid(True)  # Add grid for better readability
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

# Show the plot
plt.show()

# Define specific date ranges
date_ranges = [
    ("2000-01-01", "2006-12-31"),
    ("2007-01-01", "2009-12-31"),
    ("2010-01-01", "2020-12-31"),
    ("2021-01-01", str(plot_df.index.max().date()))  # Till the latest available date
]

# Create subplots
fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=False)  # 4 subplots

for i, (start, end) in enumerate(date_ranges):
    ax = axes[i]  # Get the current subplot
    subset = plot_df.loc[start:end]  # Filter data for the given date range

    sns.lineplot(data=subset, x=subset.index, y="close_bac", ax=ax, label=f"{start[:4]} - {end[:4]}",color='green')

    # Format x-axis for better readability
    ax.xaxis.set_major_locator(mdates.YearLocator())  # Major ticks every year
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))  # Minor ticks every 6 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as Year-Month

    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=45)

    ax.set_ylabel("BAC Closing Price")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5)

# Set main title
plt.suptitle("BAC Stock Price Trends (Segmented by Time Periods)", fontsize=14)

plt.tight_layout()
plt.show()

plot_df.columns

plt.figure(figsize=(12,6))
#plt.plot(plot_df['date_ordinal'], plot_df['close_bac'], label='BAC Close Price', color='blue')
#plt.plot(plot_df['date_ordinal'], plot_df['close_gspc'], label='S&P 500 Close Price', color='red')
sns.lineplot(data=plot_df, x=plot_df.index, y=plot_df['close_bac'], label="BAC Close Price", color="orange")
sns.lineplot(data=plot_df, x=plot_df.index, y=plot_df['close_gspc'], label="S&P 500 Close Price")

plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("BAC vs S&P 500 Closing Price Trends")
plt.legend()
plt.grid(True)
plt.show()

"""#### Visualizing Variable Relationships"""

#Check Stationarity
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series):
    ad_result = adfuller(series)
    print("ADF Statistic:", ad_result[0])
    print("p-value:",ad_result[1])
    print("Critical value:",ad_result[4])
    if ad_result[1] <= 0.05:
        print("The data is stationary.")
    else:
        print("The data is not stationary.")

check_stationarity(df['close_bac'])

df.columns

#Checking relationships between multiple variables.
sns.pairplot(df[['close_bac', 'volume_bac', 'close_gspc', 'volume_gspc', 'gdp',
       'unemployment_rate', 'interest_rates', 'inflation', 'housing_market',
       'federal_debt', 'nonfarm_payrolls', 'money_supply']], plot_kws={'color': 'red'})
plt.savefig('/content/drive/MyDrive/capstone/variable_relationships.png')
plt.show()

# Select only the target variable and other features
target_variable = 'close_bac'
features = [col for col in df.columns if col != target_variable]  # Exclude target from features

# Create pairplot for only target vs features
sns.pairplot(df, x_vars=features, y_vars=[target_variable], kind='scatter', plot_kws={'alpha':0.5, 'color':'blue'})

plt.suptitle("Pairwise Relationships Between Close_BAC and Other Variables", y=1.02)
plt.show()



# List of variables to plot against `close_bac`
variables = ['close_bac', 'volume_bac', 'close_gspc', 'volume_gspc', 'gdp',
       'unemployment_rate', 'interest_rates', 'inflation', 'housing_market',
       'federal_debt', 'nonfarm_payrolls', 'money_supply']

# Set up the figure and axes
fig, axes = plt.subplots(1, len(variables), figsize=(20, 5))

for i, var in enumerate(variables):
    sns.regplot(x=df[var], y=df['close_bac'], scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=axes[i])
    axes[i].set_title(f'close_bac vs {var}')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('close_bac')

plt.tight_layout()
plt.show()

"""### Using only positive trend variables might lead to biased prediction, reducing model accuracy. Instead of pre-selecting only positive trend variables, we will let XGBoost decide the important features."""



"""# SPLIT THE DATA

Data split into train and test

A time series split is used to prepare the data before modeling along with defining potential predictors as y and target as X. Time Series cross-validator provides train/test indices to split time series data samples that are observed at fixed time intervals, in train/test sets. This class can be used to cross-validate time series data samples that are observed at fixed time intervals. (2025, scikit-learn)
"""

# Define potential predictors as (X) and target as (y)
X = df.drop(columns=["close_bac"])
y = df["close_bac"]

from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)  # 5 splits for cross-validation

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("Training set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])

#DO NOT RUN  - Not required for XGBoost


# Feature scaling using StandardScaler (important for XGBoost)
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)
# Convert scaled data back to DataFrame
#X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
#X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

model = XGBRegressor()

model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# plot feature importance
plot_importance(model)
plt.show()

"""The importance is calculated based on weight, which is the number of times a feature appears in a tree."""

from xgboost import plot_tree
plot_tree(model)
plt.show()

mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")

mape = mean_absolute_percentage_error(y_test, y_pred)
print('MAPE: %.3f' % mape)


# Evaluate model on test data
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5


print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

max_close_bac = df['close_bac'].max()
print(f"Highest value of close_bac in historical data: {max_close_bac}")



###########NOT REQUIRED#####################
# Convert predictions and actual values into a DataFrame
results_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})

# Print first 10 rows
print(results_df.head(10))

#############NOT REQUIRED#################
# Print first 10 rows
print(results_df.head(10))

xgb_opt_feature = XGBRegressor(
        n_estimators=500, learning_rate=0.01, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        gamma=0.1, random_state=42
    )
xgb_opt_feature.fit(X_train, y_train)

# Predict and evaluate
y_pred = xgb_opt_feature.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
plot_importance(xgb_opt_feature)
plt.show()
print("\n\n")
print(f"Mean Absolute Error: {mae}")

# Make predictions on test set
y_pred = xgb_opt_feature.predict(X_test)

# Evaluate model on test data
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5


print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

"""### Lower RMSE value could be due to irrelevantfeatures, which needs to be removed for better model performance"""

import shap


# Create an explainer object using the trained model and the same dataset used for training
explainer = shap.Explainer(model, X_train)

# Calculate SHAP values
shap_values = explainer(X_train)  # Use the same data used for training

# Create a summary plot to visualize feature importance
shap.summary_plot(shap_values, X_train)  # Use the same data for plotting

"""the bar plots above are just summary statistics from the values shown in the beeswarm plots below."""

shap.plots.beeswarm(shap_values.abs, color="shap_red")

"""Key Points from SHAP Summary Plot:

Negative SHAP values (left) indicate a lowering effect on the prediction.

Positive SHAP values (right) indicate an increasing effect on the prediction.

interest_rates, gdp, and close_gspc have a significant impact on the model’s predictions.
unemployment_rate, inflation, and nonfarm_payrolls also have a notable influence
"""

# Feature Importance
feature_importance = xgb_opt_feature.feature_importances_
feature_names = X_train.columns

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance, y=feature_names, palette="viridis")
plt.title("Feature Importance using XGBoost")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

"""# LSTM with lagged target variable"""

# Load the dataset
rnn_df = pd.read_csv('/content/drive/MyDrive/capstone/merged_bac_gspc_macro_data2.csv', parse_dates=['date'])

# Convert the 'date' column to datetime format if not already
rnn_df['date'] = pd.to_datetime(rnn_df['date'])

# Create a complete date range from the min to max date
all_dates = pd.date_range(start=rnn_df['date'].min(), end=rnn_df['date'].max(), freq='B')  # 'B' for business days

# Find missing dates
missing_dates = all_dates.difference(rnn_df['date'])

# Display missing dates
print("Missing Dates:")
print(missing_dates)
print(len(missing_dates))

print(len(rnn_df))

# Ensure 'date' is in datetime format
#rnn_df['date'] = pd.to_datetime(rnn_df['date'])

# Create a complete date range from min to max date (business days)
#all_dates = pd.date_range(start=rnn_df['date'].min(), end=rnn_df['date'].max(), freq='B')

# Reindex with all dates to introduce missing dates
#rnn_df = rnn_df.set_index('date').reindex(all_dates)

# Forward fill missing values
#rnn_df = rnn_df.ffill()

# Reset index to restore 'date' column
#rnn_df = rnn_df.reset_index().rename(columns={'index': 'date'})

#print("Missing values filled using forward fill.")

"""### The above mentioned forwardfill imputation did not make any difference in the accuracy. Missing dates are a concerning factor, that may lead to inaccurate predictions."""

print(len(rnn_df))

print(rnn_df.columns)

#without macroeconomic or s&p 500

#rnn_df.drop(columns=['close_gspc', 'high_gspc', 'low_gspc', 'open_gspc', 'volume_gspc',
      # 'gdp', 'consumer_spending', 'unemployment_rate', 'interest_rates',
     #  'inflation', 'housing_market', 'federal_debt', 'nonfarm_payrolls',
     #  'money_supply'], inplace=True)

#Final attempt
rnn_df.drop(columns=['high_bac', 'low_bac', 'open_bac', 'high_gspc', 'low_gspc', 'open_gspc', 'consumer_spending','money_supply','federal_debt',
                   'housing_market','volume_bac','volume_gspc'], inplace=True)

# Ensure 'date' is in datetime format
rnn_df['date'] = pd.to_datetime(rnn_df['date'])

print(rnn_df.columns)

# Scale the target variable for LSTM (same as done for XGBoost)
scaler = MinMaxScaler(feature_range=(0, 1))
rnn_df['close_bac_scaled'] = scaler.fit_transform(rnn_df[['close_bac']])

# Prepare lagged features for LSTM
n_lags =5
def create_lag_features_lstm(df, target_column, n_lags=5):
    lagged_data = df.copy()
    for i in range(1, n_lags + 1):
        # Use the scaled target column name here
        lagged_data[f'{target_column}_lag_{i}'] = lagged_data[target_column].shift(i)
    lagged_data.dropna(inplace=True)
    return lagged_data

# Pass 'close_bac_scaled' (the scaled target column) to the function
lagged_df = create_lag_features_lstm(rnn_df, 'close_bac_scaled', n_lags)

# Split the dataset into features (X) and target (y)
feature_columns = [f'close_bac_scaled_lag_{i}' for i in range(1, n_lags + 1)]  # Use scaled column names
X = lagged_df[feature_columns].values
y = lagged_df['close_bac_scaled'].values

# Train-test split (80-20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape data for LSTM [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Prepare the data (assuming you have already split into X_train, X_test, etc.)
# Normalize your data if not already done

# Create the LSTM model
model_rnn = Sequential()

model_rnn.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_rnn.add(Dropout(0.2))

model_rnn.add(LSTM(units=50, return_sequences=False))
model_rnn.add(Dropout(0.2))

model_rnn.add(Dense(units=1))

# Compile the model
model_rnn.compile(optimizer='adam', loss='mean_squared_error')

# Set up EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping
history = model_rnn.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping], verbose=1)

# Evaluate the model
test_loss = model_rnn.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

# 4. **Make Predictions**
predictions = model_rnn.predict(X_test)

# Inverse transform the predictions and actual values
predictions = scaler.inverse_transform(predictions)
actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))

# 5. **Evaluate the Model**
mae = mean_absolute_error(actual_values, predictions)
rmse = np.sqrt(mean_squared_error(actual_values, predictions))
r2 = r2_score(actual_values, predictions)
mape = mean_absolute_percentage_error(actual_values, predictions)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R^2 Score: {r2}")
print(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")

# 6. **Plot the Results**
plt.figure(figsize=(14, 10))
plt.plot(rnn_df['date'][:train_size], scaler.inverse_transform(rnn_df['close_bac_scaled'][:train_size].values.reshape(-1, 1)).flatten(), label="Train", color='blue')
plt.plot(rnn_df['date'][train_size:train_size+len(actual_values)], actual_values.flatten(), label="Test Actual", color='green')
plt.plot(rnn_df['date'][train_size:train_size+len(predictions)], predictions.flatten(), label="Test Predicted", color='red')
plt.legend()
plt.xlabel("Date")
plt.ylabel("Close Bac Value")
plt.title("LSTM Model - Actual vs Predicted Test Values")
plt.show()

"""Future prediction - LSTM"""

# Forecast for the next 90 days (3 months)
forecast_steps = 90
forecast_values = []

# Prepare the last sequence from the test data for input
last_sequence = X_test[-1].reshape(1, X_test.shape[1], 1)
test_data = rnn_df.iloc[train_size:]

# Create a range of dates for the forecast
forecast_start_date = test_data['date'].iloc[-1] + pd.Timedelta(days=1)
forecast_dates = pd.date_range(forecast_start_date, periods=forecast_steps, freq='D')

# Generate forecasted values for the next 90 days
for step in range(forecast_steps):
    forecast = model_rnn.predict(last_sequence)[0]
    forecast_values.append(forecast)

    # Update last_sequence with the new prediction for the next step
    last_sequence = np.roll(last_sequence, shift=-1, axis=1)
    last_sequence[0, -1, 0] = forecast  # Update the last feature with forecasted value

# Convert forecasted values to a DataFrame
forecast_values_df = pd.DataFrame(forecast_values, columns=['Forecasted'], index=forecast_dates)

print("Forecast Start Date:", forecast_start_date)

print("Test Data Dates Range:", test_data['date'].min(), "to", test_data['date'].max())

# Forecast for the next 90 days (3 months)
forecast_steps = 90
forecast_values = []

# Ensure X_test and y_test are properly sliced
test_dates = rnn_df['date'].iloc[train_size + n_lags:]  # Ensure test data dates match y_test length

# Prepare the last sequence from the test data for input
last_sequence = X_test[-1].reshape(1, X_test.shape[1], 1)

# Create a range of dates for the forecast
forecast_start_date = test_dates.iloc[-1] + pd.Timedelta(days=1)
forecast_dates = pd.date_range(forecast_start_date, periods=forecast_steps, freq='D')

# Generate forecasted values for the next 90 days
for step in range(forecast_steps):
    forecast = model_rnn.predict(last_sequence)[0]  # Predict next step
    forecast_values.append(forecast)

    # Update last_sequence with the new prediction for the next step
    last_sequence = np.roll(last_sequence, shift=-1, axis=1)
    last_sequence[0, -1, 0] = forecast  # Update last feature with forecasted value

# Convert forecasted values to a DataFrame
forecast_values_df = pd.DataFrame(forecast_values, columns=['Forecasted'], index=forecast_dates)

# Inverse transform both test and forecasted values
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))  # Transform test values
forecast_values_df['Forecasted'] = scaler.inverse_transform(np.array(forecast_values).reshape(-1, 1))  # Transform forecast

# Plot test data and forecasted values
plt.figure(figsize=(12, 6))

# Plot actual test data
plt.plot(test_dates, y_test_inv, label="Test Data", color='green')

# Plot forecasted data
plt.plot(forecast_values_df.index, forecast_values_df['Forecasted'], label="Forecasted Data", color='orange')

# Labels and title
plt.xlabel("Date")
plt.ylabel("Close Bac Value")
plt.title("Test Data vs Forecasted Values for Next 3 Months")
plt.legend()
plt.show()

# Print forecasted values with dates
print(forecast_values_df)

### Inverse the scaling

# Import the necessary module
import matplotlib.dates as mdates

# Inverse transform train, test, and forecasted values
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
forecast_values_df['Forecasted'] = scaler.inverse_transform(np.array(forecast_values).reshape(-1, 1))

# Extract train dates
train_dates = rnn_df['date'].iloc[:train_size]

# Plot all data
plt.figure(figsize=(12, 6))

# Plot train data
plt.plot(train_dates, y_train_inv, label="Train Data", color='black')

# Plot actual test data
plt.plot(test_dates, y_test_inv, label="Test Data", color='green')

#plot predicted data
plt.plot(rnn_df['date'][train_size:train_size+len(predictions)], predictions.flatten(), label="Test Predicted", color='pink')

# Plot forecasted data
plt.plot(forecast_values_df.index, forecast_values_df['Forecasted'], label="Forecasted Data", color='orange')

# Customize x-axis to show all years
plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))  # Show every year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Display as 'YYYY'
plt.xticks(rotation=45)

# Add labels and title
plt.xlabel("Date")
plt.ylabel("Close Bac Value")
plt.title("Train Data, Test Data, and Forecasted Values for Next 3 Months")
plt.legend()
plt.show()

# Plot all data
plt.figure(figsize=(12, 8))

# Plot train data
#plt.plot(train_dates, y_train_inv, label="Train Data", color='black')

# Plot actual test data
plt.plot(test_dates, y_test_inv, label="Test Data", color='black')

#plot predicted data
plt.plot(rnn_df['date'][train_size:train_size+len(predictions)], predictions.flatten(), label="Test Predicted", color='pink')

# Plot forecasted data
plt.plot(forecast_values_df.index, forecast_values_df['Forecasted'], label="Forecasted Data", color='orange')

# Customize x-axis to show all years
plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))  # Show every year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Display as 'YYYY'
plt.xticks(rotation=45)

# Add labels and title
plt.xlabel("Date")
plt.ylabel("Close Bac Value")
plt.title("BAC Stock - LSTM Model Test Data, Prediction and Forecasted Values")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))



# Plot actual test data
plt.plot(test_dates, y_test_inv, label="Test Data", color='green')

#plot predicted data
plt.plot(rnn_df['date'][train_size:train_size+len(predictions)], predictions.flatten(), label="Test Predicted", color='red')

# Plot forecasted data
plt.plot(forecast_values_df.index, forecast_values_df['Forecasted'], label="Forecasted Data", color='orange')

# Customize x-axis to show all years
plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))  # Show every year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Display as 'YYYY'
plt.xticks(rotation=45)

# Add labels and title
plt.xlabel("Date")
plt.ylabel("Close Bac Value")
plt.title("Train Data, Test Data, and Forecasted Values for Next 3 Months")
plt.legend()
plt.show()

print(forecast_values_df.head(50))

forecast_values_df.to_csv('/content/drive/MyDrive/capstone/forecast_values_df.csv', index=True) #index=True to include the date index in the CSV

"""# Optimized Model"""

from tensorflow.keras.layers import LSTM, Dense, Dropout
#LSTM optimized
model_rnn2 = Sequential()

model_rnn2.add(Input(shape=(X_train.shape[1], 1)))

model_rnn2.add(LSTM(units=50, return_sequences=True))  #allows the next LSTM layer to receive sequences
model_rnn2.add(Dropout(0.2))

model_rnn2.add(LSTM(units=50, return_sequences=False)) #No further LSTM layer following
model_rnn2.add(Dropout(0.2))

model_rnn2.add(Dense(units=1))

from tensorflow.keras.callbacks import EarlyStopping
# Compile the model
model_rnn2.compile(optimizer='adam', loss='mean_squared_error')

# Set up EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with early stopping
history2 = model_rnn2.fit(X_train, y_train, epochs=20, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping], verbose=1)

# Evaluate the model
test_loss2 = model_rnn2.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss2}')

# Display the model summary
model_rnn2.summary()

# 4. **Make Predictions**
predictions = model_rnn2.predict(X_test)

# Inverse transform the predictions and actual values
predictions = scaler.inverse_transform(predictions)
actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))

# 5. **Evaluate the Model**
mae = mean_absolute_error(actual_values, predictions)
rmse = np.sqrt(mean_squared_error(actual_values, predictions))
r2 = r2_score(actual_values, predictions)
mape = mean_absolute_percentage_error(actual_values, predictions)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R^2 Score: {r2}")
print(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")



actual_values

import pandas as pd

# Create a DataFrame for comparison
comparison_df = pd.DataFrame({'Actual': actual_values.flatten(), 'Predicted': predictions.flatten()})
print(comparison_df.head(20))  # Display first 20 results

print(comparison_df.tail(50))

comparison_df.to_csv('/content/drive/MyDrive/capstone/comparison_df.csv', index=False)

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Forecast for the next 90 days (3 months)
forecast_steps = 90
forecast_values = []

# Ensure X_test and y_test are properly sliced
test_dates = rnn_df['date'].iloc[train_size + n_lags:]  # Ensure test data dates match y_test length

# Prepare the last sequence from the test data for input
last_sequence = X_test[-1].reshape(1, X_test.shape[1], 1)

# Create a range of dates for the forecast
forecast_start_date = test_dates.iloc[-1] + pd.Timedelta(days=1)
forecast_dates = pd.date_range(forecast_start_date, periods=forecast_steps, freq='D')

# Generate forecasted values for the next 90 days
for step in range(forecast_steps):
    forecast = model_rnn2.predict(last_sequence)[0]  # Predict next step
    forecast_values.append(forecast)

    # Update last_sequence with the new prediction for the next step
    last_sequence = np.roll(last_sequence, shift=-1, axis=1)
    last_sequence[0, -1, 0] = forecast  # Update last feature with forecasted value

# Convert forecasted values to a DataFrame
forecast_values_df = pd.DataFrame(forecast_values, columns=['Forecasted'], index=forecast_dates)

# Inverse transform both test and forecasted values
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))  # Transform test values
forecast_values_df['Forecasted'] = scaler.inverse_transform(np.array(forecast_values).reshape(-1, 1))  # Transform forecast

# Plot test data and forecasted values
plt.figure(figsize=(12, 6))

# Plot actual test data
plt.plot(test_dates, y_test_inv, label="Test Data", color='green')

# Plot forecasted data
plt.plot(forecast_values_df.index, forecast_values_df['Forecasted'], label="Forecasted Data", color='orange')

# Labels and title
plt.xlabel("Date")
plt.ylabel("Close Bac Value")
plt.title("Test Data vs Forecasted Values for Next 3 Months")
plt.legend()
plt.show()

import matplotlib.dates as mdates  # Import the mdates module

plt.figure(figsize=(12, 6))
# Plot actual test data
plt.plot(test_dates, y_test_inv, label="Test Data", color='green')

#plot predicted data
plt.plot(rnn_df['date'][train_size:train_size+len(predictions)], predictions.flatten(), label="Test Predicted", color='red')

# Plot forecasted data
plt.plot(forecast_values_df.index, forecast_values_df['Forecasted'], label="Forecasted Data", color='orange')

# Customize x-axis to show all years
plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))  # Show every year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Display as 'YYYY'
plt.xticks(rotation=45)

# Add labels and title
plt.xlabel("Date")
plt.ylabel("Close Bac Value")
plt.title("Test Data,prediction and Forecasted Values for Next 3 Months")
plt.legend()
plt.show()

# Plot all data
plt.figure(figsize=(12, 6))

train_dates = rnn_df['date'].iloc[:train_size]  # Define train_dates using rnn_df
# Plot train data
plt.plot(train_dates, y_train_inv, label="Train Data", color='blue')

# Plot actual test data
plt.plot(test_dates, y_test_inv, label="Test Data", color='green')

#plot predicted data
plt.plot(rnn_df['date'][train_size:train_size+len(predictions)], predictions.flatten(), label="Test Predicted", color='red')

# Plot forecasted data
plt.plot(forecast_values_df.index, forecast_values_df['Forecasted'], label="Forecasted Data", color='orange')

# Customize x-axis to show all years
plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))  # Show every year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Display as 'YYYY'
plt.xticks(rotation=45)

# Add labels and title
plt.xlabel("Date")
plt.ylabel("Close Bac Value")
plt.title("Train, Test, and Forecasted Values-LSTM Model")
plt.legend()
plt.show()

print(forecast_values_df.head(50))

forecast_values_df.to_csv('/content/drive/MyDrive/capstone/forecast_values_df.csv', index=True) #index=True to include the date index in the CSV

plt.figure(figsize=(12, 6))


# Plot forecasted data
plt.plot(forecast_values_df.index, forecast_values_df['Forecasted'], label="Forecasted Data", color='orange')

# Customize x-axis to show months and dates
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Show every month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format as 'Mon YYYY'
plt.gca().xaxis.set_minor_locator(mdates.DayLocator(interval=10))  # Show day ticks every 10 days
plt.xticks(rotation=45)  # Rotate labels for better visibility

# Add labels and title
plt.xlabel("Date")
plt.ylabel("Close BAC Value")
plt.title("Test Data, Prediction, and Forecasted Values for Next 3 Months")
plt.legend()
plt.show()

forecast_values_df.to_csv('/content/drive/MyDrive/capstone/forecast_values_df.csv', index=True) #index=True to include the date index in the CSV

# Save to CSV
comparison_df.to_csv("/content/drive/MyDrive/capstone/BAC_predictions.csv", index=False)

"""### This study confirms that historical stock data and macroeconomic indicators can predict BAC’s stock price trends. While XGBoost provides feature importance insights, LSTM models demonstrate superior predictive performance. Future research should explore alternative macroeconomic indicators, additional data sources, and hyperparameter tuning to improve forecasting accuracy further.

### While the LSTM model yielded strong metrics, **it is not recommended for use without further refinement**. BAC stock price performance analysis, along with market indices and macroeconomic indicators, needed a more advanced modeling approach. Future study should focus on adding additional data, advanced feature selection techniques and expertise in feature engineering to enhance predictive accuracy and reliability. By leveraging more sophisticated modeling techniques and enhanced feature engineering, the predictive capability can be significantly improved, resulting in more accurate and reliable forecasting models that support data-driven decision-making in dynamic investment strategies.
"""

























