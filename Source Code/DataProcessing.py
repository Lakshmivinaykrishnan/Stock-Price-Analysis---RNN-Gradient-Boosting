
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