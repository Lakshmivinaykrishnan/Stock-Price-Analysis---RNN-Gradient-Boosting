#  Stock Price Analytics and Forecasting

This end-to-end pipeline focuses on analyzing and forecasting stock prices. The workflow includes extensive data extraction, feature engineering, feature selection using XGBoost, and forecasting using an LSTM neural network. The analysis is performed using publicly available stock data for Bank of America (BAC).

---
## Tech Stack for data manipulation, statistical analysis, machine learning, and deep learning tasks
- Python
- TensorFlow 
- Keras
- Xgboost
- Pandas
- Numpy
- Scikit-learn
- Statsmodels
- Shap
- Matplotlib
- Seaborn

##   Data Extraction & Preprocessing

🔹 **Sources Used**
- yfinance API
- FRED API 

🔹 **Key Steps**
-	Extracted daily OHLCV data for BAC and S&P 500 
-	Extracted  macroeconomic data
-	Reset the index and formatted the columns 
- Merged multiple dataframes by date
- Handled missing values using forward-fill imputation
- Saved merged data for downstream use
-	Scaled features using MinMaxScaler for neural network readiness
> ⚠️ This step was the most time-consuming due to inconsistencies and missing macroeconomic entries.
---

##  Feature Selection with XGBoost

To reduce dimensionality and retain only the most informative features:

- Trained an XGBoost regressor to identify top contributing features
- Plotted and analyzed weighted feature importances
- Dropped low-importance features to reduce model complexity
  
---
## Feature Engineering and Scaling
- Scaling the Target Variable
- Lagged features were generated to help the LSTM model capture past price movements
- Data Preparation for LSTM:dataset was split into features (X) and target (y)
  
## Time Series Forecasting Using RNN (LSTM)

- Used a deep learning approach to model temporal dependencies in stock prices
- Building and Training the LSTM Model
- Model Evaluation and Prediction
- Plotting the Results
- Future Forecasting
- Optimized Model

## Model Architecture
- Architecture: Input → LSTM → Dropout  → LSTM → Dropout → Dense

## Forecast Output
- Generated predictions for the next 90 days based on the trained model
- Plotted actual vs predicted values for test data, alongside forecasted values




