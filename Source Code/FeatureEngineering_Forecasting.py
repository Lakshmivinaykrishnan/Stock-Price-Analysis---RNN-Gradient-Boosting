
# Time series Forecasting Using RNN


# LSTM with lagged target variable

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