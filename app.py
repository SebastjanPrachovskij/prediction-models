from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)
CORS(app)

def preprocess_data(series):
    """Prepares time series data for LSTM/GRU by scaling and creating sequences."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
    return scaled_data, scaler

def create_dataset(data, look_back=1):
    """Transforms data into sequences for model training."""
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

def printErrors(actual, predictions, model_name, crypto_name, label):
    if len(actual) != len(predictions):
        print(f"Error: Inconsistent data lengths. Actual: {len(actual)}, Predictions: {len(predictions)}")
        return  # Exit the function if lengths mismatch

    print(label)
    mse_value = mean_squared_error(actual, predictions)
    print(f"MSE: {mse_value}")
    print(f"SQRT(MSE): {math.sqrt(mse_value)}")
    print(f"MAE: {mean_absolute_error(actual, predictions)}")
    print(f"RMSE: {math.sqrt(mse_value)}")
    mape_value = np.mean(np.abs((actual - predictions) / actual)) * 100 if np.any(actual) else float('inf')
    print(f"MAPE: {mape_value}%")

@app.route('/predict/<method>', methods=['POST'])
def predict(method):
    data = request.get_json()
    series = pd.Series(data['series'])
    

    if series.empty:
        return jsonify({"error": "The series is empty"}), 400

    series.index = pd.to_datetime(series.index)
    last_date = series.index[-1]

    if method == 'sarimax':
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=25)
        forecast_dates = [last_date + timedelta(days=x) for x in range(1, 10)]
        forecast_with_dates = {str(date.date()): float(value) for date, value in zip(forecast_dates, forecast)}

        if not series.empty:
            printErrors(series.values, forecast[:len(series)], 'SARIMAX', 'Models', 'SARIMAX Error Metrics')

        return jsonify(forecast_with_dates)

    elif method in ['lstm', 'gru']:
        scaled_data, scaler = preprocess_data(series)
        look_back = 10
        X, Y = create_dataset(scaled_data, look_back)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        model = Sequential()
        if method == 'lstm':
            model.add(LSTM(50, input_shape=(look_back, 1)))
        elif method == 'gru':
            model.add(GRU(50, input_shape=(look_back, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, Y, epochs=5, batch_size=1, verbose=2)

        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)
        prediction_dates = [last_date + timedelta(days=x) for x in range(1, len(predictions) + 1)]
        predictions_with_dates = {str(date.date()): float(prediction) for date, prediction in zip(prediction_dates, predictions.flatten())}

        if not series.empty:
            actual_scaled, _ = preprocess_data(series)
            _, actual_Y = create_dataset(actual_scaled, look_back)
            printErrors(actual_Y, predictions[:len(actual_Y)], method.upper(), 'Models', f'{method.upper()} Error Metrics')

        return jsonify(predictions_with_dates)

    else:
        return jsonify({"error": "Invalid method"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
