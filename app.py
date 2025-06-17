import pandas as pd
from flask import Flask, jsonify, request
from prophet import Prophet
import os

app = Flask(__name__)

# Load CSV data and prepare for forecasting
def load_data():
    # Adjust the file path as necessary
    data = pd.read_csv('digital_marketing_data.csv')
    data['ds'] = pd.to_datetime(data['Week Date'])
    forecast_data = data[['ds', 'Revenue']]
    forecast_data.rename(columns={'Revenue': 'y'}, inplace=True)
    return forecast_data

# Train Prophet model
def train_model(forecast_data):
    model = Prophet()
    model.fit(forecast_data)
    return model

# Make predictions
def predict(model, periods=12):
    future = model.make_future_dataframe(forecast_data, periods=periods, freq='W')
    forecast = model.predict(future)
    forecast_data_to_write = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    return forecast_data_to_write

@app.route('/forecast', methods=['GET'])
def forecast():
    # Load data
    forecast_data = load_data()

    # Train the model
    model = train_model(forecast_data)

    # Predict for the next 12 weeks (can be adjusted via query parameter)
    periods = int(request.args.get('periods', 12))  # Default is 12
    forecast_result = predict(model, periods)

    # Convert forecast to JSON
    forecast_json = forecast_result.to_dict(orient='records')
    return jsonify(forecast_json)

@app.route('/')
def index():
    return 'Welcome to the Forecast API! Use /forecast to get the forecast.'

if __name__ == '__main__':
    app.run(debug=True)
