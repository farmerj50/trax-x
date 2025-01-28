# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout


# def preprocess_for_lstm(data, features, target, time_steps=50):
#     """
#     Prepares data for LSTM model by creating sequences.
    
#     :param data: DataFrame with historical stock data
#     :param features: List of feature column names
#     :param target: Target column name
#     :param time_steps: Number of time steps in each sequence
#     :return: X (features), y (target) arrays, scaler
#     """
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_scaled = scaler.fit_transform(data[features])
    
#     X, y = [], []
#     for i in range(time_steps, len(data_scaled)):
#         X.append(data_scaled[i - time_steps:i])
#         y.append(data_scaled[i, data.columns.get_loc(target)])
    
#     return np.array(X), np.array(y), scaler

# def create_lstm_model(input_shape):
#     """
#     Creates an LSTM model for stock prediction.
    
#     :param input_shape: Shape of the input data (time_steps, features)
#     :return: Compiled LSTM model
#     """
#     model = Sequential()
#     model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=50, return_sequences=False))
#     model.add(Dropout(0.2))
#     model.add(Dense(units=1))  # Output layer (predicting 1 value)

#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# def train_lstm_model(data, features, target, time_steps=50):
#     """
#     Train an LSTM model on the provided data.
    
#     :param data: DataFrame with historical stock data
#     :param features: List of feature column names
#     :param target: Target column name
#     :param time_steps: Number of time steps in each sequence
#     :return: Trained model and scaler for inverse transformations
#     """
#     X, y, scaler = preprocess_for_lstm(data, features, target, time_steps)
#     model = create_lstm_model((X.shape[1], X.shape[2]))

#     model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
#     return model, scaler

# def predict_next_day(model, recent_data, scaler, features):
#     """
#     Predict stock price for the next day using the trained model.
    
#     :param model: Trained LSTM model
#     :param recent_data: Recent stock data for prediction
#     :param scaler: Scaler used for preprocessing
#     :param features: List of feature column names
#     :return: Predicted stock price
#     """
#     recent_scaled = scaler.transform(recent_data[features])
#     recent_input = np.expand_dims(recent_scaled, axis=0)  # Reshape for LSTM input
#     prediction = model.predict(recent_input)
#     return scaler.inverse_transform(prediction)[0][0]  # Inverse transform to original scale
