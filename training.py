import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Load the dataset
data = pd.read_csv('dataset.csv')  # Make sure to provide the correct path

# Feature selection
features = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 
            'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
X = data[features].values

# Target
y = data['FTR'].apply(lambda x: 1 if x == 'H' else 0 if x == 'A' else 2).values

# Normalizing the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for LSTM [samples, time steps, features]
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Split the data
train_size = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Building the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(3, activation='softmax'))  # 3 classes for Home win, Away win, Draw
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Save the best model
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[checkpoint])
