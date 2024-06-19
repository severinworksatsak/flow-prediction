# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:01:41 2024

@author: LES
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from models.utility import generate_sequences

# Example DataFrame with time series data
data = {
    'rainfall': np.random.rand(1000),
    'temperature': np.random.rand(1000),
    'river_flow': np.random.rand(1000)
}
df = pd.DataFrame(data)

# Normalize the data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Define parameters
n_features = df.shape[1]
lookback = 6  # Past timesteps
forecast_horizon = 8  # Future timesteps

# Create input and output sequences
X = []
y = []

for i in range(lookback, len(df) - forecast_horizon):
    X.append(df_scaled[i-lookback:i])
    y.append(df_scaled[i:i+forecast_horizon, 2])  # Assuming river_flow is the 3rd column

X = np.array(X)
y = np.array(y)

print("Input shape: ", X.shape)  # (samples, lookback, n_features)
print("Output shape: ", y.shape)  # (samples, forecast_horizon)


df_scaled.shape[0]

df_seq = generate_sequences(df, target_var='river_flow', n_lookback=12, n_ahead=8)
