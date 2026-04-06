import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from pathlib import Path

# Config
WINDOW_SIZE = 7
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
HIDDEN_SIZE = 64

# Features (subset for LSTM signal)
FEATURES = ['pm25', 'fire_count_total', 'temperature_2m_mean', 'relative_humidity_2m_mean', 'wind_speed_10m_mean', 'precipitation_sum', 'pressure_msl_mean']

class PM25LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(PM25LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take last output
        return out

def prepare_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, :-1]) # Features
        y.append(data[i+window_size, -1])     # Target (already shifted)
    return np.array(X), np.array(y)

def train():
    df = pd.read_csv("data/processed/merged/dataset.csv", parse_dates=["date"])
    df['target'] = df['pm25'].shift(-1)
    df = df.dropna(subset=['target'])
    
    data = df[FEATURES + ['target']].values
    
    # Split (last 2 years for test)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    
    X_train, y_train = prepare_sequences(train_data_scaled, WINDOW_SIZE)
    X_test, y_test = prepare_sequences(test_data_scaled, WINDOW_SIZE)
    
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1)
    
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
    
    model = PM25LSTM(len(FEATURES), HIDDEN_SIZE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    best_mae = float('inf')
    
    print("Training LSTM...")
    for epoch in range(EPOCHS):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            preds = model(X_test_tensor)
            # Inverse scale the target
            # Scaler has (target_col) at the end. index is -1.
            preds_unscaled = preds.numpy() * scaler.scale_[-1] + scaler.mean_[-1]
            y_test_unscaled = y_test_tensor.numpy() * scaler.scale_[-1] + scaler.mean_[-1]
            
            mae = mean_absolute_error(y_test_unscaled, preds_unscaled)
            if mae < best_mae:
                best_mae = mae
                
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], MAE: {mae:.4f}")

    # Persistence baseline for the SAME test set
    y_today_unscaled = test_data[WINDOW_SIZE:, 0] # First column is today's PM2.5
    persist_mae = mean_absolute_error(y_test_unscaled, y_today_unscaled)
    
    print("\n" + "="*50)
    print(f"LSTM RESULTS (Test Set):")
    print(f"Best Model MAE: {best_mae:.4f}")
    print(f"Persistence MAE: {persist_mae:.4f}")
    print(f"Improvement   : {(persist_mae - best_mae)/persist_mae*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    train()
