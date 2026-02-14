import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
import joblib  # Used to save the Scaler for future inference/live trading


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class Config:
    feature_dir = "final_features"
    model_save_path = "best_lstm_model.pth"
    scaler_save_path = "feature_scaler.pkl"  # Save Scaler for future use

    train_start_date = "2016-01-01"
    train_end_date = "2022-12-31"
    test_start_date = "2023-01-01"
    test_end_date = "2025-12-31"

    seq_len = 30
    hidden_size = 16
    num_layers = 1
    dropout = 0.5
    batch_size = 512
    learning_rate = 0.0005
    epochs = 30
    patience = 1000

    # Input Features (Must match the columns in your CSV files)
    feature_cols = [
        'sma100_gap',
        'sma60_gap',
        'rsi14',           # Core Trend
        'bb_position',     # Relative Position (Overbought/Oversold)
        'k_upper',
    ]

    # Output Label (Multi-task capable)
    label_cols = ['label_10d']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 2. Data Preprocessing Functions (Load & Scale)
# ==============================================================================
def load_and_scale_data():
    """
    Reads all CSVs, splits into Train/Test, and applies RobustScaling.
    (Includes caching mechanism: slow on first run, fast on subsequent runs)
    """
    # Define cache file name
    cache_file = "processed_tensors.pt"

    # --- 1. Check for cache. If exists, load directly (Fast load) ---
    if os.path.exists(cache_file):
        print(f"⚡ Cache file {cache_file} found, loading directly...")
        try:
            # Load dictionary
            data = torch.load(cache_file)
            return data['train'], data['test']
        except Exception as e:
            print(f"⚠️ Cache load failed ({e}), reloading CSVs...")

    # --- 2. If no cache, execute original CSV loading logic ---
    print("🐢 No cache found. Starting to load all stock data (This may take a while)...")

    train_data_list = []
    test_data_list = []

    files = [f for f in os.listdir(Config.feature_dir) if f.endswith('.csv')]

    for file in tqdm(files):
        path = os.path.join(Config.feature_dir, file)
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)

            # Split training and testing sets
            train_mask = (df.index >= Config.train_start_date) & (df.index <= Config.train_end_date)
            test_mask = (df.index >= Config.test_start_date) & (df.index <= Config.test_end_date)

            if train_mask.sum() > Config.seq_len:
                train_data_list.append(df.loc[train_mask])

            if test_mask.sum() > Config.seq_len:
                test_data_list.append(df.loc[test_mask])

        except Exception as e:
            print(f"Skipping {file}: {e}")

    # Merge into a large DataFrame for Scaler training
    print("Merging data for normalization...")
    full_train_df = pd.concat(train_data_list)

    # Initialize Scaler and Fit only on the training set
    print("Fitting RobustScaler on Training Data...")
    scaler = RobustScaler()
    scaler.fit(full_train_df[Config.feature_cols])

    # Save Scaler for future real-time use
    joblib.dump(scaler, Config.scaler_save_path)
    print(f"Scaler saved to {Config.scaler_save_path}")

    # Transform Data
    def transform_list(df_list, scaler):
        processed_data = []
        for df in tqdm(df_list):
            scaled_features = scaler.transform(df[Config.feature_cols])
            labels = df[Config.label_cols].values
            processed_data.append((scaled_features, labels))
        return processed_data

    print("Transforming Training Data...")
    train_processed = transform_list(train_data_list, scaler)

    print("Transforming Test Data...")
    test_processed = transform_list(test_data_list, scaler)

    # --- 3. After processing, save the cache file! ---
    print(f"💾 Saving data cache to {cache_file} ...")
    torch.save({'train': train_processed, 'test': test_processed}, cache_file)

    return train_processed, test_processed


# ==============================================================================
# 3. Dataset Class (Built from processed in-memory data)
# ==============================================================================
class InMemoryDataset(Dataset):
    def __init__(self, processed_data_list, seq_len):
        self.X_data = []
        self.y_data = []

        # Create sliding windows
        for features, labels in processed_data_list:
            num_samples = len(features) - seq_len
            if num_samples <= 0: continue

            # For speed, we generate window indices directly here.
            # To speed up training, we expand into Tensors here (consumes more RAM, but training is faster).
            # Based on 500 stocks x 10 years, data size is approx 300MB-500MB, RAM is sufficient.
            
            # Using a step size to reduce redundancy (optional)
            step_size = 5

            for i in range(0, num_samples, step_size):
                self.X_data.append(features[i : i+seq_len])
                self.y_data.append(labels[i+seq_len])

        self.X_data = torch.tensor(np.array(self.X_data), dtype=torch.float32)
        self.y_data = torch.tensor(np.array(self.y_data), dtype=torch.float32)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]


# ==============================================================================
# 4. LSTM Model
# ==============================================================================
class MultiTaskLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(MultiTaskLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ==============================================================================
# 5. Main Training Loop
# ==============================================================================
def train():
    # 1. Load and normalize data
    train_list, test_list = load_and_scale_data()

    # 2. Build Datasets
    print("Building Datasets (Sliding Windows)...")
    train_dataset = InMemoryDataset(train_list, Config.seq_len)
    test_dataset = InMemoryDataset(test_list, Config.seq_len)

    print(f"Train Samples: {len(train_dataset)}, Test Samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0)

    # 3. Initialize Model
    model = MultiTaskLSTM(
        input_dim=len(Config.feature_cols),
        hidden_dim=Config.hidden_size,
        num_layers=Config.num_layers,
        output_dim=len(Config.label_cols),
        dropout=Config.dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # 4. Training Loop
    best_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'test_loss': []}

    print(f"\n🚀 Start Training on {device}...")

    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                test_loss += criterion(preds, y_batch).item()

        avg_test_loss = test_loss / len(test_loader)

        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        scheduler.step(avg_test_loss)

        print(f"Epoch {epoch + 1:02d} | Train Loss: {avg_train_loss:.5f} | Test Loss: {avg_test_loss:.5f}")

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.model_save_path)
            print("  --> Model Saved!")
        else:
            patience_counter += 1
            if patience_counter >= Config.patience:
                print("🛑 Early Stopping!")
                break

    # 5. Plotting
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['test_loss'], label='Test')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train()
