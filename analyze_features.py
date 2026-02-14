import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
# Import configuration, model, and data processing functions from train.py
# Ensure your training script is named train.py
from train import Config, MultiTaskLSTM, load_and_scale_data

# Set font for plotting
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 1. Data Preparation Function (Fixed weights loading error + Added fault tolerance)
# ==============================================================================
def load_test_data():
    print("Preparing test set data...")

    cache_file = "processed_tensors.pt"
    test_data = None

    # Attempt 1: Load from cache
    if os.path.exists(cache_file):
        print(f"✅ Cache file {cache_file} found, attempting to load...")
        try:
            # --- Key Fix: Added weights_only=False ---
            # This is necessary because we are loading a dictionary containing lists/tensors, not just model weights.
            data = torch.load(cache_file, weights_only=False)
            test_data = data['test']
        except Exception as e:
            print(f"⚠️ Cache load failed ({e}), switching to CSV reload...")

    # Attempt 2: If no cache or load failed, reload from CSV
    if test_data is None:
        print("🐢 Calling train.py function to reload CSVs (This will take a while)...")
        _, test_data = load_and_scale_data()

    # Construct Test Set Tensor
    print("Constructing Sliding Windows...")
    X_data = []
    y_data = []
    seq_len = Config.seq_len

    for features, labels in tqdm(test_data):
        num_samples = len(features) - seq_len
        if num_samples <= 0: continue

        # Simple loop splitting
        for i in range(num_samples):
            X_data.append(features[i: i + seq_len])
            y_data.append(labels[i + seq_len])

    if not X_data:
        print("❌ Error: Test set data is empty!")
        return None, None

    return torch.tensor(np.array(X_data), dtype=torch.float32), torch.tensor(np.array(y_data), dtype=torch.float32)


# ==============================================================================
# 2. Permutation Importance Calculation Core
# ==============================================================================
def calculate_permutation_importance(model, X, y, feature_names):
    model.eval()
    criterion = nn.MSELoss()
    device = next(model.parameters()).device

    # 1. Calculate Baseline Loss
    batch_size = 1024
    num_samples = len(X)
    baseline_loss = 0.0

    print("Calculating Baseline Loss...")
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            X_batch = X[i:i + batch_size].to(device)
            y_batch = y[i:i + batch_size].to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            baseline_loss += loss.item() * len(X_batch)

    baseline_loss /= num_samples
    print(f"📊 Baseline Test Set Loss: {baseline_loss:.6f}")

    # 2. Shuffle features one by one
    importances = {}

    print("Analyzing feature importance (This may take a few minutes)...")
    for i, feature_name in enumerate(tqdm(feature_names)):
        X_permuted = X.clone()

        # Shuffle the i-th feature across all samples
        idx = torch.randperm(num_samples)
        X_permuted[:, :, i] = X_permuted[idx, :, i]

        permuted_loss = 0.0
        with torch.no_grad():
            for j in range(0, num_samples, batch_size):
                X_batch = X_permuted[j:j + batch_size].to(device)
                y_batch = y[j:j + batch_size].to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                permuted_loss += loss.item() * len(X_batch)

        permuted_loss /= num_samples

        # Importance = Corrupted Loss - Original Loss
        importances[feature_name] = permuted_loss - baseline_loss

    return importances


# ==============================================================================
# 3. Main Execution
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    X_test, y_test = load_test_data()
    if X_test is None: exit()

    # 2. Load Model
    print(f"Loading model: {Config.model_save_path} ...")
    model = MultiTaskLSTM(
        input_dim=len(Config.feature_cols),
        hidden_dim=Config.hidden_size,
        num_layers=Config.num_layers,
        output_dim=len(Config.label_cols),
        dropout=Config.dropout
    ).to(device)

    try:
        # --- Key Fix: Added weights_only=False ---
        # Note: Depending on your PyTorch version and how the model was saved, you might need weights_only=True or False.
        # Since the train script saved state_dict (weights only), usually True is safer, 
        # but if you saved the full model object, False is required.
        model.load_state_dict(torch.load(Config.model_save_path, map_location=device, weights_only=False))
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print(f"❌ Model file {Config.model_save_path} not found. Please run train.py first!")
        exit()
    except RuntimeError as e:
        print(f"❌ Model structure mismatch! Check if Config in train.py was modified.\nDetail: {e}")
        exit()

    # 3. Run Analysis
    importances = calculate_permutation_importance(model, X_test, y_test, Config.feature_cols)

    # 4. Plotting and Output
    df_imp = pd.DataFrame(list(importances.items()), columns=['Feature', 'Increase_in_Loss'])
    df_imp = df_imp.sort_values(by='Increase_in_Loss', ascending=True)

    print("\n======== Feature Importance Ranking (Low to High) ========")
    print(df_imp)

    # Identify useless features (Minimal increase in Loss, or negative)
    useless_features = df_imp[df_imp['Increase_in_Loss'] <= 1e-6]['Feature'].tolist()
    print(f"\n🗑️ Suggested useless features to remove ({len(useless_features)}):")
    print(useless_features)

    plt.figure(figsize=(10, 12))
    plt.barh(df_imp['Feature'], df_imp['Increase_in_Loss'], color='skyblue')
    plt.xlabel('Increase in Loss (Permutation Importance)')
    plt.title('LSTM Feature Importance Analysis')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.tight_layout()
    plt.show()
