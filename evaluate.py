import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ----------------------
# Model Definitions
# ----------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoderModel(nn.Module):
    def __init__(self, d_model=32, nhead=2, num_layers=1, seq_len=60, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embed(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.fc(x)

# ----------------------
# Argument Parser
# ----------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Transformer model on test data")
    parser.add_argument('--ticker', type=str, required=True,
                        help='Ticker name matching split_<ticker> folder')
    parser.add_argument('--seq_len', type=int, default=60,
                        help='Input sequence length')
    parser.add_argument('--clean_dir', type=str, default='cleanDataset',
                        help='Directory containing split_<ticker> data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Where to load models')
    parser.add_argument('--output_dir', type=str, default='evaluationResults',
                        help='Where to save evaluation results')
    return parser.parse_args()

# ----------------------
# Load Trained Model
# ----------------------

def load_model(args, device):
    model = TransformerEncoderModel(
        d_model=32, nhead=2, num_layers=1,
        seq_len=args.seq_len, dropout=0.1
    )
    model_path = os.path.join(args.checkpoint_dir, f"model_{args.ticker}.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ----------------------
# Main Evaluation Flow
# ----------------------

def main():
    # Parse arguments
    args = parse_args()

    # Reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Directory for split data
    split_dir = os.path.join(args.clean_dir, f'split_{args.ticker}')

    # ----- Fit scaler on TRAIN data only -----
    train_path = os.path.join(split_dir, 'train.npy')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train data not found at {train_path}")
    train_series = np.load(train_path)
    scaler = MinMaxScaler()
    scaler.fit(train_series.reshape(-1, 1))

    # ----- Load test data -----
    test_path = os.path.join(split_dir, 'test.npy')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")
    test_series = np.load(test_path)
    if len(test_series) < (args.seq_len + 1):
        raise ValueError(f"Test data too short for seq_len={args.seq_len}")

    # Prepare inputs and targets for 1-step ahead evaluation
    inputs, targets = [], []
    for i in range(len(test_series) - args.seq_len):
        inputs.append(test_series[i:i + args.seq_len])
        targets.append(test_series[i + args.seq_len])
    inputs = np.array(inputs)
    targets = np.array(targets)

    # Device and model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args, device)

    # Make predictions
    predictions = []
    with torch.no_grad():
        for seq in inputs:
            x_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
            y_pred = model(x_tensor).cpu().numpy().flatten()[0]
            predictions.append(y_pred)
    predictions = np.array(predictions)

    # Inverse transform
    predictions_inv = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    targets_inv = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

    # Calculate metrics
    mse = mean_squared_error(targets_inv, predictions_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_inv, predictions_inv)

    # Print and save metrics
    print(f"Evaluation Metrics for {args.ticker}: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    metrics_df = pd.DataFrame({'Metric': ['MSE','RMSE','MAE'], 'Value': [mse, rmse, mae]})
    metrics_df.to_csv(os.path.join(args.output_dir, f"{args.ticker}_metrics.csv"), index=False)

    # Plot Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(targets_inv, label='Actual')
    plt.plot(predictions_inv, label='Predicted', linestyle='--')
    plt.title(f"{args.ticker} Actual vs Predicted Prices")
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, f"{args.ticker}_actual_vs_predicted.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"✅ Metrics and plot saved in {args.output_dir}")

if __name__ == '__main__':
    main()
