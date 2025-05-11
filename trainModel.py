import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer model on preprocessed stock data")
    parser.add_argument('--ticker', type=str, required=True, help='Ticker name matching split_<ticker> folder')
    parser.add_argument('--seq_len', type=int, default=60, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=1, help='Output prediction length')  # Dự báo 1 ngày/lần
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')  # Tăng epochs, nhưng có early stopping
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience on validation loss')
    parser.add_argument('--clean_dir', type=str, default='cleanDataset', help='Directory containing split_<ticker> data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Where to save models')
    return parser.parse_args()

class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len, pred_len):
        self.series = series
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.samples = []
        n = len(series)
        for i in range(n - seq_len - pred_len + 1):
            x = series[i:i+seq_len]
            y = series[i+seq_len:i+seq_len+pred_len]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):  # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerEncoderModel(nn.Module):
    def __init__(self, d_model=32, nhead=2, num_layers=1, seq_len=60, pred_len=1, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embed = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)  # Output layer tối ưu, chỉ lấy 1 giá trị

    def forward(self, x):
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        x = self.embed(x)    # (batch, seq_len, d_model)
        x = self.pos_enc(x)  # Add positional encoding
        x = self.transformer(x)  # (batch, seq_len, d_model)
        x = x[:, -1, :]  # Lấy bước cuối cùng
        out = self.fc(x)  # (batch, 1)
        return out

def train():
    args = parse_args()
    split_dir = os.path.join(args.clean_dir, f'split_{args.ticker}')
    train_path = os.path.join(split_dir, 'train.npy')
    val_path = os.path.join(split_dir, 'val.npy')

    train_series = np.load(train_path)
    val_series = np.load(val_path) if os.path.exists(val_path) else np.array([])

    use_val = len(val_series) >= (args.seq_len + args.pred_len)
    if not use_val:
        print(f"⚠️ Not enough validation data for {args.ticker}, skipping validation.")

    train_ds = TimeSeriesDataset(train_series, args.seq_len, args.pred_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = None
    if use_val:
        val_ds = TimeSeriesDataset(val_series, args.seq_len, args.pred_len)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device('cpu')
    model = TransformerEncoderModel(
        d_model=32, nhead=2, num_layers=1,
        seq_len=args.seq_len, pred_len=args.pred_len,
        dropout=0.1
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss = None
        if use_val and val_loader and len(val_loader) > 0:
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    val_loss_sum += criterion(pred, y).item()
            val_loss = val_loss_sum / len(val_loader)

        if val_loss is not None:
            print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        else:
            print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.6f}")

        current_metric = val_loss if val_loss is not None else train_loss
        if current_metric < best_val:
            best_val = current_metric
            epochs_no_improve = 0
            save_path = os.path.join(args.checkpoint_dir, f"model_{args.ticker}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model to {save_path}")
        else:
            epochs_no_improve += 1
            if use_val and epochs_no_improve >= args.patience:
                print(f"⏹ Early stopping at epoch {epoch} (no improvement in {args.patience} epochs)")
                break

if __name__ == '__main__':
    train()