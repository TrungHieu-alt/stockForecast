import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoderModel(nn.Module):
    def __init__(self, d_model=32, nhead=2, num_layers=1, seq_len=60, pred_len=1, dropout=0.1):
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
        out = self.fc(x)
        return out

def parse_args():
    parser = argparse.ArgumentParser(description="Recursive daily forecast for stock prices")
    parser.add_argument('--ticker', type=str, required=True, help='Ticker name matching cleaned csv & model')
    parser.add_argument('--seq_len', type=int, default=60)
    parser.add_argument('--forecast_days', type=int, default=750, help='Total days to forecast')  # Tăng lên 750 ngày
    parser.add_argument('--clean_dir', type=str, default='cleanDataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='forecastResults')
    return parser.parse_args()

def load_model(args, device):
    model = TransformerEncoderModel(
        d_model=32, nhead=2, num_layers=1,
        seq_len=args.seq_len, pred_len=1, dropout=0.1
    )
    path = os.path.join(args.checkpoint_dir, f"model_{args.ticker}.pt")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.train()  # Giữ dropout hoạt động để tạo randomness (MC Dropout)
    return model

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.clean_dir, f"{args.ticker}_cleaned.csv")
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    prices = df['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices).flatten().tolist()

    device = torch.device('cpu')
    model = load_model(args, device)

    # Tính độ biến động lịch sử để kiểm soát xu hướng
    historical_changes = np.diff(prices.flatten())
    change_std = np.std(historical_changes)
    max_change = 3 * change_std  # Giới hạn thay đổi tối đa (3 sigma)

    # Recursive daily forecast
    window = scaled[-args.seq_len:]
    preds = []
    for i in range(args.forecast_days):
        x = torch.FloatTensor(window[-args.seq_len:]).unsqueeze(0).to(device)
        with torch.no_grad():
            y = model(x).cpu().numpy().flatten()
        next_val = y[0]
        preds.append(next_val)
        window.append(next_val)

    forecast_prices = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    # Thêm nhiễu ngẫu nhiên để mô phỏng biến động (tăng độ nhiễu cho dự báo dài hạn)
    forecast_prices += np.random.normal(0, 0.02 * forecast_prices.std(), forecast_prices.shape)

    # Kiểm soát xu hướng: Giới hạn thay đổi hàng ngày dựa trên biến động lịch sử
    for i in range(1, len(forecast_prices)):
        change = forecast_prices[i] - forecast_prices[i-1]
        if abs(change) > max_change:
            forecast_prices[i] = forecast_prices[i-1] + np.sign(change) * max_change

    last_date = df['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=args.forecast_days, freq='D')

    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Historical')
    plt.plot(future_dates, forecast_prices, linestyle='--', label='Forecast')
    plt.title(f"{args.ticker} Price Forecast (Recursive 1-day, 750 days)")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

    result_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast_prices})
    out_csv = os.path.join(args.output_dir, f"{args.ticker}_forecast.csv")
    result_df.to_csv(out_csv, index=False)
    print(f"✅ Forecast results saved to {out_csv}")

if __name__ == '__main__':
    main()