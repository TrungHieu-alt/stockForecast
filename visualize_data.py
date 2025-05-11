import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize stock price data")
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol (e.g., VNM)')
    parser.add_argument('--clean_dir', type=str, default='cleanDataset', help='Directory containing cleaned CSV')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Directory to save visualizations')
    return parser.parse_args()

def main():
    args = parse_args()
    # Sử dụng đường dẫn tuyệt đối dựa trên thư mục gốc của dự án
    base_dir = os.path.dirname(os.path.abspath(__file__))
    clean_dir = os.path.join(base_dir, args.clean_dir)
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Xác định đường dẫn file CSV
    csv_path = os.path.join(clean_dir, f"{args.ticker}_cleaned.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}. Please run collect_data.py first.")

    # Load data
    df = pd.read_csv(csv_path, parse_dates=['Date'])

    # Plot price over time (similar to forecast.py)
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Historical', color='blue')
    plt.title(f"{args.ticker} Price History")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"{args.ticker}_price_history.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"✅ Visualization saved to {plot_path}")

if __name__ == '__main__':
    main()