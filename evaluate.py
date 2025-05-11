import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Chuẩn hóa lại theo đúng cách preprocess


raw = pd.read_csv("cleanDataset/HoaPhat_cleaned.csv")
train = np.load("cleanDataset/split_HoaPhat/train.npy")
val = np.load("cleanDataset/split_HoaPhat/val.npy")
test = np.load("cleanDataset/split_HoaPhat/test.npy")
scaler = MinMaxScaler()
true_scaled = scaler.fit_transform(raw['Close'].values.reshape(-1, 1)).flatten()

full_scaled = np.concatenate([train, val, test])
plt.figure(figsize=(10, 4))
plt.plot(raw['Date'], raw['Close'], label='Original')
plt.plot(raw['Date'], true_scaled * (raw['Close'].max() - raw['Close'].min()) + raw['Close'].min(),
         label='Rescaled (from true MinMax)', linestyle='--')
plt.legend()
plt.title("Check preprocessing result")
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()
plt.show()
