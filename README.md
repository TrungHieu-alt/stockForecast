# 📈 **Dự báo Tài chính với Transformer**

Một mô hình dự báo giá cổ phiếu dựa trên kiến trúc Transformer cho các công ty Việt Nam (FPT, Hòa Phát, MB, VNM, HAGL). Dự án bao gồm tiền xử lý dữ liệu, trực quan hóa, huấn luyện mô hình, dự báo và đánh giá kết quả.

---

## 🚀 **Mục lục**

* [Tổng quan dự án](#tổng-quan-dự-án)
* [Tính năng](#tính-năng)
* [Cài đặt](#cài-đặt)
* [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)

  * [Tiền xử lý dữ liệu](#tiền-xử-lý-dữ-liệu)
  * [Trực quan hóa dữ liệu](#trực-quan-hóa-dữ-liệu)
  * [Huấn luyện mô hình](#huấn-luyện-mô-hình)
  * [Dự báo](#dự-báo)
  * [Đánh giá mô hình](#đánh-giá-mô-hình)
* [Cấu trúc thư mục](#cấu-trúc-thư-mục)
* [Kết quả](#kết-quả)
* [Đóng góp](#đóng-góp)
* [Giấy phép](#giấy-phép)
* [Liên hệ](#liên-hệ)

---

## 🌟 **Tổng quan dự án**

Dự án này sử dụng **Transformer Encoder** để dự báo giá cổ phiếu cho 5 công ty Việt Nam: **FPT**, **Hòa Phát**, **MB**, **VNM** và **HAGL**. Quy trình bao gồm:

1. **Thu thập & tiền xử lý dữ liệu**: Dữ liệu lịch sử giá cổ phiếu được chuẩn hóa và chia thành tập huấn luyện, kiểm tra và kiểm định.
2. **Trực quan hóa dữ liệu**: Hiển thị xu hướng giá cổ phiếu.
3. **Huấn luyện mô hình**: Huấn luyện mô hình Transformer với độ dài chuỗi đầu vào (`seq_len`) tùy chỉnh cho từng cổ phiếu.
4. **Dự báo**: Dự đoán giá cổ phiếu trong 750 ngày tiếp theo (3 năm).
5. **Đánh giá mô hình**: Đánh giá hiệu quả mô hình bằng các chỉ số MSE, RMSE và MAE trên tập kiểm định và dự báo.

---

## ✨ **Tính năng**

* **Kiến trúc Transformer**: Sử dụng Transformer Encoder cho bài toán chuỗi thời gian.
* **Tham số linh hoạt**: Cho phép tùy chỉnh `seq_len` và `pred_len`.
* **Đánh giá toàn diện**: Bao gồm MSE, RMSE, MAE và đồ thị so sánh Thực tế vs. Dự đoán.
* **Dự báo dài hạn**: Dự báo lên tới 750 ngày.
* **Trực quan hóa dữ liệu**: Hiển thị xu hướng lịch sử giá cổ phiếu.

---

## 🛠 **Cài đặt**

### Yêu cầu

* Python 3.8+
* Các thư viện cần thiết: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `torch`, `tqdm`

### Các bước cài đặt

1. Clone kho lưu trữ:

   ```bash
   git clone https://github.com/yourusername/financial-forecasting-transformer.git
   cd financial-forecasting-transformer
   ```

2. Cài đặt các thư viện:

   ```bash
   pip install -r requirements.txt
   ```

3. Đảm bảo thư mục `data` chứa các file CSV của giá cổ phiếu lịch sử cho FPT, Hòa Phát, MB, VNM và HAGL.

---

## 📊 **Hướng dẫn sử dụng**

### Tiền xử lý dữ liệu

Chuẩn hóa dữ liệu và chia thành tập huấn luyện (80%), kiểm tra (10%) và kiểm định (10%).

```bash
python preprocess.py --ticker FPT
python preprocess.py --ticker HoaPhat
python preprocess.py --ticker MB
python preprocess.py --ticker VNM
python preprocess.py --ticker HAGL
```

### Trực quan hóa dữ liệu

Hiển thị xu hướng giá cổ phiếu. Kết quả lưu tại `visualizations/{ticker}_price_history.png`.

```bash
python visualize_data.py --ticker FPT
python visualize_data.py --ticker HoaPhat
python visualize_data.py --ticker MB
python visualize_data.py --ticker VNM
python visualize_data.py --ticker HAGL
```

### Huấn luyện mô hình

Huấn luyện mô hình Transformer với độ dài chuỗi và số epoch tùy chỉnh.

```bash
python trainModel.py --ticker FPT --seq_len 60 --pred_len 1 --epochs 100
python trainModel.py --ticker HoaPhat --seq_len 60 --pred_len 1 --epochs 100
python trainModel.py --ticker MB --seq_len 60 --pred_len 1 --epochs 100
python trainModel.py --ticker VNM --seq_len 30 --pred_len 1 --epochs 100
python trainModel.py --ticker HAGL --seq_len 30 --pred_len 1 --epochs 100
```

### Dự báo

Chạy mô hình để dự báo giá cổ phiếu trong tương lai. Kết quả được lưu tại thư mục `results/forecast/`.

```bash
python forecast.py --ticker HoaPhat --seq_len 60 --forecast_days 750
python forecast.py --ticker FPT --seq_len 60 --forecast_days 750
python forecast.py --ticker MB --seq_len 60 --forecast_days 750
python forecast.py --ticker VNM --seq_len 30 --forecast_days 750
python forecast.py --ticker HAGL --seq_len 30 --forecast_days 750
```

### Đánh giá mô hình

Tính toán sai số MSE, RMSE, MAE và trực quan hóa kết quả.

```bash
python evaluate.py --ticker FPT
python evaluate.py --ticker HoaPhat
python evaluate.py --ticker MB
python evaluate.py --ticker VNM
python evaluate.py --ticker HAGL
```

Kết quả sẽ được lưu trong thư mục `results/evaluation/` dưới dạng biểu đồ so sánh `Actual vs. Predicted` và các chỉ số sai số.

---

## 📁 **Cấu trúc thư mục**

```
financial-forecasting-transformer/
├── cleanDataset/              # Dữ liệu đã tiền xử lý và chia nhỏ
├── checkpoints/               # Các mô hình train được
├── evaluationResults/         # Kết quả đánh giá các mô hình
├── forecastResults/           # Kết quả dự báo
├── rawdata/                   # Dữ liệu gốc CSV
├── visualizations/            # Biểu đồ lịch sử giá
├── preprocess.py              # Tiền xử lý dữ liệu
├── visualize_data.py          # Trực quan hóa dữ liệu
├── trainModel.py              # Huấn luyện mô hình
├── forecast.py                # Dự báo giá
├── evaluate.py                # Đánh giá mô hình
└── requirements.txt           # Các thư viện cần thiết
```

---

## 📈 **Kết quả**

* Mô hình Transformer hoạt động tốt trên các tập dữ liệu đã chuẩn hóa.
* RMSE thấp và biểu đồ dự báo bám sát xu hướng thực tế.
* Mô hình có khả năng học được các xu hướng trung hạn đến dài hạn của thị trường chứng khoán Việt Nam.


