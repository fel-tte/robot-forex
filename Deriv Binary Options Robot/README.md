# Deriv Binary Options Robot 🤖

Robot giao dịch tự động Binary Options trên nền tảng **Deriv**, sử dụng chiến lược **RSI + Momentum**.  
Kiến trúc kế thừa từ khoá học `robot-forex` (Redis signal bus + Scheduler).

---

## 📁 Cấu trúc thư mục

```
Deriv Binary Options Robot/
├── config.py        # Cấu hình: API token, symbol, tham số giao dịch
├── deriv_data.py    # Lấy dữ liệu nến từ Deriv WebSocket API
├── strategy.py      # Tính RSI + Momentum → sinh tín hiệu → Redis
├── deriv_trade.py   # Đọc tín hiệu từ Redis → đặt lệnh CALL/PUT
├── robot.py         # Robot chính: kết hợp tất cả + scheduler
└── README.md        # Tài liệu này
```

---

## ⚙️ Cài đặt

### 1. Cài thư viện Python

```bash
pip install websockets pandas redis schedule
```

### 2. Cài và khởi động Redis

```bash
# Ubuntu/Debian
sudo apt install redis-server
sudo service redis-server start

# macOS
brew install redis
brew services start redis
```

### 3. Lấy Deriv API Token

1. Đăng nhập tại [app.deriv.com](https://app.deriv.com)
2. Vào **Settings → API Token**
3. Tạo token với quyền **Trade** và **Read**
4. Copy token và dán vào `config.py`:

```python
DERIV_API_TOKEN = "your_real_token_here"
```

> ⚠️ **Khuyến nghị**: Hãy test với tài khoản **Demo** trước khi dùng tài khoản thật.

---

## 🚀 Khởi chạy

```bash
cd "Deriv Binary Options Robot"

# Chạy robot hoàn chỉnh (phân tích + đặt lệnh tự động)
python robot.py

# Hoặc chạy từng module riêng để kiểm tra:
python deriv_data.py    # Test lấy dữ liệu nến
python strategy.py      # Test tính tín hiệu RSI + Momentum
python deriv_trade.py   # Test đặt lệnh (cần token hợp lệ)
```

---

## 📊 Chiến lược giao dịch

| Tín hiệu | Điều kiện | Lệnh |
|----------|-----------|------|
| **MUA (CALL)** | RSI vừa vượt lên khỏi vùng quá bán (< 30) **VÀ** Momentum > 0 | Đặt hợp đồng CALL |
| **BÁN (PUT)**  | RSI vừa rơi xuống từ vùng quá mua (> 70) **VÀ** Momentum < 0 | Đặt hợp đồng PUT  |
| Không có | Các trường hợp còn lại | Không đặt lệnh |

---

## ⚙️ Tuỳ chỉnh trong `config.py`

| Tham số | Mô tả | Mặc định |
|---------|-------|---------|
| `SYMBOL` | Mã thị trường | `R_100` (Volatility 100) |
| `GRANULARITY` | Khung thời gian nến (giây) | `60` (1 phút) |
| `RSI_OVERSOLD` | Ngưỡng quá bán | `30` |
| `RSI_OVERBOUGHT` | Ngưỡng quá mua | `70` |
| `TRADE_AMOUNT` | Số tiền mỗi lệnh (USD) | `10` |
| `CONTRACT_DURATION` | Thời hạn hợp đồng | `5` phút |
| `SCAN_INTERVAL_SECONDS` | Tần suất quét thị trường | `60` giây |

---

## 🏗️ Kiến trúc

```
┌─────────────────┐    candles    ┌──────────────────┐
│   Deriv API     │ ─────────────▶│  deriv_data.py   │
│ (WebSocket)     │               └────────┬─────────┘
└─────────────────┘                        │ DataFrame
                                           ▼
                                  ┌──────────────────┐
                                  │   strategy.py    │
                                  │  RSI + Momentum  │
                                  └────────┬─────────┘
                                           │ signal dict
                                           ▼
                                  ┌──────────────────┐
                                  │      Redis       │
                                  │  (hash signal)   │
                                  └────────┬─────────┘
                                           │ read signal
                                           ▼
                                  ┌──────────────────┐    order    ┌─────────────────┐
                                  │  deriv_trade.py  │ ───────────▶│   Deriv API     │
                                  │  CALL / PUT      │             │ (WebSocket)     │
                                  └──────────────────┘             └─────────────────┘
```

---

## ⚠️ Cảnh báo rủi ro

- Binary Options là hình thức giao dịch **rủi ro cao**, có thể mất toàn bộ vốn đặt cược.
- Robot này chỉ là **công cụ học tập**. Không đảm bảo lợi nhuận.
- Luôn test kỹ với tài khoản **Demo** trước.
- Không đầu tư số tiền bạn không thể chấp nhận mất.
