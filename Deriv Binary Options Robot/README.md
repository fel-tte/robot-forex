# Deriv Binary Options Robot 🤖 — Hệ thống Tự Vận Hành

Robot giao dịch **tự vận hành hoàn toàn** trên nền tảng **Deriv**, được xây dựng theo kiến trúc khoá học `robot-forex`.

> Robot **tự quyết định làm gì trước**, **tự chọn điểm vào lệnh tốt nhất**,  
> **tự điều phối tài nguyên** và **gần như tự vận hành hoàn toàn**.

---

## 📁 Cấu trúc thư mục

```
Deriv Binary Options Robot/
├── config.py        # Tất cả cấu hình: API, symbol, risk, autonomous params
├── deriv_data.py    # Lấy dữ liệu nến từ Deriv WebSocket API
├── brain.py         # 🧠 Bộ não: quét nhiều thị trường, tính điểm tín hiệu 0-100
├── risk_manager.py  # 🛡️  Quản lý rủi ro: stake động, giới hạn lỗ, cooldown
├── logger.py        # 📝 Nhật ký giao dịch: CSV + Redis, thống kê hiệu suất
├── strategy.py      # Chiến lược đơn (RSI+Momentum) — dùng độc lập nếu cần
├── deriv_trade.py   # Đặt lệnh CALL/PUT, chờ kết quả từ Deriv API
├── robot.py         # 🤖 Vòng lặp tự vận hành chính — chỉ cần chạy file này
└── README.md        # Tài liệu này
```

---

## ⚙️ Cài đặt

### 1. Cài thư viện Python

```bash
pip install websockets pandas numpy redis
```

### 2. Cài và khởi động Redis

```bash
# Ubuntu/Debian
sudo apt install redis-server && sudo service redis-server start

# macOS
brew install redis && brew services start redis
```

### 3. Lấy Deriv API Token

1. Đăng nhập tại [app.deriv.com](https://app.deriv.com)
2. Vào **Settings → API Token**
3. Tạo token với quyền **Trade** và **Read**
4. Dán vào `config.py`:

```python
DERIV_API_TOKEN = "your_real_token_here"
```

> ⚠️ Hãy test với tài khoản **Demo** trước.

---

## 🚀 Khởi chạy

```bash
cd "Deriv Binary Options Robot"

# Chạy robot tự vận hành
python robot.py
```

---

## 🏗️ Kiến trúc Tự Vận Hành

```
                    ┌──────────────────────────────────────┐
                    │         robot.py  (vòng lặp)         │
                    │                                      │
   ① Tự quyết      │   brain.pick_best_entry()            │◄─── SCAN_SYMBOLS
      làm gì trước │      quét R_10/R_25/R_50/R_75/R_100  │
                    │      tính điểm 0-100 cho từng thị   │
                    │      trường, chọn điểm cao nhất      │
                    │                                      │
   ② Tự chọn       │   [brain.py] score = RSI(30pt)       │
      điểm vào     │              + Momentum(20pt)         │
      tốt nhất     │              + MACD(25pt)             │
                    │              + Bollinger(25pt)        │
                    │                                      │
   ③ Tự điều phối  │   risk_manager.can_trade()           │
      tài nguyên   │      - kiểm tra giới hạn lỗ ngày     │
                    │      - kiểm tra cooldown chuỗi thua  │
                    │   risk_manager.compute_stake()       │
                    │      - score≥80 → 5% số dư           │
                    │      - score 60-79 → 3% số dư        │
                    │                                      │
   ④ Tự vận hành   │   deriv_trade.place_and_wait()       │──► Deriv API
                    │      đặt CALL/PUT, chờ kết quả       │
                    │   logger.log()                       │──► trade_log.csv
                    │   risk_manager.update_after_trade()  │──► Redis state
                    │                                      │
                    │   [tự phục hồi lỗi, lặp vô tận]     │
                    └──────────────────────────────────────┘
```

---

## ⚙️ Tuỳ chỉnh trong `config.py`

### Cấu hình cơ bản

| Tham số | Mô tả | Mặc định |
|---------|-------|---------|
| `SYMBOL` | Symbol mặc định | `R_100` |
| `SCAN_SYMBOLS` | Danh sách thị trường tự quét | `R_10 … R_100` |
| `GRANULARITY` | Khung thời gian nến (giây) | `60` (1 phút) |
| `CONTRACT_DURATION` | Thời hạn hợp đồng | `5m` |
| `SCAN_INTERVAL_SECONDS` | Chu kỳ quét | `60` giây |

### Cấu hình tự vận hành

| Tham số | Mô tả | Mặc định |
|---------|-------|---------|
| `MIN_SIGNAL_SCORE` | Điểm tối thiểu để đặt lệnh (0-100) | `60` |
| `RISK_MAX_DAILY_LOSS_PCT` | Dừng khi lỗ X% số dư trong ngày | `20%` |
| `RISK_MAX_CONSECUTIVE_LOSS` | Cooldown sau N lần thua liên tiếp | `5` |
| `RISK_COOLDOWN_MINUTES` | Thời gian cooldown | `30 phút` |
| `STAKE_PCT_HIGH` | Stake khi score≥80 | `5% số dư` |
| `STAKE_PCT_MEDIUM` | Stake khi score 60-79 | `3% số dư` |
| `STAKE_MIN_USD` / `STAKE_MAX_USD` | Giới hạn stake | `1–50 USD` |

---

## 📊 Hệ thống tính điểm tín hiệu (brain.py)

| Chỉ báo | Điểm tối đa | Điều kiện tối đa |
|---------|-------------|-----------------|
| RSI crossover | 30 | RSI vừa vượt ngưỡng quá bán/mua |
| Momentum | 20 | Z-score momentum cao |
| MACD histogram | 25 | MACD histogram vừa đổi chiều |
| Bollinger Bands | 25 | Giá chạm/vượt dải Bollinger |
| **Tổng** | **100** | |

---

## ⚠️ Cảnh báo rủi ro

- Binary Options là hình thức giao dịch **rủi ro cao**, có thể mất toàn bộ vốn.
- Robot này là **công cụ học tập** — không đảm bảo lợi nhuận.
- Luôn test với tài khoản **Demo** trước khi dùng tiền thật.
- Không đầu tư số tiền bạn không thể chấp nhận mất.

