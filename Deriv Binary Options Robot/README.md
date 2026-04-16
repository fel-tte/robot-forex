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

## 🏗️ Kiến trúc Operator System

```
  BẠN                            HỆ THỐNG TỰ VẬN HÀNH
  ──────                          ────────────────────────────────────────────
  👁️ Giám sát                 ①  Tự phát hiện sóng hồi trong sóng chính
  màn hình                        (detect_swings + _main_direction)
                               ②  Tự đo lường độ sâu sóng hồi
                                   (Fibonacci 23.6 / 38.2 / 50 / 61.8 / 78.6%)
                               ③  Tự giới hạn rủi ro
                                   (lỗ ngày ≤20%, cooldown sau 5 lần thua)
                               ④  Tự điều phối tài nguyên
                                   (stake = f(score, balance))
                               ⑤  Tự tìm điểm vào an toàn nhất
                                   (cuối sóng hồi + Fib zone + S/R cluster)
                               ⑥  Tự thoát an toàn nhất
                                   (TP = đỉnh/đáy sóng chính trước)

                    ┌──────────────────────────────────────────────┐
                    │              robot.py (vòng lặp)             │
                    │                                              │
  get_balance() ───▶│  ① risk.can_trade(balance)                  │
                    │                                              │
  Deriv API    ───▶│  ② brain.pick_best_entry()                  │
  (5 symbols)       │      ├─ deriv_data.fetch_candles()          │
                    │      ├─ wave_analyzer.analyze_waves()       │
                    │      │    ├─ detect_swings()  (ZigZag)      │
                    │      │    ├─ fibonacci_levels()             │
                    │      │    ├─ find_sr_levels()               │
                    │      │    └─ → WaveContext                  │
                    │      └─ _score_signal() → MarketSignal      │
                    │           Tầng1: RSI+Mom+MACD+BB  (max 60) │
                    │           Tầng2: Fib+Depth+SR    (max 40) │
                    │                                              │
                    │  ③ risk.compute_stake(score, balance)       │
                    │                                              │
  Deriv API    ◀───│  ④ place_and_wait(dir, symbol, stake)       │
                    │      └─ subscribe proposal_open_contract    │
                    │                                              │
                    │  ⑤ logger.log(TradeRecord)  → CSV + Redis  │
                    │  ⑥ risk.update_after_trade(won, pnl)       │
                    └──────────────────────────────────────────────┘
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

### Cấu hình phân tích sóng (wave_analyzer.py)

| Tham số | Mô tả | Mặc định |
|---------|-------|---------|
| `WAVE_SWING_ORDER` | Cửa sổ rolling phát hiện đỉnh/đáy | `5 nến` |
| `WAVE_MIN_SIZE_PCT` | Kích thước tối thiểu của sóng chính | `0.5% giá` |
| `WAVE_CORRECTION_MIN` | Sóng hồi tối thiểu (% sóng chính) | `20%` |
| `WAVE_CORRECTION_MAX` | Sóng hồi tối đa trước khi tính là đảo chiều | `80%` |
| `WAVE_FIB_TOLERANCE` | Dung sai xác nhận "tại vùng Fib" | `±1.5%` |

---

## 📊 Hệ thống tính điểm tín hiệu (brain.py)

### Tầng 1 — Chỉ báo kỹ thuật (max 60 điểm)

| Chỉ báo | Điểm | Điều kiện tối đa |
|---------|------|-----------------|
| RSI crossover | 18 | RSI vừa vượt ngưỡng quá bán/mua |
| Momentum (z-score) | 12 | Momentum mạnh bất thường |
| MACD histogram | 15 | MACD histogram vừa đổi chiều |
| Bollinger Bands | 15 | Giá chạm/vượt dải Bollinger |

### Tầng 2 — Phân tích sóng (max 40 điểm)

| Tiêu chí | Điểm | Chi tiết |
|---------|------|---------|
| Fibonacci 61.8% | 20 | Vùng vàng nhất |
| Fibonacci 50.0% | 15 | Vùng trung tâm |
| Fibonacci 38.2% | 10 | Vùng nông |
| Độ sâu lý tưởng (38-62%) | 10 | Tối đa tại 50% |
| S/R cluster | 10 | Pivot cluster xác nhận |
| **Tổng tối đa** | **100** | |

> 💡 Khi không có sóng hồi: điểm tối đa = 60. Khi có sóng hồi tại Fib 61.8% + S/R: điểm tối đa = 100.

---

## ⚠️ Cảnh báo rủi ro

- Binary Options là hình thức giao dịch **rủi ro cao**, có thể mất toàn bộ vốn.
- Robot này là **công cụ học tập** — không đảm bảo lợi nhuận.
- Luôn test với tài khoản **Demo** trước khi dùng tiền thật.
- Không đầu tư số tiền bạn không thể chấp nhận mất.

