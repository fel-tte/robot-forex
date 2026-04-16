# ============================================================
# Deriv Binary Options Robot - Cấu hình
# ============================================================
# Hướng dẫn lấy API Token:
# 1. Đăng nhập tài khoản Deriv tại https://app.deriv.com
# 2. Vào Settings > API Token
# 3. Tạo token với quyền "Trade" và "Read"
# 4. Dán token vào biến DERIV_API_TOKEN bên dưới
# ============================================================

# --- Deriv API ---
DERIV_API_TOKEN = "YOUR_DERIV_API_TOKEN"   # Thay bằng API token thực
DERIV_APP_ID    = 1089                     # App ID mặc định (demo). Tạo app tại https://api.deriv.com/app-registration
DERIV_WS_URL    = f"wss://ws.binaryws.com/websockets/v3?app_id={DERIV_APP_ID}"

# --- Thị trường giao dịch ---
# Các symbol phổ biến trên Deriv:
#   Volatility Index : R_10, R_25, R_50, R_75, R_100
#   Crash/Boom      : CRASH1000, BOOM1000, CRASH500, BOOM500
#   Forex (Binary)  : frxEURUSD, frxGBPUSD, frxUSDJPY
SYMBOL = "R_100"  # Volatility 100 Index

# --- Tham số chiến lược ---
CANDLE_COUNT       = 100    # Số nến lịch sử dùng để tính chỉ báo
GRANULARITY        = 60     # Khung thời gian nến (giây): 60=1m, 300=5m, 900=15m, 3600=1h
RSI_PERIOD         = 14     # Chu kỳ RSI
RSI_OVERSOLD       = 30     # RSI < ngưỡng này → tín hiệu MUA
RSI_OVERBOUGHT     = 70     # RSI > ngưỡng này → tín hiệu BÁN
MOMENTUM_PERIOD    = 10     # Chu kỳ Momentum

# --- Tham số lệnh (Binary Options) ---
TRADE_AMOUNT       = 10     # Số tiền đặt cược (USD)
TRADE_CURRENCY     = "USD"
CONTRACT_DURATION  = 5      # Thời hạn hợp đồng
CONTRACT_DURATION_UNIT = "m"  # Đơn vị: t=giây, m=phút, h=giờ, d=ngày

# --- Redis ---
REDIS_HOST    = "localhost"
REDIS_PORT    = 6379
REDIS_DB      = 0
REDIS_HASH_KEY = "Deriv_Binary_Signal"

# --- Scheduler ---
SCAN_INTERVAL_SECONDS = 60  # Kiểm tra tín hiệu mỗi N giây
