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

# ============================================================
# Hệ thống TỰ VẬN HÀNH
# ============================================================

# --- Danh sách thị trường tự quét ---
# Robot sẽ tự chọn thị trường tốt nhất trong danh sách này
SCAN_SYMBOLS = ["R_10", "R_25", "R_50", "R_75", "R_100"]

# --- Ngưỡng chất lượng tín hiệu ---
# Robot chỉ đặt lệnh khi điểm tín hiệu >= ngưỡng này (0-100)
MIN_SIGNAL_SCORE = 60

# --- Quản lý rủi ro tự động ---
RISK_MAX_DAILY_LOSS_PCT  = 0.20   # Dừng giao dịch khi lỗ >= 20% số dư ban đầu trong ngày
RISK_MAX_CONSECUTIVE_LOSS = 5     # Dừng tạm thời sau N lần thua liên tiếp
RISK_COOLDOWN_MINUTES     = 30    # Nghỉ bao nhiêu phút sau chuỗi thua

# --- Quản lý kích thước lệnh tự động ---
# Dựa trên điểm tín hiệu (0-100) và số dư tài khoản
STAKE_PCT_HIGH   = 0.05   # score >= 80 → 5% số dư
STAKE_PCT_MEDIUM = 0.03   # score 60-79 → 3% số dư
STAKE_PCT_LOW    = 0.02   # score < 60  → 2% số dư (fallback)
STAKE_MIN_USD    = 1.0    # Lệnh tối thiểu (USD)
STAKE_MAX_USD    = 50.0   # Lệnh tối đa (USD)

# --- Redis keys cho trạng thái tự vận hành ---
REDIS_STATE_KEY   = "Deriv_Robot_State"    # Hash: trạng thái rủi ro
REDIS_LOG_KEY     = "Deriv_Trade_Log"      # List: lịch sử lệnh (JSON)

# --- File log giao dịch ---
TRADE_LOG_FILE = "trade_log.csv"

# ============================================================
# Hệ thống PHÂN TÍCH SÓNG (Wave Analyzer — Operator System)
# ============================================================

# Cửa sổ rolling để phát hiện đỉnh/đáy (Swing High/Low)
WAVE_SWING_ORDER = 5

# Kích thước tối thiểu của sóng chính (% so với giá hiện tại)
# Sóng nhỏ hơn ngưỡng này bị bỏ qua
WAVE_MIN_SIZE_PCT = 0.005       # 0.5% giá

# Biên sóng hồi hợp lệ: [min%, max%] của sóng chính
# < 20%  → chưa đủ sâu để tính là sóng hồi
# > 80%  → có thể là đảo chiều, không phải hồi
WAVE_CORRECTION_MIN = 0.20      # 20%
WAVE_CORRECTION_MAX = 0.80      # 80%

# Dung sai xác nhận "tại vùng Fibonacci" (±% khoảng cách sóng)
WAVE_FIB_TOLERANCE = 0.015      # ±1.5%
