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

# ============================================================
# SIMULATOR — Tự mô phỏng (Self-Simulate)
# ============================================================

# Số nến tải về cho backtest (nhiều hơn CANDLE_COUNT để đủ walk-forward)
SIM_CANDLE_COUNT      = 200

# Số nến sau điểm vào để xác định kết quả thắng/thua
# (5 nến × GRANULARITY 60s = 5 phút ~ CONTRACT_DURATION)
SIM_LOOKAHEAD_CANDLES = 5

# Tỉ lệ payout binary options (85% → thắng nhận 85%, thua mất 100%)
SIM_PAYOUT_RATIO      = 0.85

# Stake giả dùng khi mô phỏng
SIM_STAKE_USD         = 10.0

# ============================================================
# LEARNER — Tự học (Self-Learn)
# ============================================================

# Cần ít nhất N lệnh trong lịch sử mới học
LEARNER_MIN_HISTORY     = 20

# Học lại sau mỗi N chu kỳ vận hành
LEARNER_INTERVAL_CYCLES = 10

# Win rate < ngưỡng này → điều kiện tín hiệu bị đánh dấu "yếu"
LEARNER_WEAK_WIN_RATE   = 0.45   # 45%

# ============================================================
# PREDICTOR — Tự dự đoán (Self-Predict)
# ============================================================

# Xác suất thắng tối thiểu để predictor cho phép vào lệnh
PREDICT_MIN_WIN_PROB      = 0.54

# Mức tự tin tối thiểu để vào lệnh
PREDICT_MIN_CONFIDENCE    = 0.30

# Ngưỡng ATR tương đối (ATR / price) xác định biến động cao/thấp
PREDICT_HIGH_VOLATILITY_ATR = 0.005   # > 0.5% → biến động cao
PREDICT_LOW_VOLATILITY_ATR  = 0.001   # < 0.1% → biến động thấp

# ============================================================
# DECISION ENGINE — Điều khiển nhịp vận hành
# ============================================================

# Chạy backtest simulation cho tất cả markets khi khởi động
ENGINE_RUN_SIM_ON_START = True

# Thời gian nghỉ (giây) khi self-heal phát hiện lỗi liên tiếp
HEAL_COOLDOWN_SECONDS   = 60

# ============================================================
# SCALER — Tự scale (Self-Scale)
# ============================================================

# Cần ít nhất N lệnh để đủ cơ sở scale
SCALE_MIN_TRADES      = 15

# Win rate >= ngưỡng này → mở rộng pool thị trường
SCALE_HIGH_WIN_RATE   = 65.0

# Win rate < ngưỡng này → thu hẹp pool thị trường
SCALE_LOW_WIN_RATE    = 45.0

# Kiểm tra scale mỗi N chu kỳ
SCALE_INTERVAL_CYCLES = 20

# ============================================================
# PIPELINE — Dây chuyền điều phối vận hành
# ============================================================

# Số lệnh tối đa trong hàng đợi cùng lúc
PIPELINE_MAX_QUEUE_DEPTH     = 3

# Số lệnh tối đa đang chờ xử lý trong 1 cửa sổ thời gian
# (giới hạn tải — rate limiting)
PIPELINE_RATE_WINDOW_SECONDS = 300    # 5 phút
PIPELINE_RATE_MAX_TRADES     = 3      # Tối đa 3 lệnh / 5 phút

# Khoảng cách tối thiểu giữa 2 lệnh liên tiếp (giây)
# Ngăn "đặt lệnh liên tục" — load spacing
PIPELINE_MIN_TRADE_GAP_SECONDS = 30

# Điểm quyền hạn tối thiểu để lệnh vượt qua cổng xác nhận
# Tổng điểm quyền hạn = signal_score_gate + predictor_gate + risk_gate
# Mỗi cổng đóng góp True/False → tổng tối đa 3
PIPELINE_MIN_AUTHORITY_GATES  = 2     # Cần ít nhất 2/3 cổng thông qua

# Kích thước cửa sổ đo lường (giây)
PIPELINE_METRICS_WINDOW_SECONDS = 3600   # Tính metrics trên 1 giờ gần nhất

# ============================================================
# MEMORY BRAIN — Redis là bộ não trung tâm ghi nhớ Win/Loss
# ============================================================

# Số lệnh tối thiểu trên một mẫu (fingerprint) để xét luật cứng
MEMORY_MIN_SAMPLES_FOR_RULE  = 3

# Tỉ lệ thua tối thiểu để đưa fingerprint vào danh sách chặn cứng
# Fingerprint có loss_rate >= ngưỡng này → luật cứng: BLOCK
MEMORY_HARD_BLOCK_LOSS_RATE  = 0.20      # >= 20% thua → chặn cứng (Hard Rule bắt buộc theo pipeline)

# Tỉ lệ thắng tốt để tăng ưu tiên (priority boost) cho fingerprint
MEMORY_STRONG_WIN_RATE       = 0.65      # >= 65% thắng → bonus ưu tiên

# Số fingerprint tối đa lưu trong Redis (FIFO — cũ nhất bị xóa)
MEMORY_MAX_PATTERNS          = 500

# Redis key prefix cho từng pattern
REDIS_MEMORY_PREFIX          = "Deriv_Mem:"          # + fingerprint

# Redis key lưu danh sách luật cứng (JSON list)
REDIS_MEMORY_RULES_KEY       = "Deriv_Mem_Rules"

# Redis key lưu tổng hợp thống kê memory (JSON)
REDIS_MEMORY_STATS_KEY       = "Deriv_Mem_Stats"
