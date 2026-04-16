"""
strategy.py
===========
Chiến lược giao dịch: RSI + Momentum

Logic:
  - Tín hiệu MUA (CALL): RSI vượt lên từ vùng quá bán (< RSI_OVERSOLD)
                          VÀ Momentum dương (giá đang tăng)
  - Tín hiệu BÁN (PUT) : RSI rơi xuống từ vùng quá mua (> RSI_OVERBOUGHT)
                          VÀ Momentum âm (giá đang giảm)

Kết quả được lưu vào Redis hash để module deriv_trade.py đọc và đặt lệnh.
"""

import redis
import pandas as pd
from datetime import datetime

import config
import deriv_data


# ------------------------------------------------------------------
# Hàm tính chỉ báo
# ------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """Tính RSI (Relative Strength Index)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_momentum(series: pd.Series, period: int = config.MOMENTUM_PERIOD) -> pd.Series:
    """Tính Momentum = close(t) - close(t - period)."""
    return series - series.shift(period)


# ------------------------------------------------------------------
# Hàm sinh tín hiệu
# ------------------------------------------------------------------

def generate_signal(df: pd.DataFrame) -> dict:
    """
    Phân tích DataFrame nến và trả về dict tín hiệu:
    {
        'Symbol'      : str,
        'Buy_Signal'  : str ('True' / 'False'),
        'Sell_Signal' : str ('True' / 'False'),
        'RSI'         : str,
        'Momentum'    : str,
        'Insertdate'  : str,
    }
    """
    df = df.copy()
    df["rsi"]      = compute_rsi(df["close"])
    df["momentum"] = compute_momentum(df["close"])

    last = df.iloc[-1]
    prev = df.iloc[-2]

    rsi_now  = last["rsi"]
    mom_now  = last["momentum"]

    # Tín hiệu MUA: RSI vừa vượt ngưỡng quá bán từ dưới lên + momentum dương
    buy_signal = (
        prev["rsi"] < config.RSI_OVERSOLD
        and rsi_now >= config.RSI_OVERSOLD
        and mom_now > 0
    )

    # Tín hiệu BÁN: RSI vừa rơi dưới ngưỡng quá mua từ trên xuống + momentum âm
    sell_signal = (
        prev["rsi"] > config.RSI_OVERBOUGHT
        and rsi_now <= config.RSI_OVERBOUGHT
        and mom_now < 0
    )

    signal = {
        "Symbol"     : config.SYMBOL,
        "Buy_Signal" : str(buy_signal),
        "Sell_Signal": str(sell_signal),
        "RSI"        : f"{rsi_now:.4f}",
        "Momentum"   : f"{mom_now:.4f}",
        "Insertdate" : datetime.now().isoformat(),
    }

    print(
        f"[{signal['Insertdate']}] "
        f"RSI={signal['RSI']} | Momentum={signal['Momentum']} | "
        f"BUY={signal['Buy_Signal']} | SELL={signal['Sell_Signal']}"
    )
    return signal


def save_signal_to_redis(signal: dict, r: redis.Redis) -> None:
    """Lưu tín hiệu vào Redis hash (xoá cũ trước khi lưu mới)."""
    r.delete(config.REDIS_HASH_KEY)
    r.hset(config.REDIS_HASH_KEY, mapping=signal)
    print(f"[{datetime.now()}] Đã lưu tín hiệu vào Redis hash='{config.REDIS_HASH_KEY}'")


# ------------------------------------------------------------------
# Hàm chính: lấy dữ liệu → phân tích → lưu Redis
# ------------------------------------------------------------------

def scan_market() -> None:
    """Lấy dữ liệu thị trường, tính tín hiệu và đẩy vào Redis."""
    r = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB)

    try:
        df = deriv_data.fetch_candles()
    except Exception as exc:
        print(f"[LỖI] Không thể lấy dữ liệu: {exc}")
        return

    signal = generate_signal(df)
    save_signal_to_redis(signal, r)


# ------------------------------------------------------------------
# Chạy trực tiếp để kiểm tra
# ------------------------------------------------------------------
if __name__ == "__main__":
    scan_market()
