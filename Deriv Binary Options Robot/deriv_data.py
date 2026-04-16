"""
deriv_data.py
=============
Lấy dữ liệu nến (OHLCV) từ Deriv WebSocket API và lưu vào Redis.
"""

import asyncio
import json
import redis
import pandas as pd
import websockets
from datetime import datetime

import config


def fetch_candles(symbol: str = config.SYMBOL,
                  count: int = config.CANDLE_COUNT,
                  granularity: int = config.GRANULARITY) -> pd.DataFrame:
    """
    Gọi đồng bộ để lấy dữ liệu nến từ Deriv WebSocket API.

    Returns
    -------
    pd.DataFrame với các cột: open, high, low, close, epoch
    """
    return asyncio.run(_async_fetch_candles(symbol, count, granularity))


async def _async_fetch_candles(symbol: str, count: int, granularity: int) -> pd.DataFrame:
    request = {
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": count,
        "end": "latest",
        "granularity": granularity,
        "start": 1,
        "style": "candles",
    }

    async with websockets.connect(config.DERIV_WS_URL) as ws:
        await ws.send(json.dumps(request))
        response = json.loads(await ws.recv())

    if "error" in response:
        raise RuntimeError(f"Deriv API lỗi: {response['error']['message']}")

    candles = response.get("candles", [])
    if not candles:
        raise ValueError("Không có dữ liệu nến trả về từ Deriv.")

    df = pd.DataFrame(candles)
    df["epoch"] = pd.to_datetime(df["epoch"], unit="s")
    df = df.rename(columns={"epoch": "datetime"})
    df = df[["datetime", "open", "high", "low", "close"]].astype(
        {"open": float, "high": float, "low": float, "close": float}
    )
    return df


def save_candles_to_redis(df: pd.DataFrame,
                          r: redis.Redis,
                          key: str = "Deriv_Candles") -> None:
    """Lưu DataFrame nến vào Redis dưới dạng JSON string."""
    r.set(key, df.to_json(orient="records", date_format="iso"))
    print(f"[{datetime.now()}] Đã lưu {len(df)} nến vào Redis key='{key}'")


def load_candles_from_redis(r: redis.Redis,
                             key: str = "Deriv_Candles") -> pd.DataFrame:
    """Đọc DataFrame nến từ Redis."""
    raw = r.get(key)
    if raw is None:
        raise KeyError(f"Không tìm thấy dữ liệu tại Redis key='{key}'")
    df = pd.read_json(raw, orient="records")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


# -------------------------------------------------------
# Chạy trực tiếp để kiểm tra
# -------------------------------------------------------
if __name__ == "__main__":
    print(f"Đang lấy dữ liệu {config.SYMBOL} từ Deriv...")
    df = fetch_candles()
    print(df.tail(5).to_string())

    r = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB)
    save_candles_to_redis(df, r)
    print("Hoàn tất.")
