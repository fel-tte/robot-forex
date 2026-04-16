"""
deriv_trade.py
==============
Đọc tín hiệu từ Redis và đặt lệnh Binary Options (CALL / PUT) qua Deriv WebSocket API.

Luồng hoạt động:
  1. Đọc hash tín hiệu từ Redis
  2. Nếu Buy_Signal == 'True'  → đặt lệnh CALL (dự đoán giá tăng)
  3. Nếu Sell_Signal == 'True' → đặt lệnh PUT  (dự đoán giá giảm)
  4. Xoá hash khỏi Redis sau khi đặt lệnh
"""

import asyncio
import json
import redis
from datetime import datetime

import websockets

import config


# ------------------------------------------------------------------
# Hàm đặt lệnh qua Deriv WebSocket API
# ------------------------------------------------------------------

async def _place_contract(contract_type: str, symbol: str) -> dict:
    """
    Kết nối Deriv WS, xác thực token và đặt hợp đồng Binary Options.

    Parameters
    ----------
    contract_type : 'CALL' (mua) hoặc 'PUT' (bán)
    symbol        : ký hiệu thị trường, vd. 'R_100'

    Returns
    -------
    dict phản hồi từ Deriv API
    """
    async with websockets.connect(config.DERIV_WS_URL) as ws:
        # Bước 1: Xác thực
        auth_req = {"authorize": config.DERIV_API_TOKEN}
        await ws.send(json.dumps(auth_req))
        auth_res = json.loads(await ws.recv())

        if "error" in auth_res:
            raise PermissionError(
                f"Xác thực Deriv thất bại: {auth_res['error']['message']}"
            )

        print(
            f"[{datetime.now()}] Đăng nhập thành công: "
            f"balance={auth_res['authorize'].get('balance')} {auth_res['authorize'].get('currency')}"
        )

        # Bước 2: Đặt lệnh
        buy_req = {
            "buy": "1",
            "price": config.TRADE_AMOUNT,
            "parameters": {
                "amount"        : config.TRADE_AMOUNT,
                "basis"         : "stake",
                "contract_type" : contract_type,
                "currency"      : config.TRADE_CURRENCY,
                "duration"      : config.CONTRACT_DURATION,
                "duration_unit" : config.CONTRACT_DURATION_UNIT,
                "symbol"        : symbol,
            },
        }
        await ws.send(json.dumps(buy_req))
        buy_res = json.loads(await ws.recv())

    return buy_res


def place_contract(contract_type: str, symbol: str) -> None:
    """Wrapper đồng bộ cho _place_contract."""
    result = asyncio.run(_place_contract(contract_type, symbol))

    if "error" in result:
        print(f"[LỖI] Đặt lệnh thất bại: {result['error']['message']}")
    else:
        buy_info = result.get("buy", {})
        print(
            f"[{datetime.now()}] ✅ Đặt lệnh {contract_type} thành công!\n"
            f"   Contract ID : {buy_info.get('contract_id')}\n"
            f"   Buy price   : {buy_info.get('buy_price')} {config.TRADE_CURRENCY}\n"
            f"   Payout      : {buy_info.get('payout')} {config.TRADE_CURRENCY}\n"
            f"   Start time  : {buy_info.get('start_time')}\n"
        )


# ------------------------------------------------------------------
# Hàm chính: đọc Redis → đặt lệnh
# ------------------------------------------------------------------

def order_from_signal(signal: dict) -> None:
    """Phân tích tín hiệu và đặt lệnh tương ứng."""
    symbol      = signal.get("Symbol", config.SYMBOL)
    buy_signal  = signal.get("Buy_Signal", "False")
    sell_signal = signal.get("Sell_Signal", "False")

    if buy_signal == "True":
        print(f"[{datetime.now()}] 📈 Tín hiệu MUA → đặt lệnh CALL ({symbol})")
        place_contract("CALL", symbol)
    elif sell_signal == "True":
        print(f"[{datetime.now()}] 📉 Tín hiệu BÁN → đặt lệnh PUT ({symbol})")
        place_contract("PUT", symbol)
    else:
        print(f"[{datetime.now()}] Không có tín hiệu giao dịch.")


def entry_deriv() -> None:
    """Đọc tín hiệu từ Redis và đặt lệnh trên Deriv."""
    r = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB)
    raw = r.hgetall(config.REDIS_HASH_KEY)

    if not raw:
        print(f"[{datetime.now()}] Không có dữ liệu trong Redis hash='{config.REDIS_HASH_KEY}'")
        return

    # Decode bytes → str
    signal = {k.decode("utf-8"): v.decode("utf-8") for k, v in raw.items()}
    print(f"[{datetime.now()}] Đọc tín hiệu: {signal}")

    order_from_signal(signal)

    # Xoá tín hiệu sau khi xử lý để tránh đặt lệnh trùng
    r.delete(config.REDIS_HASH_KEY)
    print(f"[{datetime.now()}] Đã xoá hash '{config.REDIS_HASH_KEY}' khỏi Redis.")


# ------------------------------------------------------------------
# Chạy trực tiếp để kiểm tra
# ------------------------------------------------------------------
if __name__ == "__main__":
    entry_deriv()
