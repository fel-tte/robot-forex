"""
deriv_trade.py
==============
Đặt lệnh Binary Options (CALL / PUT) qua Deriv WebSocket API và theo dõi kết quả.

Luồng hoạt động:
  1. Xác thực token, lấy số dư tài khoản
  2. Đặt hợp đồng CALL hoặc PUT với kích thước lệnh được truyền vào
  3. Chờ hợp đồng kết thúc, lấy kết quả (thắng/thua, P&L)
  4. Trả về dict kết quả cho robot.py

API tham khảo: https://api.deriv.com/
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Optional

import websockets

import config


# ------------------------------------------------------------------
# Xác thực và lấy số dư
# ------------------------------------------------------------------

async def _get_balance_async() -> Optional[float]:
    """Xác thực và lấy số dư tài khoản hiện tại."""
    async with websockets.connect(config.DERIV_WS_URL) as ws:
        await ws.send(json.dumps({"authorize": config.DERIV_API_TOKEN}))
        res = json.loads(await ws.recv())
        if "error" in res:
            raise PermissionError(f"Xác thực Deriv thất bại: {res['error']['message']}")
        return float(res["authorize"].get("balance", 0))


def get_balance() -> float:
    """Lấy số dư tài khoản (đồng bộ)."""
    return asyncio.run(_get_balance_async())


# ------------------------------------------------------------------
# Đặt lệnh và chờ kết quả
# ------------------------------------------------------------------

async def _place_and_wait_async(contract_type: str,
                                 symbol: str,
                                 stake: float) -> dict:
    """
    Đặt hợp đồng Binary Options và chờ kết thúc để lấy kết quả.

    Returns
    -------
    dict với các key:
      contract_id, won (bool), buy_price, sell_price, payout, pnl, profit, status
    """
    async with websockets.connect(config.DERIV_WS_URL) as ws:
        # Bước 1: Xác thực
        await ws.send(json.dumps({"authorize": config.DERIV_API_TOKEN}))
        auth_res = json.loads(await ws.recv())
        if "error" in auth_res:
            raise PermissionError(f"Xác thực thất bại: {auth_res['error']['message']}")

        balance = float(auth_res["authorize"].get("balance", 0))
        print(
            f"[{datetime.now()}] Đăng nhập thành công: "
            f"balance={balance} {auth_res['authorize'].get('currency')}"
        )

        # Bước 2: Đặt lệnh
        buy_req = {
            "buy": "1",
            "price": stake,
            "parameters": {
                "amount"        : stake,
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

        if "error" in buy_res:
            raise RuntimeError(f"Đặt lệnh thất bại: {buy_res['error']['message']}")

        buy_info    = buy_res.get("buy", {})
        contract_id = buy_info.get("contract_id", "")
        buy_price   = float(buy_info.get("buy_price", stake))
        payout      = float(buy_info.get("payout", 0))

        print(
            f"[{datetime.now()}] ✅ Đặt lệnh {contract_type} thành công! "
            f"contract_id={contract_id} stake={buy_price} payout={payout}"
        )

        # Bước 3: Theo dõi hợp đồng đến khi kết thúc
        await ws.send(json.dumps({
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "subscribe": 1,
        }))

        sell_price = 0.0
        status     = "open"
        while True:
            msg = json.loads(await ws.recv())
            if "error" in msg:
                print(f"[WARN] Lỗi khi theo dõi: {msg['error']['message']}")
                break
            poc = msg.get("proposal_open_contract", {})
            status = poc.get("status", "open")
            if status in ("sold", "won", "lost"):
                sell_price = float(poc.get("sell_price", 0))
                break
            # Chờ cập nhật tiếp theo
            await asyncio.sleep(1)

    won = sell_price > 0 and sell_price >= buy_price
    pnl = sell_price - buy_price

    return {
        "contract_id": str(contract_id),
        "won"        : won,
        "buy_price"  : buy_price,
        "sell_price" : sell_price,
        "payout"     : payout,
        "pnl"        : round(pnl, 2),
        "status"     : status,
    }


def place_and_wait(contract_type: str,
                   symbol: str,
                   stake: float) -> dict:
    """
    Đặt hợp đồng và chờ kết quả (đồng bộ).

    Parameters
    ----------
    contract_type : 'CALL' hoặc 'PUT'
    symbol        : mã thị trường, vd. 'R_100'
    stake         : số tiền đặt cược (USD)

    Returns
    -------
    dict kết quả (xem _place_and_wait_async)
    """
    return asyncio.run(_place_and_wait_async(contract_type, symbol, stake))


# ------------------------------------------------------------------
# Chạy trực tiếp để kiểm tra
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Đang lấy số dư...")
    bal = get_balance()
    print(f"Số dư: {bal} {config.TRADE_CURRENCY}")
