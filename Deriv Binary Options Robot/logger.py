"""
logger.py
=========
Nhật ký giao dịch tự động.

Chức năng:
  - Ghi mỗi lệnh vào file CSV (trade_log.csv)
  - Đồng thời đẩy vào Redis List để truy vấn nhanh
  - Tính win-rate, tổng P&L, và các chỉ số hiệu suất
"""

import csv
import json
import os
import redis
from datetime import datetime
from dataclasses import dataclass, asdict

import config


# ------------------------------------------------------------------
# Dataclass ghi nhận một lệnh
# ------------------------------------------------------------------

@dataclass
class TradeRecord:
    timestamp:    str     # ISO datetime
    symbol:       str
    direction:    str     # 'CALL' / 'PUT'
    signal_score: float
    stake:        float   # Số tiền đặt
    payout:       float   # Khoản nhận về (0 nếu thua)
    pnl:          float   # Lãi/lỗ thực tế
    won:          bool
    contract_id:  str     = ""
    rsi:          float   = 0.0
    momentum:     float   = 0.0
    macd_hist:    float   = 0.0
    bb_position:  float   = 0.0


# ------------------------------------------------------------------
# Logger
# ------------------------------------------------------------------

_CSV_HEADER = [
    "timestamp", "symbol", "direction", "signal_score",
    "stake", "payout", "pnl", "won",
    "contract_id", "rsi", "momentum", "macd_hist", "bb_position",
]


class TradeLogger:
    """
    Ghi nhật ký giao dịch vào CSV và Redis.

    Sử dụng:
        logger = TradeLogger()
        logger.log(record)
        stats = logger.get_stats()
    """

    def __init__(self,
                 csv_path: str = config.TRADE_LOG_FILE,
                 redis_key: str = config.REDIS_LOG_KEY) -> None:
        self._csv_path  = csv_path
        self._redis_key = redis_key
        self._r = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
        )
        self._ensure_csv()

    def _ensure_csv(self) -> None:
        """Tạo file CSV với header nếu chưa tồn tại."""
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_HEADER)
                writer.writeheader()

    def log(self, record: TradeRecord) -> None:
        """Ghi một lệnh vào CSV và Redis."""
        row = asdict(record)

        # Ghi vào CSV
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_HEADER)
            writer.writerow(row)

        # Đẩy vào Redis List (giữ tối đa 500 bản ghi)
        self._r.lpush(self._redis_key, json.dumps(row))
        self._r.ltrim(self._redis_key, 0, 499)

        status = "✅ THẮNG" if record.won else "❌ THUA"
        print(
            f"[Logger] {status} | {record.symbol} {record.direction} | "
            f"stake={record.stake:.2f} payout={record.payout:.2f} "
            f"P&L={record.pnl:+.2f} USD | score={record.signal_score}"
        )

    def get_stats(self) -> dict:
        """Tính các chỉ số hiệu suất từ lịch sử trong Redis."""
        raw_list = self._r.lrange(self._redis_key, 0, -1)
        if not raw_list:
            return {"message": "Chưa có dữ liệu giao dịch."}

        records = [json.loads(r) for r in raw_list]
        total   = len(records)
        wins    = sum(1 for r in records if r.get("won"))
        total_pnl = sum(r.get("pnl", 0) for r in records)
        win_rate  = wins / total * 100 if total else 0

        # Tính profit factor
        gross_win  = sum(r["pnl"] for r in records if r.get("pnl", 0) > 0)
        gross_loss = abs(sum(r["pnl"] for r in records if r.get("pnl", 0) < 0))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

        return {
            "total_trades" : total,
            "wins"         : wins,
            "losses"       : total - wins,
            "win_rate_pct" : round(win_rate, 2),
            "total_pnl"    : round(total_pnl, 2),
            "gross_win"    : round(gross_win, 2),
            "gross_loss"   : round(gross_loss, 2),
            "profit_factor": round(profit_factor, 4),
        }

    def print_stats(self) -> None:
        stats = self.get_stats()
        if "message" in stats:
            print(f"[Logger] {stats['message']}")
            return
        print(
            f"\n{'='*50}\n"
            f"📊 HIỆU SUẤT GIAO DỊCH\n"
            f"{'='*50}\n"
            f"  Tổng lệnh      : {stats['total_trades']}\n"
            f"  Thắng / Thua   : {stats['wins']} / {stats['losses']}\n"
            f"  Tỉ lệ thắng    : {stats['win_rate_pct']}%\n"
            f"  Tổng P&L       : {stats['total_pnl']:+.2f} USD\n"
            f"  Gross Win      : +{stats['gross_win']:.2f} USD\n"
            f"  Gross Loss     : -{stats['gross_loss']:.2f} USD\n"
            f"  Profit Factor  : {stats['profit_factor']}\n"
            f"{'='*50}"
        )


# ------------------------------------------------------------------
# Chạy trực tiếp để xem thống kê
# ------------------------------------------------------------------
if __name__ == "__main__":
    logger = TradeLogger()
    logger.print_stats()
