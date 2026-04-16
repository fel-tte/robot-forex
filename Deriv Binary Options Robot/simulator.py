"""
simulator.py
============
Tự mô phỏng — Self-Simulate

Chạy engine phân tích trên dữ liệu lịch sử theo kiểu walk-forward:
  - Duyệt từng nến, gọi engine chấm điểm trên window lịch sử
  - Mô phỏng kết quả lệnh: thắng nếu giá đúng hướng sau N nến
  - Trả về SimResult với win_rate, profit_factor, expectancy

Hệ thống dùng simulator để:
  ① Kiểm tra xem chiến lược có khả thi trước khi vào LIVE
  ② Cung cấp dữ liệu nền để Learner phân tích
  ③ Tự đánh giá lại khi hiệu suất thực tế thay đổi
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

import config


# ──────────────────────────────────────────────────────────────────
# Dataclass kết quả mô phỏng
# ──────────────────────────────────────────────────────────────────

@dataclass
class SimTrade:
    candle_idx:  int
    symbol:      str
    direction:   str
    score:       float
    entry_price: float
    exit_price:  float
    won:         bool
    pnl:         float
    wave_active: bool
    fib_zone:    str
    rsi:         float


@dataclass
class SimResult:
    symbol:        str
    total_trades:  int
    wins:          int
    losses:        int
    win_rate_pct:  float
    total_pnl:     float
    profit_factor: float
    expectancy:    float   # P&L trung bình mỗi lệnh
    trades:        list[SimTrade] = field(default_factory=list)

    def is_viable(self) -> bool:
        """
        Chiến lược khả thi nếu:
          - Win rate >= 52%  (đủ để có lợi với payout 85%)
          - Profit Factor  >= 1.2
          - Ít nhất 5 lệnh mô phỏng
        """
        return (
            self.total_trades >= 5
            and self.win_rate_pct >= 52.0
            and self.profit_factor >= 1.2
        )


# ──────────────────────────────────────────────────────────────────
# Hàm mô phỏng chính
# ──────────────────────────────────────────────────────────────────

def simulate(
    df: pd.DataFrame,
    symbol: str = "SIM",
    min_score: Optional[float] = None,
    lookahead_candles: Optional[int] = None,
    payout_ratio: Optional[float] = None,
) -> SimResult:
    """
    Chạy walk-forward simulation trên DataFrame.

    Parameters
    ----------
    df                : DataFrame với cột close (ít nhất 80 nến)
    symbol            : tên thị trường dùng để gán nhãn
    min_score         : ngưỡng điểm để "đặt lệnh" (mặc định = config.MIN_SIGNAL_SCORE)
    lookahead_candles : số nến sau điểm vào để xác định thắng/thua
    payout_ratio      : tỉ lệ payout binary options

    Returns
    -------
    SimResult với đầy đủ chỉ số hiệu suất
    """
    # Import lazy để tránh circular import
    from brain import _score_signal   # noqa: PLC0415

    min_score         = min_score         if min_score         is not None else config.MIN_SIGNAL_SCORE
    lookahead_candles = lookahead_candles if lookahead_candles is not None else config.SIM_LOOKAHEAD_CANDLES
    payout_ratio      = payout_ratio      if payout_ratio      is not None else config.SIM_PAYOUT_RATIO
    stake             = config.SIM_STAKE_USD

    trades: list[SimTrade] = []
    warmup = 60   # Cần ít nhất 60 nến để tính đủ chỉ báo

    for i in range(warmup, len(df) - lookahead_candles):
        window = df.iloc[: i + 1].copy()

        try:
            sig = _score_signal(window)
            sig.symbol = symbol
        except Exception:
            continue

        if not sig.is_tradeable():
            continue
        if sig.score < min_score:
            continue

        entry_price = float(df.iloc[i]["close"])
        exit_price  = float(df.iloc[i + lookahead_candles]["close"])

        if sig.direction == "CALL":
            won = exit_price > entry_price
        elif sig.direction == "PUT":
            won = exit_price < entry_price
        else:
            continue

        pnl = round(stake * payout_ratio if won else -stake, 2)

        wave_active = bool(sig.wave and sig.wave.correction_active)
        fib_zone    = sig.wave.fib_zone if sig.wave else "NONE"

        trades.append(SimTrade(
            candle_idx  = i,
            symbol      = symbol,
            direction   = sig.direction,
            score       = sig.score,
            entry_price = entry_price,
            exit_price  = exit_price,
            won         = won,
            pnl         = pnl,
            wave_active = wave_active,
            fib_zone    = fib_zone,
            rsi         = sig.rsi,
        ))

    if not trades:
        return SimResult(
            symbol=symbol, total_trades=0, wins=0, losses=0,
            win_rate_pct=0.0, total_pnl=0.0, profit_factor=0.0,
            expectancy=0.0, trades=[],
        )

    total     = len(trades)
    wins      = sum(1 for t in trades if t.won)
    total_pnl = sum(t.pnl for t in trades)
    win_rate  = wins / total * 100

    gross_win  = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
    expectancy    = total_pnl / total

    return SimResult(
        symbol        = symbol,
        total_trades  = total,
        wins          = wins,
        losses        = total - wins,
        win_rate_pct  = round(win_rate, 2),
        total_pnl     = round(total_pnl, 2),
        profit_factor = round(profit_factor, 4),
        expectancy    = round(expectancy, 4),
        trades        = trades,
    )


# ──────────────────────────────────────────────────────────────────
# Chạy trực tiếp để kiểm tra
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import deriv_data
    for sym in config.SCAN_SYMBOLS:
        print(f"\nĐang backtest {sym}...")
        try:
            df     = deriv_data.fetch_candles(symbol=sym, count=config.SIM_CANDLE_COUNT)
            result = simulate(df, symbol=sym)
            status = "✅ KHẢ THI" if result.is_viable() else "⚠️  KHÔNG KHẢ THI"
            print(
                f"  {status} | trades={result.total_trades} "
                f"WR={result.win_rate_pct:.1f}% "
                f"PF={result.profit_factor:.2f} "
                f"PnL={result.total_pnl:+.2f}"
            )
        except Exception as exc:
            print(f"  Lỗi: {exc}")
