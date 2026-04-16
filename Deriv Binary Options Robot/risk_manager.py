"""
risk_manager.py
===============
Tự điều phối tài nguyên và bảo vệ tài khoản.

Chức năng:
  - Tính kích thước lệnh tự động theo điểm tín hiệu + số dư tài khoản
  - Giới hạn lỗ theo ngày (auto-pause khi vượt ngưỡng)
  - Cooldown sau chuỗi thua liên tiếp
  - Lưu / phục hồi trạng thái qua Redis (bền vững qua restart)
"""

import json
import redis
from datetime import datetime, date, timedelta
from dataclasses import dataclass, asdict

import config


# ------------------------------------------------------------------
# Trạng thái rủi ro (được lưu vào Redis)
# ------------------------------------------------------------------

@dataclass
class RiskState:
    # Được đặt lại mỗi ngày
    trade_date:         str   = ""       # ISO date của ngày hiện tại
    daily_pnl:          float = 0.0      # Lãi/lỗ trong ngày (USD)
    trades_today:       int   = 0        # Số lệnh đã đặt hôm nay
    wins_today:         int   = 0
    losses_today:       int   = 0

    # Tích luỹ toàn thời gian
    consecutive_losses: int   = 0        # Số lần thua liên tiếp hiện tại
    total_trades:       int   = 0
    total_wins:         int   = 0
    total_pnl:          float = 0.0

    # Cooldown
    paused_until:       str   = ""       # ISO datetime, "" = không bị pause


# ------------------------------------------------------------------
# Risk Manager
# ------------------------------------------------------------------

class RiskManager:
    """
    Quản lý rủi ro tự động cho robot Deriv.

    Khởi tạo:
        rm = RiskManager()

    Trước khi đặt lệnh:
        if rm.can_trade():
            stake = rm.compute_stake(signal.score, balance)

    Sau khi lệnh kết thúc:
        rm.update_after_trade(won=True, pnl=+8.50)
    """

    def __init__(self) -> None:
        self._r = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
        )
        self.state = self._load_state()
        self._reset_if_new_day()

    # ----------------------------------------------------------
    # Persist state
    # ----------------------------------------------------------

    def _load_state(self) -> RiskState:
        raw = self._r.get(config.REDIS_STATE_KEY)
        if raw:
            try:
                data = json.loads(raw)
                return RiskState(**{k: v for k, v in data.items() if k in RiskState.__dataclass_fields__})
            except Exception:
                pass
        return RiskState()

    def _save_state(self) -> None:
        self._r.set(config.REDIS_STATE_KEY, json.dumps(asdict(self.state)))

    # ----------------------------------------------------------
    # Daily reset
    # ----------------------------------------------------------

    def _reset_if_new_day(self) -> None:
        today = date.today().isoformat()
        if self.state.trade_date != today:
            self.state.trade_date   = today
            self.state.daily_pnl    = 0.0
            self.state.trades_today = 0
            self.state.wins_today   = 0
            self.state.losses_today = 0
            self._save_state()
            print(f"[RiskManager] Ngày mới ({today}) — đặt lại thống kê ngày.")

    # ----------------------------------------------------------
    # Kiểm tra có được phép giao dịch không
    # ----------------------------------------------------------

    def can_trade(self, balance: float = 0.0) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).

        Kiểm tra các điều kiện:
          1. Đang trong thời gian cooldown?
          2. Lỗ ngày vượt quá giới hạn?
        """
        self._reset_if_new_day()

        # 1. Cooldown
        if self.state.paused_until:
            pause_dt = datetime.fromisoformat(self.state.paused_until)
            if datetime.now() < pause_dt:
                remaining = int((pause_dt - datetime.now()).total_seconds() / 60)
                return False, f"Đang cooldown, còn {remaining} phút"
            else:
                self.state.paused_until = ""
                self._save_state()
                print("[RiskManager] Hết cooldown — tiếp tục giao dịch.")

        # 2. Giới hạn lỗ ngày (chỉ kiểm tra nếu có balance)
        if balance > 0:
            max_daily_loss = balance * config.RISK_MAX_DAILY_LOSS_PCT
            if self.state.daily_pnl <= -max_daily_loss:
                return False, (
                    f"Đã lỗ {abs(self.state.daily_pnl):.2f} USD trong ngày "
                    f"(giới hạn {max_daily_loss:.2f} USD)"
                )

        return True, "OK"

    # ----------------------------------------------------------
    # Tính kích thước lệnh
    # ----------------------------------------------------------

    def compute_stake(self, signal_score: float, balance: float) -> float:
        """
        Tính kích thước lệnh dựa trên điểm tín hiệu và số dư.

        Phân bổ:
          score >= 80 → 5% số dư
          score 60-79 → 3% số dư
          score < 60  → 2% số dư

        Kết quả được giới hạn trong [STAKE_MIN_USD, STAKE_MAX_USD].
        """
        if signal_score >= 80:
            pct = config.STAKE_PCT_HIGH
        elif signal_score >= 60:
            pct = config.STAKE_PCT_MEDIUM
        else:
            pct = config.STAKE_PCT_LOW

        raw = balance * pct
        stake = max(config.STAKE_MIN_USD, min(config.STAKE_MAX_USD, raw))
        return round(stake, 2)

    # ----------------------------------------------------------
    # Cập nhật sau khi lệnh kết thúc
    # ----------------------------------------------------------

    def update_after_trade(self, won: bool, pnl: float) -> None:
        """
        Cập nhật trạng thái sau kết quả lệnh.

        Parameters
        ----------
        won : True nếu thắng, False nếu thua
        pnl : Lãi (+) hoặc lỗ (-) tính bằng USD
        """
        self._reset_if_new_day()

        self.state.daily_pnl    += pnl
        self.state.total_pnl    += pnl
        self.state.trades_today += 1
        self.state.total_trades += 1

        if won:
            self.state.wins_today      += 1
            self.state.total_wins      += 1
            self.state.consecutive_losses = 0
            print(f"[RiskManager] ✅ Thắng | P&L ngày: {self.state.daily_pnl:+.2f} USD")
        else:
            self.state.losses_today        += 1
            self.state.consecutive_losses  += 1
            print(
                f"[RiskManager] ❌ Thua | "
                f"thua liên tiếp: {self.state.consecutive_losses} | "
                f"P&L ngày: {self.state.daily_pnl:+.2f} USD"
            )
            # Kích hoạt cooldown nếu thua quá nhiều lần liên tiếp
            if self.state.consecutive_losses >= config.RISK_MAX_CONSECUTIVE_LOSS:
                pause_until = datetime.now() + timedelta(minutes=config.RISK_COOLDOWN_MINUTES)
                self.state.paused_until = pause_until.isoformat()
                print(
                    f"[RiskManager] ⏸️  Thua {self.state.consecutive_losses} lần liên tiếp — "
                    f"cooldown đến {pause_until.strftime('%H:%M:%S')}"
                )

        self._save_state()

    # ----------------------------------------------------------
    # Thống kê
    # ----------------------------------------------------------

    def summary(self) -> str:
        """Trả về chuỗi tóm tắt hiệu suất."""
        win_rate = (
            self.state.total_wins / self.state.total_trades * 100
            if self.state.total_trades > 0 else 0
        )
        return (
            f"📊 Thống kê | "
            f"Tổng lệnh: {self.state.total_trades} | "
            f"Thắng: {self.state.total_wins} ({win_rate:.1f}%) | "
            f"P&L tổng: {self.state.total_pnl:+.2f} USD | "
            f"P&L hôm nay: {self.state.daily_pnl:+.2f} USD"
        )


# ------------------------------------------------------------------
# Chạy trực tiếp để kiểm tra
# ------------------------------------------------------------------
if __name__ == "__main__":
    rm = RiskManager()
    print(rm.summary())
    print("can_trade:", rm.can_trade(balance=100))
    print("stake (score=85, balance=100):", rm.compute_stake(85, 100))
    print("stake (score=70, balance=100):", rm.compute_stake(70, 100))
    print("stake (score=55, balance=100):", rm.compute_stake(55, 100))
