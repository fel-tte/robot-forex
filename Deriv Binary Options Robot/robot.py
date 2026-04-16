"""
robot.py
========
ENTRY POINT — Hệ thống tự vận hành hoàn toàn.

Bạn không còn điều hành. Bạn đang GIÁM SÁT.

Toàn bộ logic nằm trong DecisionEngine:
  tự chọn việc  — LIVE / PAPER / LEARNING / PAUSED
  tự quyết định — brain + predictor + learner
  tự mô phỏng   — paper cycle, walk-forward backtest
  tự hành động  — đặt lệnh thật khi đủ điều kiện
  tự sửa lỗi    — circuit breaker, retry, mode fallback
  tự học        — adaptive score threshold + stake multiplier
  tự dự đoán    — win probability với confidence score
  tự scale      — mở rộng / thu hẹp pool thị trường

Khởi động:
    python robot.py
"""

from decision_engine import DecisionEngine


def main() -> None:
    engine = DecisionEngine()
    engine.run()


if __name__ == "__main__":
    main()
