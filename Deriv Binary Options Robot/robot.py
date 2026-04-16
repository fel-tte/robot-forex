"""
robot.py
========
Robot trading TỰ VẬN HÀNH cho Deriv Binary Options.

Hệ thống tự:
  ① Quyết định làm gì trước  — quét tất cả thị trường, ưu tiên cơ hội tốt nhất
  ② Chọn điểm vào lệnh tốt nhất — tính điểm 0-100 qua 4 chỉ báo kỹ thuật
  ③ Điều phối tài nguyên    — quản lý kích thước lệnh, giới hạn lỗ ngày, cooldown
  ④ Tự vận hành hoàn toàn   — vòng lặp vô tận, tự ghi nhật ký, tự phục hồi lỗi

Khởi động:
    python robot.py
"""

import time
from datetime import datetime

import config
from brain        import pick_best_entry
from risk_manager import RiskManager
from deriv_trade  import get_balance, place_and_wait
from logger       import TradeLogger, TradeRecord


# ------------------------------------------------------------------
# Hàm thực thi một chu kỳ giao dịch
# ------------------------------------------------------------------

def run_cycle(risk: RiskManager, logger: TradeLogger) -> None:
    """
    Một chu kỳ tự vận hành:
      1. Lấy số dư → kiểm tra rủi ro
      2. Quét thị trường → chọn tín hiệu tốt nhất
      3. Tính kích thước lệnh
      4. Đặt lệnh và chờ kết quả
      5. Ghi nhật ký + cập nhật trạng thái rủi ro
    """
    ts = datetime.now()
    print(f"\n{'='*60}")
    print(f"[{ts}] 🤖 Bắt đầu chu kỳ giao dịch")
    print(f"{'='*60}")

    # --- Bước 1: Lấy số dư tài khoản ---
    try:
        balance = get_balance()
        print(f"[{datetime.now()}] 💰 Số dư: {balance} {config.TRADE_CURRENCY}")
    except Exception as exc:
        print(f"[LỖI] Không lấy được số dư: {exc} — bỏ qua chu kỳ này.")
        return

    # --- Bước 2: Kiểm tra điều kiện giao dịch ---
    allowed, reason = risk.can_trade(balance=balance)
    if not allowed:
        print(f"[RiskManager] 🚫 Không giao dịch: {reason}")
        return

    # --- Bước 3: Quét thị trường và chọn điểm vào tốt nhất ---
    print(f"\n[Brain] Đang quét {len(config.SCAN_SYMBOLS)} thị trường...")
    try:
        best = pick_best_entry()
    except Exception as exc:
        print(f"[LỖI] Quét thị trường thất bại: {exc} — bỏ qua chu kỳ này.")
        return

    if best is None:
        print("[Brain] Không có tín hiệu đủ điều kiện — chờ chu kỳ tiếp theo.")
        print(risk.summary())
        return

    # --- Bước 4: Tính kích thước lệnh ---
    stake = risk.compute_stake(best.score, balance)
    print(
        f"[Brain] 🎯 Tín hiệu: {best.symbol} {best.direction} "
        f"| score={best.score} | stake={stake} USD"
    )

    # --- Bước 5: Đặt lệnh và chờ kết quả ---
    print(f"[Trade] Đang đặt lệnh {best.direction} trên {best.symbol}...")
    try:
        result = place_and_wait(best.direction, best.symbol, stake)
    except Exception as exc:
        print(f"[LỖI] Đặt lệnh thất bại: {exc}")
        return

    won    = result["won"]
    pnl    = result["pnl"]
    payout = result.get("payout", 0)

    # --- Bước 6: Ghi nhật ký ---
    record = TradeRecord(
        timestamp    = ts.isoformat(),
        symbol       = best.symbol,
        direction    = best.direction,
        signal_score = best.score,
        stake        = stake,
        payout       = payout,
        pnl          = pnl,
        won          = won,
        contract_id  = result.get("contract_id", ""),
        rsi          = best.rsi,
        momentum     = best.momentum,
        macd_hist    = best.macd_hist,
        bb_position  = best.bb_position,
    )
    logger.log(record)

    # --- Bước 7: Cập nhật trạng thái rủi ro ---
    risk.update_after_trade(won=won, pnl=pnl)

    # --- Tóm tắt hiệu suất ---
    print(f"\n{risk.summary()}")
    logger.print_stats()


# ------------------------------------------------------------------
# Vòng lặp tự vận hành chính
# ------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  🤖 Deriv Binary Options — Hệ thống TỰ VẬN HÀNH")
    print(f"  Thị trường quét  : {', '.join(config.SCAN_SYMBOLS)}")
    print(f"  Điểm tối thiểu   : {config.MIN_SIGNAL_SCORE}/100")
    print(f"  Giới hạn lỗ ngày : {config.RISK_MAX_DAILY_LOSS_PCT*100:.0f}%")
    print(f"  Cooldown thua N  : {config.RISK_MAX_CONSECUTIVE_LOSS} lần → {config.RISK_COOLDOWN_MINUTES} phút")
    print(f"  Chu kỳ quét      : {config.SCAN_INTERVAL_SECONDS}s")
    print("=" * 60)

    risk   = RiskManager()
    logger = TradeLogger()

    print(f"\n[Khởi động] {risk.summary()}")
    logger.print_stats()
    print(f"\n[{datetime.now()}] Robot đang chạy. Nhấn Ctrl+C để dừng.\n")

    while True:
        try:
            run_cycle(risk, logger)
        except KeyboardInterrupt:
            print(f"\n[{datetime.now()}] Robot dừng theo yêu cầu.")
            print(risk.summary())
            logger.print_stats()
            break
        except Exception as exc:
            print(f"[LỖI nghiêm trọng] {exc} — tự phục hồi sau 30 giây...")
            time.sleep(30)
            continue

        print(f"\n[{datetime.now()}] Nghỉ {config.SCAN_INTERVAL_SECONDS}s trước chu kỳ tiếp theo...")
        time.sleep(config.SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
