"""
robot.py
========
Hệ thống OPERATOR — Bạn không còn điều hành, bạn đang GIÁM SÁT.

Hệ thống tự:
  ① Phát hiện sóng hồi trong sóng chính khi gặp cản mạnh
  ② Đo lường độ sâu sóng hồi (Fibonacci 38.2/50/61.8%)
  ③ Giới hạn rủi ro tự động (lỗ ngày, cooldown chuỗi thua)
  ④ Điều phối tài nguyên (stake động theo chất lượng tín hiệu)
  ⑤ Tìm điểm vào an toàn nhất (cuối sóng hồi + Fib + S/R)
  ⑥ Xác định điểm thoát an toàn nhất (TP = đỉnh/đáy sóng chính)

Bạn chỉ cần xem màn hình — hệ thống tự quyết định tất cả.

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
# Banner giám sát — in sau mỗi chu kỳ
# ------------------------------------------------------------------

def _wave_banner(sig) -> str:
    """Tạo chuỗi hiển thị trạng thái sóng cho màn hình giám sát."""
    w = sig.wave
    if w is None:
        return "  🌊 Sóng: Không có dữ liệu"
    if not w.correction_active:
        return f"  🌊 Xu hướng {w.main_direction} — không có sóng hồi"
    return (
        f"  🌊 Sóng chính : {w.main_direction}  |  Kích thước: {w.main_wave_size:.4f}\n"
        f"  🔄 Sóng hồi   : {w.correction_depth_pct:.1f}%  |  Fib zone: {w.fib_zone}"
        f"  |  S/R: {'✓' if w.at_support_resistance else '✗'}\n"
        f"  🎯 Vào lệnh   : {w.entry_direction}  |  Điểm sóng: {w.entry_score}/40\n"
        f"  📍 TP / SL    : {w.tp_price}  /  {w.sl_price}"
    )


# ------------------------------------------------------------------
# Một chu kỳ vận hành
# ------------------------------------------------------------------

def run_cycle(risk: RiskManager, logger: TradeLogger) -> None:
    """
    Một chu kỳ tự vận hành:
      1. Lấy số dư → kiểm tra rủi ro
      2. Quét thị trường → phân tích sóng → chọn điểm vào tốt nhất
      3. Tính kích thước lệnh theo điểm tín hiệu + số dư
      4. Đặt lệnh và chờ kết quả
      5. Ghi nhật ký + cập nhật trạng thái rủi ro
    """
    ts = datetime.now()
    print(f"\n{'='*65}")
    print(f"  👁️  OPERATOR MONITOR  |  {ts.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")

    # ── Bước 1: Lấy số dư ─────────────────────────────────────────
    try:
        balance = get_balance()
        print(f"  💰 Số dư: {balance:.2f} {config.TRADE_CURRENCY}")
    except Exception as exc:
        print(f"  [LỖI] Không lấy được số dư: {exc}")
        return

    # ── Bước 2: Kiểm tra rủi ro ───────────────────────────────────
    allowed, reason = risk.can_trade(balance=balance)
    if not allowed:
        print(f"  🚫 Hệ thống tạm dừng: {reason}")
        return

    # ── Bước 3: Quét thị trường + phân tích sóng ──────────────────
    print(f"\n  [Scanning {len(config.SCAN_SYMBOLS)} markets + wave analysis]")
    try:
        best = pick_best_entry()
    except Exception as exc:
        print(f"  [LỖI] Quét thất bại: {exc}")
        return

    if best is None:
        print("  ⏳ Không có tín hiệu đủ điều kiện — chờ chu kỳ tiếp theo.")
        print(f"\n  {risk.summary()}")
        return

    # ── Hiển thị bảng giám sát ────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  🏆 {best.symbol}  |  {best.direction}  |  Điểm tổng: {best.score}/100")
    print(f"  📊 RSI={best.rsi:.1f}  Momentum={best.momentum:.4f}  BB={best.bb_position:.3f}")
    print(_wave_banner(best))
    print(f"{'─'*65}")

    # ── Bước 4: Tính kích thước lệnh ──────────────────────────────
    stake = risk.compute_stake(best.score, balance)
    print(f"  💵 Stake: {stake:.2f} USD  (score={best.score})")

    # ── Bước 5: Đặt lệnh và chờ kết quả ──────────────────────────
    print(f"  ⚡ Đang đặt lệnh {best.direction} trên {best.symbol}...")
    try:
        result = place_and_wait(best.direction, best.symbol, stake)
    except Exception as exc:
        print(f"  [LỖI] Đặt lệnh thất bại: {exc}")
        return

    won    = result["won"]
    pnl    = result["pnl"]
    payout = result.get("payout", 0)

    # ── Bước 6: Ghi nhật ký ───────────────────────────────────────
    tp_price = best.wave.tp_price if best.wave else 0.0
    sl_price = best.wave.sl_price if best.wave else 0.0
    fib_zone = best.wave.fib_zone if best.wave else "NONE"
    wave_score = best.wave.entry_score if best.wave else 0.0
    correction = best.wave.correction_active if best.wave else False

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

    # ── Bước 7: Cập nhật rủi ro ───────────────────────────────────
    risk.update_after_trade(won=won, pnl=pnl)

    # ── Tóm tắt chu kỳ ────────────────────────────────────────────
    status = "✅ THẮNG" if won else "❌ THUA"
    print(f"\n  {status}  |  P&L: {pnl:+.2f} USD")
    if correction and tp_price and sl_price:
        print(f"  📌 TP mục tiêu = {tp_price}  |  SL an toàn = {sl_price}  |  Fib = {fib_zone}")
    print(f"\n  {risk.summary()}")
    logger.print_stats()


# ------------------------------------------------------------------
# Vòng lặp Operator chính
# ------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 65)
    print("  👁️  OPERATOR SYSTEM — DERIV BINARY OPTIONS")
    print("  Bạn đang GIÁM SÁT. Hệ thống tự vận hành hoàn toàn.")
    print("=" * 65)
    print(f"  Thị trường quét  : {', '.join(config.SCAN_SYMBOLS)}")
    print(f"  Điểm tối thiểu   : {config.MIN_SIGNAL_SCORE}/100")
    print(f"  Sóng hồi biên    : {config.WAVE_CORRECTION_MIN*100:.0f}% – {config.WAVE_CORRECTION_MAX*100:.0f}%")
    print(f"  Fibonacci tol.   : ±{config.WAVE_FIB_TOLERANCE*100:.1f}%")
    print(f"  Giới hạn lỗ ngày : {config.RISK_MAX_DAILY_LOSS_PCT*100:.0f}%")
    print(f"  Cooldown thua    : {config.RISK_MAX_CONSECUTIVE_LOSS} lần → {config.RISK_COOLDOWN_MINUTES} phút")
    print(f"  Chu kỳ quét      : {config.SCAN_INTERVAL_SECONDS}s")
    print("=" * 65)

    risk   = RiskManager()
    logger = TradeLogger()

    print(f"\n  {risk.summary()}")
    logger.print_stats()
    print(f"\n  [{datetime.now()}] Hệ thống ONLINE. Nhấn Ctrl+C để dừng.\n")

    while True:
        try:
            run_cycle(risk, logger)
        except KeyboardInterrupt:
            print(f"\n  [{datetime.now()}] Operator ngắt hệ thống.")
            print(f"  {risk.summary()}")
            logger.print_stats()
            break
        except Exception as exc:
            print(f"  [LỖI nghiêm trọng] {exc} — tự phục hồi sau 30 giây...")
            time.sleep(30)
            continue

        print(f"\n  ⏳ Nghỉ {config.SCAN_INTERVAL_SECONDS}s trước chu kỳ tiếp theo...\n")
        time.sleep(config.SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
