"""
robot.py
========
Robot trading tự động cho Deriv Binary Options.

Luồng hoạt động:
  Mỗi SCAN_INTERVAL_SECONDS giây:
    1. [Strategy]    Lấy dữ liệu nến từ Deriv API → tính RSI + Momentum → lưu tín hiệu vào Redis
    2. [Trade]       Đọc tín hiệu từ Redis → nếu có BUY/SELL → đặt lệnh CALL/PUT trên Deriv
    3. Lặp lại

Khởi động:
    python robot.py
"""

import schedule
import time
from datetime import datetime

import config
from strategy import scan_market
from deriv_trade import entry_deriv


def run_robot() -> None:
    """Một vòng lặp của robot: phân tích thị trường rồi đặt lệnh."""
    print(f"\n{'='*60}")
    print(f"[{datetime.now()}] 🤖 Robot đang chạy...")
    print(f"{'='*60}")

    # Bước 1: Phân tích thị trường và ghi tín hiệu vào Redis
    scan_market()

    # Bước 2: Đọc tín hiệu từ Redis và đặt lệnh
    entry_deriv()


def main() -> None:
    print("=" * 60)
    print("  Deriv Binary Options Robot")
    print(f"  Symbol   : {config.SYMBOL}")
    print(f"  Amount   : {config.TRADE_AMOUNT} {config.TRADE_CURRENCY}")
    print(f"  Duration : {config.CONTRACT_DURATION}{config.CONTRACT_DURATION_UNIT}")
    print(f"  Interval : {config.SCAN_INTERVAL_SECONDS}s")
    print("=" * 60)

    # Chạy ngay lần đầu
    run_robot()

    # Lên lịch chạy định kỳ
    schedule.every(config.SCAN_INTERVAL_SECONDS).seconds.do(run_robot)

    print(f"\n[{datetime.now()}] Bộ lịch đã khởi động. Nhấn Ctrl+C để dừng.\n")
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
