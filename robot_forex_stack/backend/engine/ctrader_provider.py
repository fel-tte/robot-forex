"""
CTrader Live Data Provider.

Kết nối tới cTrader Open API (TCP + Protobuf) qua thư viện ctrader-open-api
(Spotware chính thức) để:
  1. Xác thực OAuth2 (Application Auth → Account Auth).
  2. Tìm symbolId từ tên symbol (vd: EURUSD).
  3. Tải 500 cây nến lịch sử (historical trendbars).
  4. Subscribe live spot events và xây nến realtime từ ticks.

Twisted reactor chạy trong daemon thread riêng; toàn bộ shared state được
bảo vệ bằng threading.Lock — FastAPI asyncio loop đọc an toàn mọi lúc.

Biến môi trường bắt buộc:
  CTRADER_CLIENT_ID      – Client ID của app đăng ký tại connect.spotware.com
  CTRADER_CLIENT_SECRET  – Client Secret của app
  CTRADER_ACCESS_TOKEN   – OAuth2 access token của tài khoản giao dịch
  CTRADER_REFRESH_TOKEN  – OAuth2 refresh token (dùng để tự gia hạn)
  CTRADER_ACCOUNT_ID     – ID tài khoản cTrader số (vd: 17093718)

Biến môi trường tuỳ chọn:
  CTRADER_SYMBOL         – Tên symbol (mặc định: EURUSD)
  CTRADER_TIMEFRAME      – Khung thời gian nến (mặc định: M5)
  CTRADER_LIVE           – "true" = live server, "false" = demo (mặc định: true)
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests as _requests

logger = logging.getLogger(__name__)

# ── Hằng số ────────────────────────────────────────────────────────────────── #

_PRICE_DIVISOR = 100_000          # Tất cả giá trong cTrader API = integer / 100000
_OAUTH_TOKEN_URL = "https://connect.spotware.com/apps/token"
_LIVE_HOST = "live.ctraderapi.com"
_DEMO_HOST = "demo.ctraderapi.com"
_PORT = 5035

# Mapping tên timeframe → (tên enum ProtoOATrendbarPeriod, giây/nến)
_TIMEFRAME_MAP: dict[str, tuple[str, int]] = {
    "M1":  ("M1",  60),
    "M5":  ("M5",  300),
    "M15": ("M15", 900),
    "M30": ("M30", 1800),
    "H1":  ("H1",  3600),
    "H4":  ("H4",  14400),
    "D1":  ("D1",  86400),
    "W1":  ("W1",  604800),
}


# ── Data classes ───────────────────────────────────────────────────────────── #

@dataclass
class OHLCVBar:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BrokerStatus:
    """Trạng thái kết nối tới cTrader."""
    provider_type: str = "MOCK"          # MOCK | CTRADER
    connected: bool = False
    app_authenticated: bool = False
    account_authenticated: bool = False
    history_loaded: bool = False
    symbol: str = ""
    symbol_id: int = 0
    timeframe: str = ""
    live: bool = False
    last_error: str = ""
    last_tick_ts: float = 0.0
    bars_loaded: int = 0
    account_id: int = 0


# ── Thread-safe storage ────────────────────────────────────────────────────── #

class _BarStorage:
    """Thread-safe deque cho completed bars + current incomplete bar."""

    def __init__(self, maxlen: int = 1000) -> None:
        self._lock = threading.Lock()
        self._bars: Deque[OHLCVBar] = deque(maxlen=maxlen)
        self._current: Optional[OHLCVBar] = None
        self._bid: float = 0.0
        self._ask: float = 0.0

    def bulk_add(self, bars: List[OHLCVBar]) -> None:
        with self._lock:
            for b in bars:
                self._bars.append(b)

    def update_tick(self, bid: float, ask: float, ts: float, candle_seconds: int) -> None:
        candle_ts = float(int(ts // candle_seconds) * candle_seconds)
        with self._lock:
            self._bid = bid
            self._ask = ask
            if self._current is None:
                self._current = OHLCVBar(
                    timestamp=candle_ts,
                    open=bid, high=bid, low=bid, close=bid, volume=0.0,
                )
            elif self._current.timestamp != candle_ts:
                # Nến hiện tại đã đóng → chuyển vào completed
                self._bars.append(self._current)
                self._current = OHLCVBar(
                    timestamp=candle_ts,
                    open=bid, high=bid, low=bid, close=bid, volume=0.0,
                )
            else:
                c = self._current
                if bid > c.high:
                    c.high = bid
                if bid < c.low:
                    c.low = bid
                c.close = bid

    def get_bars(self, limit: int) -> List[OHLCVBar]:
        with self._lock:
            result = list(self._bars)
            if self._current is not None:
                result = result + [self._current]
            return result[-limit:]

    def get_price(self) -> Tuple[float, float]:
        with self._lock:
            return self._bid, self._ask

    def size(self) -> int:
        with self._lock:
            return len(self._bars)


# ── OAuth2 token refresh (HTTP) ────────────────────────────────────────────── #

def _http_refresh_token(client_id: str, client_secret: str, refresh_token: str) -> dict:
    """Gọi Spotware Connect để đổi refresh_token → access_token mới."""
    resp = _requests.post(
        _OAUTH_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# ── CTrader Data Provider ─────────────────────────────────────────────────── #

class CTraderDataProvider:
    """
    Dữ liệu thật từ cTrader Open API — thay thế hoàn toàn MockDataProvider.

    Giao diện public giống hệt MockDataProvider:
      get_candles(), get_current_price(), get_spread_points(), advance()
    Cộng thêm:
      status   — BrokerStatus (trạng thái kết nối)
      is_ready — bool (đã tải lịch sử và đang nhận ticks)
    """

    def __init__(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        live: Optional[bool] = None,
    ) -> None:
        # ── Credentials (chỉ lấy từ env, không lưu vào DB) ──
        self._client_id = os.environ.get("CTRADER_CLIENT_ID", "").strip()
        self._client_secret = os.environ.get("CTRADER_CLIENT_SECRET", "").strip()
        self._access_token = os.environ.get("CTRADER_ACCESS_TOKEN", "").strip()
        self._refresh_token = os.environ.get("CTRADER_REFRESH_TOKEN", "").strip()
        self._account_id = int(os.environ.get("CTRADER_ACCOUNT_ID", "0"))

        # ── Cấu hình ──
        self.symbol = (
            symbol or os.environ.get("CTRADER_SYMBOL", "EURUSD")
        ).upper()
        self.timeframe = (
            timeframe or os.environ.get("CTRADER_TIMEFRAME", "M5")
        ).upper()
        _live_env = os.environ.get("CTRADER_LIVE", "true").lower()
        self._live: bool = live if live is not None else (_live_env == "true")

        tf_info = _TIMEFRAME_MAP.get(self.timeframe, ("M5", 300))
        self._period_name: str = tf_info[0]
        self._candle_seconds: int = tf_info[1]

        # ── State ──
        self._storage = _BarStorage(maxlen=1000)
        self._symbol_id: int = 0
        self._token_expires_at: float = 0.0

        self.status = BrokerStatus(
            provider_type="CTRADER",
            symbol=self.symbol,
            timeframe=self.timeframe,
            live=self._live,
            account_id=self._account_id,
        )

        # ── Sync primitives ──
        self._ready_event = threading.Event()
        self._client = None  # sẽ được gán trong _run()

        # ── Validate bắt buộc ──
        missing = [
            k for k, v in {
                "CTRADER_CLIENT_ID": self._client_id,
                "CTRADER_CLIENT_SECRET": self._client_secret,
                "CTRADER_ACCESS_TOKEN": self._access_token,
                "CTRADER_ACCOUNT_ID": str(self._account_id),
            }.items() if not v
        ]
        if missing:
            raise ValueError(
                f"Thiếu env vars bắt buộc cho cTrader: {', '.join(missing)}. "
                "Xem hướng dẫn trong robot_forex_stack/.env.example"
            )

        # ── Khởi động Twisted reactor trong daemon thread ──
        self._reactor_thread = threading.Thread(
            target=self._run_reactor, daemon=True, name="CTraderReactor"
        )
        self._reactor_thread.start()
        logger.info(
            "CTraderDataProvider khởi động: symbol=%s tf=%s live=%s account=%d",
            self.symbol, self.timeframe, self._live, self._account_id,
        )

    # ── Public API (giống MockDataProvider) ──────────────────────────────── #

    @property
    def is_ready(self) -> bool:
        return self.status.history_loaded

    @property
    def is_live(self) -> bool:
        """True nếu đang kết nối tới live server, False nếu demo."""
        return self._live

    def wait_ready(self, timeout: float = 60.0) -> bool:
        """Chờ cho tới khi lịch sử được tải xong (blocking)."""
        return self._ready_event.wait(timeout)

    def get_candles(self, limit: int = 200, timeframe: str = "M5") -> pd.DataFrame:
        """Trả về DataFrame OHLCV gồm `limit` cây nến gần nhất."""
        bars = self._storage.get_bars(limit)
        if not bars:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "datetime"]
            )
        df = pd.DataFrame([
            {
                "timestamp": b.timestamp,
                "open":      b.open,
                "high":      b.high,
                "low":       b.low,
                "close":     b.close,
                "volume":    b.volume,
            }
            for b in bars
        ])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        return df

    def get_current_price(self) -> Tuple[float, float]:
        """Trả về (bid, ask) realtime từ cTrader spot feed."""
        bid, ask = self._storage.get_price()
        if bid <= 0:
            # Chưa nhận tick nào — trả về giá nến cuối
            bars = self._storage.get_bars(1)
            if bars:
                mid = bars[-1].close
                bid = round(mid - 0.00010, 5)
                ask = round(mid + 0.00010, 5)
        return bid, ask

    def get_spread_points(self) -> float:
        """Trả về spread hiện tại tính bằng pips (0.0001 = 1 pip)."""
        bid, ask = self._storage.get_price()
        if bid > 0 and ask > bid:
            return round((ask - bid) / 0.0001, 1)
        return 1.5  # fallback

    def advance(self) -> None:
        """No-op — dữ liệu tới qua push events, không cần gọi thủ công."""
        pass

    # ── Static indicator helpers (giống hệt MockDataProvider) ────────────── #

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        high = df["high"]
        low  = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        val = tr.rolling(period).mean().iloc[-1]
        return float(val) if not math.isnan(val) else 0.0

    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
        return df[column].ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_fractals(
        df: pd.DataFrame, period: int = 2
    ) -> Tuple[pd.Series, pd.Series]:
        n = len(df)
        frac_highs = pd.Series(np.nan, index=df.index)
        frac_lows  = pd.Series(np.nan, index=df.index)
        for i in range(period, n - period):
            window_h = df["high"].iloc[i - period: i + period + 1]
            if df["high"].iloc[i] == window_h.max():
                frac_highs.iloc[i] = df["high"].iloc[i]
            window_l = df["low"].iloc[i - period: i + period + 1]
            if df["low"].iloc[i] == window_l.min():
                frac_lows.iloc[i] = df["low"].iloc[i]
        return frac_highs, frac_lows

    @staticmethod
    def calculate_sma(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
        return df[column].rolling(period).mean()

    # ── Twisted reactor (background thread) ──────────────────────────────── #

    def _run_reactor(self) -> None:
        """Khởi động Twisted reactor trong daemon thread."""
        try:
            from twisted.internet import reactor
            from ctrader_open_api import Client, TcpProtocol

            host = _LIVE_HOST if self._live else _DEMO_HOST
            self._client = Client(host, _PORT, TcpProtocol)
            self._client.setConnectedCallback(self._on_connected)
            self._client.setDisconnectedCallback(self._on_disconnected)
            self._client.setMessageReceivedCallback(self._on_message)
            self._client.startService()
            logger.info("CTrader: đang kết nối tới %s:%d …", host, _PORT)
            reactor.run(installSignalHandlers=False)
        except ImportError:
            err = (
                "ctrader-open-api chưa được cài đặt. "
                "Chạy: pip install ctrader-open-api"
            )
            logger.error(err)
            self.status.last_error = err
        except Exception as exc:
            logger.error("CTrader reactor lỗi: %s", exc, exc_info=True)
            self.status.last_error = str(exc)

    # ── Callback: kết nối / ngắt kết nối ─────────────────────────────────── #

    def _on_connected(self, client, _) -> None:
        logger.info("CTrader: TCP đã kết nối. Gửi app auth…")
        self.status.connected = True
        self.status.last_error = ""
        self._maybe_refresh_token()
        self._send_app_auth(client)

    def _on_disconnected(self, client, reason) -> None:
        logger.warning("CTrader: mất kết nối — %s", reason)
        self.status.connected = False
        self.status.app_authenticated = False
        self.status.account_authenticated = False

    # ── Callback: nhận message ────────────────────────────────────────────── #

    def _on_message(self, client, message) -> None:
        try:
            from ctrader_open_api import Protobuf
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOAApplicationAuthRes,
                ProtoOAAccountAuthRes,
                ProtoOAGetTrendbarsRes,
                ProtoOASymbolsListRes,
                ProtoOASubscribeSpotsRes,
                ProtoOASpotEvent,
                ProtoOARefreshTokenRes,
            )
            from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import (
                ProtoErrorRes,
            )

            pt = message.payloadType

            if   pt == ProtoOAApplicationAuthRes().payloadType:
                self._handle_app_auth(client, Protobuf.extract(message, ProtoOAApplicationAuthRes))
            elif pt == ProtoOAAccountAuthRes().payloadType:
                self._handle_account_auth(client, Protobuf.extract(message, ProtoOAAccountAuthRes))
            elif pt == ProtoOASymbolsListRes().payloadType:
                self._handle_symbols(client, Protobuf.extract(message, ProtoOASymbolsListRes))
            elif pt == ProtoOAGetTrendbarsRes().payloadType:
                self._handle_trendbars(client, Protobuf.extract(message, ProtoOAGetTrendbarsRes))
            elif pt == ProtoOASubscribeSpotsRes().payloadType:
                logger.info("CTrader: spot subscription xác nhận cho symbolId=%d", self._symbol_id)
            elif pt == ProtoOASpotEvent().payloadType:
                self._handle_spot(Protobuf.extract(message, ProtoOASpotEvent))
            elif pt == ProtoOARefreshTokenRes().payloadType:
                self._handle_token_refresh(Protobuf.extract(message, ProtoOARefreshTokenRes))
            elif pt == ProtoErrorRes().payloadType:
                err = Protobuf.extract(message, ProtoErrorRes)
                msg = f"errorCode={err.errorCode} | {getattr(err, 'description', '')}"
                logger.error("CTrader API error: %s", msg)
                self.status.last_error = msg
        except Exception as exc:
            logger.error("CTrader _on_message lỗi: %s", exc, exc_info=True)

    # ── Auth flow ─────────────────────────────────────────────────────────── #

    def _send_app_auth(self, client) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAApplicationAuthReq
        req = ProtoOAApplicationAuthReq()
        req.clientId = self._client_id
        req.clientSecret = self._client_secret
        client.send(req)

    def _handle_app_auth(self, client, _res) -> None:
        logger.info("CTrader: ứng dụng đã xác thực.")
        self.status.app_authenticated = True
        self._send_account_auth(client)

    def _send_account_auth(self, client) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAAccountAuthReq
        req = ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = self._account_id
        req.accessToken = self._access_token
        client.send(req)

    def _handle_account_auth(self, client, _res) -> None:
        logger.info("CTrader: tài khoản %d đã xác thực.", self._account_id)
        self.status.account_authenticated = True
        self._request_symbols(client)

    # ── Symbol list ───────────────────────────────────────────────────────── #

    def _request_symbols(self, client) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOASymbolsListReq
        req = ProtoOASymbolsListReq()
        req.ctidTraderAccountId = self._account_id
        req.includeArchivedSymbols = False
        client.send(req)

    def _handle_symbols(self, client, res) -> None:
        target = self.symbol.upper()
        for sym in res.symbol:
            name = getattr(sym, "symbolName", "").upper()
            if name == target:
                self._symbol_id = sym.symbolId
                self.status.symbol_id = sym.symbolId
                logger.info("CTrader: tìm thấy %s → symbolId=%d", self.symbol, self._symbol_id)
                break
        if self._symbol_id == 0:
            err = f"Symbol '{self.symbol}' không tìm thấy trong danh sách."
            logger.error("CTrader: %s", err)
            self.status.last_error = err
            return
        self._request_trendbars(client)

    # ── Historical trendbars ──────────────────────────────────────────────── #

    def _request_trendbars(self, client) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAGetTrendbarsReq
        from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATrendbarPeriod
        req = ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = self._account_id
        req.symbolId = self._symbol_id
        req.period = ProtoOATrendbarPeriod.Value(self._period_name)
        req.count = 500
        client.send(req)
        logger.info(
            "CTrader: yêu cầu 500 nến lịch sử %s cho symbolId=%d…",
            self.timeframe, self._symbol_id,
        )

    def _handle_trendbars(self, client, res) -> None:
        bars: List[OHLCVBar] = []
        for tb in res.trendbar:
            low    = tb.low / _PRICE_DIVISOR
            open_  = (tb.low + tb.deltaOpen)  / _PRICE_DIVISOR
            close  = (tb.low + tb.deltaClose) / _PRICE_DIVISOR
            high   = (tb.low + tb.deltaHigh)  / _PRICE_DIVISOR
            volume = float(tb.volume) / 100.0
            ts     = float(tb.utcTimestampInMinutes) * 60.0
            bars.append(OHLCVBar(
                timestamp=ts,
                open=round(open_, 5), high=round(high, 5),
                low=round(low, 5),   close=round(close, 5),
                volume=round(volume, 2),
            ))
        # Sắp xếp tăng dần theo thời gian
        bars.sort(key=lambda b: b.timestamp)
        self._storage.bulk_add(bars)

        self.status.history_loaded = True
        self.status.bars_loaded = len(bars)
        logger.info("CTrader: đã tải %d nến lịch sử %s.", len(bars), self.timeframe)
        self._ready_event.set()
        self._subscribe_spots(client)

    # ── Live spot subscription ─────────────────────────────────────────────── #

    def _subscribe_spots(self, client) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOASubscribeSpotsReq
        req = ProtoOASubscribeSpotsReq()
        req.ctidTraderAccountId = self._account_id
        req.symbolId.append(self._symbol_id)
        client.send(req)
        logger.info("CTrader: subscribe live spots cho symbolId=%d…", self._symbol_id)

    def _handle_spot(self, event) -> None:
        if event.symbolId != self._symbol_id:
            return
        # Giá trong event là integer; nếu trường không có (ask có thể vắng) → 0
        bid = float(event.bid) / _PRICE_DIVISOR if event.bid else 0.0
        ask = float(event.ask) / _PRICE_DIVISOR if event.ask else 0.0
        if bid <= 0:
            return
        if ask <= bid:
            # ask không có trong event này — ước tính từ spread điển hình
            ask = round(bid + 0.00015, 5)
        ts = time.time()
        self._storage.update_tick(bid, ask, ts, self._candle_seconds)
        self.status.last_tick_ts = ts

    # ── Token refresh ──────────────────────────────────────────────────────── #

    def _maybe_refresh_token(self) -> None:
        """Làm mới access_token nếu sắp hết hạn (còn < 5 phút)."""
        if not self._refresh_token:
            return
        if self._token_expires_at > time.time() + 300:
            return  # Còn hợp lệ
        try:
            data = _http_refresh_token(
                self._client_id, self._client_secret, self._refresh_token
            )
            self._access_token = data["access_token"]
            self._refresh_token = data.get("refresh_token", self._refresh_token)
            expires_in = int(data.get("expires_in", 3600))
            self._token_expires_at = time.time() + expires_in
            logger.info("CTrader: access token đã làm mới (hết hạn sau %ds).", expires_in)
        except Exception as exc:
            logger.warning("CTrader: làm mới token thất bại — %s", exc)

    def _handle_token_refresh(self, res) -> None:
        self._access_token = res.accessToken
        self._token_expires_at = time.time() + 3600
        logger.info("CTrader: access token đã làm mới qua API.")
