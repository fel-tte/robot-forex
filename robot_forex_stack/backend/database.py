"""
Database — SQLite + SQLAlchemy for persisting settings and trade history.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(os.path.dirname(__file__), "data", "robot_forex.db"),
)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class SettingsModel(Base):
    __tablename__ = "settings"
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(128), unique=True, nullable=False)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TradeModel(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, index=True)
    trade_id = Column(String(32), unique=True, nullable=False)
    symbol = Column(String(16), nullable=False)
    direction = Column(String(8), nullable=False)
    lot_size = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    sl = Column(Float, nullable=False)
    tp = Column(Float, nullable=False)
    entry_mode = Column(String(32), nullable=False)
    open_time = Column(Float, nullable=False)
    close_time = Column(Float, nullable=True)
    close_price = Column(Float, nullable=True)
    pnl = Column(Float, default=0.0)
    status = Column(String(16), default="OPEN")
    remaining_lots = Column(Float, default=0.0)
    be_moved = Column(Boolean, default=False)
    grid_level = Column(Integer, default=0)
    comment = Column(String(256), default="")
    meta_json = Column(Text, default="{}")


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created at %s", DB_PATH)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Settings helpers ───────────────────────────────────────────────────── #

def load_settings(db: Session) -> Optional[Dict[str, Any]]:
    row = db.query(SettingsModel).filter_by(key="robot_settings").first()
    if row:
        return json.loads(row.value)
    return None


def save_settings(db: Session, settings_dict: Dict[str, Any]) -> None:
    row = db.query(SettingsModel).filter_by(key="robot_settings").first()
    if row:
        row.value = json.dumps(settings_dict)
        row.updated_at = datetime.utcnow()
    else:
        row = SettingsModel(key="robot_settings", value=json.dumps(settings_dict))
        db.add(row)
    db.commit()


# ── Trade helpers ──────────────────────────────────────────────────────── #

def save_trade(db: Session, trade_dict: Dict[str, Any]) -> None:
    existing = db.query(TradeModel).filter_by(trade_id=trade_dict["trade_id"]).first()
    if existing:
        for k, v in trade_dict.items():
            if hasattr(existing, k):
                setattr(existing, k, v)
    else:
        row = TradeModel(**{k: v for k, v in trade_dict.items() if k != "meta"})
        row.meta_json = json.dumps(trade_dict.get("meta", {}))
        db.add(row)
    db.commit()


def get_all_trades(db: Session, page: int = 1, page_size: int = 50) -> List[Dict[str, Any]]:
    offset = (page - 1) * page_size
    rows = (
        db.query(TradeModel)
        .order_by(TradeModel.open_time.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )
    return [_trade_to_dict(r) for r in rows]


def get_trade_count(db: Session) -> int:
    return db.query(TradeModel).count()


def _trade_to_dict(row: TradeModel) -> Dict[str, Any]:
    return {
        "trade_id": row.trade_id,
        "symbol": row.symbol,
        "direction": row.direction,
        "lot_size": row.lot_size,
        "entry_price": row.entry_price,
        "sl": row.sl,
        "tp": row.tp,
        "entry_mode": row.entry_mode,
        "open_time": row.open_time,
        "close_time": row.close_time,
        "close_price": row.close_price,
        "pnl": row.pnl,
        "status": row.status,
        "remaining_lots": row.remaining_lots,
        "be_moved": row.be_moved,
        "grid_level": row.grid_level,
        "comment": row.comment,
    }
