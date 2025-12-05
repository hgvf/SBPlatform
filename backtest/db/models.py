# db/models.py
from sqlalchemy import (
    Column, Integer, String, Text, Date, Float, ForeignKey, TIMESTAMP
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Strategy(Base):
    __tablename__ = "tbl_strategy"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(50), unique=True, nullable=False)
    description = Column(Text)
    strategy_func = Column(String(100), nullable=False)
    param_path = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship
    backtests = relationship("BacktestHistory", back_populates="strategy", cascade="all, delete")

    def __repr__(self):
        return f"<Strategy(name={self.strategy_name}, func={self.strategy_func})>"


class BacktestHistory(Base):
    __tablename__ = "tbl_backtest_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(Integer, ForeignKey("tbl_strategy.id", ondelete="CASCADE"), nullable=False)
    backtest_name = Column(String(50), nullable=False)
    ticker = Column(String(20), nullable=False)
    test_date = Column(Date, nullable=False)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    market = Column(String(5), nullable=False)
    total_return = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    trades_count = Column(Integer)
    report_path = Column(Text)
    custom_params_path = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    # Relationship
    strategy = relationship("Strategy", back_populates="backtests")

    def __repr__(self):
        return f"<BacktestHistory(ticker={self.ticker}, date={self.test_date}, strategy_id={self.strategy_id})>"
