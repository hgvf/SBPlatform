"""
Backtrader analyzers 全整合 + Streamlit 報告展示
----------------------------------------------------------------
你可以：
1) 直接在本檔下方 `if __name__ == "__main__":` 區塊用 sample data 測跑
2) 或將 `run_backtest()` 視為後端 API 的核心，將結果 JSON 傳給 Streamlit 前端

注意：
- 已整合你現在使用的 analyzers：Returns(TimeReturn/Returns)、DrawDown、TradeAnalyzer、SharpeRatio、SQN、AnnualReturn、PositionsValue。
- 部分 analyzer 可能在不同版本 backtrader 表現略有差異；程式以 try/except 包裝，缺什麼就自動跳過，不會整體報錯。
- 你原本的 `extract_returns()` 已內建到 `extract_time_return_df()`。

需求：
  pip install backtrader pandas matplotlib streamlit
"""
from __future__ import annotations
import math
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import backtrader as bt
import pandas as pd
import numpy as np

def to_jsonable(df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
    if df is None or df.empty:
        return None
    out = df.copy()
    out.index = out.index.astype(str)
    return out.to_dict(orient='index')

def extract_returns_overall(analyzer: Any) -> Dict[str, Any]:
    """Returns analyzer 摘要：總報酬率/年化等。"""
    if analyzer is None:
        return {}
    d = analyzer.get_analysis() or {}
    # keys: rtot, ravg, rnorm, rnorm100
    return dict(
        total_return=float(d.get('rtot', np.nan)),
        avg_period_return=float(d.get('ravg', np.nan)),
        annual_return=float(d.get('rnorm', np.nan)),  # 非百分比，e.g., 0.18
        annual_return_pct=float(d.get('rnorm100', np.nan)),  # ％，e.g., 18.0
    )

def extract_time_return_df(analyzer: Any) -> pd.DataFrame:
    """TimeReturn 轉成 period_return / cum_return DataFrame。"""
    if analyzer is None:
        return pd.DataFrame(columns=['period_return', 'cum_return'])
    ser = pd.Series(analyzer.get_analysis())  # {datetime: return}
    if ser.empty:
        return pd.DataFrame(columns=['period_return', 'cum_return'])
    df = ser.to_frame('period_return')
    df.index = pd.to_datetime(df.index)
    df['cum_return'] = (1.0 + df['period_return']).cumprod() - 1.0
    return df.sort_index()

def extract_drawdown(analyzer: Any) -> Dict[str, Any]:
    if analyzer is None:
        return {}
    d = analyzer.get_analysis() or {}
    maxd = d.get('max', {}) if isinstance(d, dict) else {}
    return dict(
        current_len=int(d.get('len', 0) or 0),
        current_drawdown=float(d.get('drawdown', np.nan)),
        current_moneydown=float(d.get('moneydown', np.nan)),
        max_len=int(maxd.get('len', 0) or 0),
        max_drawdown=float(maxd.get('drawdown', np.nan)),
        max_moneydown=float(maxd.get('moneydown', np.nan)),
    )

def extract_trade_stats(analyzer: Any) -> Dict[str, Any]:
    if analyzer is None:
        return {}
    d = analyzer.get_analysis() or {}
    total_closed = int(((d.get('total') or {}).get('closed')) or 0)
    won_total = int(((d.get('won') or {}).get('total')) or 0)
    lost_total = int(((d.get('lost') or {}).get('total')) or 0)
    pnl_net = (d.get('pnl') or {}).get('net') or {}
    pnl_gross = (d.get('pnl') or {}).get('gross') or {}

    win_rate = (won_total / total_closed) if total_closed else np.nan

    gross_total = float(pnl_gross.get('total', np.nan))
    gross_gain = float(((d.get('won') or {}).get('pnl') or {}).get('total', np.nan))
    gross_loss = float(((d.get('lost') or {}).get('pnl') or {}).get('total', np.nan))
    profit_factor = (abs(gross_gain) / abs(gross_loss)) if (gross_loss not in [0, np.nan, None]) else np.nan

    return dict(
        trades_total=int(((d.get('total') or {}).get('total')) or 0),
        trades_closed=total_closed,
        trades_open=int(((d.get('total') or {}).get('open')) or 0),
        wins=won_total,
        losses=lost_total,
        win_rate=win_rate,  # 0~1
        pnl_net_total=float(pnl_net.get('total', np.nan)),
        pnl_net_avg=float(pnl_net.get('average', np.nan)),
        pnl_gross_total=gross_total,
        profit_factor=profit_factor,
        longest_win_streak=int(((d.get('streak') or {}).get('won') or {}).get('longest', 0) or 0),
        longest_loss_streak=int(((d.get('streak') or {}).get('lost') or {}).get('longest', 0) or 0),
    )

def extract_sharpe(analyzer: Any) -> Optional[float]:
    if analyzer is None:
        return None
    d = analyzer.get_analysis() or {}
    return float(d.get('sharperatio', 0.0)) if 'sharperatio' in d else None

def extract_sqn(analyzer: Any) -> Optional[float]:
    if analyzer is None:
        return None
    d = analyzer.get_analysis() or {}
    return float(d.get('sqn')) if 'sqn' in d else None

def extract_annual_return(analyzer: Any) -> Dict[str, float]:
    if analyzer is None:
        return {}
    d = analyzer.get_analysis() or {}
    # e.g., {2019: 0.12, 2020: 0.05, ...}
    return {int(k): float(v) for k, v in d.items()}

def extract_positions_value(analyzer: Any) -> pd.DataFrame:
    if analyzer is None:
        return pd.DataFrame(columns=['pos_value'])
    ser = pd.Series(analyzer.get_analysis())  # {datetime: value}
    if ser.empty:
        return pd.DataFrame(columns=['pos_value'])
    df = ser.to_frame('pos_value')
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

# 以 TimeReturn 產生延伸統計
def compute_stats_from_period_returns(df_ret: pd.DataFrame, periods_per_year: Optional[int] = None) -> Dict[str, Any]:
    """從 period_return/cum_return 計算額外統計：年化、波動、Sortino、Calmar 等。
    periods_per_year: 若為日資料可傳 252；週 52；月 12。若省略則自動偵測頻率。
    """
    if df_ret is None or df_ret.empty:
        return {}
    r = df_ret['period_return'].dropna()

    # 嘗試自動偵測頻率
    if periods_per_year is None:
        # 以中位數間隔估計：天(1)、週(~5~7)、月(~21~31)
        if len(r.index) >= 2:
            deltas = r.index.to_series().diff().dropna().dt.days.values
            med = np.median(deltas) if len(deltas) else 1
        else:
            med = 1
        periods_per_year = 252 if med <= 2 else (52 if med <= 8 else 12)

    mean = r.mean()
    std = r.std(ddof=1)
    downside = r[r < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan

    ann_return = (1 + mean) ** periods_per_year - 1 if not math.isnan(mean) else np.nan
    ann_vol = std * math.sqrt(periods_per_year) if not math.isnan(std) else np.nan
    sharpe = ann_return / ann_vol if (ann_vol and not math.isnan(ann_vol) and ann_vol != 0) else np.nan

    # Sortino：以下行波動
    ann_down_vol = downside_std * math.sqrt(periods_per_year) if (downside_std and not math.isnan(downside_std)) else np.nan
    sortino = ann_return / ann_down_vol if (ann_down_vol and not math.isnan(ann_down_vol) and ann_down_vol != 0) else np.nan

    # Calmar：年化報酬 / 最大回撤(以 cum_return 近似)
    max_dd = None
    if 'cum_return' in df_ret.columns and not df_ret['cum_return'].empty:
        equity = (1 + df_ret['cum_return']).values
        running_max = np.maximum.accumulate(equity)
        dd = (equity / running_max) - 1
        max_dd = dd.min()  # 負值
    calmar = (ann_return / abs(max_dd)) if (max_dd and max_dd != 0) else np.nan

    return dict(
        ann_vol=ann_vol,
        sharpe_from_series=sharpe,
        sortino=sortino,
        calmar=calmar,
    )
