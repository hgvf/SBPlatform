import datetime as _dt
import os
import json
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
from typing import Dict, Any

from src.backtest.strategies.test import MySignalStrategy
from src.backtest.strategy_utils import TWStockCommInfo, ATRRiskSizer
from src.backtest.data_utils import PandasDataWithMetric, ensure_datetime_index, validate_columns, load_dataframe
from src.backtest.analyzers_utils import (
    extract_returns_overall,
    extract_time_return_df,
    extract_drawdown,
    extract_trade_stats,
    extract_sharpe,
    extract_sqn,
    extract_annual_return,
    extract_positions_value,
    compute_stats_from_period_returns,
    to_jsonable
)

def _get_bt_param_keys(strategy_cls) -> set:
    """拿到該 Strategy 所有 params 名稱（支援 tuple/dict 舊寫法）"""
    try:
        return set(strategy_cls.params._getkeys())  # backtrader 內建
    except Exception:
        p = getattr(strategy_cls, 'params', ())
        keys = []
        if isinstance(p, dict):
            keys = list(p.keys())
        else:
            for item in p:
                if isinstance(item, tuple) and item:
                    keys.append(item[0])
                else:
                    keys.append(item)
        return set(keys)
    
def run_backtest(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    datafeed = PandasDataWithMetric(dataname=df,
                                    datetime=None, # None 表示使用 index 作為 datetime
                                    open='Open',
                                    high='High',
                                    low='Low',
                                    close='Close',
                                    volume='Volume',
                                    openinterest=-1,  # 沒有 openinterest 資料
    )

    cerebro = bt.Cerebro()
    cerebro.adddata(datafeed)
    
    # Add cerebro stragegy
    m_strategy = kwargs.get("m_strategy", None) # strategy manager
    allowed = _get_bt_param_keys(m_strategy.getStrategy())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}

    cerebro.addstrategy(
        m_strategy.getStrategy(),
        **filtered
    )

    # Broker 設置
    cerebro.broker.setcash(float(kwargs.get("startcash", "1000000"))) # default: 1M dollars
    
    comm = TWStockCommInfo(buy_fee_rate=float(kwargs.get("buy_fee_rate", "0")), 
                            sell_fee_rate=float(kwargs.get("sell_fee_rate", "0")), 
                            sell_tax_rate=float(kwargs.get("sell_tax_rate", "0")))
    cerebro.broker.addcommissioninfo(comm)
    cerebro.broker.set_slippage_perc(perc=float(kwargs.get("slippage", "0")))
    cerebro.broker.set_coc(True)

    # Sizer
    if bool(kwargs.get("use_atr_sizer", "0")):
        cerebro.addsizer(ATRRiskSizer, risk_perc=0.02)

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.Returns, _name='ret')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='tret')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sr')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='ta')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, riskfreerate=0.01)
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual')

    res = cerebro.run()
    strat = res[0]

    # 分析報告
    ret = extract_returns_overall(getattr(strat.analyzers, 'ret', None))
    tret_df = extract_time_return_df(getattr(strat.analyzers, 'tret', None))
    dd = extract_drawdown(getattr(strat.analyzers, 'dd', None))
    ta = extract_trade_stats(getattr(strat.analyzers, 'ta', None))
    sharpe = extract_sharpe(getattr(strat.analyzers, 'sharpe', None))
    sqn = extract_sqn(getattr(strat.analyzers, 'sqn', None))
    annual = extract_annual_return(getattr(strat.analyzers, 'annual', None))

    series_stats = compute_stats_from_period_returns(tret_df)

    payload = dict(
        returns_overall=ret,
        timereturn=to_jsonable(tret_df),
        drawdown=dd,
        trade=ta,
        sharpe_ratio=sharpe,
        sqn=sqn,
        annual_return=annual,
        series_stats=series_stats,
        meta=dict(riskfreerate=0.01),
        final_value=cerebro.broker.getvalue() # 取得最終資產
    )
    
    return payload

def plot_equity_curve(tret_df: pd.DataFrame):
    rcParams['font.sans-serif'] = ['Microsoft JhengHei']

    if tret_df is None or tret_df.empty:
        return None
    
    fig = plt.figure(figsize=(10, 4))
    
    # out = ((1 + tret_df['period_return']).cumprod()-1)*100
    (tret_df['cum_return']*100).plot()
    plt.title('Equity Curve (Cumulative Return)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.tight_layout()
    return fig

def make_snapshot_image(report: dict) -> bytes:
    # 取資料
    ro = report.get('returns_overall') or {}
    dd = report.get('drawdown') or {}
    ta = report.get('trade') or {}
    timeret = report.get('timereturn') or {}
    import pandas as pd, numpy as np

    # 準備 time return df（若沒有就空）
    tret_df = pd.DataFrame(timeret).T if timeret else pd.DataFrame()
    if not tret_df.empty:
        tret_df.index = pd.to_datetime(tret_df.index)
        tret_df = tret_df.sort_index()
        equity = (1 + tret_df['cum_return']).rename('equity') if 'cum_return' in tret_df else (1 + tret_df['period_return']).cumprod()
        running_max = equity.cummax()
        dd_series = equity / running_max - 1
    else:
        equity = None
        dd_series = None

    # 畫布
    fig = plt.figure(figsize=(10, 12), dpi=150)
    gs = GridSpec(6, 2, figure=fig, height_ratios=[0.6, 0.6, 2.2, 2.2, 1.8, 1.6], hspace=0.9, wspace=0.4)

    # 標題
    ax0 = fig.add_subplot(gs[0, :])
    ax0.axis('off')
    ax0.text(0, 0.9, 'Backtest Report', fontsize=18, fontweight='bold')
    ax0.text(0, 0.55, 'KPIs', fontsize=12, fontweight='bold')

    def pct(x):
        import numpy as np
        return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.2f}%"

    kpi_lines = [
        f"Total Return: {pct(ro.get('total_return'))}",
        f"Annualized Return: {pct(ro.get('annual_return'))}",
        f"Sharpe: {report.get('sharpe_ratio') or (report.get('series_stats') or {}).get('sharpe_from_series')}",
        f"Max Drawdown: {pct((-abs(dd.get('max_drawdown') or 0)/100) if dd.get('max_drawdown') and dd.get('max_drawdown')>1 else dd.get('max_drawdown'))}",
        f"Win Rate: {pct(ta.get('win_rate'))}",
    ]
    ax0.text(0, 0.18, "\n".join(kpi_lines), fontsize=10, family="monospace")

    # Equity Curve
    ax1 = fig.add_subplot(gs[2, :])
    if equity is not None and not equity.empty:
        ax1.plot(equity.index, equity.values)
        ax1.set_title('Equity Curve (Cumulative Return)')
        ax1.set_xlabel('Date'); ax1.set_ylabel('Equity (1+cum_return)')
    else:
        ax1.text(0.5, 0.5, 'No TimeReturn data', ha='center', va='center')
        ax1.set_axis_off()

    # Drawdown
    ax2 = fig.add_subplot(gs[3, :])
    if dd_series is not None and not dd_series.empty:
        ax2.fill_between(dd_series.index, dd_series.values, 0, alpha=0.3, step='mid')
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date'); ax2.set_ylabel('Drawdown')
    else:
        ax2.text(0.5, 0.5, 'No Drawdown series', ha='center', va='center')
        ax2.set_axis_off()

    # Annual return（表格簡版）
    annual = report.get('annual_return') or {}
    ax3 = fig.add_subplot(gs[4, 0])
    ax3.axis('off'); ax3.set_title('Annual Returns', loc='left')
    if annual:
        rows = [[str(y), pct(v)] for y, v in sorted(annual.items())]
        table = ax3.table(cellText=rows, colLabels=['Year', 'Return'], loc='center')
        table.auto_set_font_size(False); table.set_fontsize(8)
        table.scale(1, 1.2)
    else:
        ax3.text(0.5, 0.5, '—', ha='center', va='center')

    # Trade stats（表格簡版）
    ax4 = fig.add_subplot(gs[4, 1])
    ax4.axis('off'); ax4.set_title('Trade Statistics', loc='left')
    if ta:
        vals = [
            ['Total Trades', ta.get('trades_total', '-')],
            ['Closed Trades', ta.get('trades_closed', '-')],
            ['Win Rate', pct(ta.get('win_rate'))],
            ['Profit Factor', '-' if ta.get('profit_factor') is None else f"{ta.get('profit_factor'):.2f}"],
            ['Net PnL (Total)', ta.get('pnl_net_total', '-')],
            ['Avg PnL', ta.get('pnl_net_avg', '-')],
        ]
        table = ax4.table(cellText=vals, colLabels=['Metric', 'Value'], loc='center')
        table.auto_set_font_size(False); table.set_fontsize(8)
        table.scale(1, 1.2)
    else:
        ax4.text(0.5, 0.5, '—', ha='center', va='center')

    # 生成 PNG bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def _fmt_pct(x):
    return f"{x*100:.2f}%" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "-"

def build_markdown_summary(report: dict) -> str:
    ro = report.get('returns_overall') or {}; dd = report.get('drawdown') or {}; ta = report.get('trade') or {}
    ss = report.get('series_stats') or {}; annual = report.get('annual_return') or {}
    ts = _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        "# Backtest Report",
        f"_Generated at: {ts}_",
        "",
        "## KPIs",
        f"- Total Return: {_fmt_pct(ro.get('total_return'))}",
        f"- Annualized Return: {_fmt_pct(ro.get('annual_return'))}",
        f"- Sharpe Ratio: {report.get('sharpe_ratio') or ss.get('sharpe_from_series')}",
        f"- Max Drawdown: {_fmt_pct((-abs(dd.get('max_drawdown') or 0)/100) if dd.get('max_drawdown') and dd.get('max_drawdown')>1 else dd.get('max_drawdown'))}",
        f"- Win Rate: {_fmt_pct(ta.get('win_rate'))}",
        "",
        "## Annual Return",
    ]
    if annual:
        lines += ["Year | Return", ":--:|--:"] + [f"{y} | {_fmt_pct(v)}" for y, v in sorted(annual.items())]
        lines.append("")
    lines += [
        "## Trade Statistics",
        f"- Total Trades: {ta.get('trades_total', '-')}",
        f"- Closed Trades: {ta.get('trades_closed', '-')}",
        f"- Win Rate: {_fmt_pct(ta.get('win_rate'))}",
        f"- Profit Factor: {ta.get('profit_factor') if not isinstance(ta.get('profit_factor'), float) or not np.isnan(ta.get('profit_factor')) else '-'}",
        f"- Net PnL (Total): {ta.get('pnl_net_total', '-')}",
        f"- Avg PnL: {ta.get('pnl_net_avg', '-')}",
        f"- Longest Win Streak: {ta.get('longest_win_streak', '-')}",
        f"- Longest Loss Streak: {ta.get('longest_loss_streak', '-')}",
        "",
    ]
    return "".join(lines)

def save_report_bundle(report: dict, 
                       ticker: str,
                       backtest_name: str = "",
                       out_dir: str = "reports") -> Dict[str, str]:
    """把快照 PNG、Markdown、JSON、CSV 同時寫到磁碟，回傳檔案路徑字典。"""

    os.makedirs(out_dir, exist_ok=True)
    stamp = _dt.datetime.now().strftime('%Y%m%d')
    if backtest_name == "":
        backtest_name = f"backtest_{stamp}_{ticker}"
    run_dir = os.path.join(out_dir, backtest_name)
    os.makedirs(run_dir, exist_ok=True)

    # 1) PNG
    png_bytes = make_snapshot_image(report)
    p_png = os.path.join(run_dir, 'Report.png')
    with open(p_png, 'wb') as f: f.write(png_bytes)

    # 2) Markdown
    md = build_markdown_summary(report)
    p_md = os.path.join(run_dir, 'Backtest_Report.md')
    with open(p_md, 'w', encoding='utf-8') as f: f.write(md)

    # 3) JSON
    p_json = os.path.join(run_dir, 'Backtest_Report.json')
    with open(p_json, 'w', encoding='utf-8') as f: json.dump(report, f, ensure_ascii=False, indent=2)

    # 4) CSV（time return / trade stats 視情況）
    timeret = report.get('timereturn') or {}
    if timeret:
        df_timeret = pd.DataFrame(timeret).T
        df_timeret.index.name = 'date'
        df_timeret.to_csv(os.path.join(run_dir, 'time_return.csv'))
    ta = report.get('trade') or {}
    if ta:
        pd.DataFrame(ta, index=[0]).T.to_csv(os.path.join(run_dir, 'trade_stats.csv'))

    return dict(dir=run_dir, png=p_png, md=p_md, json=p_json), run_dir
