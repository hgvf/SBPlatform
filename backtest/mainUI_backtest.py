import os
import numpy as np
import datetime
import pandas as pd
import json
import time
import streamlit as st

from dotenv import load_dotenv
from influxdb_client import InfluxDBClient
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select

from src.backtest.backtest_utils import run_backtest, plot_equity_curve, save_report_bundle
from src.backtest.data_utils import load_dataframe
from src.backtest.strategy_manager import strategyManager
from db.influxDB_client import influxdb_client
from db.postgreSQL_session import db_session
from db.models import Strategy, BacktestHistory

def validate_inputs():
    errors = []
    if not ticker:
        errors.append("請輸入股票代碼")
    if not (startt and endt and startt < endt):
        errors.append("開始日期必須早於結束日期")
    if not selected_strategy or selected_strategy == '---':
        errors.append("請選擇策略")
    if not backtest_name:
        errors.append("請輸入本次回測命名")
    
    # For 技術面指標
    if baseline_metric_error != 0:
        errors.append("請充分選擇技術面指標的進場/出場策略")
    
    return errors

# 讀取環境變數
print("Loading .env file...")
load_dotenv(".env", override=True)

# create influxDB client
print("Connecting to InfluxDB...")
influx_client = influxdb_client()

# get postgreSQL session
print("Connecting to PostgreSQL...")
sess_db = db_session()
db = next(sess_db)

try:
    baseline_metric_error = 0

    # UI title
    st.set_page_config(page_title="Backtrader 回測系統", layout="wide")
    st.title("TBD 回測系統")

    tab_baseline_metric, tab_backtest, tab_backtestHis, tab_trade, tab_screener = st.tabs([
        "基本技術面指標回測 - 參數設定",
        "回測結果報告",
        "查看歷史回測結果",
        "自動化下單",
        "找小鬼股"
    ])

    # UI sidebar
    with st.sidebar:
        st.header("資料來源")
        src = st.radio("選擇來源", ["系統資料庫", "自行匯入 csv"], index=0)

        up = None
        if src == '自行匯入 csv':
            up = st.file_uploader("上傳含有 OHLCV 之 CSV 檔案。", type=["csv"])

        st.markdown("---")
        st.header("個股代碼")
        market = st.radio("股票市場", ["台股", "美股"], index=0)
        ticker = st.text_input(
            "代碼 ❗",
            value="",
            placeholder="請輸入股票代碼",
            help="台股: 數字; 美股: 英文",
            max_chars=30,
            label_visibility="visible",   # "hidden" / "collapsed"
        )

        st.markdown("---")
        st.header("回測時間區間")
        startt = st.date_input(
            "開始日期 ❗",
            value=datetime.date(2025, 1, 1),
            min_value=datetime.date(1990, 1, 1),
            max_value=datetime.date(2030, 12, 31)
        )

        curdate = datetime.date.today()
        endt = st.date_input(
            "結束日期 ❗",
            value=curdate - datetime.timedelta(days=1),
            min_value=datetime.date(1990, 1, 1),
            max_value=curdate - datetime.timedelta(days=1)
        )

        st.markdown("---")
        st.header("策略選擇")
        
        try:
            strategy = db.scalars(select(Strategy.strategy_name)).all()
        except:
            strategy = ["---"]

        selected_strategy = st.selectbox("選擇回測策略 ❗", strategy)

        st.header("風控參數")
        atr_period = st.number_input("ATR 天數", min_value=1, value=14, help="ATR（Average True Range)，衡量波動度，數值越高代表價格波動越大。")
        stop_loss_pct = st.number_input("固定停損百分地 (單位: %)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, help="跌落進場價幾%強制平倉止損。") / 100.0
        atr_stop_mult = st.number_input("ATR * 倍數停損", min_value=0.0, value=2.0, step=0.1, help="跌落進場價 (倍數 * ATR) 塊強制平倉止損。")
        time_stop_bars = st.number_input("時間停損 (單位: 天)", min_value=0, value=0, help="持有股票幾天後強制平倉止損。")

        st.markdown("---")
        st.header("資金/交易成本/滑價")
        dollars = st.radio("幣別", ["NTD", "USD"], index=0)
        startcash = st.number_input("起始現金", min_value=0.0, value=1_000_000.0, step=1000.0)
        buy_fee = st.number_input("買進手續費 (單位: 元)", min_value=0.0, value=0.0015, step=0.0001)
        sell_fee = st.number_input("賣出手續費 (單位: 元)", min_value=0.0, value=0.0015, step=0.0001)
        sell_tax = st.number_input("證交稅 (單位: %)", min_value=0.0, value=0.0, step=0.01, help="台股: 0.3%; 美股: 0.0%") / 100.0
        slippage = st.number_input("滑價 (單位: %)", min_value=0.0, value=0.0, step=0.01, help="套用滑價，避免成交價不如預期。") / 100.0

        st.markdown("---")
        st.header("Sizer（可選）")
        use_atr_sizer = st.checkbox("ATR Risk (2%)", help="(本金 * 2%) / (ATR * 倍數)")

        st.markdown("---")
        st.header("其他")
        backtest_name = st.text_input(
            "本次回測命名 ❗",
            placeholder="ex. 鴻圖大展",
            help="請輸入純文字，最多50個字元",
            max_chars=50,
            label_visibility="visible",   # "hidden" / "collapsed"
        )

        memo = st.text_input(
            "回測備註",
            placeholder="Write anything...",
            help="其他你覺得需要記錄的，最多100字",
            max_chars=100,
            label_visibility="visible",
        )

        st.markdown("---")
        run_disabled = len(validate_inputs()) > 0
        run_btn = st.button("🚀 開始回測", disabled=run_disabled, help="請先填完必要欄位")
        save_btn = st.button("保存回測結果", disabled=not run_btn, help="回測完成後，才能保存結果")

    # 技術面指標回測 - 參數設定頁面
    with tab_baseline_metric:
        # 進場策略定義
        ENTRY_STRATEGY_DEFINITIONS = {
            "均線類指標": {
                "ma_golden_cross": {
                    "name": "均線黃金交叉",
                    "description": "快線向上突破慢線",
                    "function": "signal_ma_golden_cross",
                    "params": {
                        "fast_period": {"label": "快線週期", "type": "number", "default": 5, "min": 1, "max": 50},
                        "slow_period": {"label": "慢線週期", "type": "number", "default": 20, "min": 1, "max": 200}
                    }
                },
                "ma_bullish": {
                    "name": "均線多頭排列",
                    "description": "快線在慢線之上",
                    "function": "signal_ma_bullish",
                    "params": {
                        "fast_period": {"label": "快線週期", "type": "number", "default": 5, "min": 1, "max": 50},
                        "slow_period": {"label": "慢線週期", "type": "number", "default": 20, "min": 1, "max": 200}
                    }
                },
                "price_above_ma": {
                    "name": "價格突破均線",
                    "description": "價格向上突破均線",
                    "function": "signal_price_above_ma",
                    "params": {
                        "ma_period": {"label": "均線週期", "type": "number", "default": 20, "min": 1, "max": 200},
                        "ma_type": {"label": "均線類型", "type": "select", "default": "SMA", "options": ["SMA", "EMA"]}
                    }
                },
                "ma_slope_up": {
                    "name": "均線向上傾斜",
                    "description": "均線持續上升",
                    "function": "signal_ma_slope_up",
                    "params": {
                        "ma_period": {"label": "均線週期", "type": "number", "default": 20, "min": 1, "max": 200},
                        "ma_type": {"label": "均線類型", "type": "select", "default": "SMA", "options": ["SMA", "EMA"]},
                        "lookback": {"label": "回看期間", "type": "number", "default": 3, "min": 1, "max": 10}
                    }
                }
            },
            "動量指標 - RSI": {
                "rsi_oversold_cross": {
                    "name": "RSI超賣後突破",
                    "description": "RSI從超賣區向上突破",
                    "function": "signal_rsi_oversold_cross",
                    "params": {
                        "rsi_period": {"label": "RSI週期", "type": "number", "default": 14, "min": 2, "max": 50},
                        "threshold": {"label": "超賣閾值", "type": "number", "default": 30, "min": 10, "max": 50}
                    }
                },
                "rsi_bullish": {
                    "name": "RSI多頭區間",
                    "description": "RSI處於健康多頭區",
                    "function": "signal_rsi_bullish",
                    "params": {
                        "rsi_period": {"label": "RSI週期", "type": "number", "default": 14, "min": 2, "max": 50},
                        "lower": {"label": "下限", "type": "number", "default": 40, "min": 20, "max": 60},
                        "upper": {"label": "上限", "type": "number", "default": 70, "min": 60, "max": 90}
                    }
                }
            },
            "動量指標 - MACD": {
                "macd_golden_cross": {
                    "name": "MACD黃金交叉",
                    "description": "MACD線向上突破信號線",
                    "function": "signal_macd_golden_cross",
                    "params": {}
                },
                "macd_bullish": {
                    "name": "MACD多頭",
                    "description": "MACD線在信號線之上",
                    "function": "signal_macd_bullish",
                    "params": {}
                },
                "macd_hist_positive": {
                    "name": "MACD柱狀圖轉正",
                    "description": "柱狀圖由負轉正",
                    "function": "signal_macd_hist_positive",
                    "params": {}
                }
            },
            "動量指標 - KD": {
                "kd_golden_cross": {
                    "name": "KD黃金交叉",
                    "description": "K線向上突破D線",
                    "function": "signal_kd_golden_cross",
                    "params": {}
                },
                "kd_oversold_cross": {
                    "name": "KD超賣區黃金交叉",
                    "description": "KD在超賣區向上交叉",
                    "function": "signal_kd_oversold_cross",
                    "params": {
                        "threshold": {"label": "超賣閾值", "type": "number", "default": 20, "min": 10, "max": 30}
                    }
                },
                "kd_bullish": {
                    "name": "KD多頭",
                    "description": "K線在D線之上",
                    "function": "signal_kd_bullish",
                    "params": {}
                }
            },
            "動量指標 - 其他": {
                "cci_oversold_cross": {
                    "name": "CCI超賣後突破",
                    "description": "CCI從超賣區向上突破",
                    "function": "signal_cci_oversold_cross",
                    "params": {
                        "threshold": {"label": "超賣閾值", "type": "number", "default": -100, "min": -200, "max": -50}
                    }
                },
                "cci_bullish": {
                    "name": "CCI多頭區間",
                    "description": "CCI大於0",
                    "function": "signal_cci_bullish",
                    "params": {}
                },
                "roc_positive": {
                    "name": "ROC轉正",
                    "description": "變動率指標轉正",
                    "function": "signal_roc_positive",
                    "params": {}
                },
                "stochrsi_golden_cross": {
                    "name": "StochRSI黃金交叉",
                    "description": "隨機RSI的K線突破D線",
                    "function": "signal_stochrsi_golden_cross",
                    "params": {}
                },
                "stochrsi_oversold_cross": {
                    "name": "StochRSI超賣後突破",
                    "description": "StochRSI從超賣區向上突破",
                    "function": "signal_stochrsi_oversold_cross",
                    "params": {
                        "threshold": {"label": "超賣閾值", "type": "number", "default": 0.2, "min": 0.1, "max": 0.4, "step": 0.1}
                    }
                },
                "willr_oversold_cross": {
                    "name": "Williams %R超賣後突破",
                    "description": "威廉指標從超賣區向上突破",
                    "function": "signal_willr_oversold_cross",
                    "params": {
                        "threshold": {"label": "超賣閾值", "type": "number", "default": -80, "min": -100, "max": -50}
                    }
                }
            },
            "布林通道指標": {
                "bb_lower_bounce": {
                    "name": "布林下軌反彈",
                    "description": "價格從下軌反彈",
                    "function": "signal_bb_lower_bounce",
                    "params": {}
                },
                "bb_squeeze_break": {
                    "name": "布林收縮突破",
                    "description": "通道收縮後突破中軌",
                    "function": "signal_bb_squeeze_break",
                    "params": {}
                },
                "bb_percent_b_bullish": {
                    "name": "布林%B多頭",
                    "description": "%B指標多頭",
                    "function": "signal_bb_percent_b_bullish",
                    "params": {
                        "threshold": {"label": "閾值", "type": "number", "default": 0.5, "min": 0.2, "max": 0.8, "step": 0.1}
                    }
                }
            },
            "成交量指標 - MFI": {
                "mfi_oversold_cross": {
                    "name": "MFI超賣後突破",
                    "description": "資金流量從超賣區向上突破",
                    "function": "signal_mfi_oversold_cross",
                    "params": {
                        "threshold": {"label": "超賣閾值", "type": "number", "default": 20, "min": 10, "max": 30}
                    }
                },
                "mfi_bullish": {
                    "name": "MFI多頭區間",
                    "description": "MFI處於健康多頭區",
                    "function": "signal_mfi_bullish",
                    "params": {
                        "lower": {"label": "下限", "type": "number", "default": 40, "min": 20, "max": 60},
                        "upper": {"label": "上限", "type": "number", "default": 80, "min": 60, "max": 90}
                    }
                }
            }
        }

        # 出場策略定義
        EXIT_STRATEGY_DEFINITIONS = {
            "均線類出場": {
                "ma_death_cross": {
                    "name": "均線死亡交叉",
                    "description": "快線向下跌破慢線",
                    "function": "exit_ma_death_cross",
                    "params": {
                        "fast_period": {"label": "快線週期", "type": "number", "default": 5, "min": 1, "max": 50},
                        "slow_period": {"label": "慢線週期", "type": "number", "default": 20, "min": 1, "max": 200}
                    }
                },
                "ma_bearish": {
                    "name": "均線空頭排列",
                    "description": "快線在慢下之下",
                    "function": "exit_ma_bearish",
                    "params": {
                        "fast_period": {"label": "快線週期", "type": "number", "default": 5, "min": 1, "max": 50},
                        "slow_period": {"label": "慢線週期", "type": "number", "default": 20, "min": 1, "max": 200}
                    }
                },
                "price_below_ma": {
                    "name": "價格跌破均線",
                    "description": "價格向下跌破均線",
                    "function": "exit_price_below_ma",
                    "params": {
                        "ma_period": {"label": "均線週期", "type": "number", "default": 20, "min": 1, "max": 200},
                        "ma_type": {"label": "均線類型", "type": "select", "default": "SMA", "options": ["SMA", "EMA"]}
                    }
                },
                "ma_slope_down": {
                    "name": "均線向下傾斜",
                    "description": "均線持續下降",
                    "function": "exit_ma_slope_down",
                    "params": {
                        "ma_period": {"label": "均線週期", "type": "number", "default": 20, "min": 1, "max": 200},
                        "ma_type": {"label": "均線類型", "type": "select", "default": "SMA", "options": ["SMA", "EMA"]},
                        "lookback": {"label": "回看期間", "type": "number", "default": 3, "min": 1, "max": 10}
                    }
                }
            },
            "動量指標出場 - RSI, MACD, KD": {
                "rsi_overbought": {
                    "name": "RSI超買",
                    "description": "RSI進入超買區",
                    "function": "exit_rsi_overbought_cross",
                    "params": {
                        "rsi_period": {"label": "RSI週期", "type": "number", "default": 14, "min": 2, "max": 50},
                        "threshold": {"label": "超買閾值", "type": "number", "default": 70, "min": 60, "max": 90}
                    }
                },
                "rsi_bearish":{
                    "name": "RSI空頭",
                    "description": "RSI 處於空頭區間",
                    "function": "exit_rsi_bearish",
                    "params": {
                        "rsi_period": {"label": "RSI週期", "type": "number", "default": 14, "min": 2, "max": 50},
                        "lower": {"label": "下限", "type": "number", "default": 30, "min": 20, "max": 50},
                        "upper": {"label": "上限", "type": "number", "default": 60, "min": 50, "max": 80}
                    }
                },
                "macd_death_cross": {
                    "name": "MACD死亡交叉",
                    "description": "MACD線向下跌破信號線",
                    "function": "exit_macd_death_cross",
                    "params": {}
                },
                "macd_bearish": {
                    "name": "MACD 空頭",
                    "description": "MACD線在信號線之下",
                    "function": "exit_macd_bearish",
                    "params": {}
                },
                "macd_hist_negative": {
                    "name": "MACD柱狀圖轉負",
                    "description": "柱狀圖由正轉負",
                    "function": "exit_macd_hist_negative",
                    "params": {}
                },
                "kd_death_cross": {
                    "name": "KD死亡交叉",
                    "description": "K線向下跌破D線",
                    "function": "exit_kd_death_cross",
                    "params": {}
                },
                "kd_overbought": {
                    "name": "KD超買",
                    "description": "KD進入超買區且死亡交叉",
                    "function": "exit_kd_overbought_cross",
                    "params": {
                        "threshold": {"label": "超買閾值", "type": "number", "default": 80, "min": 70, "max": 90}
                    }
                },
                "kd_bearish": {
                    "name": "KD空頭",
                    "description": "K線在D線之下",
                    "function": "exit_kd_bearish",
                    "params": {}
                }
            },
            "動量指標出場 - 其他": {
                "cci_overbought_cross": {
                    "name": "CCI超買後跌破",
                    "description": "CCI從超買區向下跌破",
                    "function": "exit_cci_overbought_cross",
                    "params": {
                        "threshold": {"label": "超買閾值", "type": "number", "default": 100, "min": 20, "max": 200}
                    }
                },
                "cci_bearish": {
                    "name": "CCI空頭區間",
                    "description": "CCI小於0",
                    "function": "exit_cci_bearish",
                    "params": {}
                },
                "roc_negative": {
                    "name": "ROC轉負",
                    "description": "變動率指標轉負",
                    "function": "exit_roc_negative",
                    "params": {}
                },
                "stochrsi_death_cross": {
                    "name": "StochRSI死亡交叉",
                    "description": "隨機RSI的K線跌破D線",
                    "function": "exit_stochrsi_death_cross",
                    "params": {}
                },
                "stochrsi_overbought_cross": {
                    "name": "StochRSI超買後跌破",
                    "description": "StochRSI從超買區向下跌破",
                    "function": "exit_stochrsi_overbought_cross",
                    "params": {
                        "threshold": {"label": "超買閾值", "type": "number", "default": 0.8, "min": 0.5, "max": 1.0, "step": 0.1}
                    }
                },
                "willr_overbought_cross": {
                    "name": "Williams %R超買後跌破",
                    "description": "威廉指標從超買區向下跌破",
                    "function": "exit_willr_overbought_cross",
                    "params": {
                        "threshold": {"label": "超買閾值", "type": "number", "default": -20, "min": -50, "max": 0}
                    }
                }
            },
            "布林通道出場": {
                "bb_upper_touch": {
                    "name": "觸及布林上軌",
                    "description": "價格觸及或突破上軌",
                    "function": "exit_bb_upper_bounce",
                    "params": {}
                },
                "bb_squeeze_break_down": {
                    "name": "布林通道收縮",
                    "description": "布林通道收縮後向下突破",
                    "function": "exit_bb_squeeze_break_down",
                    "params": {}
                },
                "bb_percent_b_high": {
                    "name": "布林%B過高",
                    "description": "%B指標超過閾值",
                    "function": "exit_bb_percent_b_bearish",
                    "params": {
                        "threshold": {"label": "閾值", "type": "number", "default": 0.8, "min": 0.6, "max": 1.2, "step": 0.1}
                    }
                }
            },
            "成交量指標出場 - MFI": {
                "mfi_overbought_cross": {
                    "name": "MFI超買後跌破",
                    "description": "資金流量從超買區向下跌破",
                    "function": "exit_mfi_overbought_cross",
                    "params": {
                        "threshold": {"label": "超買閾值", "type": "number", "default": 80, "min": 50, "max": 100}
                    }
                },
                "mfi_bearish": {
                    "name": "MFI空頭區間",
                    "description": "MFI處於健康空頭區",
                    "function": "exit_mfi_bearish",
                    "params": {
                        "lower": {"label": "下限", "type": "number", "default": 20, "min": 0, "max": 40},
                        "upper": {"label": "上限", "type": "number", "default": 60, "min": 40, "max": 80}
                    }
                }
            }
        }

        # 初始化 session state
        if 'selected_entry_strategies' not in st.session_state:
            st.session_state.selected_entry_strategies = []
        if 'selected_exit_strategies' not in st.session_state:
            st.session_state.selected_exit_strategies = []

        subtab = st.radio(
            "選擇分類",
            ["🎯 進場策略", "🚪 出場策略", "📊 配置總覽"],
            horizontal=True
        )
        st.markdown('---')

        # ==================== 進場策略 Tab ====================
        if subtab == '🎯 進場策略':
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("選擇進場指標")
                
                for category, strategies in ENTRY_STRATEGY_DEFINITIONS.items():
                    with st.expander(f"📊 {category}", expanded=False):
                        for strategy_id, strategy_info in strategies.items():
                            col_btn, col_desc = st.columns([3, 5])
                            
                            with col_btn:
                                if st.button(
                                    f"➕ {strategy_info['name']}", 
                                    key=f"entry_btn_{strategy_id}",
                                    use_container_width=True
                                ):
                                    new_strategy = {
                                        "id": strategy_id,
                                        "category": category,
                                        "name": strategy_info['name'],
                                        "function": strategy_info['function'],
                                        "params": {}
                                    }
                                    
                                    for param_name, param_config in strategy_info['params'].items():
                                        new_strategy['params'][param_name] = param_config['default']
                                    
                                    st.session_state.selected_entry_strategies.append(new_strategy)
                                    st.rerun()
                            
                            with col_desc:
                                st.caption(strategy_info['description'])
            
            with col2:
                st.subheader("✅ 已選進場策略")
                st.caption(f"共 {len(st.session_state.selected_entry_strategies)} 個")
                
                if len(st.session_state.selected_entry_strategies) == 0:
                    st.info("👈 請從左側選擇進場策略")
                else:
                    for idx, strategy in enumerate(st.session_state.selected_entry_strategies):
                        with st.container():
                            col_title, col_delete = st.columns([5, 1])
                            
                            with col_title:
                                st.markdown(f"**{idx + 1}. {strategy['name']}**")
                                st.caption(f"類別: {strategy['category']}")
                            
                            with col_delete:
                                if st.button("🗑️", key=f"del_entry_{idx}"):
                                    st.session_state.selected_entry_strategies.pop(idx)
                                    st.rerun()
                            
                            strategy_def = None
                            for cat_strategies in ENTRY_STRATEGY_DEFINITIONS.values():
                                if strategy['id'] in cat_strategies:
                                    strategy_def = cat_strategies[strategy['id']]
                                    break
                            
                            if strategy_def and strategy_def['params']:
                                with st.container():
                                    st.markdown("##### 參數設定")
                                    
                                    for param_name, param_config in strategy_def['params'].items():
                                        if param_config['type'] == 'number':
                                            step = param_config.get('step', 1)
                                            strategy['params'][param_name] = st.number_input(
                                                param_config['label'],
                                                min_value=param_config['min'],
                                                max_value=param_config['max'],
                                                value=strategy['params'].get(param_name, param_config['default']),
                                                step=step,
                                                key=f"entry_param_{idx}_{param_name}"
                                            )
                                        elif param_config['type'] == 'select':
                                            strategy['params'][param_name] = st.selectbox(
                                                param_config['label'],
                                                options=param_config['options'],
                                                index=param_config['options'].index(
                                                    strategy['params'].get(param_name, param_config['default'])
                                                ),
                                                key=f"entry_param_{idx}_{param_name}"
                                            )
                            else:
                                st.caption("此策略無需設定參數")
                            
                            st.markdown("---")

        # ==================== 出場策略 Tab ====================
        elif subtab == '🚪 出場策略':
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("選擇出場指標")
                
                for category, strategies in EXIT_STRATEGY_DEFINITIONS.items():
                    with st.expander(f"🚪 {category}", expanded=False):
                        for strategy_id, strategy_info in strategies.items():
                            col_btn, col_desc = st.columns([3, 5])
                            
                            with col_btn:
                                if st.button(
                                    f"➕ {strategy_info['name']}", 
                                    key=f"exit_btn_{strategy_id}",
                                    use_container_width=True
                                ):
                                    new_strategy = {
                                        "id": strategy_id,
                                        "category": category,
                                        "name": strategy_info['name'],
                                        "function": strategy_info['function'],
                                        "params": {}
                                    }
                                    
                                    for param_name, param_config in strategy_info['params'].items():
                                        new_strategy['params'][param_name] = param_config['default']
                                    
                                    st.session_state.selected_exit_strategies.append(new_strategy)
                                    st.rerun()
                            
                            with col_desc:
                                st.caption(strategy_info['description'])
            
            with col2:
                st.subheader("✅ 已選出場策略")
                st.caption(f"共 {len(st.session_state.selected_exit_strategies)} 個")
                
                if len(st.session_state.selected_exit_strategies) == 0:
                    st.info("👈 請從左側選擇出場策略")
                else:
                    for idx, strategy in enumerate(st.session_state.selected_exit_strategies):
                        with st.container():
                            col_title, col_delete = st.columns([5, 1])
                            
                            with col_title:
                                st.markdown(f"**{idx + 1}. {strategy['name']}**")
                                st.caption(f"類別: {strategy['category']}")
                            
                            with col_delete:
                                if st.button("🗑️", key=f"del_exit_{idx}"):
                                    st.session_state.selected_exit_strategies.pop(idx)
                                    st.rerun()
                            
                            strategy_def = None
                            for cat_strategies in EXIT_STRATEGY_DEFINITIONS.values():
                                if strategy['id'] in cat_strategies:
                                    strategy_def = cat_strategies[strategy['id']]
                                    break
                            
                            if strategy_def and strategy_def['params']:
                                with st.container():
                                    st.markdown("##### 參數設定")
                                    
                                    for param_name, param_config in strategy_def['params'].items():
                                        if param_config['type'] == 'number':
                                            step = param_config.get('step', 1)
                                            strategy['params'][param_name] = st.number_input(
                                                param_config['label'],
                                                min_value=param_config['min'],
                                                max_value=param_config['max'],
                                                value=strategy['params'].get(param_name, param_config['default']),
                                                step=step,
                                                key=f"exit_param_{idx}_{param_name}"
                                            )
                                        elif param_config['type'] == 'select':
                                            strategy['params'][param_name] = st.selectbox(
                                                param_config['label'],
                                                options=param_config['options'],
                                                index=param_config['options'].index(
                                                    strategy['params'].get(param_name, param_config['default'])
                                                ),
                                                key=f"exit_param_{idx}_{param_name}"
                                            )
                            else:
                                st.caption("此策略無需設定參數")
                            
                            st.markdown("---")

        # ==================== 配置總覽 Tab ====================
        elif subtab == '📊 配置總覽':
            st.subheader("📋 策略配置總覽")
            
            col_stats1, col_stats2 = st.columns(2)
            with col_stats1:
                st.metric("進場策略數量", len(st.session_state.selected_entry_strategies))
            with col_stats2:
                st.metric("出場策略數量", len(st.session_state.selected_exit_strategies))
            
            st.markdown("---")
            
            # 檢查配置完整性
            if len(st.session_state.selected_entry_strategies) == 0:
                baseline_metric_error += 1
                st.warning("⚠️ 請至少選擇一個進場策略")
            
            if len(st.session_state.selected_exit_strategies) == 0:
                baseline_metric_error += 1
                st.warning("⚠️ 請至少選擇一個出場策略")
            
            if len(st.session_state.selected_entry_strategies) > 0 and len(st.session_state.selected_exit_strategies) > 0:
                baseline_metric_error = 0
                st.success("✅ 策略配置完整，可以進行回測")
                
                # 匯出完整配置
                st.markdown("### 📤 匯出完整配置")
                
                full_config = {
                    "entry_strategies": [
                        {
                            "function": s['function'],
                            "name": s['name'],
                            "category": s['category'],
                            "params": s['params']
                        }
                        for s in st.session_state.selected_entry_strategies
                    ],
                    "exit_strategies": [
                        {
                            "function": s['function'],
                            "name": s['name'],
                            "category": s['category'],
                            "params": s['params']
                        }
                        for s in st.session_state.selected_exit_strategies
                    ]
                }
                
                col_json, col_download = st.columns([3, 1])
                
                with col_json:
                    st.json(full_config)
                
                with col_download:
                    config_str = json.dumps(full_config, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="💾 下載配置",
                        data=config_str,
                        file_name="backtest_config.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            # 清空按鈕
            st.markdown("---")
            col_clear1, col_clear2, col_clear3 = st.columns([1, 1, 1])
            
            with col_clear1:
                if st.button("🗑️ 清空進場策略", use_container_width=True):
                    st.session_state.selected_entry_strategies = []
                    st.rerun()
            
            with col_clear2:
                if st.button("🗑️ 清空出場策略", use_container_width=True):
                    st.session_state.selected_exit_strategies = []
                    st.rerun()
            
            with col_clear3:
                if st.button("🗑️ 清空全部策略", type="primary", use_container_width=True):
                    st.session_state.selected_entry_strategies = []
                    st.session_state.selected_exit_strategies = []
                    st.rerun()

    # 主頁內容
    with tab_backtest:
        # 跑回測
        if run_btn:
            # 檢查回測命名，不能重複
            backtest_hisName = db.scalars(select(BacktestHistory.backtest_name)).all()
            if backtest_name not in backtest_hisName:
                with st.spinner('執行回測中…'):
                    df = load_dataframe(up, 
                                        bucket=os.getenv("INFLUX_BUCKET"),
                                        ohlcv_measurement=os.getenv("INFLUX_MEASUREMENT_OHLCV"),
                                        metric_measurement=os.getenv("INFLUX_MEASUREMENT_METRIC"),
                                        org=os.getenv("INFLUX_ORG", ""),
                                        influx_client=influx_client,
                                        ticker=ticker,
                                        startt=startt,
                                        endt=endt,
                                        market='TW' if market == '台股' else 'US')
                    
                    # 初始化策略
                    strategy_func = db.execute(
                        select(Strategy.strategy_func)
                        .where(Strategy.strategy_name == selected_strategy)
                    ).scalars().all()

                    # TODO: 寫到這
                    m_strategy = strategyManager(strategy_func[0])
                    report = run_backtest(df,
                                          m_strategy=m_strategy,
                                          entry_signals=[signal['function'] for signal in st.session_state.selected_entry_strategies],
                                          entry_params=[signal['params'] for signal in st.session_state.selected_entry_strategies],
                                          exit_signals=[signal['function'] for signal in st.session_state.selected_exit_strategies],
                                          exit_params=[signal['params'] for signal in st.session_state.selected_exit_strategies],
                                          atr_period=atr_period,
                                          stop_loss_pct=stop_loss_pct,
                                          atr_stop_mult=atr_stop_mult,
                                          time_stop_bars=time_stop_bars,
                                          startcash=startcash,
                                          buy_fee_rate=buy_fee,
                                          sell_fee_rate=sell_fee,
                                          sell_tax_rate=sell_tax,
                                          slippage=slippage,
                                          use_atr_sizer=use_atr_sizer)

                # === KPI 區 ===
                st.subheader("KPIs")
                cols = st.columns(5)
                ro = report.get('returns_overall') or {}
                dd = report.get('drawdown') or {}
                ta = report.get('trade') or {}
                ss = report.get('series_stats') or {}

                def pct(x):
                    return (f"{x*100:.2f}%" if x is not None and not np.isnan(x) else "-")

                with cols[0]:
                    st.metric("總報酬率", pct(ro.get('total_return')))
                with cols[1]:
                    st.metric("年化報酬率(Returns)", pct(ro.get('annual_return')))
                with cols[2]:
                    st.metric("最終資產", int(report.get("final_value")))
                with cols[3]:
                    drawdown_str = pct(-abs(dd.get('max_drawdown') or 0)/100 if dd.get('max_drawdown') and dd.get('max_drawdown')>1 else dd.get('max_drawdown'))  # dd 來源單位可能為 % 或 小數，盡力顯示
                    st.metric("最大回撤/天數", f"{drawdown_str}, {dd.get('max_len') or 0}天")
                with cols[4]:
                    st.metric("勝率", pct(ta.get('win_rate')), help=f"交易次數: {ta.get('trades_total')}, 最長連勝/連敗次數: {ta.get('longest_win_streak')}/{ta.get('longest_lose_streak')}")

                # === 其他回測指標 ===
                st.markdown("---")
                st.subheader("其他回測指標")

                other_cols = st.columns(2)
                with other_cols[0]:
                    st.metric("Sharpe", f"{report.get('sharpe_ratio') or ss.get('sharpe_from_series'):.2f}", help="風險調整後報酬，若 < 1 可能要調整策略 (越大越好)")
                with other_cols[1]:
                    st.metric("SQN", f"{round(report.get('sqn'), 1)}", help="衡量交易系統穩定與效率的品質指標 (< 2.5: 不太好，> 3: 不錯的策略)")
                
                # === 累積報酬 (使用你的 plot_equity_curve) & 回撤曲線 ===
                st.subheader("資金曲線 / 回撤 (%)", help="左: 累積報酬 (每日報酬連乘); 右: 回撤幅度, 單位: % (0: 沒有回撤，不斷創新高)")
                timeret = report.get('timereturn') or {}
                if timeret:
                    tret_df = pd.DataFrame(timeret).T
                    tret_df.index = pd.to_datetime(tret_df.index)
                    tret_df = tret_df.sort_index()

                    col1, col2 = st.columns([2,1])
                    with col1:
                        fig = plot_equity_curve(tret_df)
                        if fig is not None:
                            st.pyplot(fig)
                        else:
                            st.info("TimeReturn 資料不足，無法繪製資金曲線。")
                    with col2:
                        # 以 cum_return 計算回撤
                        if 'cum_return' in tret_df.columns and not tret_df['cum_return'].empty:
                            equity = (1 + tret_df['cum_return'])
                            running_max = equity.cummax()
                            dd_series = equity / running_max - 1
                            dd_series = dd_series * 100
                            st.area_chart(dd_series.rename('drawdown'))
                        else:
                            st.info("沒有 cum_return 可用於回撤圖。")
                else:
                    st.info("沒有可用的 TimeReturn 數據。")

                df_cols = st.columns(2)

                # === 年度報酬表 ===
                with df_cols[0]:
                    st.subheader("年度報酬")
                    annual = report.get('annual_return') or {}
                    if annual:
                        dfy = pd.DataFrame.from_dict(annual, orient='index', columns=['return']).sort_index()
                        dfy['單年度報酬率 (%)'] = dfy['return'] * 100
                        dfy = dfy.drop(columns=['return'])
                        st.dataframe(dfy.style.format({'單年度報酬率 (%)': '{:.2f}%'}))
                    else:
                        st.info("沒有年度報酬資料。")

                # === 交易統計 ===
                with df_cols[1]:
                    # === 進階統計(由 series 推導) ===
                    st.subheader("進階統計")
                    st.write("Sharpe (不考慮複利): 只拿整段報酬序列的平均穩定度")
                    st.write("年化波動: 一年內報酬平均有多大的上下起伏，考量報酬率的標準差 (>30%: 高波動)")
                    st.write("Sortino: 年化報酬 / 年化下行波動，指考量下跌標準差，更貼近投資指風險 (>1: 較佳)")
                    st.write("Calmar: 年化報酬 / 最大回撤，更直覺反應回撤風險，每承受1單位最大回撤可以產生多少年化報酬 (>0: 較佳)")
                    if ss:
                        df_s = pd.DataFrame(ss, index=[0])
                        df_s = df_s.rename(columns={
                            'ann_vol': '年化波動',
                            'sharpe_from_series': 'Sharpe (不考慮複利)',
                            'sortino': 'Sortino',
                            'calmar': 'Calmar'}).T
                        st.dataframe(df_s.rename(columns={0: 'Value'}))
                    else:
                        st.info("沒有進階統計")

                st.subheader("交易統計")
                if ta:
                    df_ta = pd.DataFrame(ta, index=[0])
                    df_ta = df_ta.rename(columns={
                        'trades_total': "總交易次數",
                        "trades_closed": "利用收盤價交易的次數",
                        "trades_open": "盤中交易的次數",
                        "wins": "交易賺錢次數",
                        "losses": "交易賠錢的次數",
                        "win_rate": "交易勝率",
                        "pnl_net_total": "總淨損益",
                        "pnl_net_avg": "平均淨損義",
                        "pnl_gross_total": "總毛損益",
                        "profit_factor": "盈虧比",
                        "longest_win_streak": "最大連勝次數",
                        "longest_loss_streak": "最大連敗次數",
                    }).T

                    st.dataframe(df_ta.rename(columns={0: 'Value'}))
                else:
                    st.info("沒有交易統計資料。")

                # 原始 JSON
                with st.expander("原始報告 JSON"):
                    st.code(json.dumps(report, ensure_ascii=False, indent=2)[:20000])

            # 保存回測結果
            if save_btn:
                paths, result_dir = save_report_bundle(report,
                                        out_dir='./report',
                                        ticker=ticker,
                                        backtest_name=backtest_name)
                st.success('已保存到本機資料夾')
                st.code("".join([f"{k}: {v}" for k, v in paths.items()]))

                # 把相關結果存到 PostgreSQL
                total_return = pct(ro.get('total_return'))
                max_drawdown = pct(-abs(dd.get('max_drawdown') or 0)/100 if dd.get('max_drawdown') and dd.get('max_drawdown')>1 else dd.get('max_drawdown'))
                sharpe_ratio = f"{report.get('sharpe_ratio') or ss.get('sharpe_from_series'):.2f}"
                trades_count = df_ta.get('total', {}).get('total', 0)

                try:
                    custom_params = {
                        "atr_period": atr_period,
                        "stop_loss_pct": stop_loss_pct,
                        "atr_stop_mult": atr_stop_mult,
                        "time_stop_bars": time_stop_bars,
                        "dollars": dollars,
                        "startcash": startcash,
                        "buy_fee": buy_fee,
                        "sell_fee": sell_fee,
                        "sell_tax": sell_tax,
                        "slippage": slippage,
                        "use_atr_sizer": use_atr_sizer,
                    }
                    with open(os.path.join(result_dir, "custom_params.json"), 'w') as f:
                        json.dump(custom_params, f, indent=2)

                    toInsert = BacktestHistory(
                        backtest_name=backtest_name,
                        ticker=ticker,
                        test_date=datetime.date.today(),
                        start_date=startt,
                        end_date=endt,
                        market=market,
                        description=memo,                    
                        total_return=total_return,
                        max_drawdown=max_drawdown,
                        sharpe_ratio=sharpe_ratio,
                        trades_count=trades_count,
                        report_path=result_dir,
                        custom_params_path=str(os.path.join(result_dir, "custom_params.json"))
                    )

                    # insert to db
                    db.add(toInsert)        # insert
                    db.flush()              # id
                    db.refresh(toInsert)    # created_at
                    db.commit()
                except IntegrityError:
                    db.rollback()
                    raise
        else:
            st.warning("回測 session 名稱已經重複!")

    # 查看歷史回測結果
    with tab_backtestHis:
        st.subheader("選擇策略")
        
        try:
            backtest_hisName = db.scalars(select(BacktestHistory.backtest_name)).all()
        except:
            backtest_hisName = ['---']
        
        selected_his_strategy = st.selectbox("選擇回顧策略代碼", backtest_hisName)

        st.markdown("---")
        st.subheader("內容")

        # TODO: 內容
        
    # 自動化下單
    with tab_trade:
        st.subheader("自動化下單 (台股)")

        try:
            strategy_candidate = db.scalars(select(Strategy.strategy_name)).all()
        except:
            strategy_candidate = ["---"]

        traded_strategy = st.selectbox("選擇下單策略代碼", strategy_candidate)

        st.markdown("---")
        ticker = st.text_input(
            "股票代碼",
            value="",
            placeholder="ex. 2330",
            max_chars=30,
            label_visibility="visible",   # "hidden" / "collapsed"
        )

        # TODO: 其他參數

    # TODO: 找小鬼股
    with tab_screener:
        st.subheader("找小鬼股")

finally:
    sess_db.close()