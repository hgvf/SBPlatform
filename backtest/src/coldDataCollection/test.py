# -*- coding: utf-8 -*-
"""
Query last-1y OHLCV for 2330 & TSLA from InfluxDB, compute tech indicators, and print.
Requires: pip install influxdb-client pandas
"""

import os
import math
import json
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone
from influxdb_client import InfluxDBClient
from typing import List, Dict, Any

load_dotenv("../../../.env")

# ========= 連線設定：改成你的 =========
INFLUXDB_UI_PORT   = os.getenv("INFLUXDB_UI_PORT", "8086")
INFLUXDB_URL = f"http://influxdb:{INFLUXDB_UI_PORT}"
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "local-dev-token")
INFLUX_ORG   = os.getenv("INFLUX_ORG", "quant")
BUCKET       = os.getenv("INFLUX_BUCKET", "TBD")

# ========= measurement 與 tag/欄位設定（照你的庫調整）=========
OHLCV_MEASUREMENT  = os.getenv("INFLUX_MEASUREMENT_OHLCV", "tbl_OHLCV")   # 例如：tbl_OHLCV_TWUS 或 tbl_OHLCV
METRIC_MEASUREMENT  = os.getenv("INFLUX_MEASUREMENT_METRIC", "tbl_metricSeries")   # 例如：tbl_OHLCV_TWUS 或 tbl_OHLCV

OHLCV_FIELD = ['Open', 'High', 'Low', 'Close', 'Volume']
METRIC_FIELD = ['SMA_5', "SMA_20", "SMA_60", "SMA_120"]
TAG_COLS     = ["Ticker", "Interval", "Market"]

# 你要查的代號（依你實際存的樣子改：例如 "2330" 或 "2330.TW"）
toSearch = [
    {"Ticker": "0050.TW", 'Market': "TW", "Interval": "1d"},
    {"Ticker": "AACB", 'Market': "US", "Interval": "1d"}
]

# ========= 從 InfluxDB 抓近一年 OHLCV =========
def fetch_ohlcv_df(client: InfluxDBClient, symbol: Dict[str, Any]) -> pd.DataFrame:
    """
    回傳欄位：[_time, Ticker, Open, High, Low, Close, Volume]，index 設為 UTC 時間
    """

    fields_array = ", ".join([f'"{f}"' for f in OHLCV_FIELD])
    fields_array = f"[{fields_array}]"
    keep_cols_json = json.dumps(["_time"] + OHLCV_FIELD + TAG_COLS)

    flux = f"""
        from(bucket: "{BUCKET}")
        |> range(start: -1y)
        |> filter(fn: (r) => r._measurement == "{OHLCV_MEASUREMENT}")
        |> filter(fn: (r) => contains(value: r._field, set: {fields_array}))
        |> filter(fn: (r) => r.Ticker == "{symbol['Ticker']}")
        |> filter(fn: (r) => exists r.Interval and r.Interval == "{symbol['Interval']}")
        |> filter(fn: (r) => r.Market == "{symbol['Market']}")
        |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> keep(columns: {keep_cols_json})
        |> sort(columns: ["_time"])
    """
    
    tables = client.query_api().query_data_frame(flux, org=INFLUX_ORG)
    
    if isinstance(tables, list) and len(tables) == 0:
        raise RuntimeError(f"No data returned for {symbol}")

    # query_data_frame 可能回傳 DataFrame 或 list[DF]；統一成一個 DF
    if isinstance(tables, list):
        df = pd.concat(tables, ignore_index=True)
    else:
        df = tables

    # 清掉 Influx 多餘欄位，確保只有我們要的
    keep_cols = ["_time"] + OHLCV_FIELD
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # 轉時間 & 設 index
    df["_time"] = pd.to_datetime(df["_time"], utc=True)
    df = df.set_index("_time").sort_index()

    # 數值強制轉型
    for c in OHLCV_FIELD:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 去掉全 NA 的列
    df = df.dropna(subset=["Close"])
    return df

# ========= 從 InfluxDB 抓近一年四種均線 =========
def fetch_sma_df(client: InfluxDBClient, symbol: Dict[str, Any]) -> pd.DataFrame:
    """
    回傳欄位：[_time, Ticker, SMA_5, SMA_20, SMA_60, SMA_120, SMA_240]，index 設為 UTC 時間
    """

    fields_array = ", ".join([f'"{f}"' for f in METRIC_FIELD])
    fields_array = f"[{fields_array}]"
    keep_cols_json = json.dumps(["_time"] + METRIC_FIELD + TAG_COLS)

    flux = f"""
        from(bucket: "{BUCKET}")
        |> range(start: -1y)
        |> filter(fn: (r) => r._measurement == "{METRIC_MEASUREMENT}")
        |> filter(fn: (r) => contains(value: r._field, set: {fields_array}))
        |> filter(fn: (r) => r.Ticker == "{symbol['Ticker']}")
        |> filter(fn: (r) => exists r.Interval and r.Interval == "{symbol['Interval']}")
        |> filter(fn: (r) => r.Market == "{symbol['Market']}")
        |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> keep(columns: {keep_cols_json})
        |> sort(columns: ["_time"])
    """
    
    tables = client.query_api().query_data_frame(flux, org=INFLUX_ORG)

    if isinstance(tables, list) and len(tables) == 0:
        raise RuntimeError(f"No data returned for {symbol}")

    # query_data_frame 可能回傳 DataFrame 或 list[DF]；統一成一個 DF
    if isinstance(tables, list):
        df = pd.concat(tables, ignore_index=True)
    else:
        df = tables

    # 清掉 Influx 多餘欄位，確保只有我們要的
    keep_cols = ["_time"] + METRIC_FIELD
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # 轉時間 & 設 index
    df["_time"] = pd.to_datetime(df["_time"], utc=True)
    df = df.set_index("_time").sort_index()

    # 數值強制轉型
    for c in METRIC_FIELD:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# ========= 主程式 =========
def main():
    with InfluxDBClient(url=INFLUXDB_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
        for tgt in toSearch:
            print("=" * 80)
            print(f"[{tgt['Ticker']}] last 1y daily OHLCV + indicators")
            
            ohlcv_df = fetch_ohlcv_df(client, tgt)
            print("OHLCV 前五列資料: ")
            print(ohlcv_df.head())

            metric_df = fetch_sma_df(client, tgt)
            print("SMA 前五列資料: ")
            print(metric_df.head())

if __name__ == "__main__":
    main()
