import pandas as pd
import time
import os
import yfinance as yf
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from typing import List, Dict, Any

def safe_history(ticker: str, **kwargs) -> pd.DataFrame:
    """
    根據 ticker name 爬取股價
    """

    for i in range(3):  # 重試
        try:
            return yf.Ticker(ticker).history(**kwargs)
        except Exception as e:
            if i == 2:
                raise
            time.sleep(1.5 * (i+1))
            
def process_ohlcv(all_prices: pd.DataFrame,
                  market: str,
                  interval: str,
                  ohlcv_field: List[str]) -> pd.DataFrame:
    """
    單獨 filter out OHLCV 資料
    """

    # 先存 OHLCV
    ohlcv_required_cols = ohlcv_field + ["Ticker", "timestamp"]

    ohlcv_missing = [c for c in ohlcv_required_cols if c not in all_prices.columns]
    if ohlcv_missing:
        raise ValueError(f"Dataframe 缺少欄位: {ohlcv_missing}")
    
    # 轉換時間欄位
    all_prices['timestamp'] = pd.to_datetime(all_prices['timestamp'], errors='raise')
    if all_prices['timestamp'].dt.tz is None:
        all_prices['timestamp'] = all_prices['timestamp'].dt.tz_localize("UTC+8")

    # 數值欄位轉換為 float / int
    for c in ['Open', "Close", "High", "Low", "Volume"]:
        all_prices[c] = pd.to_numeric(all_prices[c], errors='raise')

    # Ticker 轉為 string
    all_prices['Ticker'] = all_prices["Ticker"].astype(str)

    # filter out the dataframe
    ohlcv_df = all_prices[ohlcv_required_cols]

    # Timestamp 設為 index
    all_prices = all_prices.set_index("timestamp")

    # add the appendix columns: Interval & Market
    ohlcv_df['Interval'] = [interval for _ in range(len(ohlcv_df))]
    ohlcv_df['Market'] = [market for _ in range(len(ohlcv_df))]

    return ohlcv_df

def process_metric(all_prices: pd.DataFrame,
                   market: str,
                   interval: str,
                   metric_field: List[str]) -> pd.DataFrame:
    """
    單獨 filter out 技術面指標 columns
    """

    metric_required_cols = metric_field + ["Ticker", "timestamp"]
    metric_missing = [c for c in metric_required_cols if c not in all_prices.columns]
    if metric_missing:
        raise ValueError(f"Dataframe 缺少欄位: {metric_missing}")
    
    # 轉換時間欄位
    all_prices['timestamp'] = pd.to_datetime(all_prices['timestamp'], errors='raise')
    if all_prices['timestamp'].dt.tz is None:
        all_prices['timestamp'] = all_prices['timestamp'].dt.tz_localize("UTC+8")

    # 數值欄位轉換為 float / int
    for c in metric_required_cols:
        if c in ['Ticker', 'timestamp']:
            continue

        all_prices[c] = pd.to_numeric(all_prices[c], errors='raise')

    # Ticker 轉為 string
    all_prices['Ticker'] = all_prices["Ticker"].astype(str)

    # filter out the dataframe
    metric_df = all_prices[metric_required_cols]

    # Timestamp 設為 index
    all_prices = all_prices.set_index("timestamp")    

    # add the appendix columns: Interval & Market
    metric_df['Interval'] = [interval for _ in range(len(metric_df))]
    metric_df['Market'] = [market for _ in range(len(metric_df))]

    return metric_df

def save2influxDB(df: pd.DataFrame,
                  measurement: str,
                  field_list: List[str],
                  payload: Dict[str, Any]):
    """
    把指定 dataframe 存到指定 influxDB
    """

    client = InfluxDBClient(url=payload["INFLUX_URL"], 
                            token=payload["INFLUX_TOKEN"], 
                            org=payload["INFLUX_ORG"])
    write_api = client.write_api(write_options=SYNCHRONOUS)
    
    df = df.rename(columns={"timestamp": "_time"})
    df['_time'] = df['_time'].dt.tz_convert('UTC')
        
    points = []
    for _, row in df.iterrows():        
        point = Point(measurement)

        # 加入 tags
        for tag in ['Ticker', 'Interval', 'Market']:
            point = point.tag(tag, row[tag])
        
        # 加入 fields（自動跳過 NaN）
        has_valid_field = False
        for field in field_list:
            value = row[field]
            if pd.notna(value):  # 只寫入非 NaN 的值
                point = point.field(field, float(value))
                has_valid_field = True
        
        # 只有當至少有一個有效的 field 時才寫入
        if has_valid_field:
            point = point.time(row['_time'])
            points.append(point)

        # 每 1000 筆寫入一次
        if len(points) >= 1000:
            write_api.write(bucket=payload["INFLUX_BUCKET"], org=payload["INFLUX_ORG"], record=points)
            points = []
    
    # 寫入剩餘的資料
    if points:
        write_api.write(bucket=payload["INFLUX_BUCKET"], org=payload["INFLUX_ORG"], record=points)

    client.close()
