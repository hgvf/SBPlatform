import pandas as pd
import backtrader as bt
import numpy as np
import json
from typing import Dict, Any, List
from datetime import datetime, timezone, timedelta, date

class PandasDataWithMetric(bt.feeds.PandasData):
    """
    額外支援技術面指標
    """

    lines = (
        'SMA_5', 'EMA_5',
        'SMA_20', 'EMA_20',
        'SMA_60', 'EMA_60',
        'SMA_120', 'EMA_120',
        'SMA_240', 'EMA_240',
        'RSI_14',
        'MACD', 'MACD_signal', 'MACD_hist',
        'KD_K', 'KD_D',
        'BB_upper', 'BB_middle', 'BB_lower',
        'BB_percent_b', 'BB_bandwidth',
        'CCI_20', 'WILLR_14', 'MFI_14',
        'ROC_12',
        'STOCHRSI_K', 'STOCHRSI_D',
    )

    params = tuple(
        (name, -1)
        for name in (
            'SMA_5', 'EMA_5',
            'SMA_20', 'EMA_20',
            'SMA_60', 'EMA_60',
            'SMA_120', 'EMA_120',
            'SMA_240', 'EMA_240',
            'RSI_14',
            'MACD', 'MACD_signal', 'MACD_hist',
            'KD_K', 'KD_D',
            'BB_upper', 'BB_middle', 'BB_lower',
            'BB_percent_b', 'BB_bandwidth',
            'CCI_20', 'WILLR_14', 'MFI_14',
            'ROC_12',
            'STOCHRSI_K', 'STOCHRSI_D',
        )
    )

def ensure_datetime_index(df: pd.DataFrame, time_col: str = 'timestamp') -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
    else:
        raise ValueError("DataFrame 必須有 DatetimeIndex 或包含 'datetime' 欄位")
    
    df = df.sort_index()
    return df

def validate_columns(df: pd.DataFrame, use_external_signal: bool = False) -> None:
    required = {'open', 'high', 'low', 'close', 'volume'}
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        raise ValueError(f"缺少必要欄位: {missing}")
    if use_external_signal and 'signal' not in df.columns:
        raise ValueError("use_external_signal=True 時，必須提供 'signal' 欄位（1=買, -1=賣/平, 0/NaN=無）")

def to_rfc3339(date_input, tz_offset_hours=0) -> str:
    """
    將 streamlit 的 date_input (datetime.date) 轉成 RFC3339 格式
    tz_offset_hours: 時區偏移（例如台灣為 +8）
    """

    # 如果輸入是 datetime.date，就轉成 datetime
    if isinstance(date_input, date) and not isinstance(date_input, datetime):
        dt = datetime.combine(date_input, datetime.min.time())
    elif isinstance(date_input, datetime):
        dt = date_input
    else:
        raise TypeError("date_input 必須是 datetime.date 或 datetime.datetime 物件")

    # 設定時區
    tz = timezone(timedelta(hours=tz_offset_hours))
    dt = dt.replace(tzinfo=tz)

    # 轉成 RFC3339 字串
    return dt.isoformat()

def get_fieldName(type: str) -> List[str]:
    """
    依照類型 (OHLCV or Metric) 回傳資料庫的欄位名稱
    """

    assert type in ['OHLCV', 'Metric'], "get_fieldName() 中的 type 必須為 ['OHLCV', 'Metric'] 其一。"

    if type == 'OHLCV':
        return ['Open', 'High', 'Low', 'Close', 'Volume']
    elif type == 'Metric':
        return ['SMA_5', 'EMA_5', 'SMA_20', 'EMA_20', 'SMA_60', 'EMA_60', 'SMA_120', 'EMA_120', 'SMA_240', 'EMA_240', 
                'RSI_7', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist', 
                'KD_K', 'KD_D', 
                'BB_upper', 'BB_middle', 'BB_lower', 'BB_percent_b', 'BB_bandwidth', 
                'CCI_20', 'WILLR_14', 'MFI_14', 'ROC_12', 'STOCHRSI_K', 'STOCHRSI_D']

def get_genericColName() -> List[str]:
    """
    通用 tag columns
    """

    return ["Ticker", "Interval", "Market"]

def get_query(ticker: str,
              startt: str,
              endt: str,
              market: str,
              measurement: str,
              bucket: str,
              fields: List[str],
              tags: List[str],
              interval: str = '1d') -> str:
    """
    回傳 influxDB query，會從 tbl_OHLCV & tbl_metricSeries 取資料
    """

    fields_array = ", ".join([f'"{f}"' for f in fields])
    fields_array = f"[{fields_array}]"
    keep_cols_json = json.dumps(["_time"] + fields + tags)

    flux = f"""
        from(bucket: "{bucket}")
        |> range(start: time(v: "{startt}"), stop: time(v: "{endt}"))
        |> filter(fn: (r) => r._measurement == "{measurement}")
        |> filter(fn: (r) => r.Ticker == "{ticker}")
        |> filter(fn: (r) => r.Market == "{market}")
        |> filter(fn: (r) => exists r.Interval and r.Interval == "{interval}")
        |> filter(fn: (r) => contains(value: r._field, set: {fields_array}))
        |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> keep(columns: {keep_cols_json})
        |> sort(columns: ["_time"])
    """

    return flux

# 資料處理
def load_dataframe(up, **kwargs) -> pd.DataFrame:
    """
    依照
    1) 使用者匯入的 CSV
    或是
    2) 使用者所選的條件去 influxDB select 出對應資料

    Args:
    up: CSV file object (default: None)
    env: 環境變數 
    """

    # 使用者自行匯入 CSV
    if up is not None:
        df = pd.read_csv(up)
            
        # TODO: 篩選 UI: startt, endt
        startt = kwargs.get("startt")
        endt = kwargs.get("endt")

        df = ensure_datetime_index(df)
        df = df.dropna().copy()
        df.index = df.index.tz_localize(None)

        df = df.loc[startt:endt]

    # 依照條件 select from influxDB
    # 條件: 個股代碼、起始/結束時間、市場類別
    else:
        # 取得變數
        client = kwargs.get("influx_client", None)
        ticker = kwargs.get("ticker", None)
        startt = kwargs.get("startt", to_rfc3339(datetime(2010, 1, 1)))
        endt = kwargs.get("endt", to_rfc3339(date.today() - timedelta(days=1)))
        market = kwargs.get("market", None)

        if market == 'TW':
            ticker += ".TW"

        assert client is not None, "必須先連線至 influxDB"
        assert ticker is not None, "必須指定股票代碼"

        tag_cols = get_genericColName()
        ohlcv_fields = get_fieldName(type='OHLCV')
        metric_fields = get_fieldName(type="Metric")

        # 製作 influxDB query
        params = {
            'ticker': ticker,
            'startt': startt,
            'endt': endt,
            'market': market,
            'bucket': kwargs.get("bucket", ""),
        }

        # get OHLCV data
        ohlcv_query = get_query(measurement=kwargs.get("ohlcv_measurement", ""),
                                fields=ohlcv_fields,
                                tags=tag_cols,
                                **params)
        
        ohlcv = client.query_api().query_data_frame(ohlcv_query, org=kwargs.get("org", ""))

        # get metric data
        metric_query = get_query(measurement=kwargs.get("metric_measurement", ""),
                                fields=metric_fields,
                                tags=tag_cols,
                                **params)

        metric = client.query_api().query_data_frame(metric_query, org=kwargs.get("org", ""))

        # inner join on "_time"
        df = pd.merge(ohlcv, metric, on="_time", how="inner")

        df = ensure_datetime_index(df)
        df = df.dropna().copy()

    return df
