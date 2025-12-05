import pandas as pd
import os
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime

from TW import crawl_TW
from US import crawl_US

if __name__ == "__main__":
    # load environment variables
    load_dotenv("../../../.env")

    # InfluxDB 連線參數
    payload = {
        "INFLUX_URL": f"{os.getenv('INFLUX_URL', 'http://influxdb')}:{os.getenv('INFLUXDB_UI_PORT', '8086')}",
        "INFLUX_TOKEN": os.getenv("INFLUX_TOKEN", "YOUR_TOKEN_HERE"),
        "INFLUX_ORG": os.getenv("INFLUX_ORG", "YOUR_ORG"),
        "INFLUX_BUCKET": os.getenv("INFLUX_BUCKET", "YOUR_BUCKET"),

        "OHLCV_MEASUREMENT": os.getenv("INFLUX_MEASUREMENT_OHLCV", "tbl_OHLCV"),
        "METRIC_MEASUREMENT": os.getenv("INFLUX_MEASUREMENT_METRIC", "tbl_metricSeries"),

        "OHLCV_FIELD": ['Open', 'High', 'Low', 'Close', 'Volume'],
        "METRIC_FIELD": ['SMA_5', 'EMA_5', 
                        'SMA_20', 'EMA_20', 
                        'SMA_60', 'EMA_60', 
                        'SMA_120', 'EMA_120', 
                        'SMA_240', 'EMA_240', 
                        'RSI_7', 'RSI_14', 
                        'MACD_macd', 'MACD_signal', 'MACD_hist',
                        'KD_K', 'KD_D', 
                        'BB_upper', 'BB_middle', 'BB_lower', 'BB_percent_b', 'BB_bandwidth', 
                        'CCI_20', 'WILLR_14', 'MFI_14', 'ROC_12', 
                        'STOCHRSI_K', 'STOCHRSI_D',
        ]
    }    

    today = datetime.now().strftime("%Y-%m-%d")
    
    interval = os.getenv("INTERVAL", "1d")
    period = os.getenv("PERIOD", "max")
    start = os.getenv("START", '2000-01-01')

    end = os.getenv("END", today)
    end = today
    
    # # 爬取股價 + 技術面指標
    crawl_TW(payload, interval, period, start, end)
    crawl_US(payload, interval, period, start, end)
    
