import os
import pandas as pd
from influxdb_client import InfluxDBClient
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Any

class InfluxManager:
    def __init__(self, ):
        self.client = InfluxDBClient(
            url=os.getenv("INFLUX_URL", "http://influxdb:8086"),
            token=os.getenv("INFLUX_TOKEN"),
            org=os.getenv("INFLUX_ORG"),
        )
        self.query_api = self.client.query_api()
        self.bucket = os.getenv("INFLUX_BUCKET", "TBD")
        self.ohlcv_measurement = os.getenv("INFLUX_MEASUREMENT_OHLCV", "tbl_OHLCV")
        self.metric_measurement = os.getenv("INFLUX_MEASUREMENT_METRIC", "tbl_metricSeries")

    def get_manager():
        return InfluxManager()
    
    def query_symbols_by_market(self,
                                market: str,
                                time: datetime) -> list[str]:
        """
        從 InfluxDB 找出所有 symbol (ticker)，依指定市場過濾
        
        Args:
            market: 市場代碼 ("TW", "US" 等)

        Returns:
            List[str]: 該市場所有 ticker 名稱
        """
        query = f'''
            from(bucket: "{self.bucket}")
            |> range(start: -{time.days()}d)                  // 給個範圍，不可省略 range
            |> filter(fn: (r) => r._measurement == "{self.ohlcv_measurement}")
            |> filter(fn: (r) => r.market == "{market}")
            |> keep(columns: ["symbol"])
            |> distinct(column: "symbol")
            |> sort(columns: ["symbol"])
        '''
        
        df = self.query_api.query_pandas(query)
        if df.empty:
            return []
        
        # InfluxDB distinct() 會回傳 _value 欄位
        return df["_value"].dropna().unique().tolist()
    
    def query_ohlcv_by_symbols_time(self,
                                    symbol: str,
                                    interval: str,
                                    time: datetime) -> pd.DataFrame:
        """"""

        query = f'''
            from(bucket: "{self.bucket}")
            |> range(start: -{time.days()}d)
            |> filter(fn: (r) => r._measurement == "ohlcv")
            |> filter(fn: (r) => r.symbol == "{symbol}")
            |> filter(fn: (r) => r.interval == "{interval}")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> sort(columns: ["_time"])
        '''

        return self._format_dataframe(self.query_api.query_pandas(query))
