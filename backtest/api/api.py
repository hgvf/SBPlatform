from fastapi import FastAPI, Depends, HTTPException, Header, Query
from pydantic_api import TickerRecord, OHLCVRecord
from enum import Enum
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
import os

from influxManager import InfluxManager

"""
TODO:
- `GET /v1/getMetric`: 取得指定個股技術指標數據
"""

app = FastAPI(title="TBD", version="1.0.0")
influxManager = InfluxManager.get_manager()

@app.get("/v1/getTicker", response_model=List[TickerRecord])
def getAllTicker(market: str,
                 time: datetime = None):
    """
    回傳指定股票市場的所有已收錄的 ticker list
    """

    res = influxManager.query_symbols_by_market(market, time)

    return res

@app.get("/v1/getOHLCV", response_model=List[OHLCVRecord])
def getOHLCV(symbol: str,
             interval: str,
             time: datetime):
    """
    回傳指定個股 + 時間範圍內的 OHLCV data
    """

    res_df = influxManager.query_ohlcv_by_symbols_time(symbol, interval, time)

    return res_df
