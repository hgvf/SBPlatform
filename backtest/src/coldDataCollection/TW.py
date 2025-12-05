import requests
import pandas as pd
import os
from tqdm import tqdm
from typing import List, Dict, Any

from metrics import TechnicalIndicatorCalculator
from utils import safe_history, process_ohlcv, process_metric, save2influxDB

def crawl_TW(payload: Dict[str, Any],
             interval: str = "1d",
             period: str = "max",
             start: str = None,
             end: str = None):
    """
    爬取每個台股市場的 OHLCV、技術面指標

    1. 先去 TWSE 爬取所有的 ticker name
    2. 逐一爬取股價 + 計算技術面指標
    3. 存成一個大的 dataframe
    """

    # 技術指標 class
    metric_calculator = TechnicalIndicatorCalculator()

    # 上市股票
    url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL"
    response = requests.get(url)
    result = response.json()

    tickers = [r[0] for r in result['data']]

    # 要爬取的個股代碼 (+".TW")
    tickers_toCrawl = [t+".TW" for t in tickers]

    print("台股爬取中...")
    cnt = 0
    for t in tqdm(tickers_toCrawl, total=len(tickers_toCrawl)):
        try:
            df = safe_history(t, period=period, interval=interval, actions=True, start=start, end=end)
            if not df.empty:   # 有些上櫃股票資料可能缺
                df["Ticker"] = t

                # 計算技術指標，append to the dataframe "df"
                df = metric_calculator.add_indicators_to_dataframe(df)
                df = df.reset_index()
                df.rename(columns={"Date": "timestamp", "Adj Close": "adj_close"}, inplace=True)

                ohlcv = process_ohlcv(df, "TW", interval, payload['OHLCV_FIELD'])
                metric = process_metric(df, "TW", interval, payload['METRIC_FIELD'])

                if not os.path.exists('./data'):
                    os.makedirs('./data')

                # merge dataframe
                out_df = pd.merge(ohlcv, metric, on="timestamp", how="inner")

                out_df.to_csv(f"./data/TW/{t}.csv", index=False)

                # if cnt < 100:
                save2influxDB(ohlcv, payload["OHLCV_MEASUREMENT"], payload["OHLCV_FIELD"], payload)
                save2influxDB(metric, payload["METRIC_MEASUREMENT"], payload["METRIC_FIELD"], payload)
                
                cnt += 1
                
            else:
                continue
        except:
            continue
