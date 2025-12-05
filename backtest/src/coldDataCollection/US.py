import pandas as pd
import requests
import os
from io import StringIO
from tqdm import tqdm

from metrics import TechnicalIndicatorCalculator
from utils import safe_history, process_ohlcv, process_metric, save2influxDB
from typing import List, Dict, Any

def get_all_us_stocks() -> List[str]:
    """
    爬取 NASDAQ, NYSE 的所有 ticker name
    """

    all_stocks = []
    
    # 方法1: 從各大交易所網站取得
    exchanges = {
        'NASDAQ': 'https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt',
        'NYSE': 'https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt'
    }
    
    for exchange, url in exchanges.items():
        try:
            response = requests.get(url)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text), sep='|')
                df = df[:-1]  # 移除最後一行（通常是檔案結束標記）
                
                if exchange == 'NASDAQ':
                    symbols = df['Symbol'].tolist()
                else:  # NYSE and others
                    symbols = df['ACT Symbol'].tolist()
                
                for symbol in symbols:
                    all_stocks.append({
                        'symbol': symbol,
                        'exchange': exchange
                    })
        except Exception as e:
            print(f"無法從 {exchange} 取得資料: {e}")
    
    return all_stocks

def crawl_US(payload: Dict[str, Any],
             interval: str = "1d",
             period: str = "max",
             start: str = None,
             end: str = None):
    """
    爬取每個美股市場的 OHLCV、技術面指標

    1. 先爬取所有的 ticker name
    2. 逐一爬取股價 + 計算技術面指標
    3. 存成一個大的 dataframe
    """

    # 技術指標 class
    metric_calculator = TechnicalIndicatorCalculator()

    # 使用方式
    us_stocks = get_all_us_stocks()
    print(f"總共取得 {len(us_stocks)} 支美股")

    tickers = [t['symbol'] for t in us_stocks]

    print("美股爬取中...")
    cnt = 0
    for t in tqdm(tickers, total=len(tickers)):
        try:
            df = safe_history(t, period=period, interval=interval, actions=True, start=start, end=end)
            df["Ticker"] = t

            # 計算技術指標，append to the dataframe "df"
            df = metric_calculator.add_indicators_to_dataframe(df)
            df = df.reset_index()
            df.rename(columns={"Date": "timestamp", "Adj Close": "adj_close"}, inplace=True)

            ohlcv = process_ohlcv(df, "US", interval, payload['OHLCV_FIELD'])
            metric = process_metric(df, "US", interval, payload['METRIC_FIELD'])

            if not os.path.exists('./data'):
                os.makedirs('./data')
                
            # merge dataframe
            out_df = pd.merge(ohlcv, metric, on="timestamp", how="inner")

            out_df.to_csv(f"./data/US/{t}.csv", index=False)

            # if cnt < 100:
            save2influxDB(ohlcv, payload["OHLCV_MEASUREMENT"], payload["OHLCV_FIELD"], payload)
            save2influxDB(metric, payload["METRIC_MEASUREMENT"], payload["METRIC_FIELD"], payload)
            
            cnt += 1
            
        except:
            continue
