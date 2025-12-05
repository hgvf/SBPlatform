# TBD
Side Project

## Docker 建置
```shell=
# 建置環境
docker compose up -d --build

# 更新 algolab 容器類套件
make sync

# 進入 algolab bash 環境
make shell
```

## 簡易啟動 streamlit 網站
* Installation
```shell=
pip install streamlit
```
* Activate
```shell=
streamlit run mainUI_backtest.py
```

---
## 系統前置作業
* 爬取台/美股歷史股價
  ```shell=
  # 進入 docker container 
  make shell

  # 開始爬取
  cd src/algolab/coldDataCollection
  uv run python -W ignore main.py

  # 測試有無存入資料庫 (CLI)
  uv run python -W ignore test.py

  # 另一種測試方法
  # 瀏覽器輸入 https://localhost:8086
  # 參考 .env 檔案中的 influxDB 的 username & keyword 登入 influxDB UI 查看資料
  ```

## Future works
- [ ] Backtest 結合 strategy class (customize)
- [ ] telegrambot 遠端呼叫 backtest 操作 (maybe fastapi)
- [ ] DL 設計 & 訓練 strategy (by RL)
