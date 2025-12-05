# 預先蒐集台美股資料

此為系統建置時，預先蒐集股價 + 技術指標，並存入資料庫的 scripts
例行排程蒐集則不在此服務範圍

## Usage
* 同時爬取 TW + US 資料
```shell=
python main.py
```

* 測試 (爬取 2330.TW & TSLA.US)
```shell
python test.py
```
