chcp 65001
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python app.py --mode backtest --symbols BTCUSDT,ETHUSDT,SOLUSDT,LINKUSDT --interval 1h --limit 1000
PAUSE
