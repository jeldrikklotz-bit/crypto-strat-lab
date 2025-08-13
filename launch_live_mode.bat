chcp 65001
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set MPLBACKEND=TkAgg
python app.py --mode live --symbols BTCUSDT,ETHUSDT --interval 1m --limit 100 --paper true
PAUSE
