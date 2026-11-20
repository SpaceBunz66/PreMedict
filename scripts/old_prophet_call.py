"""Đối chiếu ProphetProd (plan §5): gọi test._build_csyt_forecast của code cũ ĐÚNG MỘT LẦN, chạy bằng .venv cũ.

An toàn (plan §0): psycopg2.connect bị thay bằng bản mở giao dịch READ ONLY + kiểm tra SHOW transaction_read_only = on
trước khi trả kết nối cho code cũ; không commit (đóng kết nối = rollback). Không sửa file nào của code cũ.
Ghi kết quả và dữ liệu code cũ đã đọc ra thư mục của script này.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(r'D:\VNPT\python_ai_predict-develop')
OUT = ROOT / "forecast_v2" / "runs" / "prophet_crosscheck"  # runs/ đã .gitignore
OUT.mkdir(parents=True, exist_ok=True)
MA_TINH, CSYT, PERIODS = '24', '24002', 30

for line in (ROOT / 'forecast_v2' / '.env').read_text(encoding='utf-8-sig').splitlines():  # read_pg đọc env lúc import
    if '=' in line and not line.lstrip().startswith('#'):
        k, v = line.split('=', 1)
        os.environ[k.strip()] = v.strip()

import psycopg2  # noqa: E402

_connect = psycopg2.connect
opened = []


def read_only_connect(*args, **kwargs):
    conn = _connect(*args, **kwargs)
    try:
        conn.set_session(readonly=True)
        cur = conn.cursor()
        cur.execute('SET LOCAL statement_timeout = 120000')
        cur.execute('SHOW transaction_read_only')
        if cur.fetchone()[0] != 'on':
            raise RuntimeError('Kết nối không read-only: dừng (plan §0)')
        cur.close()
    except BaseException:
        conn.close()
        raise
    opened.append(1)
    return conn


psycopg2.connect = read_only_connect
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('old_test', ROOT / 'test.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

captured = {}
_get = old.get_datamart_by_csyt


def capture(*args, **kwargs):
    df = _get(*args, **kwargs)
    captured['df'] = df
    return df


old.get_datamart_by_csyt = capture
res = old._build_csyt_forecast(MA_TINH, CSYT, PERIODS)
captured['df'].to_csv(OUT / 'old_input.csv', index=False)
(OUT / 'old_result.json').write_text(json.dumps(res, ensure_ascii=False, default=str), encoding='utf-8')
print(f'{len(opened)} kết nối, tất cả READ ONLY; last_data_date {res["last_data_date"]}; '
      f'{len(captured["df"])} dòng; train_info: { {k: v["status"] for k, v in res["train_info"].items()} }')
