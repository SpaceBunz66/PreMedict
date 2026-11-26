"""P0: trích dữ liệu từ DB ra Parquet, chỉ đọc (plan §0, §P0).

Đầu ra: data/<YYYY-MM-DD>/{daily_<năm>.parquet, facilities.parquet, provinces_config.parquet}.
"""
import io
import os
from contextlib import closing
from datetime import date
from pathlib import Path

import pandas as pd
import psycopg2

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data'
DB_VARS = ('DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD')

# Copy từ read_pg.py:283 (code cũ): 20 chỉ tiêu, cũng là hợp đồng dữ liệu đầu ra.
PREDICT_FIELDS = ['tong', 'luot_ra', 'luot_vao', 'luot_kham', 'ngoai_tru', 'noi_tru', 'cap_cuu', 'tu_vong', 'bhyt',
                  'tn_kxd', 'tn_gt', 'tn_ld', 'tn_dn', 'tn_bg', 'tn_bl', 'tn_tt', 'tn_nd', 'tn_k',
                  'dung_tuyen', 'trai_tuyen']


def load_env(path=ROOT / '.env'):
    """Nạp file .env (dòng KEY=VALUE) vào biến môi trường; biến đã đặt sẵn thì giữ nguyên."""
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():  # -sig: Notepad/PowerShell 5 ghi kèm BOM
            key, sep, value = line.partition('=')
            if sep and not key.strip().startswith('#'):
                os.environ.setdefault(key.strip(), value.strip())


def use_env(name):
    """Dùng DB trong file name (trong forecast_v2/, vd. .env.prod) thay cho .env: bỏ biến DB đang có vì load_env chỉ setdefault."""
    if not (ROOT / name).exists():
        raise FileNotFoundError(f'Không có {ROOT / name}')
    for v in DB_VARS:
        os.environ.pop(v, None)
    load_env(ROOT / name)


def connect(timeout_ms=900_000):
    """Mở kết nối nằm trong một giao dịch READ ONLY (plan §0); dừng nếu server không xác nhận read-only.

    Cấu hình như read_pg.py:9-13 nhưng không có mật khẩu mặc định. Mọi truy vấn chạy trong giao dịch
    đầu tiên này: không commit, đóng kết nối là rollback.
    """
    load_env()
    missing = [v for v in DB_VARS if not os.environ.get(v)]
    if missing:
        raise RuntimeError(f'Thiếu {", ".join(missing)}: đặt biến môi trường hoặc tạo .env theo .env.example')
    conn = psycopg2.connect(host=os.environ['DB_HOST'], port=int(os.environ['DB_PORT']), dbname=os.environ['DB_NAME'],
                            user=os.environ['DB_USER'], password=os.environ['DB_PASSWORD'])
    try:
        # BEGIN ... READ ONLY cho từng giao dịch; REPEATABLE READ để mọi file của một snapshot nhất quán.
        conn.set_session(readonly=True, isolation_level='REPEATABLE READ')
        with conn.cursor() as cur:
            # SET LOCAL thay vì SET cấp phiên: không lan sang client khác nếu DB đứng sau connection pooler.
            cur.execute('SET LOCAL statement_timeout = %s', (timeout_ms,))
            cur.execute('SHOW transaction_read_only')
            if cur.fetchone()[0] != 'on':
                raise RuntimeError('Kết nối DB không ở chế độ read-only: dừng (plan §0)')
    except BaseException:
        conn.close()
        raise
    return conn


def _frame(cur):
    return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


def run(snapshot=None):
    out = DATA_DIR / (snapshot or date.today().isoformat())
    out.mkdir(parents=True, exist_ok=True)
    sums = ', '.join(f'COALESCE(SUM({f}), 0) AS {f}' for f in PREDICT_FIELDS)
    # Cột nguồn kiểu numeric: float32 giữ chính xác số đếm tới 16 triệu và không làm tròn ngầm số lẻ.
    dtypes = {'ma_tinh': str, 'csyt_id': str, 'time_id': 'int32'} | {f: 'float32' for f in PREDICT_FIELDS}
    # DB prod: GROUP BY một năm mất ~5 phút (2023), năm 2025 lớn gấp ~6 lần nên 15 phút mặc định không đủ.
    with closing(connect(timeout_ms=3_600_000)) as conn, conn.cursor() as cur:
        cur.execute('SELECT min(time_id), max(time_id) FROM hoc.dtm_hoc_kcb_theo_ngay WHERE csyt_id IS NOT NULL')
        first, last = (int(v) for v in cur.fetchone())
        for year in range(first // 10000, last // 10000 + 1):
            # Logic gom theo ngày của get_datamart_by_csyt (read_pg.py:317-334), mở rộng cho mọi CSYT và giữ ma_tinh.
            # Không lọc time_id < hôm nay: clean.py tự cắt đoạn đệm 0. COPY không nhận tham số; year là int nên an toàn.
            buf = io.StringIO()
            cur.copy_expert(f"""COPY (
                SELECT ma_tinh, csyt_id, time_id, {sums}
                FROM hoc.dtm_hoc_kcb_theo_ngay
                WHERE csyt_id IS NOT NULL AND time_id >= {year}0101 AND time_id < {year + 1}0101
                GROUP BY ma_tinh, csyt_id, time_id
            ) TO STDOUT WITH CSV HEADER""", buf)
            buf.seek(0)
            df = pd.read_csv(buf, dtype=dtypes)
            df.to_parquet(out / f'daily_{year}.parquet', index=False)
            print(f'daily {year}: {len(df):,} dòng, {int((df[PREDICT_FIELDS] % 1 != 0).to_numpy().sum())} giá trị lẻ')

        # Không lấy email, điện thoại, người đại diện (plan §P0).
        cur.execute('SELECT ma_bv, ten_bv, ma_tinh, ma_xa, tuyen_bv, hang_bv, loai_bv, latitude, longitude '
                    'FROM hoc.dm_coso_kcb')
        fac = _frame(cur)
        fac[['latitude', 'longitude']] = fac[['latitude', 'longitude']].apply(pd.to_numeric, errors='coerce')
        text = [c for c in fac.columns if c not in ('latitude', 'longitude')]
        fac.astype({c: 'string' for c in text}).to_parquet(out / 'facilities.parquet', index=False)
        print(f'facilities: {len(fac):,} dòng')

        cur.execute('SELECT ma_tinh, trang_thai FROM hoc.config_tinh_dtm_kcb')
        _frame(cur).astype({'ma_tinh': 'string'}).to_parquet(out / 'provinces_config.parquet', index=False)
    print(f'Đã ghi {out}')
    return out
