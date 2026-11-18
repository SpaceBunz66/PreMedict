"""Mọi kết nối của forecast_v2 đều read-only (plan §0). Chỉ kiểm tra bằng SHOW, không bao giờ thử ghi."""
import os

import psycopg2
import pytest

from forecast_v2 import extract


def test_connection_is_read_only():
    extract.load_env()
    if not all(os.environ.get(v) for v in extract.DB_VARS):
        pytest.skip('Chưa cấu hình DB (.env)')
    try:
        conn = extract.connect(timeout_ms=30_000)
    except psycopg2.OperationalError as e:
        pytest.skip(f'Không kết nối được DB: {e}')
    try:
        with conn.cursor() as cur:
            cur.execute('SHOW transaction_read_only')
            assert cur.fetchone()[0] == 'on'
    finally:
        conn.close()
