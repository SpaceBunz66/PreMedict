"""Làm sạch: cộng dồn CSYT bị tách mã tỉnh, bỏ dòng 0 đệm, mask ngày thiếu, phát hiện tháng gãy ETL (plan §5)."""
import numpy as np
import pandas as pd

from forecast_v2 import clean, quality
from forecast_v2.calendar_vn import days_off, in_tet_window

C = {'open_weekday_share': 0.5, 'rampup_days': 28, 'tet_window': [-7, 10]}
DAYS = pd.date_range('2025-03-01', '2025-06-30')
LAST = pd.Timestamp('2025-06-15')


def make_raw():
    frames = []
    for i in range(10):
        until = LAST + pd.Timedelta(days=3) if i == 0 else LAST  # c0 nạp dở thêm 3 ngày
        frames.append(pd.DataFrame({'ma_tinh': '01', 'csyt_id': f'c{i}', 'ds': DAYS,
                                    'tong': np.where(DAYS <= until, 10.0, 0.0)}))  # sau đó là dòng 0 đệm
    raw = pd.concat(frames, ignore_index=True)
    # c1 đóng cửa Chủ nhật, thiếu số liệu thứ Tư 14/05
    raw.loc[(raw.csyt_id == 'c1') & ((raw.ds.dt.dayofweek == 6) | (raw.ds == '2025-05-14')), 'tong'] = 0
    # c2 chuyển sang mã tỉnh 99: 01–10/05 chia đôi giữa hai mã, từ 11/05 chỉ còn mã 99
    c2 = raw.csyt_id == 'c2'
    half = c2 & raw.ds.between('2025-05-01', '2025-05-10')
    raw.loc[half, 'tong'] /= 2
    raw.loc[c2 & (raw.ds > '2025-05-10'), 'ma_tinh'] = '99'
    return pd.concat([raw, raw[half].assign(ma_tinh='99')], ignore_index=True)


def test_clean_pipeline():
    daily, province = clean.merge_provinces(make_raw())
    assert province['c2'] == '99'
    c2 = daily[daily.csyt_id == 'c2'].set_index('ds').tong
    assert (c2[:LAST] == 10).all()

    last = clean.last_data_day(daily)
    assert last == LAST

    panel = clean.build_panel(daily, last, C, clean.etl_holes(daily, 0.5)).set_index(['csyt_id', 'ds'])
    ok = lambda c, d: panel.loc[(c, pd.Timestamp(d)), 'ok']
    assert panel.index.get_level_values('ds').max() == LAST
    assert not ok('c1', '2025-05-14')  # thứ Tư thiếu số liệu
    assert ok('c1', '2025-05-18')  # Chủ nhật CSYT đóng cửa: 0 thật
    assert not ok('c3', '2025-03-05')  # trong 28 ngày đầu
    assert ok('c3', '2025-05-14')


def test_etl_breaks():
    p = pd.DataFrame({'csyt_id': 'a', 'ds': pd.date_range('2025-01-01', '2025-12-31'), 'tong': 100.0, 'noi_tru': 20.0})
    p.loc[p.ds.dt.month == 6, 'noi_tru'] = 1.0
    b = clean.etl_breaks(p, ['noi_tru'], 0.5)
    assert [str(m) for m in b.index[b.noi_tru]] == ['2025-06']


def test_load_raw_drops_before_2021_and_extract_day_and_after(tmp_path):
    raw = pd.DataFrame({'ma_tinh': '01', 'csyt_id': 'c0', 'time_id': [20201231, 20251123, 20251124, 20251125, 20251130],
                        **{f: 1.0 for f in clean.FIELDS}})
    for name in ('2025-11-25', 'devtest'):
        (tmp_path / name).mkdir()
        raw.to_parquet(tmp_path / name / 'daily_2025.parquet')
    assert clean.load_raw(tmp_path / '2025-11-25').ds.max() == pd.Timestamp('2025-11-24')
    assert len(clean.load_raw(tmp_path / 'devtest')) == 4  # tên không phải ngày: giữ nguyên


def test_etl_holes_mask_province_gap_but_not_tet():
    days = pd.date_range('2025-01-01', '2025-06-30')
    raw = pd.concat([pd.DataFrame({'ma_tinh': t, 'csyt_id': f'{t}-{i}', 'ds': days, 'tong': 10.0})
                     for t in ('01', '02') for i in range(10)], ignore_index=True)
    april = pd.date_range('2025-04-01', '2025-04-28')
    raw.loc[raw.ds.isin(april) & (raw.ma_tinh == '02') & (raw.csyt_id != '02-0'), 'tong'] = 0  # 4 tuần chỉ 1/10 CSYT
    raw.loc[in_tet_window(raw.ds, C['tet_window']) & (raw.csyt_id != '01-0'), 'tong'] = 0  # Tết vắng cả nước: thật
    raw.loc[(raw.ds == '2025-05-18') & (raw.ma_tinh == '01') & (raw.csyt_id > '01-2'), 'tong'] = 0  # CN ít CSYT mở: nhiễu
    for f in clean.FIELDS[1:]:
        raw[f] = raw.tong
    daily, _ = clean.merge_provinces(raw)
    p = clean.build_panel(daily, days[-1], C, clean.etl_holes(daily, 0.5))
    hole = p.set_index(['csyt_id', 'ds']).hole.unstack()
    off = april[april.isin(days_off([2025]))]  # 07/04/2025 Giỗ Tổ: ngày nghỉ, không phải lỗ
    assert hole.loc[['02-0', '02-5'], april.difference(off)].all().all()  # cả tuần thứ 4, cả CSYT vẫn có số
    assert not hole.loc[['02-0', '02-5']].drop(columns=april.difference(off)).any().any()  # Tết, ngày nghỉ, trước/sau lỗ
    assert not hole.loc[['01-0', '01-5']].any().any()
    assert not p[p.csyt_id == '02-0'].set_index('ds').ok[april.difference(off)].any()

    m = quality.metrics(p.assign(outlier=False), days[-1], ['tong'], {'window_days': 365})
    assert m.loc['02-5', 'max_workday_gap'] == 0 and m.loc['02-5', 'workday_coverage'] == 1
    assert m.loc['02-5', 'month_median_tong'] == m.loc['01-5', 'month_median_tong']
