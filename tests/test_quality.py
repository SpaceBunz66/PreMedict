"""Mỗi tiêu chí chất lượng loại đúng CSYT trên dữ liệu tổng hợp (plan §5)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from forecast_v2 import clean, quality
from forecast_v2.calendar_vn import days_off

CFG = yaml.safe_load((Path(__file__).parents[1] / 'configs' / 'base.yaml').read_text(encoding='utf-8'))
END = pd.Timestamp('2026-06-07')  # Chủ nhật
DAYS = pd.date_range(end=END, periods=800)
BAD = {'gappy': 'coverage', 'gap': 'gap', 'small': 'volume', 'collapse': 'collapse', 'outliers': 'outliers',
       'shift': 'level_shift', 'stopped': 'active', 'young': 'history', 'nocoord': 'coords',
       'placeholder': 'coords', 'otherprov': 'coords'}


def tong(kind):
    """T2–T6: 40, T7: 20, CN: 5, ngày nghỉ chính thức: 0. Mỗi kind làm hỏng một tiêu chí."""
    wd = DAYS.dayofweek.to_numpy()
    off = DAYS.isin(days_off(range(2024, 2027)))
    y = np.select([off, wd < 5, wd == 5], [0, 40, 20], 5).astype(float)
    between = lambda a, b: (DAYS >= a) & (DAYS <= b)
    if kind == 'gappy':
        y[np.flatnonzero((wd < 5) & ~off)[::12]] = 0  # thiếu ~8% ngày làm việc, rải rác
    elif kind == 'gap':
        y[between('2025-11-03', '2025-11-11')] = 0  # 7 ngày làm việc liền
    elif kind == 'small':
        y /= 4
    elif kind == 'collapse':
        y[between('2025-10-01', '2025-10-31')] *= 0.2
    elif kind == 'outliers':
        y[DAYS.isin(pd.to_datetime(['2025-07-08', '2025-08-05', '2025-09-09', '2025-10-14', '2025-11-04',
                                    '2025-12-02', '2026-01-06', '2026-03-10', '2026-04-07', '2026-05-05']))] = 400
    elif kind == 'shift':
        y[DAYS >= '2025-12-01'] *= 2.5
    elif kind == 'stopped':
        y[DAYS > END - pd.Timedelta(days=30)] = 0
    elif kind == 'young':
        y[DAYS < END - pd.Timedelta(days=200)] = 0
    return np.round(y)


@pytest.fixture(scope='module')
def result():
    kinds = ['good', 'split', *BAD]
    pad = pd.date_range(END + pd.Timedelta(days=1), periods=20)  # dòng 0 đệm sau ngày cuối
    raw = pd.concat([pd.DataFrame({'ma_tinh': '24', 'csyt_id': k, 'ds': DAYS.append(pad),
                                   'tong': np.r_[tong('good' if k == 'split' else k), np.zeros(len(pad))]})
                     for k in kinds], ignore_index=True)
    for f in clean.FIELDS[1:]:
        raw[f] = raw.tong
    raw['trai_tuyen'] = 0.0
    # 'split' chuyển từ mã tỉnh cũ 34 sang 24; 25/06–05/07/2025 chia đôi giữa hai mã
    s = raw.csyt_id == 'split'
    both = s & raw.ds.between('2025-06-25', '2025-07-05')
    raw.loc[both, clean.FIELDS] /= 2
    raw.loc[s & (raw.ds < '2025-06-25'), 'ma_tinh'] = '34'
    raw = pd.concat([raw, raw[both].assign(ma_tinh='34')], ignore_index=True)

    i = np.arange(len(kinds))
    fac = pd.DataFrame({'ma_bv': kinds, 'ten_bv': kinds, 'ma_tinh': '24', 'tuyen_bv': '3', 'hang_bv': '3',
                        'latitude': 21.03 + 0.0011 * i, 'longitude': 106.07 + 0.0013 * i})  # cụm CSYT tỉnh 24
    fac.loc[fac.ma_bv == 'nocoord', 'latitude'] = np.nan
    fac.loc[fac.ma_bv == 'placeholder', ['latitude', 'longitude']] = [21.0, 106.0]
    fac.loc[fac.ma_bv == 'otherprov', 'ma_tinh'] = '01'  # khai tỉnh 01 nhưng nằm giữa cụm tỉnh 24
    return quality.evaluate(raw, fac, CFG)


def test_last_day_skips_padding(result):
    assert result[1] == END


def test_good_and_split_pass(result):
    m = result[0]
    assert m.loc['good', 'failed'] == '' and m.loc['good', 'pass_strict']
    assert m.loc['split', 'failed'] == '' and m.loc['split', 'ma_tinh'] == '24'


@pytest.mark.parametrize('kind,criterion', BAD.items())
def test_each_criterion_rejects(result, kind, criterion):
    m = result[0]
    assert criterion in m.loc[kind, 'failed'].split(','), m.loc[kind].to_dict()
    assert not m.loc[kind, 'pass_strict']


def test_relaxed_tier(result):
    assert result[0].loc['gappy', 'pass_relaxed']
