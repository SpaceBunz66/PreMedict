"""Chuẩn hóa hạng (giá trị thật gặp trong dm_coso_kcb) và bán kính theo tuyến (plan §P2)."""
import pandas as pd

from forecast_v2 import facilities

C = {'large_tuyen': ['1', '2'], 'radius_km': {'large': 12, 'small': 6}}


def test_norm_hang():
    raw = pd.Series(['4', ' 3', 'Hạng 3', 'Hạng 2', 'ĐB', 'éB', 'CH', 'ch', 'Chưa xếp hạng', '0', '82', '-4', '', None])
    assert facilities.norm_hang(raw).tolist() == ['4', '3', '3', '2', 'DB', 'DB', 'CH', 'CH', 'CH',
                                                  'unknown', 'unknown', 'unknown', 'unknown', 'unknown']


def test_static_table_radius():
    report = pd.DataFrame({'pass_relaxed': [True, True, True, False], 'pass_strict': [True, False, True, False],
                           'tuyen_bv': ['1', '2 ', '7', '1'], 'hang_bv': ['ĐB', '1', None, '2'],
                           'ma_tinh': '24', 'latitude': 21.0, 'longitude': 106.0},
                          index=pd.Index(['a', 'b', 'c', 'd'], name='csyt_id'))
    t = facilities.static_table(report, C)
    assert t.index.tolist() == ['a', 'b', 'c']  # chỉ tầng nới
    assert t.radius_km.tolist() == [12, 12, 6]
    assert t.hang.tolist() == ['DB', '1', 'unknown']
    assert t.strict.tolist() == [True, False, True]
