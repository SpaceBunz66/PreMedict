"""Khí tượng (plan §5): trọng số bán kính, giải tích lũy mưa, gộp ngày UTC+7, bỏ ô biển."""
import numpy as np
import pandas as pd
import pytest

from forecast_v2 import weather


def test_cell_weights():
    fac = pd.DataFrame({'latitude': [21.0, 21.05, 21.0], 'longitude': [105.8, 105.85, 105.85]},
                       index=pd.Index(['center', 'corner', 'edge'], name='csyt_id'))
    w = weather.cell_weights(fac, [3, 12])
    assert np.allclose(w.groupby(['csyt_id', 'radius_km']).weight.sum(), 1)
    get = lambda c, r: w[(w.csyt_id == c) & (w.radius_km == r)].set_index(['lat', 'lon']).weight.round(3).to_dict()
    assert get('center', 3) == {(21.0, 105.8): 1.0}  # hình tròn 3 km nằm gọn trong ô ~11 km
    assert set(get('corner', 3).values()) == {0.25}  # tại góc chung của 4 ô: chia đều
    assert sorted(get('edge', 3).values()) == [0.5, 0.5]
    assert len(get('center', 12)) == 9  # 12 km phủ sang các ô xung quanh


def _accumulate(rain_mm, time):
    """Tạo tp kiểu ERA5-Land (m) từ lượng mưa từng giờ: cộng dồn từ 00 UTC, giá trị 00 UTC = tổng ngày trước."""
    acc, total = [], 0.0
    for r, t in zip(rain_mm, time):
        total = r if t.hour == 1 else total + r
        acc.append(total / 1000)
    return np.array(acc)


def test_daily_utc7_and_deaccumulation():
    time = pd.date_range('2025-06-01 00:00', '2025-06-04 00:00', freq='h')
    rain = np.zeros(len(time))
    rain[time == '2025-06-02 17:00'] = 5  # mưa (16h, 17h] UTC = 23h–24h ngày 02/06 giờ VN
    rain[time == '2025-06-02 18:00'] = 3  # (17h, 18h] UTC = 0h–1h ngày 03/06 giờ VN
    rain[time.hour == 1] += 0.5  # mỗi ngày có mưa lúc 0h–1h UTC: kiểm tra mốc reset cộng dồn
    t2m = np.full(len(time), 300.0)
    t2m[time == '2025-06-02 17:00'] = 310.0  # 0h ngày 03/06 giờ VN
    tp = _accumulate(rain, time)

    assert np.allclose(weather.hourly_precip(tp, time)[1:], rain[1:])
    d = weather.daily(time, t2m[:, None], t2m[:, None], tp[:, None])
    # chỉ còn ngày VN đủ 24 giờ: 02/06 (17h UTC 01/06 -> 17h UTC 02/06) và 03/06
    assert d['rain_mm'].index.strftime('%m-%d').tolist() == ['06-02', '06-03']
    assert d['rain_mm'][0].round(6).tolist() == [5.5, 3.5]
    assert d['t_max'][0].round(2).tolist() == [26.85, 36.85]
    assert np.allclose(d['rh_mean'][0], 100)


def test_derived_runs_and_rolling():
    th = {'cold': 15, 'severe_cold': 13, 'hot': 35, 'very_hot': 37, 'heavy_rain': 50}
    w = pd.DataFrame({'csyt_id': 'a', 'radius_km': 6, 'ds': pd.date_range('2025-06-01', periods=6),
                      't_mean': 30.0, 't_max': [34, 36, 36, 34, 36, 38.0], 't_min': 25.0, 'rh_mean': 80.0,
                      'rain_mm': [0, 10, 0, 60, 0, 5.0], 'clim_t_mean': 29.0, 'clim_rh_mean': 82.0})
    d = weather.derived(w, th)
    assert d.hot_days.tolist() == [0, 1, 2, 0, 1, 2]  # số ngày nắng nóng liên tiếp
    assert d.very_hot_days.tolist() == [0, 0, 0, 0, 0, 1]
    assert d.rain_3d.tolist()[2:] == [10, 70, 60, 65]
    assert d.t_anom.iloc[0] == 1 and d.heavy_rain.tolist() == [False, False, False, True, False, False]


def test_by_radius_drops_sea_cells():
    days = pd.date_range('2025-06-01', periods=2)
    cell_daily = pd.DataFrame([[20.0, np.nan], [22.0, np.nan]], index=days)  # ô thứ 2 là biển
    cells = pd.DataFrame({'lat': [21.0, 21.1], 'lon': [105.8, 105.8]})
    weights = pd.DataFrame({'csyt_id': 'a', 'radius_km': 6, 'lat': [21.0, 21.1], 'lon': [105.8, 105.8], 'weight': [0.3, 0.7]})
    out = weather.by_radius(cell_daily, cells, weights)
    assert out[('a', 6)].tolist() == pytest.approx([20.0, 22.0])


def test_existing_area_reuses_box_containing_all_cells(tmp_path, monkeypatch):
    monkeypatch.setattr(weather, 'WEATHER_DIR', tmp_path)
    (tmp_path / '22.8_103_8.3_109.7').mkdir()
    inside = pd.DataFrame({'lat': [8.3, 22.8], 'lon': [103.0, 109.7]})  # tâm ô đúng trên biên vẫn có trong file
    assert weather.existing_area(inside) == [22.8, 103.0, 8.3, 109.7]
    assert weather.existing_area(inside.assign(lat=[8.3, 22.9])) is None
