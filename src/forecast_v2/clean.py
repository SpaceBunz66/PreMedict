"""P1: làm sạch chuỗi ngày theo CSYT (plan §P1, "Làm sạch")."""
import numpy as np
import pandas as pd

from forecast_v2.calendar_vn import days_off, in_tet_window

# Chỉ tiêu v1 (plan §P1): 5 model trực tiếp + 5 suy ra từ tong.
FIELDS = ['tong', 'luot_kham', 'ngoai_tru', 'noi_tru', 'cap_cuu', 'bhyt', 'luot_vao', 'luot_ra', 'dung_tuyen', 'trai_tuyen']
KEYS = ('ma_tinh', 'csyt_id', 'ds')


def load_raw(snapshot_dir, fields=FIELDS):
    files = sorted(snapshot_dir.glob('daily_*.parquet'))
    if not files:
        raise FileNotFoundError(f'Không có daily_*.parquet trong {snapshot_dir}: chạy extract trước')
    raw = pd.concat([pd.read_parquet(f, columns=['ma_tinh', 'csyt_id', 'time_id', *fields]) for f in files],
                    ignore_index=True)
    raw['ds'] = pd.to_datetime(raw.pop('time_id').astype(str), format='%Y%m%d')
    return raw


def merge_provinces(raw):
    """Cộng dồn một CSYT qua các ma_tinh: sáp nhập tỉnh 07/2025 chia bản ghi giữa mã cũ và mã mới.

    Trả về (daily, province); province[csyt_id] = ma_tinh của bản ghi có tong > 0 mới nhất.
    """
    fields = [c for c in raw.columns if c not in KEYS]
    daily = raw.groupby(['csyt_id', 'ds'], as_index=False)[fields].sum()
    province = raw[raw.tong > 0].sort_values('ds').groupby('csyt_id').ma_tinh.last()
    return daily, province


def last_data_day(daily, min_share=0.5):
    """Ngày cuối mà số CSYT báo cáo (tong > 0) còn >= min_share × median cùng thứ của 4 tuần trước.

    Sau ngày đó chỉ còn dòng 0 đệm hoặc tháng nạp dở. So theo cùng thứ để Chủ nhật không bị coi là dừng.
    """
    rep = daily[daily.tong > 0].groupby('ds').size()
    rep = rep.reindex(pd.date_range(rep.index.min(), rep.index.max()), fill_value=0)
    typical = pd.concat([rep.shift(7 * k) for k in range(1, 5)], axis=1).median(axis=1)
    return rep.index[(typical > 0) & (rep >= min_share * typical)].max()


def build_panel(daily, last_day, c):
    """Lịch ngày đầy đủ cho mỗi CSYT, từ ngày đầu có tong > 0 tới last_day, kèm các cột:

    off: ngày nghỉ chính thức; tet: trong cửa sổ Tết; ok: mask theo ngày (available_mask), False khi
      - ngày không nghỉ, CSYT thường mở vào thứ đó, nhưng tong = 0 (thiếu số liệu), hoặc
      - trong rampup_days ngày đầu CSYT lên hệ thống.
    Số 0 vào ngày nghỉ, hoặc vào thứ CSYT vốn đóng cửa, là 0 thật.
    """
    fields = [f for f in daily.columns if f not in KEYS]
    daily = daily[daily.ds <= last_day]
    first = daily[daily.tong > 0].groupby('csyt_id').ds.min()
    n = ((last_day - first).dt.days + 1).to_numpy()
    offset = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    grid = pd.DataFrame({'csyt_id': np.repeat(first.index.to_numpy(), n),
                         'ds': np.repeat(first.to_numpy(), n) + offset.astype('timedelta64[D]')})
    panel = grid.merge(daily, on=['csyt_id', 'ds'], how='left')
    panel[fields] = panel[fields].fillna(0).astype('float32')

    panel['off'] = panel.ds.isin(days_off(range(panel.ds.dt.year.min(), panel.ds.dt.year.max() + 1)))
    panel['tet'] = in_tet_window(panel.ds, c['tet_window'])
    weekday = panel.ds.dt.dayofweek
    # shortcut: tỉ lệ mở cửa theo thứ tính trên cả lịch sử; đổi sang cửa sổ trượt nếu CSYT đổi lịch làm việc nhiều.
    open_share = (panel.tong > 0).astype('float32').where(~panel.off).groupby([panel.csyt_id, weekday]).transform('mean')
    missing = (panel.tong == 0) & ~panel.off & (open_share >= c['open_weekday_share'])
    panel['ok'] = ~missing & (panel.groupby('csyt_id').cumcount() >= c['rampup_days'])
    return panel


def etl_breaks(panel, fields, drop):
    """(tháng × chỉ tiêu) gãy ETL toàn hệ thống: tỉ lệ chỉ tiêu/tong của tháng < (1 - drop) × median 12 tháng trước.

    Tỉ lệ so với tong nên không bị ảnh hưởng bởi số CSYT tăng dần hay tháng nạp dở. Tham chiếu là các tháng
    trước (không đặt ở giữa) để một đợt gãy kéo dài vài tháng không tự kéo tham chiếu xuống.
    shortcut: phát hiện ở cấp toàn hệ thống; tách theo tỉnh nếu có tỉnh gãy riêng.
    """
    total = panel.groupby(panel.ds.dt.to_period('M'))[['tong', *fields]].sum()
    ratio = total[fields].div(total.tong, axis=0)
    return ratio < (1 - drop) * ratio.rolling(12, min_periods=3).median().shift(1)


def _rolling_median(x, keys, window):
    r = x.groupby(keys).rolling(window, center=True, min_periods=window // 2 + 1).median()
    return r.droplevel(list(range(len(keys)))).reindex(x.index)


def outliers(panel, field, z):
    """Ngoại lai: |log1p(y) - median| > z × MAD chuẩn hóa, median/MAD trượt trên 9 tuần cùng thứ quanh ngày đó.

    Chỉ xét ngày ok, không nghỉ, ngoài Tết: số thấp vào ngày nghỉ/Tết là thật.
    """
    x = np.log1p(panel[field]).where(panel.ok & ~panel.off & ~panel.tet)
    keys = [panel.csyt_id, panel.ds.dt.dayofweek]
    med = _rolling_median(x, keys, 9)
    dev = (x - med).abs()
    scale = np.maximum(_rolling_median(dev, keys, 9) / 0.6745, 0.1)  # sàn 0.1 (≈10%): chuỗi phẳng có MAD = 0
    return dev > z * scale
