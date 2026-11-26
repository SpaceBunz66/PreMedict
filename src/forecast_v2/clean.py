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
    # Trước 2021 (DB prod) chỉ có dòng rải rác của 10 CSYT tỉnh 40 (285 lượt), lịch nghỉ calendar/ cũng bắt đầu từ 2021.
    raw = raw[raw.ds >= '2021-01-01']
    # Snapshot tên YYYY-MM-DD = ngày extract: ngày đó còn đang nạp (DB prod), sau đó chỉ có dòng 0 đệm hoặc ngày ra sai.
    extracted = pd.to_datetime(snapshot_dir.name, format='%Y-%m-%d', errors='coerce')
    return (raw if pd.isna(extracted) else raw[raw.ds < extracted]).reset_index(drop=True)


def merge_provinces(raw):
    """Cộng dồn một CSYT qua các ma_tinh: sáp nhập tỉnh 07/2025 chia bản ghi giữa mã cũ và mã mới.

    Trả về (daily, province); province[csyt_id] = ma_tinh của bản ghi có tong > 0 mới nhất, cũng là cột ma_tinh
    của daily (etl_holes cần tỉnh của mọi CSYT).
    """
    fields = [c for c in raw.columns if c not in KEYS]
    daily = raw.groupby(['csyt_id', 'ds'], as_index=False)[fields].sum()
    province = raw[raw.tong > 0].sort_values('ds').groupby('csyt_id').ma_tinh.last()
    daily['ma_tinh'] = daily.csyt_id.map(province)
    return daily, province


def reporting_share(daily):
    """Theo ngày: số CSYT báo cáo (tong > 0) / median cùng thứ của 4 tuần trước (NaN khi chưa có tham chiếu).

    So theo cùng thứ để Chủ nhật không bị coi là dừng. Chỉ dùng quá khứ nên không nhìn thấy tương lai.
    """
    rep = daily[daily.tong > 0].groupby('ds').size()
    rep = rep.reindex(pd.date_range(rep.index.min(), rep.index.max()), fill_value=0)
    typical = pd.concat([rep.shift(7 * k) for k in range(1, 5)], axis=1).median(axis=1)
    return rep / typical.where(typical > 0)


def last_data_day(daily, min_share=0.5):
    """Ngày cuối mà reporting_share >= min_share. Sau ngày đó chỉ còn dòng 0 đệm hoặc tháng nạp dở."""
    share = reporting_share(daily)
    return share.index[share >= min_share].max()


def etl_holes(daily, min_share):
    """Lỗ nạp theo tỉnh: DataFrame (csyt_id, ds) của mọi CSYT trong tỉnh vào ngày số CSYT báo cáo (tong > 0) của tỉnh
    < min_share × max cùng thứ của 8 tuần trước. Chỉ dùng quá khứ. daily: mọi CSYT, có cột ma_tinh (merge_provinces).

    DB prod có lỗ cả tỉnh kéo dài ~4 tuần (07/2026, 28/09–19/10/2026): tham chiếu là max 8 tuần thay vì median 4 tuần
    như reporting_share để chính lỗ không kéo tham chiếu xuống. Ngày nghỉ, Tết vắng là thật: build_panel bỏ qua.
    shortcut: tỉnh của CSYT là mã mới nhất cho cả lịch sử; tách theo mã từng ngày nếu cần lỗ của tỉnh cũ trước 07/2025.
    """
    act = daily[daily.tong > 0]
    rep = act.groupby(['ds', 'ma_tinh']).size().unstack(fill_value=0)
    rep = rep.reindex(pd.date_range(rep.index.min(), rep.index.max()), fill_value=0)
    ref = pd.concat([rep.shift(7 * k) for k in range(1, 9)]).groupby(level=0).max()
    hole = (rep < min_share * ref).astype('float32')
    # T7, CN ít CSYT mở nên số đếm nhiễu (~27% tỉnh × ngày cuối tuần bị gắn nhầm): là lỗ khi T6 trước và T2 sau đều lỗ.
    hole.loc[hole.index.dayofweek >= 5] = np.nan
    hole = ((hole.ffill() == 1) & (hole.bfill() == 1)).rename_axis(index='ds', columns='ma_tinh').stack()
    hole = hole[hole].reset_index()[['ds', 'ma_tinh']]
    csyt = act.groupby('csyt_id').ma_tinh.first().reset_index()
    return hole.merge(csyt, on='ma_tinh')[['csyt_id', 'ds']]


def build_panel(daily, last_day, c, holes):
    """Lịch ngày đầy đủ cho mỗi CSYT, từ ngày đầu có tong > 0 tới last_day, kèm các cột:

    off: ngày nghỉ chính thức; tet: trong cửa sổ Tết; hole: lỗ nạp theo tỉnh (holes từ etl_holes), ngoài ngày nghỉ
    và Tết; ok: mask theo ngày (available_mask), False khi
      - ngày không nghỉ, CSYT thường mở vào thứ đó, nhưng tong = 0 (thiếu số liệu), hoặc
      - hole (số của ngày đó có thể nạp dở, kể cả khi tong > 0), hoặc
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
    panel = grid.merge(daily[['csyt_id', 'ds', *fields]], on=['csyt_id', 'ds'], how='left')
    panel[fields] = panel[fields].fillna(0).astype('float32')

    panel['off'] = panel.ds.isin(days_off(range(panel.ds.dt.year.min(), panel.ds.dt.year.max() + 1)))
    panel['tet'] = in_tet_window(panel.ds, c['tet_window'])
    keys = pd.MultiIndex.from_frame(panel[['csyt_id', 'ds']])
    panel['hole'] = keys.isin(pd.MultiIndex.from_frame(holes[['csyt_id', 'ds']])) & ~panel.off & ~panel.tet
    weekday = panel.ds.dt.dayofweek
    # shortcut: tỉ lệ mở cửa theo thứ tính trên cả lịch sử; đổi sang cửa sổ trượt nếu CSYT đổi lịch làm việc nhiều.
    open_share = ((panel.tong > 0).astype('float32').where(~panel.off & ~panel.hole)
                  .groupby([panel.csyt_id, weekday]).transform('mean'))
    missing = (panel.tong == 0) & ~panel.off & (open_share >= c['open_weekday_share'])
    panel['ok'] = ~missing & ~panel.hole & (panel.groupby('csyt_id').cumcount() >= c['rampup_days'])
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
