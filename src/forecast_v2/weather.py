"""P3: khí tượng ERA5-Land theo bán kính quanh CSYT (plan §P3).

Lưới ERA5-Land 0,1°, tâm ô tại bội số của 0,1°. Biến theo giờ (UTC): t2m, d2m (K), tp (m, cộng dồn từ 00 UTC;
giá trị lúc 00 UTC là tổng của cả ngày hôm trước).

Tải: mỗi tháng một file theo giờ cho khung bao các CSYT; khí hậu nền lấy từ ERA5-Land monthly means.
File thô để trong data/weather/<khung>/ (dùng chung giữa các snapshot), tải lại được và tự bỏ qua file đã có.
"""
import math
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from forecast_v2.extract import DATA_DIR

RES = 0.1  # độ
KM_PER_DEG = 111.32
LOCAL = pd.Timedelta(hours=7)


def cell_weights(fac, radii_km, step_km=0.5):
    """Trọng số ô lưới cho mỗi (CSYT, bán kính): rải điểm cách nhau step_km trong hình tròn, trọng số = tỉ lệ điểm rơi vào ô.

    Điểm đặt ở tâm các ô vuông step_km (không có điểm ở đúng tâm hình tròn) để CSYT nằm trên ranh giới ô
    được chia đều, không bị phép làm tròn kéo lệch về một phía.
    fac: index csyt_id, cột latitude, longitude. Trả về DataFrame [csyt_id, radius_km, lat, lon, weight].
    """
    out = []
    for r in radii_km:
        g = np.arange(-r + step_km / 2, r, step_km)
        dx, dy = np.meshgrid(g, g)
        inside = dx ** 2 + dy ** 2 <= r ** 2
        dx, dy = dx[inside], dy[inside]
        for csyt, lat, lon in zip(fac.index, fac.latitude, fac.longitude):
            plat = lat + dy / KM_PER_DEG
            plon = lon + dx / (KM_PER_DEG * np.cos(np.radians(lat)))
            cells = pd.DataFrame({'lat': np.round(plat / RES).astype(int), 'lon': np.round(plon / RES).astype(int)})
            w = cells.value_counts(normalize=True).rename('weight').reset_index()
            out.append(w.assign(csyt_id=csyt, radius_km=r))
    w = pd.concat(out, ignore_index=True)
    w[['lat', 'lon']] = (w[['lat', 'lon']] * RES).round(1)  # tâm ô, độ
    return w[['csyt_id', 'radius_km', 'lat', 'lon', 'weight']]


def rh_magnus(t_c, td_c):
    """Độ ẩm tương đối (%) từ nhiệt độ và điểm sương (°C), công thức Magnus (hệ số Sonntag 1990)."""
    es = lambda x: np.exp(17.62 * x / (243.12 + x))
    return 100 * es(td_c) / es(t_c)


def hourly_precip(tp, time):
    """Giải tích lũy tp (m, cộng dồn từ 00 UTC) thành lượng mưa từng giờ (mm), theo trục 0 (thời gian).

    Giá trị lúc 01 UTC là chính nó; các giờ khác (kể cả 00 UTC của ngày sau) = hiệu với giờ trước.
    """
    hours = pd.DatetimeIndex(time).hour
    prev = np.concatenate([np.full_like(tp[:1], np.nan), tp[:-1]])
    contiguous = np.r_[False, np.diff(pd.DatetimeIndex(time).asi8) == 3_600_000_000_000]
    p = np.where((hours == 1)[(...,) + (None,) * (tp.ndim - 1)], tp, tp - prev)
    p[~contiguous & (hours != 1)] = np.nan  # thiếu giờ trước: không giải được
    return np.clip(p, 0, None) * 1000  # sai số số học có thể âm rất nhỏ


def daily(time, t2m, d2m, tp):
    """Tổng hợp theo ngày giờ địa phương UTC+7 cho mảng (giờ, ô...). Chỉ giữ ngày đủ 24 giờ.

    Nhiệt độ/điểm sương là giá trị tức thời: thuộc ngày địa phương của chính thời điểm đó.
    Mưa ghi tại t là lượng mưa của giờ (t-1h, t]: thuộc ngày địa phương của đầu khoảng.
    Trả về dict biến -> DataFrame (index ngày, cột = ô).
    """
    time = pd.DatetimeIndex(time)
    flat = lambda a: a.reshape(len(time), -1)
    t_c, td_c = flat(t2m) - 273.15, flat(d2m) - 273.15
    inst = pd.DataFrame(t_c, index=(time + LOCAL).floor('D'))
    rh = pd.DataFrame(rh_magnus(t_c, td_c), index=inst.index)
    rain = pd.DataFrame(flat(hourly_precip(tp, time)), index=(time + LOCAL - pd.Timedelta(hours=1)).floor('D'))
    def full(df, how):  # ô thiếu giờ thành NaN; bỏ ngày không ô nào đủ 24 giờ (đầu/cuối file)
        g = df.groupby(level=0)
        return g.agg(how).where(g.count() == 24).dropna(how='all')

    return {'t_mean': full(inst, 'mean'), 't_max': full(inst, 'max'), 't_min': full(inst, 'min'),
            'rh_mean': full(rh, 'mean'), 'rain_mm': full(rain, 'sum')}


def by_radius(cell_daily, cells, weights):
    """Trung bình có trọng số theo bán kính. cell_daily: DataFrame (ngày × ô) của một biến; cells: DataFrame [lat, lon]
    theo thứ tự cột. Ô không có dữ liệu (biển, NaN) bị bỏ và trọng số được chuẩn hóa lại.
    Trả về DataFrame (ngày × (csyt_id, radius_km)).
    """
    col = pd.Series(range(len(cells)), index=pd.MultiIndex.from_arrays([cells.lat.round(1), cells.lon.round(1)]))
    row = col.reindex(pd.MultiIndex.from_arrays([weights.lat.round(1), weights.lon.round(1)]))
    if row.isna().any():
        raise ValueError('Có ô trong weights không nằm trong cells')
    keys = pd.MultiIndex.from_frame(weights[['csyt_id', 'radius_km']]).unique()
    w = np.zeros((len(cells), len(keys)))
    w[row.to_numpy(int), keys.get_indexer(pd.MultiIndex.from_frame(weights[['csyt_id', 'radius_km']]))] = weights.weight
    v = cell_daily.to_numpy(float)
    ok = ~np.isnan(v)
    den = ok @ w
    return pd.DataFrame((np.where(ok, v, 0) @ w) / np.where(den > 0, den, np.nan), index=cell_daily.index, columns=keys)


# ---- Tải và xử lý ----
WEATHER_DIR = DATA_DIR / 'weather'
ERA_VARS = ['2m_temperature', '2m_dewpoint_temperature', 'total_precipitation']
KEYS = ['csyt_id', 'radius_km']


def area(fac, margin):
    """Khung [Bắc, Tây, Nam, Đông] bao các CSYT cộng lề margin độ, làm tròn ra ngoài theo lưới 0,1°."""
    up, down = (lambda x: math.ceil(round(x * 10, 6)) / 10), (lambda x: math.floor(round(x * 10, 6)) / 10)
    return [up(fac.latitude.max() + margin), down(fac.longitude.min() - margin),
            down(fac.latitude.min() - margin), up(fac.longitude.max() + margin)]


def raw_dir(a):
    return WEATHER_DIR / '_'.join(f'{v:g}' for v in a)  # khung khác thì thư mục khác: không dùng nhầm file cũ


def existing_area(cells):
    """Khung đã tải trong data/weather/ chứa mọi tâm ô cần dùng (None nếu không có): tập CSYT đổi mà vẫn nằm trong khung
    cũ thì chỉ tải thêm tháng thiếu, không tải lại cả lịch sử cho khung mới."""
    for d in sorted(WEATHER_DIR.glob('*_*_*_*')):
        n, w, s, e = (float(v) for v in d.name.split('_'))
        if cells.lat.between(s, n).all() and cells.lon.between(w, e).all():
            return [n, w, s, e]


def _retrieve(dataset, request, target):
    """Tải vào thư mục tạm rồi mới đổi tên, nên file .nc đã có luôn là file tải xong. Gói zip (nhiều file nc) thì gộp."""
    import cdsapi  # chỉ cần khi tải
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        part = Path(tmp) / 'download'
        cdsapi.Client(quiet=True).retrieve(dataset, request, str(part))
        if zipfile.is_zipfile(part):
            with zipfile.ZipFile(part) as z:
                names = z.namelist()
                z.extractall(tmp)
            datasets = []
            for n in names:
                with xr.open_dataset(Path(tmp) / n) as d:
                    datasets.append(d.load())
            part = Path(tmp) / 'merged.nc'
            xr.merge(datasets, compat='override').to_netcdf(part)
        part.replace(target)


def download(a, c):
    """Tải các tháng còn thiếu và file khí hậu nền, chạy song song c['parallel'] request. Lỗi thì chạy lại để tải tiếp."""
    d = raw_dir(a)
    d.mkdir(parents=True, exist_ok=True)
    common = {'variable': ERA_VARS, 'area': a, 'data_format': 'netcdf', 'download_format': 'unarchived'}
    jobs, ready = {}, {}
    # ERA5-Land trễ ~5 ngày: tháng hiện tại chỉ xin tới hôm nay - 6 ngày; file tải trước cuối tháng + 10 ngày là tháng
    # chưa trọn, chạy lại sẽ tải lại.
    available = pd.Timestamp.today().normalize() - pd.Timedelta(days=6)
    for m in pd.period_range(c['start'], c['end'], freq='M'):
        last = min(m.days_in_month, (available - m.start_time).days + 1)
        if last < 1:
            raise ValueError(f'weather.end = {c["end"]}: tháng {m} chưa có số liệu ERA5-Land')
        jobs[d / f'hourly_{m}.nc'] = ('reanalysis-era5-land', common | {
            'year': str(m.year), 'month': f'{m.month:02d}', 'day': [f'{x:02d}' for x in range(1, last + 1)],
            'time': [f'{h:02d}:00' for h in range(24)]})
        ready[d / f'hourly_{m}.nc'] = (m.end_time + pd.Timedelta(days=10)).timestamp()
    y0, y1 = c['clim_years']
    jobs[d / f'monthly_{y0}-{y1}.nc'] = ('reanalysis-era5-land-monthly-means', common | {
        'product_type': ['monthly_averaged_reanalysis'], 'year': [str(y) for y in range(y0, y1 + 1)],
        'month': [f'{x:02d}' for x in range(1, 13)], 'time': ['00:00']})
    jobs = {t: j for t, j in jobs.items() if not t.exists() or t.stat().st_mtime < ready.get(t, 0)}
    print(f'Khung {a}: cần tải {len(jobs)} file vào {d}', flush=True)
    failed = []
    with ThreadPoolExecutor(c['parallel']) as ex:
        futures = {ex.submit(_retrieve, *job, target): target for target, job in jobs.items()}
        for f in as_completed(futures):
            name = futures[f].name
            try:
                f.result()
                print('xong', name, flush=True)
            except Exception as e:  # lỗi mạng/hàng đợi CDS: ghi lại, chạy lại lệnh sẽ tải tiếp
                failed.append(name)
                print('LỖI', name, ' | '.join(str(e).splitlines()[:3])[:400], flush=True)  # dòng 2–3 có lý do (vd. licence)
    if failed:
        raise RuntimeError(f'{len(failed)} file tải lỗi ({", ".join(sorted(failed))}): chạy lại lệnh weather để tải tiếp')
    return d


def _points(ds, cells):
    """Chuỗi tại tâm các ô cần dùng, dims (time, cell). Ô ngoài file (khung tải khác khung cần) thì báo lỗi."""
    t = 'valid_time' if 'valid_time' in ds.dims else 'time'
    p = ds[['t2m', 'd2m', 'tp']].sel(latitude=xr.DataArray(cells.lat.to_numpy(), dims='cell'),
                                      longitude=xr.DataArray(cells.lon.to_numpy(), dims='cell'),
                                      method='nearest', tolerance=0.01)
    p = p.rename({t: 'time'}).transpose('time', 'cell', ...).load()
    if p.t2m.ndim != 2:
        raise ValueError(f'File ERA5-Land có chiều lạ: {dict(p.sizes)}')
    return p


def observed(d, months, weights, cells):
    """Biến ngày (giờ VN) theo (CSYT, bán kính). Ghép 24 giờ cuối của tháng trước để ngày đầu tháng đủ 24 giờ."""
    parts, prev = [], None
    for m in months:
        with xr.open_dataset(d / f'hourly_{m}.nc') as ds:
            cur = _points(ds, cells)
        both = cur if prev is None else xr.concat([prev.isel(time=slice(-24, None)), cur], dim='time')
        prev = cur
        day = daily(both.time.values, both.t2m.values, both.d2m.values, both.tp.values)
        parts.append(pd.concat({v: by_radius(df[df.index.to_period('M') == m], cells, weights).stack(KEYS, future_stack=True)
                                for v, df in day.items()}, axis=1))
    out = pd.concat(parts)
    out.index.names = ['ds', *KEYS]
    return out.reset_index()


def climatology(path, weights, cells):
    """Khí hậu nền theo ngày trong năm: trung bình tháng (monthly means) đặt ở ngày 15, nội suy tuần hoàn ra 366 ngày,
    tương đương làm mượt khoảng ±15 ngày. rh tính từ T và Td trung bình tháng nên chỉ là xấp xỉ."""
    with xr.open_dataset(path) as ds:
        p = _points(ds, cells)
    month = pd.DatetimeIndex(p.time.values).month
    mean = lambda a: pd.DataFrame(a).groupby(month).mean()
    t, td, rain = mean(p.t2m.values - 273.15), mean(p.d2m.values - 273.15), mean(p.tp.values * 1000)  # tp: m/ngày
    mid = pd.to_datetime([f'2001-{m:02d}-15' for m in range(1, 13)]).dayofyear.to_numpy()
    doy = np.arange(1, 367)
    interp = lambda df: pd.DataFrame({k: np.interp(doy, mid, df[k], period=365) for k in df}, index=pd.Index(doy, name='doy'))
    clim = {'clim_t_mean': interp(t), 'clim_rh_mean': interp(rh_magnus(t, td)), 'clim_rain_mm': interp(rain)}
    return pd.concat({k: by_radius(v, cells, weights).stack(KEYS, future_stack=True) for k, v in clim.items()},
                     axis=1).reset_index()


def derived(w, th):
    """Biến dẫn xuất theo (CSYT, bán kính), chỉ nhìn về quá khứ. Độ trễ quan trắc (5 ngày) áp ở P5."""
    w = w.sort_values([*KEYS, 'ds']).reset_index(drop=True)
    g = w.groupby(KEYS)
    for n in (3, 7, 14, 28):
        w[f'rain_{n}d'] = g.rain_mm.transform(lambda s: s.rolling(n).sum())
    for n in (3, 7):
        w[f't_mean_{n}d'] = g.t_mean.transform(lambda s: s.rolling(n).mean())
    w['t_anom'], w['rh_anom'] = w.t_mean - w.clim_t_mean, w.rh_mean - w.clim_rh_mean
    w['dtr'] = w.t_max - w.t_min
    flags = {'cold': w.t_mean <= th['cold'], 'severe_cold': w.t_mean <= th['severe_cold'],
             'hot': w.t_max >= th['hot'], 'very_hot': w.t_max >= th['very_hot'], 'heavy_rain': w.rain_mm >= th['heavy_rain']}
    for name, f in flags.items():
        run = (~f).groupby([w.csyt_id, w.radius_km]).cumsum()
        w[name], w[f'{name}_days'] = f, f.astype(int).groupby([w.csyt_id, w.radius_km, run]).cumsum()  # số ngày liên tiếp
    return w


def run(snap, cfg):
    c = cfg['weather']
    fac = pd.read_parquet(snap / 'facilities_static.parquet')
    weights = cell_weights(fac, c['radii_km'])
    cells = weights[['lat', 'lon']].drop_duplicates().reset_index(drop=True)
    d = download(existing_area(cells) or area(fac, c['margin_deg']), c)
    obs = observed(d, pd.period_range(c['start'], c['end'], freq='M'), weights, cells)
    clim = climatology(d / f'monthly_{c["clim_years"][0]}-{c["clim_years"][1]}.nc', weights, cells)
    ratio = clim.clim_rain_mm.mean() / obs.rain_mm.mean()
    if not 0.33 < ratio < 3:  # monthly means phải là mm/ngày; lệch cả chục lần là sai đơn vị
        raise ValueError(f'Mưa khí hậu / mưa quan trắc = {ratio:.2f}: kiểm tra đơn vị tp của monthly means')
    w = derived(obs.assign(doy=obs.ds.dt.dayofyear).merge(clim, on=[*KEYS, 'doy'], how='left'), c['thresholds'])
    weights.to_parquet(snap / 'facility_cell_weights.parquet', index=False)
    clim.to_parquet(snap / 'weather_clim.parquet', index=False)
    w.to_parquet(snap / 'weather_daily.parquet', index=False)
    print(f'{w.csyt_id.nunique()} CSYT × {w.radius_km.nunique()} bán kính, {w.ds.min():%d/%m/%Y}–{w.ds.max():%d/%m/%Y}; '
          f'thiếu t_mean {w.t_mean.isna().mean():.2%}; mưa khí hậu/quan trắc {ratio:.2f}. Đã ghi {snap / "weather_daily.parquet"}')
