"""P2: thuộc tính tĩnh và bán kính khí tượng của CSYT (plan §P2).

Quy mô log và tỉ lệ mở cửa cuối tuần tính ở P5 (theo từng fold, chỉ trên dữ liệu train); vùng thêm ở P4 cùng provinces.csv.
"""
import pandas as pd

HANG = ['DB', '1', '2', '3', '4', 'CH']


def norm_hang(s):
    """Chuẩn hóa hạng BV về DB/1/2/3/4/CH; giá trị thiếu hoặc lạ (0, 82, -4...) thành 'unknown'."""
    s = s.astype('string').str.strip().str.replace(r'^Hạng\s*', '', regex=True)
    s = s.replace({'ĐB': 'DB', 'éB': 'DB', 'ch': 'CH', 'Chưa xếp hạng': 'CH'})  # 'éB': ĐB bị lỗi mã hóa trong DB
    return s.where(s.isin(HANG), 'unknown').astype(str)


def static_table(report, c):
    """Một dòng cho mỗi CSYT tầng nới (chứa tập chặt, cột strict) để mở rộng tập train không phải chạy lại."""
    s = report[report.pass_relaxed]
    tuyen = s.tuyen_bv.astype('string').str.strip().fillna('unknown').astype(str)
    return pd.DataFrame({
        'ma_tinh': s.ma_tinh,
        'latitude': s.latitude,
        'longitude': s.longitude,
        'tuyen': tuyen,
        'hang': norm_hang(s.hang_bv),
        'radius_km': tuyen.isin(c['large_tuyen']).map({True: c['radius_km']['large'], False: c['radius_km']['small']}),
        'strict': s.pass_strict,
    }).rename_axis('csyt_id')


def run(snap, cfg):
    out = static_table(pd.read_parquet(snap / 'quality_report.parquet'), cfg['facilities'])
    out.to_parquet(snap / 'facilities_static.parquet')
    print(f'{len(out)} CSYT ({out.strict.sum()} tập chặt); bán kính: {out.radius_km.value_counts().to_dict()}; '
          f'hạng: {out.hang.value_counts().to_dict()}. Đã ghi {snap / "facilities_static.parquet"}')
