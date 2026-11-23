"""A7 (plan §P8): báo cáo tổng. `python -m forecast_v2.cli report` -> runs/<giờ>_report/report.md + tet_tuyen.png.

Run dùng cho báo cáo ghi ở mục `report` của configs/base.yaml. Phần tóm tắt (docs/report_summary.md) và phụ lục thiết
lập (docs/report_config.md) viết tay; phần kết quả tính lại từ forecasts.parquet của các run, mọi model chấm trên cùng
tập điểm mask = 1 (assert). Không có TFT (chưa làm) và RAM (chưa đo).
"""
import ast
import json
import re

import numpy as np
import pandas as pd

from forecast_v2 import metrics
from forecast_v2.extract import ROOT
from forecast_v2.models.lgbm import HOLIDAY, STATIC

SERIES = {'lgbm': '#2a78d6', 'tide': '#eb6834', 'prophet_prod': '#1baf7a'}  # 3 màu đầu bảng dataviz: đạt mọi cặp
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e6e5e0'


def table(df, digits=3):
    """DataFrame (index có tên) -> bảng Markdown; số thập phân dấu phẩy, số nguyên dấu chấm như phần viết tay."""
    def cell(v):
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            return f'{v:,}'.replace(',', '.')
        if isinstance(v, (float, np.floating)):
            return '–' if np.isnan(v) else f'{v:.{digits}f}'.replace('.', ',')
        return str(v)
    df = df.reset_index()
    rows = [list(map(str, df.columns)), ['---'] * df.shape[1], *([cell(v) for v in r] for r in df.itertuples(index=False))]
    return '\n'.join('| ' + ' | '.join(r) + ' |' for r in rows)


def seconds(run):
    """Giây/gốc theo model: backtest chỉ ghi số này trong metrics.md."""
    text = (ROOT / 'runs' / run / 'metrics.md').read_text(encoding='utf-8')
    return ast.literal_eval(re.search(r'Thời gian \(giây/gốc, trung bình\): (\{.*\})', text).group(1))


def group(k):
    """Nhóm feature LightGBM cho bảng importance (cùng cách chia với --drop)."""
    if k in HOLIDAY or k == 'last_year':
        return 'lễ, Tết, âm lịch'
    if k.startswith('w_'):
        return 'khí tượng quan trắc (w_*)'
    if k.startswith('clim_'):
        return 'khí hậu (clim_*)'
    if k in STATIC:
        return 'tĩnh (CSYT, chỉ tiêu)'
    if re.fullmatch(r'(lag|mean|median|std)\d+|trend7_28|days_since_nz|same_weekday', k):
        return 'lịch sử y (lag, rolling)'
    return 'lịch cơ bản và h'


def tet_chart(e, cutoff, path):
    """Tổng `tong` theo tuyến quanh Tết: thực tế và q50 từng model, cộng trên các điểm mask = 1."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    d = e[(e.cutoff == cutoff) & (e.field == 'tong') & e.model.isin(list(SERIES))]
    g = d.groupby(['tuyen', 'ds', 'model'], observed=True)[['y', 'q50']].sum()
    tet = d.ds[d.tet_offset == 0].iloc[0]
    tuyen = sorted(d.tuyen.unique())
    fig, axes = plt.subplots(int(np.ceil(len(tuyen) / 2)), 2, figsize=(11, 3.4 * np.ceil(len(tuyen) / 2)), sharex=True)
    for ax, t in zip(axes.flat, tuyen):
        x = g.xs(t, level='tuyen')
        y = x.xs('lgbm', level='model').y  # y như nhau ở mọi model (cùng tập điểm)
        ax.plot(y.index, y, color=INK, lw=2, label='thực tế')
        for mdl, c in SERIES.items():
            q = x.xs(mdl, level='model').q50
            ax.plot(q.index, q, color=c, lw=2, label=mdl)
        ax.axvline(tet, color=MUTED, lw=1, ls='--')
        ax.set_title(f'Tuyến {t} · {d[d.tuyen == t].unique_id.nunique()} CSYT', loc='left', fontsize=11, color=INK)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.yaxis.set_major_formatter(lambda v, _: f'{v:,.0f}'.replace(',', '.'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    for ax in axes.flat[len(tuyen):]:
        ax.set_visible(False)
    axes.flat[0].annotate('mùng 1 Tết', (tet, 1), xycoords=('data', 'axes fraction'), xytext=(4, -4),
                          textcoords='offset points', va='top', fontsize=9, color=MUTED)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='upper left', ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.01, 0.995))
    fig.suptitle(f'Lượt khám/ngày (tong) quanh Tết, gốc {cutoff:%d/%m/%Y}', x=0.01, y=1.04, ha='left', fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#fcfcfb')
    plt.close(fig)


def run(cfg):
    r = cfg['report']
    runs = [r['main'], *r['tide']]
    e = pd.concat([pd.read_parquet(ROOT / 'runs' / x / 'forecasts.parquet').assign(run=x) for x in runs],
                  ignore_index=True)
    e = e[(e.available_mask > 0) & e.q50.notna()]
    m = metrics.report(e)
    tot = m.loc['tổng'].droplevel('group')
    n = tot.n.unstack('model')
    assert (n.nunique(axis=1) == 1).all(), f'các model không chấm trên cùng tập điểm (thiếu run hoặc trùng run):\n{n}'

    # Bảng xếp hạng: sắp theo WAPE validation (tiêu chí chọn); cover80 sau A5 = hệ số fit trên val, chấm trên test
    val, test = tot.xs('val', level='set'), tot.xs('test', level='set')
    cal = metrics.score(metrics.calibrate(e, metrics.interval_factors(e[e.set == 'val'])))
    ncut = e.groupby('run').cutoff.nunique()
    sec = pd.concat([pd.Series(seconds(x)) * ncut[x] for x in runs]).groupby(level=0).sum()
    board = pd.DataFrame({'WAPE val': val.wape, 'WAPE test': test.wape, 'MASE test': test.mase, 'bias test': test.bias,
                          'pinball test': test.pinball, 'cover80 test': test.cover80,
                          'cover80 test sau A5': cal.xs('test', level='set').cover80,
                          'giây/gốc': (sec / e.groupby('model').cutoff.nunique()).round().astype(int)})
    board = board.loc[val.wape.sort_values().index]
    order = list(board.index)

    v = metrics.versus(e)
    v = pd.DataFrame({'số chuỗi': v.series, 'tỉ lệ chuỗi tốt hơn': v.better_share,
                      'p (Wilcoxon)': v.p_value.map(lambda p: '< 0,0001' if p < 1e-4 else f'{p:.4f}'.replace('.', ','))})

    s = m.xs('test', level='set')
    sl = s.wape.unstack('model').reindex(s.index.droplevel('model').unique())[order]
    sl.insert(0, 'n', s.n.groupby(level=['slice', 'group']).first())
    sl = sl.rename(index={'True': 'có', 'False': 'không'}, level='group')

    rows = {}
    for label, x in {'A3 đầy đủ (run chính)': r['main'], **r['ablation']}.items():
        a = pd.read_parquet(ROOT / 'runs' / x / 'metrics.parquet').xs('lgbm', level='model')
        assert a.loc[('tổng', '')].n.sort_index().equals(tot.xs('lgbm', level='model').n.sort_index()), x
        rows[label] = {'WAPE val': a.loc[('tổng', '', 'val')].wape, 'WAPE test': a.loc[('tổng', '', 'test')].wape,
                       'Tết ±21 test': a.loc[('Tết ±21 ngày', 'True', 'test')].wape}
    abl = pd.DataFrame(rows).T.rename_axis('biến thể')

    imp = pd.read_parquet(ROOT / 'runs' / r['main'] / 'importance.parquet')
    share = imp[(imp.model == 'lgbm') & (imp.set == 'val')].groupby('feature').share.mean().sort_values(ascending=False)
    groups = share.groupby(share.index.map(group)).sum().sort_values(ascending=False).rename('tỉ lệ gain').rename_axis('nhóm')

    art = max((ROOT / 'artifacts' / 'lgbm').glob('*/manifest.json')).parent
    man = json.loads((art / 'manifest.json').read_text(encoding='utf-8'))
    man['interval_factors (A5)'] = 'có' if (art / 'interval_factors_lgbm.csv').exists() else 'KHÔNG: q10/q90 chưa hiệu chỉnh'

    out = ROOT / 'runs' / f'{pd.Timestamp.now():%Y%m%d-%H%M%S}_report'
    out.mkdir(parents=True)
    cutoff = pd.Timestamp(r['tet_cutoff'])
    tet_chart(e, cutoff, out / 'tet_tuyen.png')
    doc = lambda name: re.sub(r'^(#+) ', r'#\1 ', (ROOT / 'docs' / name).read_text(encoding='utf-8'), flags=re.M)
    csyt = e.unique_id.str.split('|').str[0].nunique()
    md = [
        f'# Báo cáo forecast_v2: dự báo 30 ngày, {csyt} CSYT, {e.unique_id.nunique()} chuỗi', '',
        f'Sinh {pd.Timestamp.now():%d/%m/%Y %H:%M} bằng `python -m forecast_v2.cli report`. Run: '
        + ', '.join(f'`{x}`' for x in runs) + f'. Mọi model chấm trên cùng {n.loc["val"].iloc[0]:,} điểm validation và '
        f'{n.loc["test"].iloc[0]:,} điểm test (mask = 1).'.replace(',', '.'), '',
        doc('report_summary.md'),
        '## Bảng xếp hạng', '',
        'Sắp theo WAPE validation (tiêu chí chọn model); test chỉ để báo cáo. tide = cấu hình s2000 (A3). cover80 = tỉ lệ '
        'y nằm trong [q10, q90], mục tiêu 0,80; "sau A5" = hệ số khoảng tin cậy fit trên validation rồi áp cho test.', '',
        table(board), '',
        '## So với ProphetProd (Wilcoxon ghép cặp theo chuỗi)', '',
        'Tổng |sai số| của mỗi chuỗi trên các gốc của tập; "tỉ lệ chuỗi tốt hơn" = tỉ lệ chuỗi model có tổng sai số nhỏ '
        'hơn ProphetProd.', '',
        table(v), '',
        '## WAPE theo lát cắt (test)', '',
        'Tổng tỉnh × chỉ tiêu: cộng dự báo CSYT lên tỉnh (bottom-up) rồi mới tính sai số. Ngày lễ: WAPE lớn vì y nhỏ.', '',
        table(sl), '',
        f'## Tết 2026 theo tuyến (gốc {cutoff:%d/%m/%Y})', '',
        '![Tết theo tuyến](tet_tuyen.png)', '',
        'Cộng trên các CSYT của tuyến, chỉ các điểm mask = 1 (như nhau ở mọi đường). Đường đứt: mùng 1 Tết. Số liệu: dòng '
        '"Tết ±21 ngày" và "tuyến" ở bảng lát cắt.', '',
        '## Ablation nhóm feature LightGBM (A2)', '',
        table(abl), '',
        '## Feature importance LightGBM (tỉ lệ gain, trung bình các gốc validation)', '',
        table(groups.to_frame()), '',
        'Top 15 feature:', '',
        table(share.head(15).rename('tỉ lệ gain').rename_axis('feature').to_frame()), '',
        '## Model production (train-final)', '',
        f'`{art.relative_to(ROOT).as_posix()}`: 3 booster (q10, q50, q90), forecast.csv, importance.csv, config.yaml.', '',
        table(pd.Series({k: v if type(v) is int else str(v) for k, v in man.items()}, name='giá trị')
              .rename_axis('mục').to_frame()), '',
        '## Phụ lục', '',
        doc('report_config.md'),
    ]
    (out / 'report.md').write_text('\n'.join(md), encoding='utf-8')
    print(table(board))
    print(f'Đã ghi {out / "report.md"}')
