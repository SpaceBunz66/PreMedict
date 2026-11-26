"""python -m forecast_v2.cli extract | quality | facilities | weather | features [--snapshot YYYY-MM-DD] [--config configs/base.yaml]
       python -m forecast_v2.cli extract --env .env.devtest --snapshot devtest-2026-11-25   # DB khác .env (prod); tên
                                                         # không dạng YYYY-MM-DD nên các lệnh khác không tự chọn
       python -m forecast_v2.cli backtest --models snaive7,mean4 [--smoke] [--drop weather,clim]  # --drop: ablation lgbm
       python -m forecast_v2.cli backtest --models tide --set models.tide.max_steps=3000 --set backtest.test=[] --tag a3
       python -m forecast_v2.cli train-final --models lgbm
       python -m forecast_v2.cli train-far [--start YYYY-MM-DD]   # dự báo xa khi dữ liệu dừng trước start (mặc định hôm nay)
       python -m forecast_v2.cli tune [--trials 10] [--smoke]   # chạy lại = chạy tiếp study
       python -m forecast_v2.cli calibrate --models lgbm,tide   # A5: hệ số khoảng tin cậy từ run backtest mới nhất
       python -m forecast_v2.cli report                         # A7: báo cáo tổng từ các run trong base.yaml (report)
       python -m forecast_v2.cli evaluate --artifact artifacts/lgbm/20261123-165545 [--snapshot YYYY-MM-DD]
                                                                # chấm dự báo của artifact bằng số thật của snapshot"""
import argparse
import sys

import yaml

from forecast_v2 import backtest, extract, facilities, features, quality, weather
from forecast_v2.extract import DATA_DIR, ROOT


def main():
    for stream in (sys.stdout, sys.stderr):  # chuyển hướng (| tee, > log) trên Windows mặc định cp1252: lỗi tiếng Việt
        stream.reconfigure(encoding='utf-8')  # stderr: thanh tiến độ tqdm
    p = argparse.ArgumentParser(prog='forecast_v2')
    p.add_argument('command', choices=['extract', 'quality', 'facilities', 'weather', 'features', 'backtest',
                                        'train-final', 'train-far', 'tune', 'calibrate', 'report', 'evaluate'])
    p.add_argument('--snapshot', help='thư mục con của data/; mặc định: hôm nay (extract), mới nhất (quality)')
    p.add_argument('--config', default=ROOT / 'configs' / 'base.yaml')
    p.add_argument('--models', default='snaive7,mean4', help='backtest: tên model, cách nhau bởi dấu phẩy')
    p.add_argument('--smoke', action='store_true', help='backtest/tune: 20 CSYT, 2 chỉ tiêu, 1 gốc (backtest.smoke)')
    p.add_argument('--drop', default='', help='backtest: nhóm feature lgbm bỏ đi (holiday, weather, clim), ablation A2')
    p.add_argument('--set', action='append', default=[], metavar='KHÓA=GIÁ_TRỊ',
                   help='backtest (kèm --tag): ghi đè khóa có sẵn trong config, vd. models.tide.max_steps=3000 hoặc '
                        'backtest.test=[] (giá trị đọc như YAML)')
    p.add_argument('--tag', default='', help='backtest: hậu tố tên run; run có tag không bị calibrate/train-final chọn')
    p.add_argument('--trials', type=int, default=10, help='tune: số trial chạy thêm; 0 = chỉ in bảng hội tụ')
    p.add_argument('--start', help='train-far: ngày đầu tiên cần dự báo (YYYY-MM-DD), mặc định hôm nay')
    p.add_argument('--env', help='extract: file cấu hình DB trong forecast_v2/ (vd. .env.prod), mặc định .env')
    p.add_argument('--artifact', help='evaluate: thư mục artifact có forecast.csv, manifest.json (vd. artifacts/lgbm/<run>)')
    args = p.parse_args()
    if args.env and args.command != 'extract':
        p.error('--env chỉ dùng cho extract')
    if (args.command == 'evaluate') != bool(args.artifact):
        p.error('evaluate cần --artifact; --artifact chỉ dùng cho evaluate')
    if args.set and (args.command != 'backtest' or not args.tag):  # lệnh khác ghi đè file snapshot dùng chung
        p.error('--set chỉ dùng cho backtest, kèm --tag (run thử không bị calibrate/train-final chọn)')
    if args.command == 'extract':
        if args.env:
            extract.use_env(args.env)
        extract.run(args.snapshot)
        return
    if args.command == 'calibrate':
        backtest.calibration(args.models.split(','))
        return
    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    for kv in args.set:
        key, value = kv.split('=', 1)
        *path, last = key.split('.')
        d = cfg
        for k in path:
            d = d.get(k, {})
        if last not in d:  # gõ sai tên khóa: báo ngay, không chạy âm thầm với giá trị cũ
            p.error(f'--set: không có khóa {key} trong {args.config}')
        d[last] = yaml.safe_load(value)
    snap = DATA_DIR / args.snapshot if args.snapshot else max(DATA_DIR.glob('20??-??-??'))
    if args.command == 'backtest':
        cfg['features']['drop'] = [g for g in args.drop.split(',') if g]
        if cfg['features']['drop'] and 'lgbm' not in args.models.split(','):
            p.error('--drop chỉ áp dụng cho lgbm')
        backtest.run(snap, cfg, args.models.split(','), args.smoke, args.tag)
        return
    if args.command == 'train-final':
        backtest.final(snap, cfg, args.models.split(','))
        return
    if args.command == 'train-far':
        backtest.far(snap, cfg, args.start)
        return
    if args.command == 'tune':
        from forecast_v2 import tune  # optuna + lightgbm: chỉ import khi cần
        tune.run(snap, cfg, args.trials, args.smoke)
        return
    if args.command == 'evaluate':
        backtest.evaluate(snap, cfg, ROOT / args.artifact)
        return
    if args.command == 'report':
        from forecast_v2 import report  # matplotlib + lightgbm: chỉ import khi cần
        report.run(cfg)
        return
    {'quality': quality, 'facilities': facilities, 'weather': weather, 'features': features}[args.command].run(snap, cfg)


if __name__ == '__main__':
    main()
