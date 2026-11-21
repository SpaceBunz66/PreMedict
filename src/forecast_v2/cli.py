"""python -m forecast_v2.cli extract | quality | facilities | weather | features [--snapshot YYYY-MM-DD] [--config configs/base.yaml]
       python -m forecast_v2.cli backtest --models snaive7,mean4 [--smoke] [--drop weather,clim]  # --drop: ablation lgbm
       python -m forecast_v2.cli backtest --models tide --set models.tide.max_steps=3000 --set backtest.test=[] --tag a3
       python -m forecast_v2.cli train-final --models lgbm
       python -m forecast_v2.cli tune [--trials 10] [--smoke]   # chạy lại = chạy tiếp study
       python -m forecast_v2.cli calibrate --models lgbm,tide   # A5: hệ số khoảng tin cậy từ run backtest mới nhất"""
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
                                        'train-final', 'tune', 'calibrate'])
    p.add_argument('--snapshot', help='thư mục con của data/; mặc định: hôm nay (extract), mới nhất (quality)')
    p.add_argument('--config', default=ROOT / 'configs' / 'base.yaml')
    p.add_argument('--models', default='snaive7,mean4', help='backtest: tên model, cách nhau bởi dấu phẩy')
    p.add_argument('--smoke', action='store_true', help='backtest/tune: 20 CSYT, 2 chỉ tiêu, 1 gốc (backtest.smoke)')
    p.add_argument('--drop', default='', help='backtest: nhóm feature lgbm bỏ đi (holiday, weather, clim), ablation A2')
    p.add_argument('--set', action='append', default=[], metavar='KHÓA=GIÁ_TRỊ',
                   help='ghi đè config, vd. models.tide.max_steps=3000 hoặc backtest.test=[] (giá trị đọc như YAML)')
    p.add_argument('--tag', default='', help='backtest: hậu tố tên run; run có tag không bị calibrate/train-final chọn')
    p.add_argument('--trials', type=int, default=10, help='tune: số trial chạy thêm; 0 = chỉ in bảng hội tụ')
    args = p.parse_args()
    if args.command == 'extract':
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
            d = d[k]  # khóa cha phải có sẵn (bắt lỗi gõ sai đường dẫn); khóa lá có thể mới, vd. hidden_size
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
    if args.command == 'tune':
        from forecast_v2 import tune  # optuna + lightgbm: chỉ import khi cần
        tune.run(snap, cfg, args.trials, args.smoke)
        return
    {'quality': quality, 'facilities': facilities, 'weather': weather, 'features': features}[args.command].run(snap, cfg)


if __name__ == '__main__':
    main()
