"""python -m forecast_v2.cli extract | quality | facilities | weather | features [--snapshot YYYY-MM-DD] [--config configs/base.yaml]
       python -m forecast_v2.cli backtest --models snaive7,mean4 [--smoke]
       python -m forecast_v2.cli train-final --models lgbm
       python -m forecast_v2.cli tune [--trials 10] [--smoke]   # chạy lại = chạy tiếp study"""
import argparse
import sys

import yaml

from forecast_v2 import backtest, extract, facilities, features, quality, weather
from forecast_v2.extract import DATA_DIR, ROOT


def main():
    sys.stdout.reconfigure(encoding='utf-8')  # stdout chuyển hướng (| tee, > log) trên Windows mặc định cp1252: lỗi khi in tiếng Việt
    p = argparse.ArgumentParser(prog='forecast_v2')
    p.add_argument('command', choices=['extract', 'quality', 'facilities', 'weather', 'features', 'backtest',
                                        'train-final', 'tune'])
    p.add_argument('--snapshot', help='thư mục con của data/; mặc định: hôm nay (extract), mới nhất (quality)')
    p.add_argument('--config', default=ROOT / 'configs' / 'base.yaml')
    p.add_argument('--models', default='snaive7,mean4', help='backtest: tên model, cách nhau bởi dấu phẩy')
    p.add_argument('--smoke', action='store_true', help='backtest/tune: 20 CSYT, 2 chỉ tiêu, 1 gốc (backtest.smoke)')
    p.add_argument('--trials', type=int, default=10, help='tune: số trial chạy thêm; 0 = chỉ in bảng hội tụ')
    args = p.parse_args()
    if args.command == 'extract':
        extract.run(args.snapshot)
        return
    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    snap = DATA_DIR / args.snapshot if args.snapshot else max(DATA_DIR.glob('20??-??-??'))
    if args.command == 'backtest':
        backtest.run(snap, cfg, args.models.split(','), args.smoke)
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
