"""python -m forecast_v2.cli extract | quality | facilities | weather | features [--snapshot YYYY-MM-DD] [--config configs/base.yaml]
       python -m forecast_v2.cli backtest --models snaive7,mean4 [--smoke]"""
import argparse

import yaml

from forecast_v2 import backtest, extract, facilities, features, quality, weather
from forecast_v2.extract import DATA_DIR, ROOT


def main():
    p = argparse.ArgumentParser(prog='forecast_v2')
    p.add_argument('command', choices=['extract', 'quality', 'facilities', 'weather', 'features', 'backtest'])
    p.add_argument('--snapshot', help='thư mục con của data/; mặc định: hôm nay (extract), mới nhất (quality)')
    p.add_argument('--config', default=ROOT / 'configs' / 'base.yaml')
    p.add_argument('--models', default='snaive7,mean4', help='backtest: tên model, cách nhau bởi dấu phẩy')
    p.add_argument('--smoke', action='store_true', help='backtest: 20 CSYT, 2 chỉ tiêu, 1 gốc (backtest.smoke)')
    args = p.parse_args()
    if args.command == 'extract':
        extract.run(args.snapshot)
        return
    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    snap = DATA_DIR / args.snapshot if args.snapshot else max(DATA_DIR.glob('20??-??-??'))
    if args.command == 'backtest':
        backtest.run(snap, cfg, args.models.split(','), args.smoke)
        return
    {'quality': quality, 'facilities': facilities, 'weather': weather, 'features': features}[args.command].run(snap, cfg)


if __name__ == '__main__':
    main()
