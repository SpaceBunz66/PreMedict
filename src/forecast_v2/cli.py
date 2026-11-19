"""python -m forecast_v2.cli extract | quality | facilities | weather | features [--snapshot YYYY-MM-DD] [--config configs/base.yaml]"""
import argparse

import yaml

from forecast_v2 import extract, facilities, features, quality, weather
from forecast_v2.extract import DATA_DIR, ROOT


def main():
    p = argparse.ArgumentParser(prog='forecast_v2')
    p.add_argument('command', choices=['extract', 'quality', 'facilities', 'weather', 'features'])
    p.add_argument('--snapshot', help='thư mục con của data/; mặc định: hôm nay (extract), mới nhất (quality)')
    p.add_argument('--config', default=ROOT / 'configs' / 'base.yaml')
    args = p.parse_args()
    if args.command == 'extract':
        extract.run(args.snapshot)
        return
    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    snap = DATA_DIR / args.snapshot if args.snapshot else max(DATA_DIR.glob('20??-??-??'))
    {'quality': quality, 'facilities': facilities, 'weather': weather, 'features': features}[args.command].run(snap, cfg)


if __name__ == '__main__':
    main()
