"""CLI --set/--tag và chọn run chính: run thử (tag, drop, smoke, chưa xong) không lẫn vào calibrate/train-final."""
import sys

import pytest

from forecast_v2 import backtest, cli


def test_latest_runs_only_main_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(backtest, 'ROOT', tmp_path)
    done = ['20261119-1_snaive7-mean4-lgbm', '20261119-2_lgbm-prophet_prod-prophet_clean', '20261120-1_lgbm_smoke',
            '20261121-1_lgbm_drop-weather', '20261121-2_lgbm_a3', '20261122-1_tide_a4-lgbm', '20261123-1_lgbm']
    for n in done + ['20261124-1_lgbm']:  # run cuối chưa có metrics.md (đang chạy hoặc lỗi giữa chừng)
        (tmp_path / 'runs' / n).mkdir(parents=True)
    for n in done:
        (tmp_path / 'runs' / n / 'metrics.md').touch()
    assert [p.name for p in backtest.latest_runs('lgbm')] == [done[0], done[1], done[-1]]
    assert backtest.latest_runs('tide') == []


def cli_cfg(monkeypatch, *argv):
    """Config mà backtest.run nhận được sau khi CLI xử lý argv."""
    got = {}
    monkeypatch.setattr(backtest, 'run', lambda snap, cfg, *a: got.update(cfg=cfg))
    monkeypatch.setattr(sys, 'argv', ['forecast_v2', *argv, '--snapshot', 'x'])
    cli.main()
    return got['cfg']


def test_drop_adds_to_config_drop(monkeypatch):
    base = cli_cfg(monkeypatch, 'backtest', '--models', 'lgbm')['features']['drop']
    assert cli_cfg(monkeypatch, 'backtest', '--models', 'lgbm', '--drop', 'holiday')['features']['drop'] == [*base, 'holiday']


def test_set_only_existing_keys_on_tagged_backtest(monkeypatch):
    cfg = cli_cfg(monkeypatch, 'backtest', '--models', 'tide', '--set', 'models.tide.hidden_size=256',
                  '--set', 'backtest.test=[]', '--tag', 'a3')
    assert cfg['models']['tide']['hidden_size'] == 256 and cfg['backtest']['test'] == []
    for argv in (['backtest', '--set', 'backtest.test=[]'],                            # thiếu --tag
                 ['quality', '--set', 'quality.max_outlier_ratio=0.05', '--tag', 'x'],  # ghi đè file snapshot
                 ['backtest', '--set', 'models.lgbm.round=1200', '--tag', 'x'],         # gõ sai khóa lá
                 ['backtest', '--set', 'modls.lgbm.rounds=1200', '--tag', 'x']):        # gõ sai khóa cha
        with pytest.raises(SystemExit):
            cli_cfg(monkeypatch, *argv)
