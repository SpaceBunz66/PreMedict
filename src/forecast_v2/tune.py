"""P8: Optuna cho LightGBM trên các gốc validation (plan §P8). Chỉ tune model q50 (điểm = WAPE gộp các gốc, điểm mask = 1);
q10/q90 giữ tham số trong config. Gốc test không được dùng ở đây.

Study lưu ở runs/optuna.db (SQLite), tên lgbm_<snapshot>: chạy lại cùng lệnh là chạy thêm --trials trial nối tiếp (kể cả sau khi tắt máy);
--trials 0 chỉ in bảng hội tụ. Trial 0 của study mới = tham số hiện tại trong config, làm mốc so sánh.
"""
import time

import numpy as np
import optuna
import pandas as pd
import yaml
from tqdm import tqdm

from forecast_v2 import backtest, features
from forecast_v2.extract import ROOT
from forecast_v2.models import lgbm


def space(trial):
    # A1 (plan §P8): bỏ quantile khỏi không gian (chậm ~10 lần, không tốt hơn ở lượt 1); trial cũ giữ trong study
    objective = 'tweedie'
    params = {'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
              'num_leaves': trial.suggest_int('num_leaves', 15, 255, log=True),
              'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 1000, log=True),
              'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
              'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
              'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True)}
    params['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1.05, 1.9)
    return objective, trial.suggest_int('rounds', 200, 1200, step=100), params


def folds(snap, cfg, smoke):
    """Dữ liệu train/dự báo của từng gốc, dựng một lần cho mọi trial (origin_step cố định nên design không đổi)."""
    st, long, daily = backtest.load(snap, cfg, smoke)
    b, c = cfg['backtest'], cfg['models']['lgbm']
    out = []
    for cutoff in map(pd.Timestamp, b['smoke']['cutoffs'] if smoke else b['val']):
        df, fs = features.fold(long, st, daily, cutoff, cfg)
        X, y, lv, train, pred, keys = lgbm.design(df, fs, cutoff, cfg['features']['horizon'], c['origin_step'])
        actual = keys[pred].merge(df[['unique_id', 'ds', 'y', 'available_mask']], on=['unique_id', 'ds'], how='left')
        ok = actual.available_mask.to_numpy() > 0
        out.append((X[train], y[train], lv[train], X[pred][ok], lv[pred][ok], actual.y.to_numpy()[ok]))
        print(f'{cutoff:%d/%m/%Y}: {train.sum():,} mẫu train, {ok.sum():,} điểm đánh giá', flush=True)
    return out


def wape(data, objective, rounds, params, bar=None):
    err = tot = 0.0
    for Xt, yt, wt, Xp, lp, ya in data:
        q = np.clip(lgbm._fit(Xt, yt, wt, objective, 0.5, params, rounds, bar).predict(Xp), 0, None) * lp
        err += np.abs(q - ya).sum()
        tot += ya.sum()
    return err / tot


def run(snap, cfg, trials, smoke=False):
    # mỗi snapshot một study: WAPE trên dữ liệu khác nhau không so được; study 'lgbm' cũ = snapshot 2026-11-18
    name = f"{'lgbm_smoke' if smoke else 'lgbm'}_{snap.name}"
    c = cfg['models']['lgbm']
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=name, storage=f'sqlite:///{(ROOT / "runs" / "optuna.db").as_posix()}',
                                direction='minimize', load_if_exists=True)
    # seed theo số trial đã có: chạy tiếp không lặp lại các trial ngẫu nhiên ban đầu
    study.sampler = optuna.samplers.TPESampler(n_startup_trials=5, seed=len(study.trials))
    if not study.trials:
        study.enqueue_trial({'objective': c['objective'], 'rounds': c['rounds'], 'tweedie_variance_power': 1.5,
                             **{k: c['params'][k] for k in ('learning_rate', 'num_leaves', 'min_data_in_leaf',
                                                            'feature_fraction', 'bagging_fraction', 'lambda_l2')}})
    if trials:
        data = folds(snap, cfg, smoke)

        def objective(trial):
            t0 = time.perf_counter()
            obj, rounds, params = space(trial)
            with tqdm(total=rounds * len(data), desc=f'trial {trial.number} ({obj}, {len(data)} gốc)', unit='vòng',
                      leave=False) as bar:
                v = wape(data, obj, rounds, c['params'] | params, bar)
            best = min([v] + [t.value for t in study.trials if t.value is not None])
            tqdm.write(f'trial {trial.number}: WAPE {v:.4f} (tốt nhất {best:.4f}), {time.perf_counter() - t0:.0f}s')
            return v

        study.optimize(objective, n_trials=trials, show_progress_bar=True)  # thanh ngoài: số trial, thời gian còn lại

    t = study.trials_dataframe(attrs=('number', 'value', 'duration', 'state', 'params'))
    t = t[t.state == 'COMPLETE'].assign(best_so_far=lambda d: d.value.cummin(),
                                        minutes=lambda d: d.duration.dt.total_seconds() / 60)
    t.to_csv(ROOT / 'runs' / f'optuna_{name}.csv', index=False)
    print(t[['number', 'value', 'best_so_far', 'minutes']].round(4).to_string(index=False))
    if len(t):
        p, base = study.best_params, t.value[t.number == 0]
        print(f'Tốt nhất: trial {study.best_trial.number}, WAPE {study.best_value:.4f}'
              + (f' (trial 0 = config hiện tại: {base.iloc[0]:.4f})' if len(base) else '')
              + '. Đoạn cho configs/base.yaml, models.lgbm:')
        print(yaml.safe_dump({'objective': p.pop('objective', 'tweedie'), 'rounds': p.pop('rounds'), 'params': p}, sort_keys=False))
