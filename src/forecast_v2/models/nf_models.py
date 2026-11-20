"""TiDE và TFT qua neuralforecast (plan §P6): một model global cho mọi chuỗi, train bằng CPU.

Dữ liệu từ features.fold: y và available_mask (loss bỏ điểm mask = 0); covariate tương lai = lịch, lễ,
khí hậu (holiday_type one-hot); covariate quá khứ = w_* (chỉ các dòng tới cutoff); static = one-hot chỉ tiêu, tuyến,
hạng + tọa độ, log_scale, weekend_open. Đầu ra MQLoss [0.1, 0.5, 0.9].

Scale tự làm, không dùng scaler theo cửa sổ của NF: scaler đó chuẩn hóa cả exog theo median của input_size ngày đầu vào,
nên tet_offset, days_to_break, doy_* mất vị trí tuyệt đối và model không học được Tết (smoke 20/11/2026). Thay bằng:
y chia mức của chuỗi (trung bình 365 ngày mask = 1 tới gốc), exog z-score toàn cục trên phần train, trọng số mẫu = mức
(loss theo đơn vị gốc, khớp WAPE, như LightGBM).
"""
import logging
import os

import numpy as np
import pandas as pd

NOT_FUTR = ['unique_id', 'csyt_id', 'field', 'ds', 'y', 'available_mask', 'holiday_type']
QUANTILES = [0.1, 0.5, 0.9]


def frames(df, static, cutoff, weather):
    """(train, futr_df, static_df, futr, hist, stat, level) đã scale. weather: danh sách cột w_*, None = mọi cột w_*."""
    hist = [k for k in df if k.startswith('w_')] if weather is None else list(weather)
    df = pd.concat([df, pd.get_dummies(df.holiday_type, prefix='ht', dtype='float32')], axis=1)
    futr = [k for k in df if k not in NOT_FUTR and not k.startswith('w_')]
    train = df.loc[df.ds <= cutoff, ['unique_id', 'ds', 'y', 'available_mask', *futr, *hist]]
    train[hist] = train.groupby('unique_id')[hist].bfill()  # vài ngày đầu chuỗi chưa có quan trắc trễ; NaN làm hỏng loss
    fut = df.loc[df.ds > cutoff, ['unique_id', 'ds', *futr]]
    ex = [*futr, *hist]
    mu, sd = train[ex].mean(), train[ex].std().replace(0, 1).fillna(1)
    train[ex] = ((train[ex] - mu) / sd).astype('float32')
    fut[futr] = ((fut[futr] - mu[futr]) / sd[futr]).astype('float32')
    seen = train[train.available_mask > 0]
    level = (seen[seen.ds > cutoff - pd.Timedelta(days=365)].groupby('unique_id').y.mean()
             .reindex(train.unique_id.unique()).fillna(seen.groupby('unique_id').y.mean()).fillna(1).clip(lower=1))
    lv = train.unique_id.map(level)
    train['y'] = train.y / lv
    train['sample_weight'] = lv / level.mean()
    s = static.set_index('unique_id')
    st = pd.get_dummies(s[['field', 'tuyen', 'hang']].astype(str), dtype='float32').join(
        s[['latitude', 'longitude', 'log_scale', 'weekend_open']].astype('float32').fillna(0))
    return train, fut, st.reset_index(), futr, hist, list(st.columns), level


def forecast_with(cls, params, df, static, cutoff, cfg, weather=None):
    import pytorch_lightning as pl
    import torch
    from neuralforecast import NeuralForecast
    from tqdm import tqdm
    from neuralforecast.losses.pytorch import MQLoss
    for name in ('lightning.pytorch', 'pytorch_lightning', 'lightning_fabric'):
        logging.getLogger(name).setLevel(logging.ERROR)
    torch.set_num_threads(os.cpu_count())

    train, fut, st, futr, hist, stat, level = frames(df, static, cutoff, weather)
    bar = tqdm(total=params['max_steps'], desc=cls.__name__, unit='step', leave=False)  # early stopping: dừng trước 100%

    class Progress(pl.Callback):  # bar lấy qua closure: NF deepcopy callback khi lưu hparams
        def on_train_batch_end(self, *args, **kwargs):
            bar.update()

    model = cls(h=cfg['features']['horizon'], futr_exog_list=futr, hist_exog_list=hist, stat_exog_list=stat,
                loss=MQLoss(quantiles=QUANTILES), scaler_type='identity', start_padding_enabled=True,
                accelerator='cpu', enable_progress_bar=False, logger=False, enable_model_summary=False,
                enable_checkpointing=False, callbacks=[Progress()], **params)
    nf = NeuralForecast(models=[model], freq='D')
    early = params.get('early_stop_patience_steps', -1) > 0
    nf.fit(df=train, static_df=st, val_size=cfg['features']['horizon'] if early else 0)
    bar.close()
    p = nf.predict(futr_df=fut).reset_index()
    lv = p.unique_id.map(level).to_numpy()
    q = {k: np.clip(p[f'{cls.__name__}-{c}'].to_numpy(float), 0, None) * lv
         for k, c in (('q10', 'lo-80.0'), ('q50', 'median'), ('q90', 'hi-80.0'))}
    # Không sắp xếp 3 phân vị: ngày lễ đầu q10 hay chưa học được (cao hơn median), sắp xếp sẽ đẩy q50 lên mức ngày
    # thường (smoke 20/11/2026: WAPE ngày lễ 21,8 thay vì 3,1). Giữ median, kẹp khoảng quanh nó.
    return pd.DataFrame({'unique_id': p.unique_id, 'ds': p.ds, 'q10': np.minimum(q['q10'], q['q50']), 'q50': q['q50'],
                         'q90': np.maximum(q['q90'], q['q50'])})


def tide(df, static, cutoff, cfg):
    from neuralforecast.models import TiDE
    return forecast_with(TiDE, cfg['models']['tide'], df, static, cutoff, cfg)
