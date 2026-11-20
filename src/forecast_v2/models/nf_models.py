"""TiDE và TFT qua neuralforecast (plan §P6): một model global cho mọi chuỗi, train bằng CPU.

Dữ liệu từ features.fold: y và available_mask (loss và robust scaler bỏ điểm mask = 0); covariate tương lai = lịch, lễ,
khí hậu (holiday_type one-hot); covariate quá khứ = w_* (chỉ các dòng tới cutoff); static = one-hot chỉ tiêu, tuyến,
hạng + tọa độ, log_scale, weekend_open. Đầu ra MQLoss [0.1, 0.5, 0.9].
"""
import logging
import os

import numpy as np
import pandas as pd

NOT_FUTR = ['unique_id', 'csyt_id', 'field', 'ds', 'y', 'available_mask', 'holiday_type']
QUANTILES = [0.1, 0.5, 0.9]


def frames(df, static, cutoff, weather):
    """(train, futr_df, static_df, futr, hist, stat). weather: danh sách cột w_*, None = mọi cột w_*."""
    hist = [k for k in df if k.startswith('w_')] if weather is None else list(weather)
    df = pd.concat([df, pd.get_dummies(df.holiday_type, prefix='ht', dtype='float32')], axis=1)
    futr = [k for k in df if k not in NOT_FUTR and not k.startswith('w_')]
    train = df.loc[df.ds <= cutoff, ['unique_id', 'ds', 'y', 'available_mask', *futr, *hist]]
    train[hist] = train.groupby('unique_id')[hist].bfill()  # vài ngày đầu chuỗi chưa có quan trắc trễ; NaN làm hỏng loss
    fut = df.loc[df.ds > cutoff, ['unique_id', 'ds', *futr]]
    s = static.set_index('unique_id')
    st = pd.get_dummies(s[['field', 'tuyen', 'hang']].astype(str), dtype='float32').join(
        s[['latitude', 'longitude', 'log_scale', 'weekend_open']].astype('float32').fillna(0))
    return train, fut, st.reset_index(), futr, hist, list(st.columns)


def forecast_with(cls, params, df, static, cutoff, cfg, weather=None):
    import pytorch_lightning as pl
    import torch
    from neuralforecast import NeuralForecast
    from tqdm import tqdm
    from neuralforecast.losses.pytorch import MQLoss
    for name in ('lightning.pytorch', 'pytorch_lightning', 'lightning_fabric'):
        logging.getLogger(name).setLevel(logging.ERROR)
    torch.set_num_threads(os.cpu_count())

    train, fut, st, futr, hist, stat = frames(df, static, cutoff, weather)
    bar = tqdm(total=params['max_steps'], desc=cls.__name__, unit='step', leave=False)  # early stopping: dừng trước 100%

    class Progress(pl.Callback):  # bar lấy qua closure: NF deepcopy callback khi lưu hparams
        def on_train_batch_end(self, *args, **kwargs):
            bar.update()

    model = cls(h=cfg['features']['horizon'], futr_exog_list=futr, hist_exog_list=hist, stat_exog_list=stat,
                loss=MQLoss(quantiles=QUANTILES), scaler_type='robust', start_padding_enabled=True,
                accelerator='cpu', enable_progress_bar=False, logger=False, enable_model_summary=False,
                enable_checkpointing=False, callbacks=[Progress()], **params)
    nf = NeuralForecast(models=[model], freq='D')
    early = params.get('early_stop_patience_steps', -1) > 0
    nf.fit(df=train, static_df=st, val_size=cfg['features']['horizon'] if early else 0)
    bar.close()
    p = nf.predict(futr_df=fut).reset_index()
    cols = [c for c in p if c.startswith(cls.__name__)]  # thứ tự cột theo phân vị: lo-80, median, hi-80
    q = np.sort(np.clip(p[cols].to_numpy(float), 0, None), axis=1)
    return pd.DataFrame({'unique_id': p.unique_id, 'ds': p.ds, 'q10': q[:, 0], 'q50': q[:, 1], 'q90': q[:, 2]})


def tide(df, static, cutoff, cfg):
    from neuralforecast.models import TiDE
    return forecast_with(TiDE, cfg['models']['tide'], df, static, cutoff, cfg)
