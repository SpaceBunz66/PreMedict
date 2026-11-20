"""TiDE/TFT qua neuralforecast: dự báo không đổi khi đổi dữ liệu sau cutoff, đủ dòng, phân vị đúng thứ tự."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('neuralforecast')
from forecast_v2.models import nf_models  # noqa: E402
from test_no_leakage import CFG, CUT, build  # noqa: E402

H = CFG['features']['horizon']
FAST = CFG | {'models': {'tide': CFG['models']['tide'] | {'max_steps': 150, 'early_stop_patience_steps': -1}}}


def test_frames_split_at_cutoff():
    _, df, st, _ = build(False)
    train, fut, s, futr, hist, stat, level = nf_models.frames(df, st, CUT, None)
    assert train.ds.max() == CUT and fut.ds.min() == CUT + pd.Timedelta(days=1) and len(fut) == 4 * H
    assert hist and not any(k.startswith('w_') for k in futr + list(fut)) and 'tet_offset' in futr
    assert not train[['y', 'sample_weight', *futr, *hist]].isna().any().any() and not s[stat].isna().any().any()
    # scale toàn cục: tet_offset giữ thứ tự tuyệt đối giữa train và tương lai (không chuẩn hóa theo cửa sổ)
    raw = df.set_index(['unique_id', 'ds']).tet_offset
    z = pd.concat([train, fut]).set_index(['unique_id', 'ds']).tet_offset.reindex(raw.index)
    assert np.allclose(np.corrcoef(raw, z)[0, 1], 1)
    assert (level >= 1).all() and set(level.index) == set(train.unique_id)


def test_tide_ignores_data_after_cutoff():
    out = [nf_models.tide(*build(perturb)[1:3], CUT, FAST) for perturb in (False, True)]
    a, b = out
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 4 * H and (a.q10 <= a.q50).all() and (a.q50 <= a.q90).all() and (a.q10 >= 0).all()

    _, df, _, _ = build(False)
    e = a.merge(df[df.available_mask > 0], on=['unique_id', 'ds'])
    assert np.abs(e.q50 - e.y).sum() / e.y.sum() < 0.3  # chuỗi tuần đều 40/20/5, 150 step: phải bắt được mẫu tuần
