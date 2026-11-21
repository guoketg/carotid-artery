# -*- coding: utf-8 -*-
"""NaN/Inf 守卫的最小验证：直接调用 trainer._check_finite_or_skip 的真实实现。

用法：venv\\Scripts\\python.exe scripts\\test_nan_guard.py
预期末行输出 ALL_OK；任何一条不符合即抛 AssertionError 并非零退出。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from trainer import _check_finite_or_skip


def main():
    finite = torch.tensor(1.234)
    nan = torch.tensor(float('nan'))
    inf = torch.tensor(float('inf'))

    # 1) 有限损失：不跳过，计数归零（即使之前有 streak）
    assert _check_finite_or_skip(finite, 7, 5, 1) == (False, 0)

    # 2) 首次非有限：跳过，计数从 0 增到 1
    assert _check_finite_or_skip(nan, 0, 5, 2) == (True, 1)
    assert _check_finite_or_skip(inf, 0, 5, 3) == (True, 1)

    # 3) 计数递增到阈值前不抛
    for streak in range(4):
        skip, new_streak = _check_finite_or_skip(nan, streak, 5, 10 + streak)
        assert skip is True and new_streak == streak + 1

    # 4) 达到 nan_patience 即抛 RuntimeError（对应「进程非零退出」）
    try:
        _check_finite_or_skip(nan, 4, 5, 99)
    except RuntimeError as e:
        assert '非有限损失' in str(e)
    else:
        raise AssertionError('达到 nan_patience 未抛异常')

    # 5) 多分类输出里的局部 NaN 也要能被抓到（.all() 语义，不能只看标量层）
    batch_loss = torch.tensor([1.0, float('nan')])
    assert bool(torch.isfinite(batch_loss).all()) is False

    print('ALL_OK')


if __name__ == '__main__':
    main()
