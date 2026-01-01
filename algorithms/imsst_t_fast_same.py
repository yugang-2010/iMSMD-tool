import numpy as np

def IMSST_T_fast_same(x, hlength, num, hop=1, use_float32=True):
    """
    与原 IMSST_T 定义尽量一致的加速版：
    - FFT 长度仍为 N（信号长度）
    - 时间列默认 hop=1（与原完全一致的采样级时间分辨率）
    - 把生成 tfr_full / omega迭代 / Ts累加 做向量化
    返回 Ts: shape (N//2, ceil(N/hop))
    """

    x = np.asarray(x)
    if x.ndim == 2:
        x = x[:, 0]
    N = x.shape[0]

    # dtype
    if use_float32:
        x = x.astype(np.float32, copy=False)
        ctype = np.complex64
    else:
        x = x.astype(np.float64, copy=False)
        ctype = np.complex128

    # force odd window length
    hlength = int(hlength)
    hlength = hlength + 1 - (hlength % 2)
    Lh = (hlength - 1) // 2

    # gaussian window (与你原来一致)
    ht = np.linspace(-0.5, 0.5, hlength, dtype=x.dtype)
    h = np.exp(-np.pi / (0.32 ** 2) * (ht ** 2)).astype(x.dtype)
    h = np.conj(h)  # 保持与你原代码的 conj(h)

    # time indices（hop=1 等价原版）
    hop = int(max(1, hop))
    t_idx = np.arange(0, N, hop, dtype=np.int32)
    tcol = t_idx.size

    # tau 与频率行索引（行坐标仍按 N 定义）
    tau = np.arange(-Lh, Lh + 1, dtype=np.int32)              # (hlength,)
    freq_idx = ((N + tau) % N).astype(np.int32)               # (hlength,)

    # 构建 time_idx (tcol, hlength)
    time_idx = t_idx[:, None] + tau[None, :]
    valid = (time_idx >= 0) & (time_idx < N)
    time_idx_clip = np.clip(time_idx, 0, N - 1)

    # windowed samples (tcol, hlength)
    vals = x[time_idx_clip] * h[Lh + tau][None, :]
    vals = np.where(valid, vals, 0)

    # 组装 tfr_full: (N, tcol)
    # 注意：这里只填充窗长那几行，其余为0，等价原来每列只写 tau 对应的行
    tfr_full = np.zeros((N, tcol), dtype=x.dtype)
    tfr_full[freq_idx[:, None], np.arange(tcol)[None, :]] = vals.T

    # FFT（原版是 np.fft.fft；这里用 rfft 取正频，结果正频部分等价）
    tfr_r = np.fft.rfft(tfr_full, axis=0).astype(ctype, copy=False)
    neta = N // 2
    tfr = tfr_r[:neta, :]  # (N//2, tcol)

    # omega（原版: diff(phase)*N/(2π)；hop>1 时要除以 hop 保持同一标定）
    phase = np.unwrap(np.angle(tfr), axis=1)
    omega = np.diff(phase, axis=1) * (N / (2 * np.pi * hop))
    omega = np.concatenate([omega, omega[:, -1][:, None]], axis=1)  # 补齐最后一列

    omega2 = omega.copy()

    # 迭代映射（向量化替代三重for）
    if num > 1:
        for _ in range(num - 1):
            k = np.rint(omega2).astype(np.int32)
            np.clip(k, 0, neta - 1, out=k)
            omega2 = np.take_along_axis(omega2, k, axis=0)

    omega2 = np.rint(np.rint(omega2 * 2) / 2).astype(np.int32)
    np.clip(omega2, 0, neta - 1, out=omega2)

    # Ts 重分配（每列一次循环，避免 eta×b 双循环）
    Ts = np.zeros_like(tfr, dtype=ctype)
    for b in range(tcol):
        k = omega2[:, b]                 # (neta,)
        tb = tfr[:, b]                   # (neta,)

        ref = np.abs(tb[k]) + 1e-12
        cond = np.abs(tb) > 0.001 * ref
        np.add.at(Ts[:, b], k[cond], tb[cond])

    Ts = Ts / (N / 2.0)
    return Ts
