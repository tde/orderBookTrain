import numpy as np
"""
строит метки на заданном интервале что цена изменилась на заддное кол-во тиков
"""

def build_barrier_labels(ms: np.ndarray,
                         mid: np.ndarray,
                         tick_size: float = 1.0,
                         theta_ticks: int = 1,
                         horizon_sec: float = 1.5) -> np.ndarray:
    """
    Triple-barrier по реальному времени t -> t+τ:
      класс 2: цена поднималась >= +θ,
      класс 0: цена падала  <= -θ,
      класс 1: ни то ни другое (flat).
    Там, где нет будущего окна, возвращаем -1.
    """
    N = len(mid)
    y = np.full(N, -1, dtype=np.int64)
    theta = theta_ticks * tick_size

    import bisect
    for i in range(N):
        t0 = ms[i]
        t_end = t0 + int(horizon_sec * 1000)
        j = bisect.bisect_right(ms, t_end, lo=i+1)
        if j <= i+1:
            continue
        m0 = mid[i]
        w = mid[i+1:j]
        if w.size == 0:
            continue
        up_hit = (w.max() - m0) >= theta
        dn_hit = (w.min() - m0) <= -theta
        if up_hit and not dn_hit:
            y[i] = 2
        elif dn_hit and not up_hit:
            y[i] = 0
        elif up_hit and dn_hit:
            # кто наступил раньше (грубо)
            up_idx = np.argmax(w == w.max())
            dn_idx = np.argmax(w == w.min())
            y[i] = 2 if up_idx < dn_idx else 0
        else:
            y[i] = 1
    return y

def make_windows(X2D: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Скользящие окна по времени (каузально):
      вход: X2D (N, F)
      выход: Xwin (N-T+1, T, F), end_idx — индексы последних точек окон
    """
    N, F = X2D.shape
    if N < T:
        raise ValueError(f"Мало данных для окна: N={N} < T={T}")
    s0, s1 = X2D.strides
    Xwin = np.lib.stride_tricks.as_strided(
        X2D, shape=(N - T + 1, T, F), strides=(s0, s0, s1)
    ).copy()
    end_idx = np.arange(T-1, N)
    return Xwin, end_idx