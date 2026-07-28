"""Triple-barrier labeling.

For each bar we look forward up to ``horizon`` bars and check which barrier is
touched first: the upper barrier (+``barrier`` fraction) or the lower barrier
(-``barrier``). Label:
    1  -> upper hit first  (a long would have won)
    0  -> lower hit first  (a short would have won)
    NaN -> neither within the horizon (ambiguous; dropped from training)

We also return the exit index and exit price for realistic PnL simulation.
The barrier should exceed round-trip costs so the label reflects a *tradable*
move rather than noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier(
    df: pd.DataFrame,
    barrier: float,
    horizon: int,
) -> pd.DataFrame:
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    label = np.full(n, np.nan)
    exit_idx = np.full(n, -1, dtype=np.int64)
    exit_price = np.full(n, np.nan)
    first_touch = np.zeros(n, dtype=np.int8)  # 1 up, -1 down, 0 timeout

    for i in range(n):
        entry = close[i]
        up = entry * (1.0 + barrier)
        dn = entry * (1.0 - barrier)
        end = min(i + horizon, n - 1)
        touched = 0
        j = i + 1
        while j <= end:
            if high[j] >= up and low[j] <= dn:
                # Both barriers in one bar: assume the adverse one first.
                touched = -1
                break
            if high[j] >= up:
                touched = 1
                break
            if low[j] <= dn:
                touched = -1
                break
            j += 1
        if touched == 1:
            label[i] = 1.0
            exit_idx[i] = j
            exit_price[i] = up
            first_touch[i] = 1
        elif touched == -1:
            label[i] = 0.0
            exit_idx[i] = j
            exit_price[i] = dn
            first_touch[i] = -1
        else:
            # Timeout: no label for training, but record timeout exit for sim.
            exit_idx[i] = end
            exit_price[i] = close[end]
            first_touch[i] = 0

    res = pd.DataFrame(
        {
            "label": label,
            "exit_idx": exit_idx,
            "exit_price": exit_price,
            "first_touch": first_touch,
        },
        index=df.index,
    )
    return res
