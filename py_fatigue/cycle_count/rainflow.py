# -*- coding: utf-8 -*-
"""
Rainflow cycle counting algorithms for fatigue analysis.

Implements:
- ASTM E1049 three-point rainflow.
- Four-point rainflow (Amzallag et al. 1994).

Also includes utilities to find turning points and extrema.

References
----------
[1] ASTM E1049, Standard Practices for Cycle Counting in Fatigue
[2] Rychlik (1987), "A new definition of the rainflow cycle counting method"
[3] Amzallag et al. (1994), "Standardization of the rainflow counting method"

See also
---------
findtp : Find indices to turning points.
findextrema : Find indices to local maxima and minima.
findcross : Find indices to level crossings.
"""

# from future imports
from __future__ import absolute_import, division, print_function

# standard imports
# import itertools
import warnings
from typing import Any, Dict, List, Tuple, Optional, Union

# non-standard imports
import numpy as np
from .._rust import astm_cycles, find_crossings

__all__ = [
    "rainflow",
    "findcross",
    "findextrema",
    "findtp",
    "findrfc_astm",
    "findrfc_fourpoint",
]

# -------------------------------------------------------------------------
# Turning point and extrema helpers
# -------------------------------------------------------------------------


def findtp(x: np.ndarray) -> np.ndarray:
    """Return reversal indices with the first and last samples included.

    Fewer than two interior reversals are treated as an empty sequence,
    following the cycle-counting convention of Py-Fatigue 2.1.1.

    Examples
    --------
    >>> findtp(np.array([0, 3, 1, 5, 0])).tolist()
    [0, 1, 2, 3, 4]
    """
    reversals = findextrema(x)
    if reversals.size < 2:
        return np.empty(0, dtype=np.int64)
    return np.r_[0, reversals, len(x) - 1]


def findcross(
    x: np.ndarray, v: float = 0.0, kind: str | None = None
) -> np.ndarray:
    """Return sample indices just before crossings of level ``v``.

    A plateau at the crossing level belongs to the preceding side, with
    the last level sample marking the crossing. Use ``"u"`` or ``"d"`` to
    select a direction, or ``None``, ``"all"`` or ``"du"`` for both.

    Examples
    --------
    >>> findcross(np.array([-2, 0, 0, 3, 0, -1])).tolist()
    [2, 4]
    >>> findcross(np.array([-2, 0, 0, 3, 0, -1]), kind="d").tolist()
    [4]
    """
    if len(x) and np.isnan(np.min(x)):
        raise ValueError("data contain NaN")
    sides = np.asarray(np.sign(np.atleast_1d(x).ravel() - v), dtype=np.int8)
    crossings = findcross_indices(sides)
    if crossings.size == 0:
        warnings.warn(f"No level v = {v:0.5g} crossings found in x")
        return crossings
    if kind in (None, "all", "du"):
        return crossings
    if kind not in ("u", "d"):
        raise ValueError(f"Unknown cycle-crossing definition! ({kind})")
    first_is_up = sides[crossings[0] + 1] > 0
    return crossings[int(first_is_up != (kind == "u")) :: 2]


def findextrema(x: np.ndarray) -> np.ndarray:
    """Return indices where consecutive sample differences change sign.

    Examples
    --------
    >>> findextrema(np.array([1, 3, 3, 0, 4])).tolist()
    [2, 3]
    """
    differences = np.diff(np.atleast_1d(x).ravel())
    return findcross(differences) + 1


def findcross_indices(xn: Any) -> np.ndarray:
    """Return indices immediately before zero crossings in signed data."""
    return find_crossings(np.ascontiguousarray(xn, dtype=np.int8))


def findrfc_astm(tp: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
    """Count cycles in a sequence of reversals using ASTM E1049.

    Parameters
    ----------
    tp : array-like
        Reversal values in acquisition order.
    t : array-like, optional
        Times at the reversals.

    Returns
    -------
    np.ndarray
        Amplitude, mean and cycle count (0.5 or 1.0) in each row.
        If ``t`` is provided, the last columns are start time and period.
        Residual half cycles follow the cycles emitted during the scan.

    Examples
    --------
    >>> findrfc_astm(np.array([0, 4, 1, 5])).tolist()
    [[1.5, 2.5, 1.0], [2.5, 2.5, 0.5]]
    """
    values = np.ascontiguousarray(np.ravel(tp), dtype=float)
    times = (
        None if t is None else np.ascontiguousarray(np.ravel(t), dtype=float)
    )
    return astm_cycles(values, times)


# -------------------------------------------------------------------------
# Four-point rainflow (Amzallag)
# -------------------------------------------------------------------------


def _extract_cycles_from_sequence(
    seq: List[Tuple[float, Optional[float], int]],
) -> Tuple[List[Dict[str, Any]], List[Tuple[float, Optional[float], int]]]:
    """
    Internal 4-point extraction following Amzallag et al. (1994).

    Parameters
    ----------
    seq : list of (value, time, index)

    Returns
    -------
    cycles : list of dict
        Extracted full cycles.
    residue_seq : list of (value, time, index)
        Remaining extrema sequence.
    """
    x = seq.copy()
    cycles: List[Dict[str, Any]] = []
    while True:
        changed = False
        if len(x) < 4:
            break
        i = 0
        while i <= len(x) - 4:
            v1, v2, v3, v4 = x[i][0], x[i + 1][0], x[i + 2][0], x[i + 3][0]
            as1, as2, as3 = abs(v2 - v1), abs(v3 - v2), abs(v4 - v3)
            if as2 <= as1 and as2 <= as3:
                p1, p2 = x[i + 1], x[i + 2]
                amp = as2 / 2.0  # amplitude (half of range)
                mean = (p1[0] + p2[0]) / 2.0
                cycles.append(
                    {
                        "amplitude": float(amp),
                        "mean": float(mean),
                        "type": 1.0,
                        "i_idx": p1[2],
                        "j_idx": p2[2],
                        "i_time": p1[1],
                        "j_time": p2[1],
                    }
                )
                del x[i + 1 : i + 3]
                changed = True
                i = max(i - 1, 0)  # step back
            else:
                i += 1
        if not changed:
            break
    return cycles, x


def _duplicate_and_select_crossing_cycles(
    residue_seq: List[Tuple[float, Optional[float], int]],
) -> List[Dict[str, Any]]:
    """
    Decompose residue by duplication and select cycles crossing boundary.

    Parameters
    ----------
    residue_seq : list of (value, time, index)

    Returns
    -------
    crossing_cycles : list of dict
    """
    m = len(residue_seq)
    if m == 0:
        return []
    dup: List[Tuple[float, Optional[float], int]] = []
    for item in residue_seq:
        dup.append(item)
    for item in residue_seq:
        dup.append(item)
    cycles_dup, _ = _extract_cycles_from_sequence(dup)
    crossing_cycles: List[Dict[str, Any]] = []
    for c in cycles_dup:
        i_idx, j_idx = c["i_idx"], c["j_idx"]
        if (i_idx < residue_seq[0][2] and j_idx >= residue_seq[0][2]) or (
            j_idx < residue_seq[0][2] and i_idx >= residue_seq[0][2]
        ):
            crossing_cycles.append(c)
    return crossing_cycles


def _process_residue(
    residue_seq: List[Tuple[float, Optional[float], int]],
) -> Tuple[List[Dict[str, Any]], List[Tuple[float, Optional[float], int]]]:
    """
    Fully decompose residue by iterative duplication until stable.

    Parameters
    ----------
    residue_seq : list of (value, time, index)

    Returns
    -------
    cycles : list of dict
        Full cycles extracted from the residue.
    residue_seq : list
        Final residue after decomposition (≤ 3 points).
    """
    all_cycles: List[Dict[str, Any]] = []
    while True:
        new_cycles = _duplicate_and_select_crossing_cycles(residue_seq)
        if not new_cycles:
            break
        all_cycles.extend(new_cycles)
        # fmt: off
        used = {c["i_idx"] for c in new_cycles} | \
               {c["j_idx"] for c in new_cycles}
        # fmt: on
        residue_seq = [pt for pt in residue_seq if pt[2] not in used]
        if len(residue_seq) < 4:
            break
    return all_cycles, residue_seq


def findrfc_fourpoint(
    tp: np.ndarray, t: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Return rainflow counted cycles using the four-point method.

    Parameters
    ----------
    tp : array-like
        vector of turning-points (NB! Only values, not sampled times)
    t : array-like, optional
        vector of sampled times

    Returns
    -------
    sig_rfc : array-like
        array of shape (n,3) or (n, 5) with:
        sig_rfc[:,0] Cycles amplitude
        sig_rfc[:,1] Cycles mean value
        sig_rfc[:,2] Cycle type, half (=0.5) or full (=1.0)
        sig_rfc[:,3] cycle_begin_time (only if t is given)
        sig_rfc[:,4] cycle_period_time (only if t is given)
    """
    sig = np.asarray(tp, dtype=float)
    n = len(sig)
    times: List[Optional[float]] = [None] * n if t is None else list(t)
    seq = [(float(sig[i]), times[i], i) for i in range(n)]

    cycles, residue_seq = _extract_cycles_from_sequence(seq)
    final_cycles = cycles.copy()
    decomposed, residue_seq = _process_residue(residue_seq)
    final_cycles.extend(decomposed)

    # convert leftover residue into half cycles
    for k in range(len(residue_seq) - 1):
        p1, p2 = residue_seq[k], residue_seq[k + 1]
        amp = abs(p2[0] - p1[0]) / 2.0  # amplitude definition
        mean = (p1[0] + p2[0]) / 2.0
        final_cycles.append(
            {
                "amplitude": amp,
                "mean": mean,
                "type": 0.5,
                "i_idx": p1[2],
                "j_idx": p2[2],
                "i_time": p1[1],
                "j_time": p2[1],
            }
        )

    rows: List[List[Any]] = []
    for c in final_cycles:
        if t is None:
            rows.append([c["amplitude"], c["mean"], c["type"]])
        else:
            begin_time = c["i_time"]
            period_time = (
                abs(c["j_time"] - c["i_time"]) * 2
                if c["i_time"] is not None and c["j_time"] is not None
                else None
            )
            rows.append(
                [c["amplitude"], c["mean"], c["type"], begin_time, period_time]
            )
    return np.asarray(rows, dtype=float)


# -------------------------------------------------------------------------
# Main rainflow function
# -------------------------------------------------------------------------


def rainflow(
    data: Union[np.ndarray, list],
    time: Optional[Union[np.ndarray, list]] = None,
    method: str = "astm",
    extended_output: bool = True,
) -> Union[np.ndarray, Tuple]:
    """
    Returns the cycle-count of the input data calculated through the
    rainflow method (ASTM or four-point).

    Parameters
    ----------
    data : Union[np.ndarray, list]
        time series or residuals sequence
    time : Optional[Union[np.ndarray, list]], optional
        sampled times, by default None
    method : str, optional
        "astm" or "fourpoint"
    extended_output : bool, optional
        If True, also return residuals.

    Returns
    -------
    Union[np.ndarray, tuple]
    """
    if isinstance(data, list):
        data = np.array(data)
    if not isinstance(data, np.ndarray):
        raise TypeError("data shall be either numpy.1darray or list")
    if time is not None:
        if isinstance(time, list):
            time = np.array(time)
        if not isinstance(time, np.ndarray):
            raise TypeError("time shall be either numpy.1darray or list")
        if len(time) != len(data):
            raise ValueError("time and data must have the same length")
    idx = findtp(data)  # [1:-1]
    tp = data[idx]

    if method.lower() == "astm":
        rfs = findrfc_astm(tp, idx)
    elif method.lower() == "fourpoint":
        rfs = findrfc_fourpoint(tp, idx)
    else:
        raise ValueError(f"Unknown rainflow method: {method}")

    # build residual signal (based on half cycles)
    res = rfs[rfs[:, 2] == 0.5]
    res_tp = res[:, -2].astype(int)
    res_tp = np.append(res_tp, [len(data) - 1])
    if extended_output:
        if time is None:
            return rfs, data[res_tp], res_tp
        return rfs, data[res_tp], res_tp, time[res_tp]
    return rfs
