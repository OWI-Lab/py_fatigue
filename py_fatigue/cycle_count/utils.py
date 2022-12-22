from __future__ import annotations

from collections import ChainMap, defaultdict
from typing import Any, DefaultDict, Iterable, Union
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from py_fatigue.damage.stress_life import get_pm
from py_fatigue.material.sn_curve import SNCurve
from py_fatigue.cycle_count.cycle_count import CycleCount
from py_fatigue.cycle_count import cycle_count


def solve_lffd(x: Any) -> Union[Any, CycleCount]:
    """Solve the low-frequency fatigue dynamics of a cycle count or return the
    object as is.

    Parameters
    ----------
    x : Any
        The object to evaluate. If it is a :class:`~py_fatigue.CycleCount`
        instance, the low-frequency fatigue dynamics is solved.
        Otherwise, the object is returned as is.

    Returns
    -------
    Any
        The object evaluated
    """
    if isinstance(x, CycleCount) and len(x.time_sequence) > 1:
        return x.solve_lffd()
    return x


def aggregate_cc(
    df: pd.DataFrame, aggr_by: str
) -> tuple[pd.DataFrame, DefaultDict[str, DefaultDict[str, list[float]]]]:
    """Aggregate a pandas dataframe by time window.
    The pandas dataframe must have a DatetimeIndex and at least one column
    whose name starts with 'CC\\_' containing :class:`~py_fatigue.CycleCount`
    instances, e.g.,

    .. list-table:: Example dataframe to be aggregated
        :widths: 30 30 30 30 30
        :header-rows: 1

        * - DateTimeIndex
          - CC_CycleCount-name-1
          - CC_CycleCount-name-2
          - CC_CycleCount-name-3
          - CC\\_...
        * - 2018-01-01 00:00:00
          - CycleCount-name-1 (01 Jan 2018, 00:00)
          - CycleCount-name-2 (01 Jan 2018, 00:00)
          - CycleCount-name-3 (01 Jan 2018, 00:00)
          - ...
        * - 2018-01-01 01:00:00
          - CycleCount-name-1 (01 Jan 2018, 01:00)
          - CycleCount-name-2 (01 Jan 2018, 01:00)
          - CycleCount-name-3 (01 Jan 2018, 01:00)
          - ...
        * - 2018-01-01 02:00:00
          - CycleCount-name-1 (01 Jan 2018, 02:00)
          - CycleCount-name-2 (01 Jan 2018, 02:00)
          - CycleCount-name-3 (01 Jan 2018, 02:00)
          - ...
        * - ⋮
          - ⋮
          - ⋮
          - ⋮
          - ⋱

    The function performs the following workflow:

    1. Perform initial checks on the input pandas dataframe
    2. Build the aggregation dictionary
    3. Aggregate the dataframe by time window, i.e. the aggregated CycleCounts
    4. Retrieve the low-frequency fatigue dynamics on the aggregated dataframe
    5. Save the residuals sequences of each aggregated CycleCount

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to cluster
    aggr_by : str
        The time window to cluster the dataframe by. It must be a valid pandas
        date offset frequency string or 'all'.
        For all the frequency string aliases offered by pandas, see:
        `pandas-timeseries.html#dateoffset-objects <shorturl.at/dgrwW>`_.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, dict[str, list]]]
        The aggregated dataframe and the residuals sequences of each aggregated
    """
    start = time.time()

    # Perform initial checks on the input pandas dataframe
    print("\33[36m1. Running checks on \33[1mdf\33[22m.\33[0m")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df must have a monotonic increasing DatetimeIndex")
    if not df.index.is_unique:
        raise ValueError("df must have a unique DatetimeIndex")
    if not df.index.inferred_type == "datetime64":
        raise ValueError("df must have a DatetimeIndex containing only dates")

    # Build the aggregation dictionary
    print("\33[36m2. Building the aggregation \33[1mdict\33[22m.")
    agg_list: list[dict[float | str, Any]] = [
        {col: cycle_count.pbar_sum}
        if isinstance(col, str) and col.startswith("CC_")
        else {col: np.nanmean}
        for col in df
    ]
    agg_dict = dict(ChainMap(*agg_list))

    # Aggregate the dataframe by time window
    print(f"3. Aggregate \33[1mdf\33[22m by \33[1m'{aggr_by}'\33[22m.\33[0m")
    if aggr_by.lower() == "all":
        df_agg = df.groupby(lambda _: True).agg(agg_dict)
    else:
        df_agg = df.groupby([df.index.to_period(aggr_by)]).agg(agg_dict)

    # Retrieving the low-frequency fatigue dynamics on the aggregated dataframe
    print("\33[36m4. Retrieving LFFD on aggregated \33[1mdf\33[22m.\33[0m")
    df_agg_rr = df_agg.applymap(solve_lffd)

    cc_cols: list[str] = [
        col for col in df_agg_rr.columns if col.startswith("CC_")
    ]

    # Saving the residuals sequences
    print("\33[36m5. Saving the \33[1mresiduals sequences\33[22m.\33[0m")
    residuals_sequence: DefaultDict[
        str, DefaultDict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    for col in cc_cols:
        for __, row in df_agg.iterrows():
            if not isinstance(row[col], CycleCount):
                continue
            if not isinstance(row[col].residuals_sequence, Iterable):
                continue
            _, res_res_seq, res_res_idx = cycle_count.calc_rainflow(
                data=np.asarray(row[col].residuals_sequence),
                extended_output=True,
            )
            if len(residuals_sequence[col]["idx"]) > 0:
                res_res_idx += residuals_sequence[col]["idx"][-1]
            residuals_sequence[col]["idx"].extend(res_res_idx.tolist())
            residuals_sequence[col]["res"].extend(res_res_seq.tolist())
    end = time.time()
    print(
        f"Elapsed time for \33[36m\33[1m'{aggr_by}'\33[0m aggregation",
        f"is {np.round(end-start, 0)}, s.\n",
    )
    return df_agg_rr, residuals_sequence


def plot_aggregated_residuals(
    res_dct: dict[str, DefaultDict[str, DefaultDict[str, list[float]]]],
    plt_prmtr: str,
    minor_grid: bool = True,
) -> tuple[plt.figure.Figure, plt.axes.Axes]:
    """Plot the aggregated residuals sequences. T

    Parameters
    ----------
    res_dct : dict[str, DefaultDict[str, DefaultDict[str, list[float]]]]
        The residuals sequences of each aggregated CycleCount as returned by
        :func:`aggregate_cc`.
    plt_prmtr : str
        The parameter to plot.
    labels : Collection[str]
        The labels of the aggregated CycleCounts.
    minor_grid : bool, optional
        Whether to plot the minor grid, by default True

    Returns
    -------
    tuple[plt.figure.Figure, plt.axes.Axes]

    """
    fig, axes = plt.subplots()

    for i, (label, res) in enumerate(res_dct.items()):
        alpha = np.round((i + 1) / len(res_dct), 2)
        axes.plot(
            res[plt_prmtr]["idx"],
            res[plt_prmtr]["res"],
            lw=alpha,
            label=label,
            alpha=alpha,
        )
    if minor_grid:
        axes.minorticks_on()
        axes.grid(visible=True, which="minor", color="#E7E6DD", linestyle=":")
    axes.set_xlabel("Residuals sequence")
    axes.set_ylabel("Residuals")
    axes.legend(
        title="Aggregated by",
        loc="lower center",
        fancybox=True,
        bbox_to_anchor=(0.5, -0.44),
        ncol=3,
        shadow=True,
    )
    plt.show()

    return fig, axes


def calc_aggregated_damage(
    df: pd.DataFrame,
    sn: Union[
        dict[str, SNCurve], DefaultDict[str, SNCurve], list[SNCurve], SNCurve
    ],
) -> pd.DataFrame:
    """Calculate the damage of each aggregated CycleCount. This function needs
    the output of :func:`aggregate_cc`.

    Parameters
    ----------
    df : pd.DataFrame
        The aggregated CycleCount as returned by :func:`aggregate_cc`.
    sn : DefaultDict[str, SNCurve]
        The S-N curves of each aggregated CycleCount.

    Returns
    -------
    pd.DataFrame
        The damage of each aggregated CycleCount as a multi-indexed dataframe.
    """
    # Check the S-N curves
    if not isinstance(sn, defaultdict) or not isinstance(sn, dict):
        if isinstance(sn, SNCurve):
            sn = {sn.name: sn}
        elif isinstance(sn, list):
            if all(isinstance(s, SNCurve) for s in sn):
                sn = {s.name: s for s in sn}
            else:
                raise TypeError(
                    "The S-N curves must be a dict, a list of SNCurve or a "
                    "SNCurve."
                )

    cc_cols: list[str] = [col for col in df.columns if col.startswith("CC_")]
    damages = pd.DataFrame()
    for _, sn_curve in sn.items():
        df_1 = df[cc_cols].applymap(
            lambda x, sk=sn_curve: np.sum(get_pm(cycle_count=x, sn_curve=sk))
        )
        df_1["sn_curve"] = f"m={sn_curve.slope}"
        damages = pd.concat([damages, df_1])
        del df_1
    dmg_mi = damages.set_index(["sn_curve", damages.index])  # type: ignore
    return dmg_mi
