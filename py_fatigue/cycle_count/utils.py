from __future__ import annotations

from collections import ChainMap, defaultdict
from typing import Any, DefaultDict, Union
import time

import numpy as np
import pandas as pd

from py_fatigue import cycle_count, CycleCount


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
        date offset frequency string.
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
        f"\nElapsed time for \33[36m\33[1m'{aggr_by}'\33[0m aggregation",
        f"is {np.round(end-start, 0)}, s.",
    )
    return df_agg_rr, residuals_sequence
