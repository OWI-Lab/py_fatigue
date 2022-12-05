# -*- coding: utf-8 -*-

r"""The following code is based on the histogram function from the numpy
package `[4]`_.

.. _[4]: https://numpy.org/doc/stable/reference/generated/numpy.histogram2d
"""

# standard imports
from __future__ import annotations
import itertools
from typing import Callable, List, Optional, Sequence, Union, TypeVar

# non-standard imports
import matplotlib
import numpy as np
import numpy.typing as npt

from py_fatigue.cycle_count import calc_rainflow
from py_fatigue.utils import make_axes
from py_fatigue.mean_stress import MeanStress
from py_fatigue.stress_range import StressRange

__all__ = [
    "make_axes",
    "make_histogram",
    "rainflow_binner",
    "binned_rainflow",
    "xy_hist2d",
]


FloatArray = TypeVar("FloatArray", bound=npt.NDArray[np.floating])


def make_histogram(
    mean_stress: np.ndarray,
    stress_range: np.ndarray,
    bins: Union[int, Sequence[int]] = 10,
    bin_range: Optional[np.ndarray] = None,
    normed: bool = False,
    weights: Optional[np.ndarray] = None,
) -> tuple:
    """
    Make a scattered-histogram plot.

    Parameters
    ----------
    mean_stress, stress_range : array_like, shape (n, )
        Input data
    bins : int or array_like or [int, int] or [array, array], optional
        The bin specification:
          * If int, the number of bins for the two dimensions (nx=ny=bins).
          * If array_like, the bin edges for the two dimensions
            (stress_range_edges=mean_stress_edges=bins).
          * If [int, int], the number of bins in each dimension
            (n_stress_range, n_mean_stress = bins).
          * If [array, array], the bin edges in each dimension
            (stress_range_edges, mean_stress_edges = bins).
          * A combination [int, array] or [array, int], where int
            is the number of bins and array is the bin edges.
    bin_range : array_like, shape(2,2), optional
        The leftmost and rightmost edges of the bins along each dimension
        (if not specified explicitly in the `bins` parameters):
        ``[[stress_range_min, stress_range_max],
        [mean_stress_min, mean_stress_max]]``. All values outside of this
        range will be considered outliers and not tallied in the histogram.
    normed : bool, optional
        If False, returns the number of samples in each bin. If True,
        returns the bin density ``bin_count / sample_count / bin_area``.
    weights : array_like, shape(N,), optional
        An array of values ``w_i`` weighing each sample
        ``(stress_range_i, mean_stress_i)``. Weights are normalized to 1
        if `normed` is True. If `normed` is False, the values of the returned
        histogram are equal to the sum of the weights belonging to the samples
        falling into each bin.

    Returns
    -------
    hist : `np.ndarray`, shape(m,m)
        Histogram matrix.
    hist_edges : ArrayArray, (shape(m+1,), shape(m+1,))
        Histogram bin edges.
    """

    if len(mean_stress) != len(stress_range):
        raise ValueError("Input arrays must be the same length")

    hist, mean_e, range_e = np.histogram2d(
        mean_stress,
        stress_range,
        bins=bins,
        range=bin_range,
        normed=normed,
        weights=weights,
    )
    hist_bin_edges = (mean_e, range_e)
    return hist, hist_bin_edges


def map_hist(
    x: np.ndarray,
    y: np.ndarray,
    h: np.ndarray,
    hist_bin_edges: tuple,
) -> np.ndarray:
    """Maps the histogram to the original data.

    Parameters
    ----------
    x : np.ndarray
        x-axis of the original data
    y : np.ndarray
        y-axis of the original data
    h : np.ndarray
        histogram
    hist_bin_edges : tuple
        histogram bin edges

    Returns
    -------
    np.ndarray
        mapped histogram
    """
    x_i = np.digitize(x, hist_bin_edges[0])
    x_i[np.where(x_i > 0)] -= 1
    y_i = np.digitize(y, hist_bin_edges[1])
    y_i[np.where(y_i > 0)] -= 1
    inds = np.ravel_multi_index(
        (x_i, y_i),
        (len(hist_bin_edges[0]) - 1, len(hist_bin_edges[1]) - 1),
        mode="clip",
    )
    vals = h.flatten()[inds]
    bads = (
        (x < hist_bin_edges[0][0])
        | (x > hist_bin_edges[0][-1])
        | (y < hist_bin_edges[1][0])
        | (y > hist_bin_edges[1][-1])
    )
    vals[bads] = np.nan
    return vals


def xy_hist2d(
    x: FloatArray,
    y: FloatArray,
    mode: str = "mountain",
    bins: Union[int, Sequence[int]] = 10,
    bin_range: Optional[FloatArray] = None,
    normed: bool = False,
    weights: Optional[FloatArray] = None,  # np.histogram2d args
    fig: Optional[matplotlib.figure.Figure] = None,
    ax: Optional[matplotlib.collections.PathCollection] = None,
    dens_func: Optional[Callable] = None,
    **kwargs: dict,
) -> tuple:
    """
    Make a scattered-histogram plot.

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data
    marker_size : int
        Size of the markers
    mode: [None | 'mountain' | 'valley' | 'clip']
        Possible values are:
            - None : The points are plotted as one scatter object, in the
                order in-which they are specified at input.
            - 'mountain' : The points are sorted/plotted in the order of
                the number of points in their 'bin'. This means that points
                in the highest density will be plotted on-top of others. This
                cleans-up the edges a bit, the points near the edges will
                overlap.
            - 'valley' : The reverse order of 'mountain'. The low density
                bins are plotted on top of the high ones.

    bins : int or array_like or [int, int] or [array, array], optional
        The bin specification:
            - If int, the number of bins for the two dimensions (nx=ny=bins).
            - If array_like, the bin edges for the two dimensions
                (x_edges = y_edges = bins).
            - If [int, int], the number of bins in each dimension
                (n_x, n_y = bins).
            - If [array, array], the bin edges in each dimension
                (x_edges, y_edges = bins).
            - A combination [int, array] or [array, int], where int
                is the number of bins and array is the bin edges.

    bin_range : array_like, shape(2,2), optional
        The leftmost and rightmost edges of the bins along each dimension
        (if not specified explicitly in the `bins` parameters):
        ``[[x_min, x_max], [y_min, y_max]]``. \
        All values outside of this range will be considered outliers
        and not tallied in the histogram.
    normed : bool, optional
        If False, returns the number of samples in each bin. If True,
        returns the bin density ``bin_count / sample_count / bin_area``.
    weights : array_like, shape(N,), optional
        An array of values ``w_i`` weighing each sample ``(y_i, x_i)``.
        Weights are normalized to 1 if `normed` is True. If `normed` is
        False, the values of the returned histogram are equal to the sum of
        the weights belonging to the samples falling into each bin.
    fig : a figure instance to add.
    ax : an axes instance to plot into.
    dens_func : function or callable (default: None)
        A function that modifies (inputs and returns) the dens
        values (e.g., np.log10). The default is to not modify the
        values.
    x_label : str, optional
        Label for the x-axis
    y_label : str, optional
        Label for the y-axis
    cbar_label : str, optional
        Label for the colorbar
    kwargs : these are all passed on to scatter.

    Returns
    -------
    figure, paths : (
        `~matplotlib.figure.Figure`,
        `~matplotlib.collections.PathCollection`
    )
        The figure with scatter instance.
    """
    fig, axes = make_axes(fig, ax)
    hist, hist_bin_edges = make_histogram(
        x, y, bins=bins, bin_range=bin_range, normed=normed, weights=weights
    )
    dens = map_hist(x, y, hist, hist_bin_edges)
    if dens_func is not None:
        dens = dens_func(dens)
    # iorder = np.empty(0, dtype=float)
    if mode == "mountain":
        iorder = np.argsort(dens)
    elif mode == "valley":
        iorder = np.argsort(dens)[::-1]
    else:
        raise ValueError("mode must be one of [None, 'mountain', 'valley']")
    x = x[iorder]
    y = y[iorder]
    dens = dens[iorder]
    im = axes.scatter(
        x,
        y,
        # edgecolors="#CCC",
        # linewidths=0.5,
        c=dens,
        **kwargs,
    )
    return fig, axes, im


def rainflow_binner(
    ranges: StressRange,
    _means: Optional[MeanStress],
    damage_tolerance_for_binning: float = 1e-3,
    damage_exponent: float = 5.0,
    max_consecutive_zeros: int = 12,
    round_decimals: int = 4,
    debug_mode: bool = True,
) -> dict:
    """Binning the rainflow cycles into mean-range bins up to a certain
    stress range which is based on the damage tolerance index. All the
    cycles beyond such stress range are saved separately for accuracy
    reasons as "large cycles".

    By default, large cycles are calculated from all the cycles
    accounting for more than 0.1% of the total damage and that have a
    discrepancy of more than 0.1% with respect to the damage that would
    be obtained from the bin center.

    Parameters
    ----------
    ranges : StressRange
        The stress range object.
    _means : Optional[MeanStress], optional
        The mean stress object. If not provided, the mean stress is
        set to zero.
    damage_tolerance_for_binning : float, optional
        tolerance for the damage when binning large cycles
        separately, by default 1e-3
    damage_exponent : float, optional
        exponent for the damage when binning large cycles separately,
        by default 5.0
    max_consecutive_zeros : int, optional
        maximum number of consecutive zeros before saving a cycle to
        large cycles array. This quantity is introduced to save storage
        space, by default 8
    round_decimals : int, optional
        number of decimals to round the output, by default 4
    debug_mode : bool, optional
        if True, print debug messages, by default False

    Returns
    -------
    out_dct : dict
        A dictionary with the following keys:
        - "nr_small_cycles": number of small cycles
        - "range_bin_lower_bound": lower bound of the range bin
        - "range_bin_width": width of the range bins
        - "mean_bin_lower_bound": lower bound of the mean bin
        - "mean_bin_width": width of the mean bins
        - "hist": histogram of the rainflow cycles
        - "lg_c": large cycles
        - "res": residuals
        - "res_sig": residuals sequence

    See Also
    --------
    make_histogram
    """
    # 1) Type of analysis: if with mean or without mean.
    # If no mean stress is desired, the method creates a dummy array of
    # zero mean stresses.
    no_mean_stress = True
    means = MeanStress(
        ranges.counts,
        np.zeros(len(ranges.values)),
        1,
    )
    if _means is not None:
        means = _means
        no_mean_stress = False

    # 2) Removing mean-range pairs below threshold (lowest bound)
    ranges, means, nr_small_cycles = rainflow_threshold(ranges, means)
    if debug_mode:  # pragma: no cover
        # a) Full cycles
        print("\033[1mInitial values\033[0m\n--------------")
        print("count cycle", ranges.counts, "\n")
        print("mean stress", means.values, "\n")
        print("stress range", ranges.values, "\n")
        # b) Full cycles
        print("\033[1mFull cycles\033[0m\n-----------")
        # print("count full",means.counts)
        print("mean full", means.full, "\n")
        print("range full", ranges.full, "\n")
        print("len mean full", len(means.full))
        print("len range full", len(ranges.full), "\n")
        # c) Residuals
        print("\033[1mResiduals\033[0m\n---------")
        print("Mean :", means.half, "\n")
        print("Range:", ranges.half, "\n")
        print("\033[1mLoop over rainflow-counted data\033[0m")
        print("-------------------------------")
    # 3) Histogram bin edges (all these attributes are in FatigueStress)
    # 3.a) Mean - lower bound
    # 3.b) Mean - upper bound
    # 3.c) Mean - edges
    # 3.d) Range - upper bound
    # 3.e) Range - edges

    # 4) Histogram
    # _, hist_bin_edges = make_histogram(
    #     means.full,
    #     ranges.full,
    #     bins=(means.bin_edges, ranges.bin_edges)
    # )

    # # Test on the edges output from numpy.histogram
    # np.testing.assert_array_equal(mean_bin_edges, hist_bin_edges[0])
    # np.testing.assert_array_equal(range_bin_edges, hist_bin_edges[1])

    # 5) Indices of the bins for mean and range using numpy.digitize, ref.
    # https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
    # -------------------------------------------------------------------------

    binned_sorted = sorted(
        zip(
            means.bins_idx,
            ranges.bins_idx,
            means.binned_values,
            ranges.binned_values,
            means.full,
            ranges.full,
        ),
        reverse=False,
    )

    # 6) Initialising the output dictionary
    out_dct = initialise_binned_rainflow_dict(
        ranges, means, nr_small_cycles, no_mean_stress
    )
    if len(ranges.full) == 0:
        return _build_binned_dict(
            ranges, means, [], [], round_decimals, no_mean_stress, out_dct
        )
    # 6) Reference damage for large cycles
    max_damage = max(ranges.full) ** damage_exponent
    # 7) Initialising output variabies: large cycles and histogram
    lg_c = []
    # the_hist: List[List[int]] = [[] for _ in range(len(means.bin_centers))]
    _hist = np.zeros((len(means.bin_centers), len(ranges.bin_centers)))
    i_ran_1 = binned_sorted[0][1]
    for en, (i_mea, i_ran, b_mea, b_ran, mea, ran) in enumerate(binned_sorted):
        # print(i_ran, i_ran_1)
        if debug_mode:  # pragma: no cover
            print("---------------------------------")
            print(f"\033[96mEnum - {en}\033[0m")
            print(f"\033[96mID   - Mean: {i_mea}, Range: {i_ran}\033[0m")
            print(f"\033[96mBin  - Mean: {b_mea}, Range: {b_ran}\033[0m")
            print(f"\033[96mReal - Mean: {mea}, Range: {ran}\033[0m")
        # 9) Large cycles identification
        ref_damage = ran**damage_exponent
        approx_damage = b_ran**damage_exponent
        if (
            ref_damage / max_damage > damage_tolerance_for_binning
            and np.abs(ref_damage - approx_damage) / ref_damage
            > damage_tolerance_for_binning
        ) or (i_ran - i_ran_1 > max_consecutive_zeros):
            lg_c.append([mea, ran])
            if debug_mode:  # pragma: no cover
                print(
                    f"\033[92mRea: ({np.round(mea, 3)},"
                    + f" {np.round(ran, 3)}),"
                    + f" Bin: ({np.round(b_mea, 3)},"
                    + f" {np.round(b_ran, 3)}"
                    + ")\033[0m Large Cycle"
                )
            continue
        # 10) Histogram identification
        if debug_mode:  # pragma: no cover
            print(
                f"\033[92mRea: ({np.round(mea, 3)},"
                + f" {np.round(ran, 3)}),"
                + f" Bin: ({np.round(b_mea, 3)},"
                + f" {np.round(b_ran, 3)}"
                + ")\033[0m Histogram Cycle"
            )
        _hist[i_mea][i_ran] += 1
        i_ran_1 = i_ran
    # 11) Histogram binning
    the_hist: List[List[float]] = []
    for hst in _hist:
        the_hist.append(np.trim_zeros(hst, trim="b").tolist())
    return _build_binned_dict(
        ranges, means, lg_c, the_hist, round_decimals, no_mean_stress, out_dct
    )


def rainflow_threshold(ranges: StressRange, means: MeanStress) -> tuple:
    """Removes the rainflow cycles below threshold for a given
    stress range and mean stress.

    Parameters
    ----------
    ranges : StressRange
        The stress range object.
    means : MeanStress
        The mean stress object.

    Returns
    -------
    tuple
        The ranges and means above threshold, as well as small cycles.
    """
    ranges_gt_lb_idx = np.where(ranges.values >= ranges.bin_lb)[0]
    nr_small_cycles = len(ranges.values) - len(ranges_gt_lb_idx)
    means.counts = means.counts[ranges_gt_lb_idx]
    means.values = means.values[ranges_gt_lb_idx]
    ranges.counts = ranges.counts[ranges_gt_lb_idx]
    ranges.values = ranges.values[ranges_gt_lb_idx]
    return ranges, means, nr_small_cycles


def initialise_binned_rainflow_dict(
    ranges: StressRange,
    means: MeanStress,
    nr_small_cycles: int,
    no_mean_stress: bool = False,
) -> dict:
    """Initialises the binned rainflow dictionary.

    Parameters
    ----------
    ranges : StressRange
        The stress range object.
    means : MeanStress
        The mean stress object.
    nr_small_cycles : int
        The number of small cycles.
    no_mean_stress : bool, optional
        Whether to ignore mean stress. Defaults to False.

    Returns
    -------
    dict
        The initialised binned rainflow dictionary.
    """
    out_dct: dict[str, Union[float, list, np.ndarray]] = {}
    out_dct["nr_small_cycles"] = nr_small_cycles
    out_dct["range_bin_lower_bound"] = ranges.bin_lb
    out_dct["range_bin_width"] = ranges.bin_width
    if not no_mean_stress:
        out_dct["mean_bin_lower_bound"] = means.bin_lb
        out_dct["mean_bin_width"] = means.bin_width
    return out_dct


def binned_rainflow(  # pylint: disable=too-many-arguments
    data: Union[np.ndarray, list],
    time: Optional[Union[np.ndarray, list]] = None,
    range_bin_lower_bound: float = 0.2,
    range_bin_width: float = 0.05,
    range_bin_upper_bound: Optional[float] = None,
    mean_bin_lower_bound: Optional[float] = None,
    mean_bin_width: float = 10.0,
    mean_bin_upper_bound: Optional[float] = None,
    damage_tolerance_for_binning: float = 1e-3,
    max_consetutive_zeros: int = 10,
    damage_exponent: float = 5.0,
    round_decimals: int = 4,
    debug_mode: bool = False,
):
    """
    Returns the cycle-count of the input data calculated through the
    :term:`rainflow<Rainflow>` method.

    Parameters
    ----------
    data : np.ndarray
        time series or residuals sequence
    time : Optional[Union[np.ndarray, list]], optional
        sampled times, by default None
    range_bin_lower_bound : float, optional
        lower bound of the range bin, by default 0.2
    range_bin_width : float, optional
        width of the range bin, by default 0.05
    range_bin_upper_bound : Optional[float], optional
        upper bound of the range bin, by default None
    mean_bin_lower_bound : Optional[float], optional
        lower bound of the mean bin, by default None
    mean_bin_width : float, optional
        width of the mean bin, by default 10
    mean_bin_upper_bound : Optional[float], optional
        upper bound of the mean bin, by default None
    damage_tolerance_for_binning : float, optional
        tolerance for the damage when binning large cycles
        separately, by default 1e-3
    damage_exponent : float, optional
        exponent for the damage when binning large cycles separately, by
        default 5.0
    round_decimals : int, optional
        number of decimals to round the output, by default 4
    debug_mode : bool, optional
        if True, print debug messages, by default False
    Returns
    -------
    Union[np.ndarray, tuple]
        - if np.ndarray:
            + rfs : np.ndarray
                rainflow
                [ampl ampl_mean nr_of_cycle cycle_begin_time cycle_period_time]
        - if tuple:
            + rfs : np.ndarray
                rainflow
                [ampl ampl_mean nr_of_cycle cycle_begin_time cycle_period_time]
            + data[res_tp] : numpy.ndarray
                the residuals signal
            + res_tp : numpy.ndarray
                the indices of the residuals signal
            + time[res_tp] : numpy.ndarray
                the time signal for residuals

    Raises
    ------
    TypeError
        data shall be numpy.ndarray or list

    See also
    --------
    calc_rainflow
    rainflow_binner
    """
    rfs, res_seq, _ = calc_rainflow(
        data=data,
        time=time,
        extended_output=True,
    )

    # return rfs, data[res_tp], res_tp, time[res_tp]

    means = MeanStress(
        _counts=rfs[:, 2],
        _values=rfs[:, 1],
        bin_width=mean_bin_width,
        _bin_lb=mean_bin_lower_bound,
        _bin_ub=mean_bin_upper_bound,
    )
    ranges = StressRange(
        _counts=rfs[:, 2],
        _values=2 * rfs[:, 0],
        bin_width=range_bin_width,
        _bin_lb=range_bin_lower_bound,
        _bin_ub=range_bin_upper_bound,
    )
    out_dct = rainflow_binner(
        ranges=ranges,
        _means=means,
        damage_tolerance_for_binning=damage_tolerance_for_binning,
        max_consecutive_zeros=max_consetutive_zeros,
        damage_exponent=damage_exponent,
        round_decimals=round_decimals,
        debug_mode=debug_mode,
    )

    out_dct["res_sig"] = res_seq
    if time is not None:
        out_dct["as_dur"] = time[-1] - time[0]

    return out_dct


def _build_binned_dict(
    ranges: StressRange,
    means: MeanStress,
    lg_c: list,
    the_hist: List[List[float]],
    round_decimals: int,
    no_mean_stress: bool,
    out_dct: dict,
):
    """
    Builds the output rainflow dictionary in py_fatigue format
    after binning the data.

    Parameters
    ----------
    ranges : StressRange
        The stress range object.
    means : MeanStress
        The mean stress object.
    lg_c : list
        The list of large cycles.
    the_hist : List[List[float]]
        The histogram.
    round_decimals : int
        The number of decimals to round the output.
    no_mean_stress : bool
        If True, the mean stress is not included in the output.
    out_dct : dict
        The output dictionary.

    Returns
    -------
    out_dct : dict
        The output dictionary.
    """

    # 11) Saving output to dictionary
    if no_mean_stress:
        out_dct["hist"] = [
            sum(x) for x in itertools.zip_longest(*the_hist, fillvalue=0)
        ]
        out_dct["lg_c"] = (
            np.asarray(np.around(lg_c, round_decimals))[:, 1]
            if len(lg_c) > 0
            else np.empty(0)
        ).tolist()
        out_dct["res"] = np.around(ranges.half, round_decimals).tolist()
    else:
        out_dct["hist"] = the_hist
        out_dct["lg_c"] = np.around(lg_c, round_decimals)
        out_dct["res"] = np.asarray(
            np.around(list(zip(means.half, ranges.half)), round_decimals)
        ).tolist()
    for key, value in out_dct.items():
        if isinstance(value, np.ndarray):
            out_dct[key] = value.tolist()

    return out_dct
