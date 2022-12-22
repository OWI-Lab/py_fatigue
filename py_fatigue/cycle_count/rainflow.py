# -*- coding: utf-8 -*-

r"""The following code is based on the Python `[1]`_ Rainflow algorithm
implemented in the WAFO toolbox `[2]`_. The method gives the following
definition of rainflow cycles, in accordance with Rychlik (1987) `[3]`_:

From each local maximum M\ :sub:`k` one shall try to reach above the same
level, in the backward (left) and forward (right) directions, with an as small
downward excursion as possible. The minima,
m\ :sup:`-`\ :sub:`k` and m\ :sup:`+`\ :sub:`k`\ , on each side are identified.
The minimum that represents the smallest deviation from the maximum M\ :sub:`k`
is defined as the corresponding rainflow minimum
m\ :sup:`RFC`\ :sub:`k`\ . The k:th rainflow cycle
is defined as (m\ :sup:`RFC`\ :sub:`k`\ , M\ :sub:`k`\ ).

.. _[1]: https://www.maths.lth.se/matstat/wafo/
.. _[2]: https://github.com/wafo-project/pywafo
.. _[3]: https://www.sciencedirect.com/science/article/abs/pii/0142112387900545

See also
---------
findtp : Find indices to turning points.
findextrema : Find indices to local maxima and minima.
"""

# from future imports
from __future__ import absolute_import, division, print_function

# standard imports
# import itertools
import warnings
from typing import Any, Optional, Union

# non-standard imports
import numpy as np
from numba import njit, int64, int8


__all__ = ["rainflow", "findextrema", "findtp", "findrfc_astm", "findcross"]


def findtp(x: np.ndarray) -> np.ndarray:

    """
    Return indices to turning points (tp) of ASTM rainflow filtered data.

    Parameters
    ----------
    x : vector
        signal to be filtered.

    Returns
    -------
    ind : Union[np.ndarray, None]
        indices to the turning points in the original sequence.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import py_fatigue.wafo.misc as wm
    >>> t = np.linspace(0,30,500).reshape((-1,1))
    >>> x = np.hstack((t, np.cos(t) + 0.3 * np.sin(5*t)))
    >>> x1 = x[0:100,:]
    >>> itp = wm.findtp(x1[:,1])
    >>> tp = x1[itp,:]
    >>> np.allclose(itp, [ 5, 18, 24, 38, 46, 57, 70, 76, 91, 98, 99])
    True

    >>> a = plt.plot(
    ... x1[:,0], x1[:,1], tp[:,0], tp[:,1], 'ro')
    >>> plt.close('all')

    See also
    ---------
    findextrema
    """

    ind = findextrema(x)

    if ind.size < 2:
        return np.empty(0, dtype=np.int64)

    # # the Nieslony approach always put the first loading point as the first
    # # turning point.
    # add the first turning point is the first of the signal
    # if ind[0] != 0:
    #     ind = np.r_[0, ind, len(x) - 1]
    # else:  # only add the last point of the signal
    #     ind = np.r_[ind, len(x) - 1]
    return np.r_[0, ind, len(x) - 1]


def findcross(x: np.ndarray, v=0.0, kind=None) -> np.ndarray:
    """
    Return indices to level v up and/or downcrossings of a vector

    Parameters
    ----------
    x : array_like
        vector with sampled values.
    v : scalar, real
        level v.
    kind : string
        defines type of crossing returned. Possible options are
        - 'd'  : downcrossings only
        - 'u'  : upcrossings only
        - None : All crossings will be returned

    Returns
    -------
    ind : array-like
        indices to the crossings in the original sequence x.

    Example
    -------
    >>> from matplotlib import pyplot as plt
    >>> import py_fatigue.wafo.misc as wm
    >>> ones = np.ones
    >>> np.allclose(findcross([0, 1, -1, 1], 0), [0, 1, 2])
    True
    >>> v = 0.75
    >>> t = np.linspace(0,7*np.pi,250)
    >>> x = np.sin(t)
    >>> ind = wm.findcross(x,v) # all crossings
    >>> np.allclose(ind, [  9,  25,  80,  97, 151, 168, 223, 239])
    True

    >>> ind2 = wm.findcross(x,v,'u')
    >>> np.allclose(ind2, [  9,  80, 151, 223])
    True
    >>> ind3 = wm.findcross(x,v,'d')
    >>> np.allclose(ind3, [  25,  97, 168, 239])
    True

    >>> t0 = plt.plot(t,x,'.',t[ind],x[ind],'r.', t, ones(t.shape)*v)
    >>> t0 = plt.plot(t[ind2],x[ind2],'o')
    >>> plt.close('all')
    """
    if len(x) > 0 and np.isnan(np.min(x)):
        raise ValueError("data contain NaN")

    xn = np.array(np.sign(np.atleast_1d(x).ravel() - v), dtype=np.int8)
    ind = findcross_indices(xn)
    if ind.size == 0:
        warnings.warn(f"No level v = {v:0.5g} crossings found in x")
        return ind

    if kind not in ("du", "all", None):
        if kind == "d":  # downcrossings only
            t_0 = int(
                xn[ind[0] + 1] > 0  # pylint: disable=unsubscriptable-object
            )
            ind = ind[t_0::2]
        elif kind == "u":  # upcrossings  only
            t_0 = int(
                xn[ind[0] + 1] < 0  # pylint: disable=unsubscriptable-object
            )
            ind = ind[t_0::2]
        else:
            raise ValueError(f"Unknown cycle-crossing definition! ({kind})")
    return ind


def findextrema(x: np.ndarray) -> np.ndarray:
    """Return indices to minima and maxima of a vector

    Parameters
    ----------
    x : vector with sampled values.

    Returns
    -------
    ind : indices to minima and maxima in the original sequence x.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import wafo.misc as wm
    >>> t = np.linspace(0,7*np.pi,250)
    >>> x = np.sin(t)
    >>> ind = wm.findextrema(x)
    >>> np.allclose(ind, [ 18,  53,  89, 125, 160, 196, 231])
    True

    >>> a = plt.plot(t,x,'.',t[ind],x[ind],'r.')
    >>> plt.close('all')
    """
    dx = np.diff(np.atleast_1d(x).ravel())
    return findcross(dx) + 1


def findcross_indices(xn: Any) -> np.ndarray:
    """Return indices to zero up and downcrossings of a vector"""
    ind = np.empty(len(xn), dtype=np.int64)
    idx = _findcross(ind, xn)
    return ind[:idx]


@njit(int64(int64[:], int8[:]))
def _findcross(ind, y):
    """Return indices to zero up and downcrossings of a vector

    Parameters
    ----------
    ind : int64[:]
        Indices of the turning points of the rainflow filter
    y : int8[:]
        The data

    Returns
    -------
    np.ndarray
        The number of indices
    """
    ix, dcross, start, v = 0, 0, 0, 0
    n = len(y)
    if n == 0:
        return ix
    if y[0] < v:
        dcross = -1  # first is a up-crossing
    elif y[0] > v:
        dcross = 1  # first is a down-crossing
    elif y[0] == v:
        # Find out what type of crossing we have next time..
        for i in range(1, n):
            start = i
            if y[i] < v:
                ind[ix] = i - 1  # first crossing is a down crossing
                ix += 1
                dcross = -1  # The next crossing is a up-crossing
                break
            if y[i] > v:
                ind[ix] = i - 1  # first crossing is a up-crossing
                ix += 1
                dcross = 1  # The next crossing is a down-crossing
                break

    for i in range(start, n - 1):
        # if (dcross == -1 and y[i] <= v and v < y[i + 1]) or (
        #     dcross == 1 and v <= y[i] and y[i + 1] < v
        # ):
        if (dcross == -1 and y[i] <= v < y[i + 1]) or (
            dcross == 1 and y[i + 1] < v <= y[i]
        ):
            ind[ix] = i
            ix += 1
            dcross = -dcross
    return ix


def _findrfc3_astm(
    array_ext: np.ndarray, a: np.ndarray, array_out: np.ndarray
) -> tuple:
    """
    Rainflow without time analysis

    Return [ampl ampl_mean nr_of_cycle]

    By Adam Nieslony
    Visit the MATLAB Central File Exchange for latest version
    http://www.mathworks.com/matlabcentral/fileexchange/3026
    """
    n = len(array_ext)
    po = 0
    # The original rainflow counting by Nieslony, unchanged
    j = -1
    c_nr1 = 1
    for i in range(n):
        j += 1
        a[j] = array_ext[i]
        while j >= 2 and abs(a[j - 1] - a[j - 2]) <= abs(a[j] - a[j - 1]):
            ampl = abs((a[j - 1] - a[j - 2]) / 2)
            mean = (a[j - 1] + a[j - 2]) / 2
            if j == 2:
                a[0] = a[1]
                a[1] = a[2]
                j = 1
                if ampl > 0:
                    array_out[po, :] = (ampl, mean, 0.5)
                    po += 1
            else:
                a[j - 2] = a[j]
                j = j - 2
                if ampl > 0:
                    array_out[po, :] = (ampl, mean, 1.0)
                    po += 1
                    c_nr1 += 1

    c_nr2 = 1
    for i in range(j):
        ampl = abs(a[i] - a[i + 1]) / 2
        mean = (a[i] + a[i + 1]) / 2
        if ampl > 0:
            array_out[po, :] = (ampl, mean, 0.5)
            po += 1
            c_nr2 += 1
    return c_nr1, c_nr2


def _findrfc5_astm(
    array_ext: np.ndarray,
    array_t: np.ndarray,
    a: np.ndarray,
    t: np.ndarray,
    array_out: np.ndarray,
) -> tuple:
    """
    Rainflow with time analysis

    returns
    [ampl ampl_mean nr_of_cycle cycle_begin_time cycle_period_time]

    By Adam Nieslony
    Visit the MATLAB Central File Exchange for latest version
    http://www.mathworks.com/matlabcentral/fileexchange/3026
    """
    n = len(array_ext)
    po = 0
    # The original rainflow counting by Nieslony, unchanged
    j = -1
    c_nr1 = 1
    for i in range(n):
        j += 1
        a[j] = array_ext[i]
        t[j] = array_t[i]
        while (j >= 2) and (abs(a[j - 1] - a[j - 2]) <= abs(a[j] - a[j - 1])):
            ampl = abs((a[j - 1] - a[j - 2]) / 2)
            mean = (a[j - 1] + a[j - 2]) / 2
            period = (t[j - 1] - t[j - 2]) * 2
            atime = t[j - 2]
            if j == 2:
                a[0] = a[1]
                a[1] = a[2]
                t[0] = t[1]
                t[1] = t[2]
                j = 1
                if ampl > 0:
                    array_out[po, :] = (ampl, mean, 0.5, atime, period)
                    po += 1
            else:
                a[j - 2] = a[j]
                t[j - 2] = t[j]
                j = j - 2
                if ampl > 0:
                    array_out[po, :] = (ampl, mean, 1.0, atime, period)
                    po += 1
                    c_nr1 += 1

    c_nr2 = 1
    for i in range(j):
        # for (i=0; i<j; i++) {
        ampl = abs(a[i] - a[i + 1]) / 2
        mean = (a[i] + a[i + 1]) / 2
        period = (t[i + 1] - t[i]) * 2
        atime = t[i]
        if ampl > 0:
            array_out[po, :] = (ampl, mean, 0.5, atime, period)
            po += 1
            c_nr2 += 1
    return c_nr1, c_nr2


def findrfc_astm(tp: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Return rainflow counted cycles

    Nieslony's Matlab implementation of the ASTM standard practice for rainflow
    counting ported to a Python C module.

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

    y1 = np.atleast_1d(tp).ravel()
    n = len(y1)
    a = np.zeros(n)
    if t is None:
        sig_rfc = np.zeros((n, 3))
        cnr = _findrfc3_astm(y1, a, sig_rfc)
    else:
        t1 = np.atleast_1d(t).ravel()
        sig_rfc = np.zeros((n, 5))
        t2 = np.zeros(n)
        cnr = _findrfc5_astm(y1, t1, a, t2, sig_rfc)
    # the sig_rfc was constructed too big in rainflow.rf3, so
    # reduce the sig_rfc array as done originally by a matlab mex c function
    # n = len(sig_rfc)
    return sig_rfc[: n - cnr[0]]


def rainflow(
    data: Union[np.ndarray, list],
    time: Optional[Union[np.ndarray, list]] = None,
    extended_output: bool = True,
) -> Union[np.ndarray, tuple]:
    """
    Returns the cycle-count of the input data calculated through the
    :term:`rainflow<Rainflow>` method.

    Parameters
    ----------
    data : Union[np.ndarray, list]
        time series or residuals sequence
    time : Optional[Union[np.ndarray, list]], optional
        sampled times, by default None
        extended_output : bool, optional
        if False it returns only the :term:`rainflow<Rainflow>`, if True
        returns also the residuals signal, by default True

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
    findtp : Find indices to turning points.
    findextrema : Find indices to local maxima and minima.
    findrfc_astm : Find rainflow cycles.
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
    rfs = findrfc_astm(data[idx], idx)
    res = rfs[rfs[:, 2] == 0.5]
    res_tp = res[:, -2].astype(int)
    res_tp = np.append(res_tp, [len(data) - 1])
    if extended_output:
        if time is None:
            return rfs, data[res_tp], res_tp
        return rfs, data[res_tp], res_tp, time[res_tp]
    return rfs
