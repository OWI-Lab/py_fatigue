import numpy as np
import numba as nb
import pytest

import py_fatigue.utils as pu
from py_fatigue.damage.stress_life import find_sn_curve_intersection
from py_fatigue.mean_stress.corrections import goodman_haigh_mean_stress_correction


def test_to_numba_dict_disable_jit_path(monkeypatch):
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    out = pu.to_numba_dict({"a": 1.0, "b": "x", 1: 2.0})
    assert out == {"a": 1.0}


def test_compile_specialized_bisect_marks_specialized_in_compat(monkeypatch):
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    pu.compile_specialized_bisect.cache_clear()

    def fn(x):
        return x - 1.0

    bisect_fn = pu.compile_specialized_bisect(fn)
    assert getattr(bisect_fn, pu.NUMBA_SPECIALIZED_ATTR, False)
    assert np.isclose(bisect_fn(0.0, 2.0, 1e-8, 100), 1.0, atol=1e-5)


def test_compile_specialized_bisect_boundary_shortcuts_in_compat(monkeypatch):
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    pu.compile_specialized_bisect.cache_clear()

    def fn(x):
        return x - 2.0

    bisect_fn = pu.compile_specialized_bisect(fn)
    assert bisect_fn(2.0, 5.0, 1e-8, 100) == 2.0
    assert bisect_fn(0.0, 2.0, 1e-8, 100) == 2.0


def test_goodman_haigh_unsorted_r_out_branch():
    amp_in = np.array([100.0, 150.0])
    mean_in = np.array([10.0, 20.0])
    r_out = np.array([0.2, -1.0])  # intentionally unsorted

    amp_out, mean_out = goodman_haigh_mean_stress_correction(
        amp_in,
        mean_in,
        r_out,
        ult_s=900.0,
        correction_exponent=1.0,
    )

    assert amp_out.shape == (2, 2)
    assert mean_out.shape == (2, 2)


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (lambda: nb.njit(lambda x: pu.scalarize_numba_result(x)), 2.0),
        (lambda: nb.njit(lambda x: pu.scalarize_numba_result((x,))), 2.0),
        (
            lambda: nb.njit(
                lambda x: pu.scalarize_numba_result(np.array([x]))
            ),
            2.0,
        ),
        (lambda: nb.njit(lambda x: pu.scalarize_numba_result([x])), 2.0),
    ],
)
def test_scalarize_numba_result_scalar_like_paths(monkeypatch, factory, expected):
    monkeypatch.delenv("NUMBA_DISABLE_JIT", raising=False)
    monkeypatch.setattr(pu.nb.config, "DISABLE_JIT", False)

    compiled = factory()
    assert compiled(2.0) == pytest.approx(expected)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: nb.njit(lambda x: pu.scalarize_numba_result((x, x + 1.0))),
        lambda: nb.njit(
            lambda x: pu.scalarize_numba_result(np.array([x, x + 1.0]))
        ),
        lambda: nb.njit(lambda x: pu.scalarize_numba_result([x, x + 1.0])),
    ],
)
def test_scalarize_numba_result_rejects_non_scalar(monkeypatch, factory):
    monkeypatch.delenv("NUMBA_DISABLE_JIT", raising=False)
    monkeypatch.setattr(pu.nb.config, "DISABLE_JIT", False)

    compiled = factory()
    with pytest.raises(TypeError):
        compiled(2.0)


def test_compile_specialized_bisect_py_func_branches(monkeypatch):
    monkeypatch.delenv("NUMBA_DISABLE_JIT", raising=False)
    monkeypatch.setattr(pu.nb.config, "DISABLE_JIT", False)
    pu.compile_specialized_bisect.cache_clear()

    def fn(x):
        return x - 1.0

    bisect_fn = pu.compile_specialized_bisect(fn)
    assert bisect_fn.__class__.__name__ == "CPUDispatcher"

    # Exercise the Python fallback body for coverage of tolerance and loop paths.
    assert bisect_fn.py_func(1.0, 2.0, 1e-8, 100) == 1.0
    assert bisect_fn.py_func(0.0, 1.0, 1e-8, 100) == 1.0
    assert np.isclose(bisect_fn.py_func(0.0, 2.0, 1e-8, 100), 1.0, atol=1e-5)


def test_warmup_numba_short_circuits_when_jit_disabled(monkeypatch):
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")

    def _explode(*args, **kwargs):
        raise AssertionError("warmup_numba should short-circuit before imports")

    monkeypatch.setattr(pu, "import_module", _explode)
    pu.warmup_numba()


def test_goodman_clip_path_with_non_linear_exponent():
    amp_in = np.array([100.0, 200.0])
    mean_in = np.array([5.0, 10.0])
    r_out = np.array([-1.0, 0.0])

    amp_out, mean_out = goodman_haigh_mean_stress_correction(
        amp_in,
        mean_in,
        r_out,
        ult_s=500.0,
        correction_exponent=2.0,
    )

    assert np.all(np.isfinite(amp_out))
    assert np.all(amp_out >= 0.0)
    assert np.all(amp_out <= 500.0)
    assert np.all(np.isfinite(mean_out))


def test_find_sn_curve_intersection_wrapper_executes():
    out = find_sn_curve_intersection(
        np.array([3.0, 5.0]),
        np.array([12.0, 15.0]),
        1e8,
        120.0,
        0.0,
        1e2,
        1e9,
    )

    assert np.isfinite(out)
    assert out > 0.0
