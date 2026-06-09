"""Benchmark numba call patterns used by py-fatigue.

The script keeps package logic unchanged. It compares current public call
paths with equivalent call patterns that hoist JIT compilation out of tight
loops, and it inventories the numba-decorated functions found in the package.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import statistics
import sys
import time
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from rich.console import Console
from rich.table import Table
from rich_argparse import RichHelpFormatter

from py_fatigue.cycle_count.rainflow import findcross, findtp
from py_fatigue.damage.crack_growth import CalcCrackGrowth, get_sif
from py_fatigue.damage.stress_life import (
    calc_theil_sn_damage,
    find_sn_curve_intersection,
)
from py_fatigue.geometry.cylinder import f_hol_cyl_01
from py_fatigue.material.crack_growth_curve import ParisCurve
from py_fatigue.material.sn_curve import SNCurve
from py_fatigue.mean_stress.corrections import (
    goodman_haigh_mean_stress_correction,
)
from py_fatigue.utils import (
    compile_specialized_bisect,
    compile_specialized_newton,
    numba_bisect,
    numba_newton,
    py_bisect,
    py_newton,
    to_numba_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "py_fatigue"


@dataclass(frozen=True)
class BenchmarkResult:
    """A single benchmark row."""

    group: str
    name: str
    calls: int
    seconds: float | None
    per_call_ms: float | None
    speedup_vs_group_base: float | None = None
    status: str = "ok"
    note: str = ""


@dataclass(frozen=True)
class InventoryItem:
    """A numba-related symbol discovered by AST inspection."""

    file: str
    line: int
    kind: str
    name: str
    mechanism: str


def quadratic_bisect(x_value: float) -> float:
    """Simple scalar residual for bisection benchmarks."""

    return x_value * x_value - 2.0


def quadratic_newton(x_value: float, target: float) -> float:
    """Simple scalar residual for Newton benchmarks."""

    return x_value * x_value - target


def time_repeated(
    func: Callable[[], Any],
    calls: int,
    repeats: int,
) -> float:
    """Return the median wall time for repeated calls."""

    timings: list[float] = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(calls):
                func()
            timings.append(time.perf_counter() - start)
    finally:
        if was_enabled:
            gc.enable()
    return statistics.median(timings)


def add_result(
    results: list[BenchmarkResult],
    group: str,
    name: str,
    calls: int,
    seconds: float | None,
    baseline: float | None = None,
    status: str = "ok",
    note: str = "",
) -> None:
    """Append a benchmark result with derived per-call and speedup values."""

    per_call_ms = None if seconds is None else seconds / calls * 1_000
    speedup = None
    if seconds is not None and baseline is not None and seconds > 0:
        speedup = baseline / seconds
    results.append(
        BenchmarkResult(
            group=group,
            name=name,
            calls=calls,
            seconds=seconds,
            per_call_ms=per_call_ms,
            speedup_vs_group_base=speedup,
            status=status,
            note=note,
        )
    )


def measure_case(
    results: list[BenchmarkResult],
    group: str,
    name: str,
    func: Callable[[], Any],
    calls: int,
    repeats: int,
    baseline: float | None = None,
) -> float | None:
    """Run a timed case and append the result."""

    try:
        elapsed = time_repeated(func, calls=calls, repeats=repeats)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        add_result(
            results,
            group,
            name,
            calls,
            None,
            baseline=baseline,
            status="error",
            note=f"{type(exc).__name__}: {exc}",
        )
        return None

    add_result(results, group, name, calls, elapsed, baseline=baseline)
    return elapsed


def benchmark_root_finders(
    calls: int,
    repeats: int,
) -> list[BenchmarkResult]:
    """Benchmark current root-finder wrappers against hoisted compilation."""

    results: list[BenchmarkResult] = []

    bisect_base = measure_case(
        results,
        "bisect-wrapper",
        "current numba_bisect(py_func)",
        lambda: numba_bisect(quadratic_bisect, 0.0, 2.0, 1e-8, 100),
        calls,
        repeats,
    )

    solver = compile_specialized_bisect(quadratic_bisect)
    solver(0.0, 2.0, 1e-8, 100)
    measure_case(
        results,
        "bisect-wrapper",
        "hoisted compile_specialized_bisect",
        lambda: solver(0.0, 2.0, 1e-8, 100),
        calls,
        repeats,
        baseline=bisect_base,
    )
    measure_case(
        results,
        "bisect-wrapper",
        "pure Python py_bisect",
        lambda: py_bisect(quadratic_bisect, 0.0, 2.0, 1e-8, 100),
        calls,
        repeats,
        baseline=bisect_base,
    )

    newton_base = measure_case(
        results,
        "newton-wrapper",
        "current numba_newton(py_func)",
        lambda: numba_newton(quadratic_newton, 1.0, 1e-8, 100, 2.0),
        calls,
        repeats,
    )

    newton_solver = compile_specialized_newton(quadratic_newton)
    newton_solver(1.0, 1e-8, 100, 2.0)
    measure_case(
        results,
        "newton-wrapper",
        "hoisted compile_specialized_newton",
        lambda: newton_solver(1.0, 1e-8, 100, 2.0),
        calls,
        repeats,
        baseline=newton_base,
    )
    measure_case(
        results,
        "newton-wrapper",
        "pure Python py_newton",
        lambda: py_newton(quadratic_newton, 1.0, 1e-8, 100, 2.0),
        calls,
        repeats,
        baseline=newton_base,
    )

    return results


def benchmark_public_paths(
    calls: int,
    repeats: int,
) -> list[BenchmarkResult]:
    """Benchmark public workflows that exercise each numba area."""

    results: list[BenchmarkResult] = []

    sn_curve = SNCurve([4, 5], [15.117, 17.146], endurance=1e9)
    stress_range = np.linspace(80.0, 240.0, 10_000)
    cycles = np.logspace(5, 9, 10_000)
    sn_curve.get_cycles(stress_range)
    sn_curve.get_stress(cycles)

    sn_base = measure_case(
        results,
        "direct-dispatchers",
        "SNCurve.get_cycles",
        lambda: sn_curve.get_cycles(stress_range),
        calls,
        repeats,
    )
    measure_case(
        results,
        "direct-dispatchers",
        "SNCurve.get_stress",
        lambda: sn_curve.get_stress(cycles),
        calls,
        repeats,
        baseline=sn_base,
    )

    paris_curve = ParisCurve(
        slope=[2.88, 5.1, 8.16, 5.1, 2.88],
        intercept=[1e-16, 1e-20, 1e-27, 1e-19, 1e-13],
        threshold=20.0,
        critical=2_000.0,
    )
    sif_range = np.linspace(20.0, 2_000.0, 10_000)
    growth_rate = np.logspace(-10, -4, 10_000)
    paris_curve.get_growth_rate(sif_range)
    paris_curve.get_sif(growth_rate)
    measure_case(
        results,
        "direct-dispatchers",
        "ParisCurve.get_growth_rate",
        lambda: paris_curve.get_growth_rate(sif_range),
        calls,
        repeats,
        baseline=sn_base,
    )
    measure_case(
        results,
        "direct-dispatchers",
        "ParisCurve.get_sif",
        lambda: paris_curve.get_sif(growth_rate),
        calls,
        repeats,
        baseline=sn_base,
    )

    signal = np.sin(np.linspace(0.0, 2_000.0, 20_000))
    signal += 0.15 * np.sin(np.linspace(0.0, 40_000.0, 20_000))
    findcross(signal)
    findtp(signal)
    measure_case(
        results,
        "direct-dispatchers",
        "rainflow.findcross",
        lambda: findcross(signal),
        calls,
        repeats,
        baseline=sn_base,
    )
    measure_case(
        results,
        "direct-dispatchers",
        "rainflow.findtp",
        lambda: findtp(signal),
        calls,
        repeats,
        baseline=sn_base,
    )

    geometry = to_numba_dict(
        {
            "initial_depth": 1.0,
            "outer_diameter": 100.0,
            "thickness": 10.0,
            "height": 200.0,
            "width_to_depth_ratio": 2.0,
        }
    )
    f_hol_cyl_01(1.0, geometry)
    get_sif(100.0, 1.0, "HOL_CYL_01", geometry)
    measure_case(
        results,
        "direct-dispatchers",
        "f_hol_cyl_01",
        lambda: f_hol_cyl_01(1.0, geometry),
        calls,
        repeats,
        baseline=sn_base,
    )
    measure_case(
        results,
        "direct-dispatchers",
        "crack_growth.get_sif",
        lambda: get_sif(100.0, 1.0, "HOL_CYL_01", geometry),
        calls,
        repeats,
        baseline=sn_base,
    )

    cg_stress = np.full(500, 100.0, dtype=np.float64)
    count_cycle = np.ones(500, dtype=np.float64)
    slope = np.array([3.0], dtype=np.float64)
    intercept = np.array([1e-12], dtype=np.float64)
    inf_geometry = to_numba_dict({"initial_depth": 1.0})

    def build_crack_growth() -> CalcCrackGrowth:
        with redirect_stdout(StringIO()):
            return CalcCrackGrowth(
                cg_stress,
                count_cycle,
                slope,
                intercept,
                0.0,
                1e9,
                "INF_SUR_00",
                inf_geometry,
            )

    build_crack_growth()
    measure_case(
        results,
        "direct-dispatchers",
        "CalcCrackGrowth construction",
        build_crack_growth,
        calls,
        repeats,
        baseline=sn_base,
    )

    small_stress = np.array([120.0, 180.0, 90.0], dtype=np.float64)
    small_counts = np.array([10.0, 8.0, 12.0], dtype=np.float64)
    calc_theil_sn_damage(small_stress, small_counts, sn_curve)
    measure_case(
        results,
        "direct-dispatchers",
        "calc_theil_sn_damage",
        lambda: calc_theil_sn_damage(small_stress, small_counts, sn_curve),
        calls,
        repeats,
        baseline=sn_base,
    )

    amp_in = np.linspace(20.0, 300.0, 40)
    mean_in = np.linspace(-10.0, 60.0, 40)
    r_out = np.array([-1.0, -0.5, 0.0, 0.5])
    goodman_haigh_mean_stress_correction(
        amp_in,
        mean_in,
        r_out,
        1_000.0,
        3.0,
    )
    measure_case(
        results,
        "direct-dispatchers",
        "goodman_haigh_mean_stress_correction",
        lambda: goodman_haigh_mean_stress_correction(
            amp_in,
            mean_in,
            r_out,
            1_000.0,
            3.0,
        ),
        max(1, calls // 5),
        repeats,
        baseline=sn_base,
    )

    try:
        find_sn_curve_intersection(
            sn_curve.slope,
            sn_curve.intercept,
            sn_curve.endurance,
            0.1,
            10.0,
            1.0,
            1e15,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        add_result(
            results,
            "direct-dispatchers",
            "find_sn_curve_intersection",
            1,
            None,
            baseline=sn_base,
            status="error",
            note=f"{type(exc).__name__}: {exc}",
        )
    else:
        measure_case(
            results,
            "direct-dispatchers",
            "find_sn_curve_intersection",
            lambda: find_sn_curve_intersection(
                sn_curve.slope,
                sn_curve.intercept,
                sn_curve.endurance,
                0.1,
                10.0,
                1.0,
                1e15,
            ),
            max(1, calls // 10),
            repeats,
            baseline=sn_base,
        )

    return results


def decorator_text(node: ast.AST) -> str:
    """Return best-effort source text for a decorator or call."""

    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover
        return type(node).__name__


def iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    """Yield Python files below the provided paths."""

    for path in paths:
        if path.is_file() and path.suffix == ".py":
            yield path
        elif path.is_dir():
            yield from sorted(path.rglob("*.py"))


def inspect_inventory() -> list[InventoryItem]:
    """Inspect package sources for numba decorators and wrapper call sites."""

    items: list[InventoryItem] = []
    for py_file in iter_python_files([PACKAGE_ROOT]):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        rel_path = py_file.relative_to(PROJECT_ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                for decorator in node.decorator_list:
                    text = decorator_text(decorator)
                    if any(
                        token in text
                        for token in ("njit", "jitclass", "vectorize")
                    ):
                        items.append(
                            InventoryItem(
                                file=rel_path,
                                line=node.lineno,
                                kind=type(node).__name__,
                                name=node.name,
                                mechanism=f"@{text}",
                            )
                        )
            if isinstance(node, ast.Assign) and isinstance(
                node.value, ast.Call
            ):
                text = decorator_text(node.value.func)
                if text.endswith(
                    (
                        "compile_specialized_bisect",
                        "compile_specialized_newton",
                    )
                ):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            items.append(
                                InventoryItem(
                                    file=rel_path,
                                    line=node.lineno,
                                    kind="Assign",
                                    name=target.id,
                                    mechanism=text,
                                )
                            )
            if isinstance(node, ast.Call):
                text = decorator_text(node.func)
                if text in {"numba_bisect", "numba_newton"}:
                    items.append(
                        InventoryItem(
                            file=rel_path,
                            line=node.lineno,
                            kind="Call",
                            name=text,
                            mechanism="generic wrapper call",
                        )
                    )
    return sorted(items, key=lambda item: (item.file, item.line, item.name))


def render_inventory(console: Console, items: list[InventoryItem]) -> None:
    """Print the numba inventory."""

    table = Table(title="Numba inventory")
    table.add_column("File")
    table.add_column("Line", justify="right")
    table.add_column("Kind")
    table.add_column("Name")
    table.add_column("Mechanism")
    for item in items:
        table.add_row(
            item.file,
            str(item.line),
            item.kind,
            item.name,
            item.mechanism,
        )
    console.print(table)


def render_results(console: Console, results: list[BenchmarkResult]) -> None:
    """Print benchmark results."""

    table = Table(title="Numba benchmark results")
    table.add_column("Group")
    table.add_column("Case")
    table.add_column("Calls", justify="right")
    table.add_column("Total (s)", justify="right")
    table.add_column("ms/call", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Status")
    table.add_column("Note")
    for result in results:
        total = "-" if result.seconds is None else f"{result.seconds:.6f}"
        per_call = (
            "-" if result.per_call_ms is None else f"{result.per_call_ms:.3f}"
        )
        speedup = (
            "-"
            if result.speedup_vs_group_base is None
            else f"{result.speedup_vs_group_base:.1f}x"
        )
        table.add_row(
            result.group,
            result.name,
            str(result.calls),
            total,
            per_call,
            speedup,
            result.status,
            result.note,
        )
    console.print(table)


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument(
        "--calls",
        type=int,
        default=10,
        help="Calls per benchmark repeat. Higher values amplify differences.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats used to compute the median wall time.",
    )
    parser.add_argument(
        "--skip-inventory",
        action="store_true",
        help="Do not print the static numba inventory.",
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Do not run timing benchmarks.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON file for machine-readable results.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the inventory and benchmarks."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    console = Console()

    inventory: list[InventoryItem] = []
    results: list[BenchmarkResult] = []

    if not args.skip_inventory:
        inventory = inspect_inventory()
        render_inventory(console, inventory)

    if not args.skip_benchmarks:
        results.extend(benchmark_root_finders(args.calls, args.repeats))
        results.extend(benchmark_public_paths(args.calls, args.repeats))
        render_results(console, results)

    if args.json_output is not None:
        payload = {
            "inventory": [asdict(item) for item in inventory],
            "results": [asdict(result) for result in results],
        }
        args.json_output.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        console.print(f"Wrote JSON results to {args.json_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
