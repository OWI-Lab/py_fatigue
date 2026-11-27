"""Exercise the local MCP through the same subprocess a host launches."""

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Iterator

import numpy as np
import mcp
import pytest

from py_fatigue import CycleCount, SNCurve
from py_fatigue import crack_growth as cg
from py_fatigue.damage.stress_life import calc_pm

pytestmark = pytest.mark.anyio

STRESS = [0, 75, -15, 50, 0, 60, -20, 90, 0, -30, 70, 0]
CURVE = cg.NasgroCurve(1e-10, 2, 0.5, 1, 25, 0, 500, 2, 0.3)
CONTROLS = dict(
    max_cycle_step=1, length_tolerance=1e-6, cycle_tolerance=1e-4, max_trials=40
)
LOADS = [cg.LoadBlock(0, 40, 8.5), cg.LoadBlock(10, 50, 10)]


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio for the SDK subprocess."""
    return "asyncio"


@pytest.fixture
async def client(tmp_path: Path):
    """Connect to the installed module from outside the checkout."""
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "output"
    inputs.mkdir()
    outputs.mkdir()
    command = shutil.which(
        "py-fatigue-mcp", path=str(Path(sys.executable).parent)
    )
    assert command is not None
    parameters = mcp.StdioServerParameters(
        command=command,
        cwd=tmp_path,
        env={
            "PY_FATIGUE_MCP_INPUT_DIR": str(inputs),
            "PY_FATIGUE_MCP_OUTPUT_DIR": str(outputs),
        },
    )
    async with mcp.Client(parameters) as connection:
        yield connection


async def call(client, name: str, **arguments) -> dict:
    """Call a tool and return its successful structured result."""
    result = await client.call_tool(name, arguments)
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return result.structured_content


async def assert_tool_error(client, name: str, match: str, **arguments) -> None:
    """Check an MCP error contains a useful explanation."""
    result = await client.call_tool(name, arguments)
    assert result.is_error
    assert match.casefold() in str(result.content).casefold()


def exported_rows(tmp_path: Path, result: dict) -> list[dict[str, str]]:
    """Read the result CSV referenced by a tool response."""
    export = result["export"]
    path = Path(export["relative_path"])
    assert path.name == str(path)
    with (tmp_path / "output" / path).open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == export["row_count"] == result["row_count"]
    return rows


async def test_lffd_recovery(client: mcp.Client) -> None:
    """Compare ordered-window recovery with the public CycleCount API."""
    windows = [
        [20, 26, 18, 35, 30, 43, 34, 45],
        [48, 43, 53, 45, 54, 40, 45, 27],
        [23, 30, 17, 23, 14, 21, 10, 23],
    ]
    timestamp = datetime(2000, 1, 1, tzinfo=timezone.utc)
    first, second, third = [
        CycleCount.from_timeseries(window, timestamp=timestamp)
        for window in windows
    ]
    combined = first + second + third
    recovered = combined.solve_lffd()

    result = await call(client, "lffd_recovery", windows=windows)
    assert result["window_count"] == 3
    assert result["windowed_total_cycles"] == combined.count_cycle.sum()
    assert result["windowed_max_stress_range"] == combined.stress_range.max()
    assert result["total_cycles"] == recovered.count_cycle.sum()
    assert result["max_stress_range"] == recovered.stress_range.max() == 44
    assert result["remaining_turning_points"] == len(
        recovered.residuals_sequence
    )
    assert result["row_count"] == len(recovered.count_cycle)
    assert len(result["preview"]) == 5
    await assert_tool_error(
        client, "lffd_recovery", "window 2", windows=[windows[0], [5, 5]]
    )
    await assert_tool_error(
        client, "lffd_recovery", "100,000 samples", windows=[[0] * 50_001] * 2
    )


async def test_rainflow_and_miner(client, tmp_path: Path) -> None:
    """Match direct cycle counting and use every row in Miner damage."""
    listed = await client.list_tools()
    assert {tool.name for tool in listed.tools} == {
        "rainflow_count",
        "miner_damage",
        "lffd_recovery",
        "ordered_crack_growth",
    }
    assert all(tool.output_schema for tool in listed.tools)

    direct = CycleCount.from_timeseries(STRESS)
    expected = np.column_stack(
        (direct.stress_range, direct.mean_stress, direct.count_cycle)
    )
    inline = await call(client, "rainflow_count", stress=STRESS)
    assert inline["export"] is None
    assert inline["row_count"] == len(expected) > 5
    assert inline["total_cycles"] == sum(direct.count_cycle)
    assert [list(row.values()) for row in inline["preview"]] == expected[
        [0, 1, 2, -2, -1]
    ].tolist()

    waveform = "stress\n" + "\n".join(map(str, STRESS))
    (tmp_path / "inputs" / "wave.csv").write_text(waveform, encoding="utf-8")
    stress_file = {"csv_path": "wave.csv"}
    csv_result = await call(
        client, "rainflow_count", stress=stress_file, export_csv=True
    )
    rows = exported_rows(tmp_path, csv_result)
    assert [[float(value) for value in row.values()] for row in rows] == (
        expected.tolist()
    )
    filename = csv_result["export"]["relative_path"]
    (tmp_path / "output" / filename).replace(tmp_path / "inputs" / filename)

    curve = dict(slope=[3, 5], intercept=[10.970, 13.617], endurance=1e8)
    contributions = calc_pm(
        direct.stress_range, direct.count_cycle, SNCurve(**curve)
    )
    csv_damage = await call(
        client,
        "miner_damage",
        source={"kind": "stress_history", "stress": stress_file},
        curve=curve,
        export_csv=True,
    )
    assert csv_damage["total_damage"] == pytest.approx(contributions.sum())
    assert len(csv_damage["preview"]) == 5
    assert [
        float(row["damage"]) for row in exported_rows(tmp_path, csv_damage)
    ] == pytest.approx(contributions)
    for cycles in (
        {"csv_path": filename},
        [
            {key: float(row[key]) for key in ("stress_range", "count_cycle")}
            for row in rows
        ],
    ):
        counted_damage = await call(
            client,
            "miner_damage",
            source={"kind": "counted_cycles", "cycles": cycles},
            curve=curve,
        )
        assert counted_damage["total_damage"] == csv_damage["total_damage"]


async def test_csv_and_validation_errors(client, tmp_path: Path) -> None:
    """Keep path access within the input root and identify invalid data."""
    inputs = tmp_path / "inputs"
    outside = tmp_path / "outside.csv"
    outside.write_text("stress\n0\n1\n", encoding="utf-8")
    (inputs / "link.csv").symlink_to(outside)
    for path in ("../outside.csv", str(outside), "link.csv"):
        await assert_tool_error(
            client, "rainflow_count", "directory", stress={"csv_path": path}
        )

    for contents, error in (
        ("other\n2\n", "header"),
        ("stress\n1,2\n", "row 2"),
        ("stress\nhello\n", "row 2"),
        ("stress\nnan\n", "finite"),
        ('stress\n"0\n', "malformed"),
    ):
        (inputs / "invalid.csv").write_text(contents, encoding="utf-8")
        await assert_tool_error(
            client, "rainflow_count", error, stress={"csv_path": "invalid.csv"}
        )
    with (inputs / "oversized.csv").open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    await assert_tool_error(
        client, "rainflow_count", "64 MiB", stress={"csv_path": "oversized.csv"}
    )
    await assert_tool_error(client, "rainflow_count", "number", stress=[0, "1"])
    await assert_tool_error(
        client,
        "miner_damage",
        "per slope",
        source={"kind": "stress_history", "stress": STRESS},
        curve={"slope": [3, 5], "intercept": 12},
    )

    empty = await call(client, "rainflow_count", stress=[0, 0])
    assert empty == dict(row_count=0, total_cycles=0, preview=[], export=None)
    await assert_tool_error(
        client, "rainflow_count", "varying stress history", stress=[0, 1]
    )
    await assert_tool_error(
        client,
        "miner_damage",
        "varying stress history",
        source={"kind": "stress_history", "stress": [0, 1, 0]},
        curve={"slope": 3, "intercept": 12},
    )
    damage = await call(
        client,
        "miner_damage",
        source={"kind": "stress_history", "stress": [0, 0]},
        curve={"slope": 3, "intercept": 12},
    )
    assert damage == dict(row_count=0, total_damage=0, preview=[], export=None)
    (tmp_path / "output").rmdir()
    await assert_tool_error(
        client, "rainflow_count", "output_dir", stress=STRESS, export_csv=True
    )


def test_csv_limits_and_failed_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remove partial output after failure and enforce the CSV row limit."""
    from py_fatigue_mcp import csv_io

    monkeypatch.setenv(csv_io.INPUT_DIR, str(tmp_path))
    monkeypatch.setenv(csv_io.OUTPUT_DIR, str(tmp_path))
    source = tmp_path / "stress.csv"
    source.write_text("stress\n0\n1\n", encoding="utf-8")
    with pytest.raises(csv_io.ToolError, match="1-row limit"):
        list(
            csv_io.read_csv_rows(
                csv_io.CsvFile(csv_path=source.name), ("stress",), 1
            )
        )

    def broken_rows() -> Iterator[tuple[float]]:
        yield (1.0,)
        raise OSError("interrupted export")

    with pytest.raises(csv_io.ToolError, match="Could not write"):
        csv_io.write_csv_rows("test", ("stress",), broken_rows(), 2)
    assert list(tmp_path.iterdir()) == [source]


async def test_ordered_crack_growth(client, tmp_path: Path) -> None:
    """Compare the ten tool cases with direct ordered solver calls."""
    geometries = {
        "TC01": cg.CenterThroughCrack(50, 5),
        "TC02": cg.EdgeThroughCrack(50, 5),
        "TC06": cg.SphereThroughCrack(100, 1, 10),
        "TC07": cg.AxialCylinderThroughCrack(100, 1, 10),
        "TC08": cg.CircumferentialCylinderThroughCrack(100, 1, 10, 2500, 0.3),
        "SC01": cg.SemiEllipticalSurfaceCrack(50, 15, 6, 10),
        "SC05": cg.CircumferentialCylinderSurfaceCrack(20, 10, 4, 5),
        "SC07": cg.RoundBarSurfaceCrack(20, 2),
        "SC08": cg.RolledBoltThreadSurfaceCrack(12, 10, 1.5),
        "SC11": cg.HoleBoreSurfaceCrack(80, 12, 12, 2.2, 2.5, 2),
    }
    (tmp_path / "inputs" / "loads.csv").write_text(
        "minimum,maximum,cycles\n0,40,8.5\n10,50,10\n", encoding="utf-8"
    )
    for case, geometry in geometries.items():
        direct = cg.simulate_crack_growth(geometry, CURVE, LOADS, **CONTROLS)
        uses_file = case == "SC01"
        result = await call(
            client,
            "ordered_crack_growth",
            geometry={"case": case, **asdict(geometry)},
            curve=asdict(CURVE),
            loads=(
                {"csv_path": "loads.csv"}
                if uses_file
                else [asdict(load) for load in LOADS]
            ),
            controls=CONTROLS,
            export_csv=uses_file,
        )
        expected = [
            json.loads(json.dumps(asdict(point))) for point in direct.history
        ]
        assert (
            result["termination"] == direct.termination == "spectrum_exhausted"
        )
        assert result["final_cycles"] == direct.final_cycles
        assert result["final_state"] == expected[-1]
        assert result["dimension_names"] == list(direct.dimension_names)
        assert result["row_count"] == len(expected)
        assert result["preview"] == expected[:3] + expected[-2:]
        if uses_file:
            rows = exported_rows(tmp_path, result)
            assert [int(row["block"]) for row in rows] == [
                point.block for point in direct.history
            ]
            assert [float(row["size_mm_2"]) for row in rows] == [
                point.sizes[1] for point in direct.history
            ]
            assert [float(row["k_max_mpa_sqrt_mm_2"]) for row in rows] == [
                point.k_max[1] for point in direct.history
            ]


async def test_crack_growth_stops_and_errors(client) -> None:
    """Distinguish admissible stops from invalid inputs and exhausted budgets."""
    name = "ordered_crack_growth"
    arguments = dict(
        geometry=dict(case="TC01", half_width=50, half_length=5),
        curve=asdict(CURVE),
        controls=CONTROLS,
        loads=[asdict(load) for load in LOADS],
    )
    result = await call(
        client,
        name,
        **(arguments | {"curve": asdict(CURVE) | {"toughness": 1}}),
    )
    assert result["termination"] == "toughness"
    assert result["final_cycles"] == result["row_count"] == 0
    assert result["final_state"] is None

    result = await call(
        client,
        name,
        **(
            arguments
            | {
                "geometry": dict(case="TC01", half_width=50, half_length=39.9),
                "curve": asdict(CURVE)
                | {"coefficient": 1e-4, "toughness": 1e6},
                "controls": CONTROLS | {"max_trials": 1000},
            }
        ),
    )
    assert result["termination"] == "geometry_limit"
    assert result["final_state"]["sizes"][0] <= 40

    await assert_tool_error(
        client,
        name,
        "trial budget",
        **(arguments | {"controls": CONTROLS | {"max_trials": 1}}),
    )
    await assert_tool_error(
        client,
        name,
        "100000",
        **(arguments | {"controls": CONTROLS | {"max_trials": 100001}}),
    )
    await assert_tool_error(
        client,
        name,
        "geometry",
        **(
            arguments
            | {"geometry": arguments["geometry"] | {"half_length": 45}}
        ),
    )
    await assert_tool_error(
        client,
        name,
        "crack_count",
        **(
            arguments
            | {
                "geometry": {
                    "case": "SC11",
                    **asdict(cg.HoleBoreSurfaceCrack(80, 12, 12, 2.2, 2.5, 1)),
                    "crack_count": True,
                }
            }
        ),
    )
