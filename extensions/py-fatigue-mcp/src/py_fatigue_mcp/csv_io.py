"""Read and export tool data within configured directories."""

import csv
import io
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Iterator
from uuid import uuid4

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

from .models import InputModel

MAX_CSV_BYTES = 64 * 1024 * 1024
INPUT_DIR = "PY_FATIGUE_MCP_INPUT_DIR"
OUTPUT_DIR = "PY_FATIGUE_MCP_OUTPUT_DIR"


class CsvFile(InputModel):
    """A CSV path relative to PY_FATIGUE_MCP_INPUT_DIR."""

    csv_path: str


class CsvExport(BaseModel):
    """The location and data row count of a complete local CSV."""

    relative_path: str
    row_count: int


def configured_directory(variable: str) -> Path:
    """Resolve the directory configured by an MCP client."""
    value = os.environ.get(variable)
    if not value:
        raise ToolError(f"Set {variable} to enable this CSV operation.")
    try:
        directory = Path(value).resolve(strict=True)
        if directory.is_dir():
            return directory
    except (OSError, RuntimeError, ValueError):
        pass
    raise ToolError(f"{variable} must point to an existing directory.")


def read_csv_rows(
    source: CsvFile, columns: tuple[str, ...], limit: int
) -> Iterator[tuple[float, ...]]:
    """Read finite numbers from required columns in their requested order."""
    directory = configured_directory(INPUT_DIR)
    try:
        relative_path = Path(source.csv_path)
        if relative_path.anchor or ".." in relative_path.parts:
            raise ToolError(
                "CSV paths must be relative to the input directory."
            )
        path = (directory / relative_path).resolve(strict=True)
        if not path.is_relative_to(directory) or not path.is_file():
            raise ToolError(
                "CSV input must be a file inside the input directory."
            )
        if path.stat().st_size > MAX_CSV_BYTES:
            raise ToolError("Input CSV exceeds the 64 MiB size limit.")
        with path.open("rb") as stream:
            payload = stream.read(MAX_CSV_BYTES + 1)
    except (OSError, RuntimeError, ValueError) as error:
        raise ToolError("Could not read the input CSV.") from error
    if len(payload) > MAX_CSV_BYTES:
        raise ToolError("Input CSV exceeds the 64 MiB size limit.")

    with io.TextIOWrapper(
        io.BytesIO(payload), encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = next(reader, [])
            if len(header) != len(set(header)) or not set(columns) <= set(
                header
            ):
                required = ", ".join(columns)
                raise ToolError(
                    f"CSV header must contain unique columns for {required}."
                )
            indices = [header.index(column) for column in columns]
            for count, row in enumerate(reader, start=1):
                if count > limit:
                    raise ToolError(
                        f"Input CSV exceeds the {limit:,}-row limit."
                    )
                if len(row) != len(header):
                    raise ToolError(
                        f"CSV row {reader.line_num} has the wrong field count."
                    )
                try:
                    values = tuple(float(row[index]) for index in indices)
                except ValueError as error:
                    raise ToolError(
                        f"CSV row {reader.line_num} has an invalid number."
                    ) from error
                if not all(map(math.isfinite, values)):
                    raise ToolError(
                        f"CSV row {reader.line_num} requires finite numbers."
                    )
                yield values
        except (csv.Error, UnicodeError) as error:
            raise ToolError(
                f"Malformed UTF-8 CSV near row {reader.line_num + 1}."
            ) from error


def write_csv_rows(
    prefix: str,
    columns: Iterable[str],
    rows: Iterable[Iterable[float]],
    row_count: int,
) -> CsvExport:
    """Publish a complete CSV under a generated name in the output directory."""
    directory = configured_directory(OUTPUT_DIR)
    filename = f"{prefix}-{uuid4().hex}.csv"
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=directory, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        temporary_path.replace(directory / filename)
    except (OSError, csv.Error) as error:
        raise ToolError("Could not write the result CSV.") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return CsvExport(relative_path=filename, row_count=row_count)
