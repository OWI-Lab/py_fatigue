import json
import re
import subprocess

BASE = "08193f9c84fcd696affdea28904f57ad9f47022c"
FILES = [
    "py_fatigue/utils.py",
    "py_fatigue/__init__.py",
    "py_fatigue/damage/stress_life.py",
    "py_fatigue/mean_stress/corrections.py",
    "py_fatigue/cycle_count/cycle_count.py",
    "py_fatigue/material/sn_curve.py",
]

with open("coverage.json", "r", encoding="utf-8") as f:
    coverage_data = json.load(f)["files"]

for file_path in FILES:
    diff_text = subprocess.check_output(
        ["git", "diff", "--unified=0", f"{BASE}..HEAD", "--", file_path],
        text=True,
    )
    added_lines = set()
    cursor = None
    for line in diff_text.splitlines():
        if line.startswith("@@"):
            match = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
            if match:
                cursor = int(match.group(1))
            continue
        if cursor is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added_lines.add(cursor)
            cursor += 1
        elif line.startswith("-") and not line.startswith("---"):
            continue
        else:
            cursor += 1

    missing_lines = set(coverage_data.get(file_path, {}).get("missing_lines", []))
    missing_patch_lines = sorted(missing_lines & added_lines)
    print(f"{file_path}: missing_patch={len(missing_patch_lines)} added={len(added_lines)}")
    if missing_patch_lines:
        print("  ", ", ".join(str(x) for x in missing_patch_lines[:80]))
