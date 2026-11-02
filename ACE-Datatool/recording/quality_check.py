import os
import csv
import fnmatch
import json
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ---------- size estimation config ----------
# Per-file-pattern expected sizes (MB per 10 minutes of recording).
# Tune these based on your actual data.  Use a very high value (e.g. 99999)
# to effectively disable the check for a particular pattern.
EXPECTED_MB_PER_10MIN: Dict[str, float] = {
    "*.svo2":                  200.0,   # ZED SVO2 raw recording
    "*_rgb.mp4":                100.0,   # realsense RGB stream
    "*_depth.mp4":             400.0,   # realsense depth stream
    "mjpeg.bin":              3000.0,   # egocap mjpeg raw byte stream
}
SIZE_MIN_RATIO = 0.7                   # flag when actual < 70% of expected
SIZE_MAX_RATIO = 10                   # flag when actual > 150% of expected
STALE_FILE_GRACE_SEC = 10.0            # file mtime must be close to end timestamp

VALID_CAMERA_TYPES = {"egocap", "realsense", "zed"}


# ---------- data structures ----------
@dataclass
class CheckResult:
    name: str
    passed: bool
    issues: List[str] = field(default_factory=list)


@dataclass
class SizeResult:
    file_path: str
    actual_mb: float
    expected_mb: float
    passed: bool
    info: str = ""


@dataclass
class SessionReport:
    session_path: str
    session_id: str
    structure: CheckResult = field(default_factory=lambda: CheckResult("structure", True))
    egocap_check: CheckResult = field(default_factory=lambda: CheckResult("egocap", True))
    realsense_check: CheckResult = field(default_factory=lambda: CheckResult("realsense", True))
    zed_check: CheckResult = field(default_factory=lambda: CheckResult("zed", True))
    size_results: List[SizeResult] = field(default_factory=list)
    duration_minutes: float = 0.0

    @property
    def all_passed(self) -> bool:
        checks = [self.structure, self.egocap_check, self.realsense_check, self.zed_check]
        return all(c.passed for c in checks) and all(s.passed for s in self.size_results)


# ---------- helpers ----------
def _find_ts_csv(session_path: Path) -> Optional[Path]:
    """Resolve task_timestamps.csv / task_timestamp.csv."""
    for name in ("task_timestamps.csv", "task_timestamp.csv"):
        p = session_path / name
        if p.is_file():
            return p
    return None


def _parse_timestamps(session_path: Path) -> Tuple[Optional[float], Optional[float], float]:
    """Return (first_start_s, last_end_s, duration_minutes) from the CSV.
    Timestamps in the CSV are **millisecond** Unix epoch.
    """
    ts_path = _find_ts_csv(session_path)
    if ts_path is None:
        return None, None, 0.0

    starts, ends = [], []
    events: List[Tuple[float, str]] = []
    with open(ts_path, newline="") as f:
        for row in csv.DictReader(f):
            ts = float(row["timestamp"])
            event = row["event"].strip()
            events.append((ts, event))
            if event == "start":
                starts.append(ts)
            elif event == "end":
                ends.append(ts)

    if not starts or not ends or not events:
        return None, None, 0.0

    last_event_ts, last_event = max(events, key=lambda item: item[0])
    if last_event == "start":
        return min(starts) / 1000.0, None, 0.0

    first_start_ms = min(starts)
    last_end_ms = max(ends)
    duration_sec = (last_end_ms - first_start_ms) / 1000.0
    return first_start_ms / 1000.0, last_end_ms / 1000.0, duration_sec / 60.0


def _collect_target_files(session_path: Path) -> List[Path]:
    """Collect *.svo2, *.mp4, **/mjpeg.bin under session_path."""
    files: List[Path] = []
    files.extend(sorted(session_path.rglob("*.svo2")))
    files.extend(sorted(session_path.rglob("*.mp4")))
    files.extend(sorted(session_path.rglob("mjpeg.bin")))
    return files


# ---------- per-camera checks ----------
def check_structure(session_path: Path) -> CheckResult:
    issues: List[str] = []

    if not (session_path / "config.json").is_file():
        issues.append("missing config.json")

    if _find_ts_csv(session_path) is None:
        issues.append("missing task_timestamps.csv")

    present = [t for t in VALID_CAMERA_TYPES if (session_path / t).is_dir()]
    if not present:
        issues.append(f"no camera dir present (need >=1 of {sorted(VALID_CAMERA_TYPES)})")

    return CheckResult("structure", len(issues) == 0, issues)


def check_egocap(session_path: Path) -> CheckResult:
    egocap_path = session_path / "egocap"
    if not egocap_path.is_dir():
        return CheckResult("egocap", True, [])                 # absence is OK

    issues: List[str] = []
    subdirs = sorted(d for d in egocap_path.iterdir() if d.is_dir())
    if not subdirs:
        issues.append(f"egocap: no data subdirectory found")
        return CheckResult("egocap", False, issues)

    required = ("imu.csv", "metadata.json", "mjpeg.bin")
    for sd in subdirs:
        for fname in required:
            if not (sd / fname).is_file():
                issues.append(f"egocap/{sd.name}: missing {fname}")

    return CheckResult("egocap", len(issues) == 0, issues)


def check_realsense(session_path: Path) -> CheckResult:
    rs_path = session_path / "realsense"
    if not rs_path.is_dir():
        return CheckResult("realsense", True, [])

    issues: List[str] = []
    rgb = sorted(rs_path.glob("*_rgb.mp4"))
    depth = sorted(rs_path.glob("*_depth.mp4"))
    csvs = sorted(rs_path.glob("*.csv"))

    if not rgb and not depth:
        issues.append("realsense: no rgb/depth .mp4 files found")
        return CheckResult("realsense", False, issues)

    if len(rgb) % 2 != 0:
        issues.append(f"realsense: odd number of rgb videos ({len(rgb)})")
    if len(depth) % 2 != 0:
        issues.append(f"realsense: odd number of depth videos ({len(depth)})")
    if len(rgb) != len(depth):
        issues.append(f"realsense: rgb count ({len(rgb)}) != depth count ({len(depth)})")
    if not csvs:
        issues.append("realsense: no .csv timestamp files found")

    return CheckResult("realsense", len(issues) == 0, issues)


def check_zed(session_path: Path) -> CheckResult:
    zed_path = session_path / "zed"
    if not zed_path.is_dir():
        return CheckResult("zed", True, [])

    issues: List[str] = []
    svo2 = sorted(zed_path.glob("*.svo2"))
    if not svo2:
        issues.append("zed: no .svo2 files found")
        return CheckResult("zed", False, issues)

    return CheckResult("zed", True, [])


def _expected_mb_for_file(fname: str, duration_minutes: float) -> Optional[float]:
    """Return expected size (MB) for a file based on its name pattern."""
    for pattern, mb_per_10min in EXPECTED_MB_PER_10MIN.items():
        if fnmatch.fnmatch(fname, pattern):
            return duration_minutes * (mb_per_10min / 10.0)
    return None


def check_file_sizes(session_path: Path, duration_minutes: float,
                     min_ratio: float = SIZE_MIN_RATIO,
                     max_ratio: float = SIZE_MAX_RATIO,
                     last_end_s: Optional[float] = None,
                     stale_grace_sec: float = STALE_FILE_GRACE_SEC) -> List[SizeResult]:
    results: List[SizeResult] = []

    for fpath in _collect_target_files(session_path):
        actual_mb = fpath.stat().st_size / (1024 * 1024)
        rel = str(fpath.relative_to(session_path))

        mtime = fpath.stat().st_mtime
        reference_time = last_end_s if (last_end_s is not None and last_end_s > 0) else time.time()
        if (reference_time - mtime) > stale_grace_sec:
            label = "结束" if (last_end_s is not None and last_end_s > 0) else "当前"
            results.append(SizeResult(
                rel,
                actual_mb,
                0.0,
                False,
                f"文件长时间未更新 – 距离{label}时间 {reference_time - mtime:.1f}秒"
            ))
            continue

        expected_mb = _expected_mb_for_file(fpath.name, duration_minutes)
        if expected_mb is None or expected_mb <= 0:
            results.append(SizeResult(rel, actual_mb, 0.0, True,
                            "no expected-size pattern matched – skipped"))
            continue

        ratio = actual_mb / expected_mb
        if ratio < min_ratio:
            results.append(SizeResult(
                rel, actual_mb, expected_mb, False,
                f"TOO SMALL – actual={actual_mb:.1f}MB  expected≈{expected_mb:.1f}MB  ratio={ratio:.1%}"
            ))
        else:
            results.append(SizeResult(
                rel, actual_mb, expected_mb, True,
                f"OK – actual={actual_mb:.1f}MB  expected≈{expected_mb:.1f}MB  ratio={ratio:.1%}"
            ))

    return results


# ---------- main inspector ----------
def inspect_session(session_path: Path,
                    min_ratio: float = SIZE_MIN_RATIO,
                    max_ratio: float = SIZE_MAX_RATIO) -> SessionReport:
    report = SessionReport(
        session_path=str(session_path),
        session_id=session_path.name,
    )

    report.structure = check_structure(session_path)

    first_start_s, last_end_s, duration_min = _parse_timestamps(session_path)
    report.duration_minutes = duration_min

    report.egocap_check = check_egocap(session_path)
    report.realsense_check = check_realsense(session_path)
    report.zed_check = check_zed(session_path)

    report.size_results = check_file_sizes(session_path, duration_min,
                                           min_ratio=min_ratio,
                                           max_ratio=max_ratio,
                                           last_end_s=last_end_s)

    return report


def _scan_sessions(root: Path) -> List[Path]:
    """Walk root and yield every leaf directory that starts with '0000'."""
    sessions: List[Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        dp = Path(dirpath)
        if dp.name.startswith("0000"):
            sessions.append(dp)
            dirnames.clear()          # stop descending inside a session
    return sorted(sessions)


# ---------- printing ----------
def print_report(report: SessionReport, verbose: bool = False):
    status = "PASS" if report.all_passed else "FAIL"
    print(f"\n{'='*72}")
    print(f"  Session: {report.session_id}   Duration: {report.duration_minutes:.1f} min   [{status}]")
    print(f"{'='*72}")

    # structure
    st = report.structure
    tag = "OK" if st.passed else "FAIL"
    print(f"  [structure]  {tag}")
    for i in st.issues:
        print(f"    - {i}")

    # cameras
    for chk in (report.egocap_check, report.realsense_check, report.zed_check):
        if chk.issues or verbose:
            tag = "OK" if chk.passed else "FAIL"
            print(f"  [{chk.name}]  {tag}")
            for i in chk.issues:
                print(f"    - {i}")

    # file sizes
    if report.size_results:
        print(f"  [file sizes]  (expected MB/10min per pattern: {json.dumps(EXPECTED_MB_PER_10MIN)})")
        for sr in report.size_results:
            tag = "OK" if sr.passed else "FAIL"
            print(f"    [{tag}]  {sr.file_path}")
            if verbose or not sr.passed:
                print(f"            {sr.info}")


def print_summary(reports: List[SessionReport]):
    passed = sum(1 for r in reports if r.all_passed)
    failed = len(reports) - passed
    print(f"\n{'='*72}")
    print(f"  TOTAL: {len(reports)} sessions   PASS: {passed}   FAIL: {failed}")
    print(f"{'='*72}")
    if failed:
        for r in reports:
            if not r.all_passed:
                print(f"  FAIL  {r.session_path}")


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(
        description="Quality inspector for ACE raw-data recording sessions."
    )
    parser.add_argument(
        "root",
        type=str,
        help="Root directory to scan (e.g. /home/dwsu/Merging/ACEDATA/guangxi/20250415-001)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed OK results as well.",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=SIZE_MIN_RATIO,
        help=f"Min acceptable size ratio (default {SIZE_MIN_RATIO})",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=SIZE_MAX_RATIO,
        help=f"Max acceptable size ratio (default {SIZE_MAX_RATIO})",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: {root} is not a valid directory.")
        return 1

    sessions = _scan_sessions(root)
    if not sessions:
        print(f"No session directories (0000*) found under {root}")
        return 0

    reports: List[SessionReport] = []
    for sp in sessions:
        r = inspect_session(sp, min_ratio=args.min_ratio, max_ratio=args.max_ratio)
        reports.append(r)
        print_report(r, verbose=args.verbose)

    print_summary(reports)
    return 0 if all(r.all_passed for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
