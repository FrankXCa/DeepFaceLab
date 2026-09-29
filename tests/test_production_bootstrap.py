"""Environment-gated REAL production-bootstrap regression test (Phase 13).

Feature: P13-RUNTIME-DEP-CLOSURE.  The packaged runtime must contain every
direct production dependency of the application's startup chain; before this
feature ``colorama`` and ``tqdm`` (both imported unconditionally at module
level by ``core/interact/interact.py``) were missing from all four runtime
locks, so ``launchers\\dfl.bat`` died with ``ModuleNotFoundError: No module
named 'colorama'`` before any CLI dispatch.

This test drives the REAL user-facing production entry chain end to end:

    launchers/dfl.bat --help

i.e. launcher (active-runtime selector validation) -> packaged interpreter
``runtime/versions/<id>/python.exe -I -B`` -> ``scripts/runtime_entry.py``
(host-environment purge, CUDA sanitation) -> ``main.py`` module-level chain
(``core.leras.nn`` -> torch, ``nn.initialize_main_env()``, ``core.pathex``,
``core.osex``, ``core.interact.interact`` -> colorama, cv2, numpy, tqdm)
-> argparse ``--help`` -> exit 0.

That is strictly stronger than ``python -c "import ..."`` (which proves a
package is importable but never the launcher/selector chain, the packaged
interpreter, or the production import order) and stronger than
``runtime_entry --self-test`` (which proves interpreter isolation and health
but executes no application code).  A missing direct production dependency
surfaces here as a launcher non-zero exit plus a traceback on stderr.

The test is skipped unless enabled, mirroring tests/test_real_runtime_build.py:

    DFL_REAL_PRODUCTION_BOOTSTRAP=1           enable (default: skip)
    DFL_PRODUCTION_BOOTSTRAP_TIMEOUT_SEC=600  per-invocation subprocess timeout

Evidence: retained logs under ``.cache/production-bootstrap-<runtime-id>/``
(developer state, gitignored) are asserted privacy-clean through the
builder's privacy scanner, exactly as the real-build test does.  The test
never modifies the runtime versions tree and leaves the selector untouched.

Retained-log sanitization (privacy regression coverage lives alongside, in
``test_privacy_sanitization_regression_cases``) never weakens detection:
transport escaping is interpreted at JSON-structure granularity (a line that
is a JSON document is decoded with ``json.loads`` and re-rendered with
semantic, single-backslash spellings; any other line is returned
byte-identical).  A blind global ``\\\\`` -> ``\\`` collapse is
deliberately NOT used: it would mask a genuinely raw UNC path
(``\\\\server\\share`` -> ``\\server\\share`` stops matching the scanner's
UNC-share rule).  After safe interpretation the builder scanner still
detects raw AND JSON-escaped UNC paths; only known locations become
logical labels and unknown drive-letter paths are redacted to ``<abs>``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from scripts import build_runtime as br

# Windows-only tests are marked individually: the privacy-sanitization
# regression cases are platform-neutral (pure string/regex + scanner) and
# run on every platform.

REPO_ROOT = Path(__file__).resolve().parents[1]
DFL_BAT = REPO_ROOT / "launchers" / "dfl.bat"
RUNTIME_ID_RE = re.compile(r"^[0-9a-f]{64}$")

# Absolute Windows/POSIX paths in retained logs must never be kept: known
# locations are replaced by logical labels, anything else by <abs> (same
# convention as tests/test_real_runtime_build.py).
_ABS_PATH_RE = re.compile(r"(?i)(?:[A-Za-z]):[\\/][^\s\"'`<>|()]*")
# UNC share path (raw, or the semantic form produced by safe JSON
# interpretation): two or more leading backslashes + server + share.  The
# builder privacy scanner detects such sequences in pre-redaction content
# (see the regression cases below); in RETAINED logs they are redacted to
# <abs> so the evidence is privacy-clean.
_UNC_RE = re.compile(r"\\\\+[^\s\\/\"'`<>|()]+(?:[\\/][^\s\\\"'`<>|()]*)*")

# Backslash spelled via chr(): keeps the regression-case literals below
# unambiguous and keeps this module free of escape sequences that the
# compiler would treat as (invalid) string escapes.
_BS = chr(92)

# Dummy hostile values (no drive letters, no cmd metacharacters): the
# launcher and runtime_entry clear/purge these regardless of content before
# any application import, so the dummies exercise the purge without needing
# real paths.
HOSTILE_PARENT_ENV = {
    "PYTHONHOME": "evil-home",
    "PYTHONPATH": "evil\\pp",
    "PYTHONSTARTUP": "evil\\su.py",
    "NN_DEVICES_INITIALIZED": "1",
    "NN_DEVICES_COUNT": "2",
    "NN_DEVICE_0": "cuda:0",
    "CUDA_PATH": "stale-cuda-toolkit",
    "CUDA_HOME": "stale-cuda-home",
    "CUDA_VISIBLE_DEVICES": "0",
}


def _render_decoded(value) -> str:
    """Render decoded JSON as reviewable text with SEMANTIC string values
    (single-backslash paths; a genuine UNC prefix keeps its two semantic
    backslashes) - no transport re-escaping, so both the known-path labels
    and the privacy scanner see the true content.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "{ " + ", ".join(
            f"{_render_decoded(key)}: {_render_decoded(item)}"
            for key, item in value.items()
        ) + " }"
    if isinstance(value, (list, tuple)):
        return "[ " + ", ".join(_render_decoded(item) for item in value) + " ]"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return str(value)


def _interpret_transport_escaping(text: str) -> str:
    """Safely interpret transport escaping without weakening detection.

    A line that parses as a JSON document is decoded with ``json.loads``
    and re-rendered with semantic (unescaped) string values: a JSON-escaped
    drive path becomes a single-backslash path, and a genuine UNC prefix
    (written with four raw backslashes in JSON text) becomes its true
    semantic spelling with two - still matching the scanner's UNC-share
    rule.  Every other line (plain help text, log headers) is returned
    byte-identical, so a raw UNC path in plain log text is never collapsed
    and stays detectable.

    Deliberately NOT a global double-backslash collapse: that would turn a
    genuine raw UNC into a single-backslash relative-looking path the
    scanner cannot see (the masking defect from the review correction pass).
    """
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                decoded = json.loads(stripped)
            except ValueError:
                out.append(line)  # not actually JSON: leave byte-identical
                continue
            out.append(_render_decoded(decoded))
        else:
            out.append(line)
    return "\n".join(out)


def _sanitize(text: str, known: dict[str, str]) -> str:
    # Safely interpret transport escaping (see _interpret_transport_escaping),
    # then label longest known paths first so the runtime interpreter is
    # labelled before its repository prefix is replaced; anything left that
    # is a file URL, a drive-letter path, or a UNC path becomes <abs> (the
    # retained evidence must be privacy-clean).
    text = _interpret_transport_escaping(text)
    for path, label in sorted(known.items(), key=lambda kv: len(kv[0]), reverse=True):
        if path:
            text = text.replace(path, label)
    text = re.sub(r"file:///(?:[A-Za-z]):[\\/][^\s\"']*", "file://<abs>", text)
    text = _ABS_PATH_RE.sub("<abs>", text)
    text = _UNC_RE.sub("<abs>", text)
    return text


def _enabled() -> bool:
    return os.environ.get("DFL_REAL_PRODUCTION_BOOTSTRAP") == "1"


def _timeout() -> int:
    return int(os.environ.get("DFL_PRODUCTION_BOOTSTRAP_TIMEOUT_SEC", "600"))


def _active_runtime_id() -> str | None:
    """The selector exactly as the launcher consumes it (data-validated)."""
    selector = REPO_ROOT / "runtime" / "active-runtime.txt"
    if not selector.is_file():
        return None
    value = selector.read_text(encoding="utf-8", errors="replace").strip()
    if not RUNTIME_ID_RE.fullmatch(value):
        return None
    return value


def _run_bat(args: list[str], cwd: Path, env: dict[str, str], timeout: int):
    # Executable-list invocation (same convention as tests/test_launchers.py):
    # CPython wraps the .bat target in `cmd /s /c "<line>"`, which preserves
    # the launcher path and each argument for this interpreter/cmd build.
    return subprocess.run(
        [str(DFL_BAT), *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows-only launcher test (dfl.bat)")
@pytest.mark.skipif(not _enabled(), reason="set DFL_REAL_PRODUCTION_BOOTSTRAP=1 to run the real production bootstrap")
def test_production_bootstrap_dfl_bat_help():
    """dfl.bat --self-test + dfl.bat --help from an arbitrary CWD against
    the active packaged runtime: the full production startup chain
    (launcher -> packaged interpreter -> runtime_entry -> main.py
    module-level imports -> argparse) must succeed with exit 0, and the
    executable the chain selected must be the promoted runtime's own
    interpreter."""
    runtime_id = _active_runtime_id()
    if runtime_id is None:
        pytest.skip(
            "no valid active runtime selector (runtime/active-runtime.txt); "
            "build and activate a verified runtime first via "
            "launchers\\dfl-setup-runtime.bat <variant> --activate"
        )
    runtime_dir = REPO_ROOT / "runtime" / "versions" / runtime_id
    python_exe = runtime_dir / "python.exe"
    assert python_exe.is_file(), (
        f"active selector names {runtime_id} but runtime/versions/{runtime_id}/python.exe "
        "is missing: the selector references a broken runtime (repair via "
        "launchers\\dfl-setup-runtime.bat <variant> --activate)"
    )

    # Arbitrary CWD: a scratch directory unrelated to the repository root,
    # proving the launcher's own ROOT derivation (~dp0..) - not the process
    # CWD - selects the interpreter and entry point.
    scratch = REPO_ROOT / ".cache" / f"production-bootstrap-{runtime_id}"
    cwd = scratch / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(HOSTILE_PARENT_ENV)
    labels = {
        str(python_exe): "<runtime-python>",
        str(REPO_ROOT): "<repo>",
        str(scratch): "<scratch>",
    }

    # -- 1. Self-test: selector -> packaged interpreter identity + isolation.
    proc_self = _run_bat(["--self-test"], cwd, env, timeout=_timeout())
    (scratch / "self-test.log").write_text(
        "$ dfl.bat --self-test (cwd=<scratch>\\cwd, hostile parent env)\n"
        "----- stdout -----\n" + _sanitize(proc_self.stdout, labels)
        + "\n----- stderr -----\n" + _sanitize(proc_self.stderr, labels),
        encoding="utf-8",
    )
    assert proc_self.returncode == 0, (
        f"self-test rc={proc_self.returncode}\nstdout={proc_self.stdout}\nstderr={proc_self.stderr}"
    )
    report = json.loads(proc_self.stdout.strip().splitlines()[-1])
    assert report["ok"] is True, report
    assert report["isolated"] is True, report
    assert report["usersite_disabled"] is True, report
    assert report["host_env_purged"] is True, report
    assert report["cuda_sanitized"] is True, report
    # Packaged-interpreter evidence: the executable the launcher chain
    # selected is the promoted runtime's own interpreter, never a host
    # Python (the launcher's python.exe is the runtime's, launched -I -B).
    executable = os.path.normcase(os.path.normpath(str(report["executable"])))
    expected_prefix = os.path.normcase(os.path.normpath(str(runtime_dir))) + os.sep
    assert executable.startswith(expected_prefix), (
        f"launcher chain selected {report['executable']!r}, not the active "
        f"runtime's interpreter under {runtime_dir}"
    )
    assert executable.endswith(os.path.normcase("python.exe")), report
    assert runtime_id in executable, report

    # -- 2. Production bootstrap: `dfl.bat --help` runs main.py's FULL
    #    module-level import chain in the packaged runtime (core.leras.nn
    #    -> torch; nn.initialize_main_env(); core.pathex; core.osex;
    #    core.interact.interact -> colorama, cv2, numpy, tqdm) and only
    #    then reaches argparse, which prints help and exits 0.  Before the
    #    dependency-closure fix this invocation died at `import colorama`
    #    with rc=1 (non-zero launcher exit, traceback on stderr).
    proc_help = _run_bat(["--help"], cwd, env, timeout=_timeout())
    (scratch / "help.log").write_text(
        "$ dfl.bat --help (cwd=<scratch>\\cwd, hostile parent env)\n"
        "----- stdout -----\n" + _sanitize(proc_help.stdout, labels)
        + "\n----- stderr -----\n" + _sanitize(proc_help.stderr, labels),
        encoding="utf-8",
    )
    combined = proc_help.stdout + proc_help.stderr
    assert proc_help.returncode == 0, (
        f"production bootstrap FAILED (rc={proc_help.returncode}); the startup "
        f"chain hit a missing or broken direct production dependency:\n"
        f"stdout={proc_help.stdout}\nstderr={proc_help.stderr}"
    )
    assert "Traceback" not in combined, combined
    assert "ModuleNotFoundError" not in combined, combined
    # argparse executed: main.py was actually run and reached CLI dispatch,
    # i.e. every pre-dispatch import succeeded in the packaged runtime.
    assert "usage:" in proc_help.stdout, proc_help.stdout
    # The top-level subcommands main.py registers (stable contract markers) -
    # ALL of them, including the internal dev_test subcommand (the real
    # `dfl.bat --help` output lists it, so the marker set keeps the
    # "all top-level subcommands" claim honest and any future loss of a
    # top-level command from the launcher's help/usage contract fails here).
    for marker in ("extract", "sort", "util", "train", "exportdfm", "merge",
                   "videoed", "facesettool", "xseg", "dev_test"):
        assert marker in proc_help.stdout, (
            f"help output lost top-level subcommand {marker!r}:\n{proc_help.stdout}"
        )

    # -- 3. Retained evidence logs must be privacy-clean: no absolute
    #    interpreter/repository/runtime/cache paths, no usernames,
    #    hostnames or private IPs (the run may use absolute paths
    #    internally; only what is retained for review is redacted).
    for log_file in sorted(scratch.glob("*.log")):
        text = log_file.read_text(encoding="utf-8")
        violations = br.privacy_violations(text)
        assert violations == [], (
            f"retained log {log_file.name} is not privacy-clean: {violations[:3]}"
        )


def test_privacy_sanitization_regression_cases():
    """Privacy scanner vs transport escaping (review-correction regression).

    The retained-log sanitizer must never mask path material from the
    builder privacy scanner (scripts/build_runtime.py privacy_violations):
    transport escaping is distinguished from path SEMANTICS - a blind
    global double-backslash collapse is exactly what the old code did and
    what this test class pins against.

    A. raw UNC path        -> passes interpretation byte-identical, and the
                              scanner's UNC-share rule still detects it;
    B. JSON-escaped UNC    -> both the escaped standalone line (left
                              byte-identical, still detectable) and the
                              safely-decoded JSON document (true semantic
                              spelling) are detected;
    C. JSON-escaped drive  -> the decoded semantic path is detected
      path                   (drive-letter + user-home rules); the RETAINED
                              (redacted) form is privacy-clean;
    D. ordinary escapes    -> no path semantics: decoding yields single
                              backslashes the scanner does not flag (no
                              false positives).
    """
    BS = _BS

    # -- A. A genuinely raw UNC path: interpretation must leave it
    #       byte-identical (no collapse) so the scanner's UNC-share rule
    #       (two backslashes + server) still detects it.  A blind global
    #       `\\` -> `\` collapse would have turned it into a
    #       single-backslash relative-looking path and masked it.
    raw_unc = BS + BS + "private-server" + BS + "share" + BS + "secret"
    assert _interpret_transport_escaping(raw_unc) == raw_unc
    assert "UNC share" in br.privacy_violations(_interpret_transport_escaping(raw_unc))

    # -- B. JSON-escaped UNC.  (1) as a standalone escaped line: not a JSON
    #       document, so it is left byte-identical and the backslash pair
    #       adjacent to the server name is still detected; (2) inside a JSON
    #       document: safely decoded to the true semantic spelling (two
    #       backslashes + server), which the scanner detects.
    escaped_unc_line = raw_unc.replace(BS, BS + BS)
    assert "UNC share" in br.privacy_violations(_interpret_transport_escaping(escaped_unc_line))
    escaped_unc_json = '{"path": "' + escaped_unc_line + '"}'
    interpreted_unc = _interpret_transport_escaping(escaped_unc_json)
    assert raw_unc in interpreted_unc  # decoded to the semantic raw UNC
    assert "UNC share" in br.privacy_violations(interpreted_unc)

    # -- C. JSON-escaped Windows drive path: safe interpretation yields the
    #       semantic path the scanner detects (drive-letter rule and
    #       user-home rule); the retained form is redacted to <abs> and is
    #       privacy-clean (no sensitive material survives redaction).
    escaped_drive_json = (
        '{"path": "C:' + BS + BS + "Users" + BS + BS + "someone" + BS + BS + 'private"}'
    )
    interpreted_drive = _interpret_transport_escaping(escaped_drive_json)
    assert "C:" + BS + "Users" + BS + "someone" + BS + "private" in interpreted_drive
    violations = br.privacy_violations(interpreted_drive)
    assert "drive-letter path" in violations
    assert "user home path" in violations
    assert br.privacy_violations(_sanitize(escaped_drive_json, {})) == []

    # -- D. Ordinary transport escaping carries no path semantics: decoding
    #       produces single-backslash sequences that must NOT trigger the
    #       scanner (no false positives from legitimate JSON escaping).
    ordinary_json = '{"note": "line1' + BS + BS + 'nline2", "sep": "a' + BS + BS + 'tb"}'
    interpreted_ordinary = _interpret_transport_escaping(ordinary_json)
    assert "line1" + BS + "nline2" in interpreted_ordinary
    assert br.privacy_violations(interpreted_ordinary) == []
    assert br.privacy_violations(_sanitize(ordinary_json, {})) == []
