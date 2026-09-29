# -----------------------------------------------------------------------------
# Phase 13 - Commit 3: isolated Windows launcher and setup-bootstrap tests.
#
# Covers the user-facing launcher layer shipped in this commit:
#   launchers/dfl.bat               normal application launcher
#   launchers/dfl-envreport.bat     thin alias -> dfl.bat envreport (Commit 4)
#   launchers/dfl-setup-runtime.bat developer setup/build helper
#   scripts/runtime_entry.py        CUDA sanitation (Commit-2 bootstrap owner,
#                                   exercised here through the launch chain)
#
# Method:
#   * Behavioural tests run the real tracked .bat files (copied unchanged
#     into a temporary fake repository tree) through cmd.exe with a fake
#     bundled interpreter (python.exe.bat) or fake host interpreters
#     (fake py.bat / python.bat on a controlled PATH for automatic
#     discovery, or an explicit DFL_SETUP_PYTHON override - which is
#     PATH DATA under the round-3 contract: one .exe interpreter path,
#     validated as data by a fallback interpreter before any use - faked
#     by a lone copy of a real python.exe, a working frozen-only
#     interpreter), capturing the child environment and arguments in log
#     files. The fake interpreter is the ONLY bundled executable
#     reachable in the launch chain, which proves the launcher never
#     falls back to a host/system/venv Python.
#   * A CURRENT runtime from runtime/versions/ (one whose builder CLI
#     `verify-runtime` check passes against the current
#     scripts/runtime_entry.py bootstrap SHA) is exercised end to end: the
#     builder CLI activates it, the real bundled interpreter runs the real
#     scripts/runtime_entry.py --self-test under a hostile environment
#     (stale CUDA variables + toolkit PATH entries, stale NN device state),
#     and the self-test JSON is asserted. This is CURRENT-MACHINE contract
#     validation (no training, self-test only), labelled as such. If no
#     current runtime exists on this machine yet (e.g. before the post-code
#     rebuild), the integration test skips.
#   * Static scans assert the launcher bodies stay free of hardcoded
#     machine paths, network use, deletion commands, unsafe argument
#     accumulation (no BFLAGS-style string re-expansion), CALL-based host
#     execution in the setup helper (rounds 2-3), caller-controlled data
#     on any `cmd /c` line, helper-file (TEMP) mechanisms in the normal
#     launcher, and - for the round-3 selector - any raw substitution of
#     selector bytes into a parsed command before the file-as-data
#     validation gates (size / line count / character set) have proven the
#     content pure lowercase hex.

# These tests are Windows-only (cmd batch semantics) and skip elsewhere.
# -----------------------------------------------------------------------------

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows-only launcher tests")

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS_DIR = REPO_ROOT / "launchers"
REAL_RUNTIME_ENTRY = REPO_ROOT / "scripts" / "runtime_entry.py"
VENV_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

RUNTIME_ID_RE = re.compile(r"^[0-9a-f]{64}$")

# The exact DFL_SETUP_PYTHON validator code embedded in
# dfl-setup-runtime.bat (the child interpreter reads the value from the
# environment - the value's bytes are never parsed by the helper). A
# static test asserts this constant equals the bat's code byte-for-byte.
VALIDATOR_CODE = (
    "import os,sys;v=os.environ.get('DFL_SETUP_PYTHON','');"
    "b=chr(34)+chr(38)+chr(124)+chr(60)+chr(62)+chr(94)+chr(37)+chr(33)"
    "+chr(40)+chr(41)+chr(59)+chr(13)+chr(10);"
    "n=chr(110)+chr(111)+chr(110)+chr(101);"
    "ok=v and not any(c in v for c in b) and v.lower().endswith('.exe')"
    " and os.path.isfile(v);"
    "sys.exit(7 if v==n else (0 if ok else 6))"
)


def base_python_exe() -> Path | None:
    """Resolve a real CPython base interpreter on this machine.

    The venv's pyvenv.cfg `home` entry points at the base
    installation. A LONE copy of that python.exe alone is NOT a
    runnable interpreter (CPython derives its prefix from the
    interpreter's own directory and aborts when the platform
    independent libraries are absent from it) - use
    `make_standalone_host` to stage a self-contained copy. Returns
    None when no resolvable base interpreter exists (functional
    override tests then skip; the security tests need no fake host
    at all)."""
    cfg = REPO_ROOT / ".venv" / "pyvenv.cfg"
    if not cfg.is_file():
        return None
    home = None
    for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("home ="):
            home = line.split("=", 1)[1].strip()
            break
    if not home:
        return None
    exe = Path(home) / "python.exe"
    return exe if exe.is_file() else None


def make_standalone_host(target_dir: Path) -> Path:
    """Stage `target_dir` as a self-contained host interpreter
    directory and return its python.exe path: a copy of the base
    CPython executable plus its sibling python*.dll runtime files,
    plus a directory junction `Lib` pointing at the base standard
    library. CPython derives its prefix from the interpreter's own
    directory and refuses to start at all (even for `-c "import
    sys"`) when the platform independent libraries are not located
    beneath it, so the directory must present a complete layout; the
    junction keeps the staging cheap (a few MB of DLLs, no
    stdlib copy). The override-host tests only ever run builtins +
    sys (frozen in the interpreter core) and the builtins-only stub
    builder, so no other layout component is required."""
    base = base_python_exe()
    if base is None:
        pytest.skip("no resolvable base CPython on this machine")
    src = Path(base)
    target_dir.mkdir(parents=True, exist_ok=True)
    exe = target_dir / src.name
    if not exe.is_file():
        exe.write_bytes(src.read_bytes())
    for name in sorted(os.listdir(src.parent)):
        if name.startswith("python") and name.endswith(".dll"):
            dst = target_dir / name
            if not dst.is_file():
                dst.write_bytes((src.parent / name).read_bytes())
    lib = target_dir / "Lib"
    if not lib.exists():
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(lib), str(src.parent / "Lib")],
            check=True,
            capture_output=True,
        )
    return exe

HOSTILE_PARENT_ENV = {
    # Dummy values (no drive letters): the launcher clears the variables
    # regardless of their content, so the dummies need no real paths.
    "PYTHONHOME": "evil-home",
    "PYTHONPATH": "evil\\pp",
    "PYTHONSTARTUP": "evil\\su.py",
    "PYTHONUSERBASE": "evil\\ub",
    "NN_DEVICES_INITIALIZED": "1",
    "NN_DEVICES_COUNT": "4",
    # NN_DEVICE_ family, uppercase ...
    "NN_DEVICE_0": "cuda:0",
    "NN_DEVICE_1": "cuda:1",
    "NN_DEVICE_ABC": "zz",
    # ... and mixed / lowercase spellings (Windows env names are
    # case-insensitive, so every spelling belongs to the family).
    "Nn_Device_Mixed": "cuda:9",
    "nn_device_lower": "cuda:10",
    # Hostile VALUE: if the launcher ever evaluated a value as command
    # text, a marker file would appear in the launcher's CWD.
    "NN_DEVICE_INJ": "1 & echo PWNED > marker-inj.txt",
    # Not part of the NN_DEVICE_ family: must pass through untouched.
    "NN_DEVICE": "keepme",
    "NN_DEVICES_X": "keepme2",
    # CUDA sanitation is owned by runtime_entry.py (Commit-2 bootstrap),
    # not by the launcher: a stale CUDA_PATH must survive the launcher
    # layer (asserted here; the runtime_entry end-to-end test asserts the
    # subsequent clearance through the real self-test).
    "CUDA_PATH": "stale-cuda-toolkit",
    "CUDA_HOME": "stale-cuda-home",
    "CUDA_VISIBLE_DEVICES": "0",
    # Unrelated variable: scoping policy is to clear only the listed
    # families, so unrelated state passes through.
    "FAKESENTINEL_X": "sent",
}


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _bat_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def _env(parent: dict | None = None, **extra) -> dict:
    env = os.environ.copy()
    if parent is not None:
        env.update(parent)
    env.update({k: v for k, v in extra.items() if v is not None})
    return env


def make_fake_repo(
    root: Path,
    *,
    selector: str | None,
    with_python: bool = True,
    with_entry: bool = True,
    with_builder_stub: bool = False,
    runtime_id: str | None = None,
    entry_source: Path | None = None,
) -> dict:
    """Build a fake repository tree containing the real launcher files."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "launchers").mkdir(parents=True, exist_ok=True)
    (root / "runtime" / "versions").mkdir(parents=True, exist_ok=True)
    for name in ("dfl.bat", "dfl-envreport.bat", "dfl-setup-runtime.bat"):
        src = LAUNCHERS_DIR / name
        assert src.is_file(), f"tracked launcher missing: {src}"
        (root / "launchers" / name).write_bytes(src.read_bytes())
    if with_entry:
        (root / "scripts").mkdir(parents=True, exist_ok=True)
        src = entry_source if entry_source is not None else None
        if src is not None and src.is_file():
            (root / "scripts" / "runtime_entry.py").write_bytes(src.read_bytes())
        else:
            (root / "scripts" / "runtime_entry.py").write_text(
                "# fake runtime entry point\n", encoding="utf-8", newline="\n"
            )
    if runtime_id is None:
        runtime_id = "ab" * 32
    # The version directory exists even without an interpreter so that the
    # "directory present, interpreter missing" case can be exercised.
    (root / "runtime" / "versions" / runtime_id).mkdir(parents=True, exist_ok=True)
    if with_python:
        vdir = root / "runtime" / "versions" / runtime_id
        # The fake interpreter is written with explicit CRLF so the quoted
        # log lines stay parseable by cmd. /I makes the environment dump
        # case-insensitive so mixed-case family names are observed too.
        vdir.joinpath("python.exe.bat").write_bytes(
            (
                "@echo off\r\n"
                'if defined FAKEINT_LOG (\r\n'
                '  >>"%FAKEINT_LOG%" echo ARGS %*\r\n'
                '  >>"%FAKEINT_LOG%" echo EXEC %~0\r\n'
                '  >>"%FAKEINT_LOG%" echo --- env ---\r\n'
                '  set | findstr /B /I "PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE PYTHONNOUSERSITE PYTHONDONTWRITEBYTECODE NN_DEVICE CUDA_PATH CUDA_HOME CUDA_VISIBLE_DEVICES FAKESENTINEL SELECTOR_EXECUTED" >>"%FAKEINT_LOG%"\r\n'
                ")\r\n"
                "if defined FAKEINT_EXIT exit /b %FAKEINT_EXIT%\r\n"
                "exit /b 0\r\n"
            ).encode("utf-8")
        )
    if with_builder_stub:
        # Stub builder executable by the override host (a self-contained
        # real CPython staged by make_standalone_host, or a real python
        # of any kind): it logs its argv (plus the running interpreter)
        # to build.log - relative to the repository root, where the
        # setup helper changes directory - and exits with the value of
        # the DFL_FAKE_BUILD_EXIT environment variable (default 0). The
        # exit-code knob is an ENVIRONMENT variable, never an option
        # value: --lock-timeout values are legitimate forwarded data in
        # other tests and must not alter the builder's exit code.
        (root / "scripts").mkdir(parents=True, exist_ok=True)
        (root / "scripts" / "build_runtime.py").write_text(
            "import os\n"
            "import sys\n"
            'log = open("build.log", "a")\n'
            'log.write("BUILD " + " ".join(sys.argv) + chr(10))\n'
            'log.write("EXE " + sys.executable + chr(10))\n'
            'try:\n'
            '    code = int(os.environ.get("DFL_FAKE_BUILD_EXIT", "0"))\n'
            "except ValueError:\n"
            "    code = 0\n"
            "log.close()\n"
            "sys.exit(code)\n",
            encoding="utf-8",
            newline="\n",
        )
    if selector is not None:
        (root / "runtime" / "active-runtime.txt").write_text(
            selector + "\n", encoding="utf-8", newline="\n"
        )
    return {"root": root, "runtime_id": runtime_id}


def run_bat(bat: Path, *args, cwd: Path, env: dict, timeout: int = 120):
    # Invoke as an executable list: CPython wraps a .bat target as
    # `cmd /s /c "<full command line>"`, which preserves the quoting of
    # both a spaced launcher path and each spaced/quoted argument
    # (verified empirically for this interpreter/cmd build). An explicit
    # `cmd /c` string line instead triggers cmd's first/last quote
    # stripping as soon as the line carries more than one quoted token,
    # which corrupts exactly the metacharacter arguments under test.
    # The test tmp-path plugin sanitises tmp directory names to
    # alphanumerics plus `._-# ` so no cmd command-syntax character
    # ((, ), &, |, ^, ...) can appear in the launcher path itself.
    return subprocess.run(
        [str(bat), *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def read_fakeint_log(log_path: Path) -> dict:
    """Parse the fake interpreter log into ARGS / EXEC / env lines."""
    if not log_path.is_file():
        return {"args": None, "exec": None, "env": {}}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    args_line = None
    exec_line = None
    env = {}
    section = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("ARGS "):
            args_line = line[len("ARGS "):]
            section = None
        elif line.startswith("EXEC "):
            exec_line = line[len("EXEC "):]
            section = None
        elif line == "--- env ---":
            section = "env"
        elif section == "env" and "=" in line:
            key, _, value = line.partition("=")
            env[key] = value
    return {"args": args_line, "exec": exec_line, "env": env}


def tree_snapshot(base: Path) -> dict:
    """Read-only recursive snapshot: relative path -> (size, mtime_ns)."""
    out = {}
    if not base.is_dir():
        return out
    for p in sorted(base.rglob("*")):
        if p.is_file():
            st = p.stat()
            out[str(p.relative_to(base))] = (st.st_size, st.st_mtime_ns)
    return out


def write_fake_host(path: Path, prefix: str) -> None:
    """Write a fake host interpreter (batch) identified by `prefix`.

    Behaviour knobs (all optional env vars, undefined = default 0/empty):
      <prefix>_PROBE_EXIT      exit code for the `import sys` probe
      <prefix>_FLOOR_EXIT      exit code for the floor probe (0 = >= 3.11,
                               3 = below floor, anything else = probe failure)
      <prefix>_VALIDATE_EXIT   exit code for the DFL_SETUP_PYTHON validator
                               code (0 = value accepted, 6 = value rejected)
      <prefix>_BUILD_EXIT      exit code for the builder invocation
      <prefix>_VER             version string for the version probe
      <prefix>_LOG             file that receives `BUILD %*` lines

    Execution-context notes (round 3): the probe and floor branches use
    `exit /b` so that a fake .bat host executed DIRECTLY by the helper's
    override probe/floor lines returns to the helper process with the
    knob's code instead of terminating it (and inside a child cmd of an
    automatic probe `cmd /c py -3 ...` the `exit /b` code becomes that
    child's exit code). The validator / version / builder branches use
    process-level `exit`: the validator and version probes always run in
    a child cmd, and the builder line is the helper's last command (no
    CALL, no child cmd on the builder path), where a process-level exit
    from a directly-nested batch propagates the code reliably (an
    `exit /b` from a directly-nested batch can lose its code on this
    cmd build, observed inside parenthesized blocks).
    """
    p = prefix.upper()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (
            "@echo off\r\n"
            f"if not defined {p}_PROBE_EXIT set \"{p}_PROBE_EXIT=0\"\r\n"
            f"if not defined {p}_FLOOR_EXIT set \"{p}_FLOOR_EXIT=0\"\r\n"
            f"if not defined {p}_VALIDATE_EXIT set \"{p}_VALIDATE_EXIT=0\"\r\n"
            f"if not defined {p}_BUILD_EXIT set \"{p}_BUILD_EXIT=0\"\r\n"
            'set "HASC=0"\r\n'
            'set "CODE="\r\n'
            ":scan\r\n"
            'if "%~1"=="" goto :act\r\n'
            'if "%~1" neq "-c" goto :noscan\r\n'
            "shift\r\n"
            "set \"CODE=%~1\"\r\n"
            "set \"HASC=1\"\r\n"
            "goto :scan\r\n"
            ":noscan\r\n"
            "shift\r\n"
            "goto :scan\r\n"
            ":act\r\n"
            'if "%HASC%"=="0" (\r\n'
            "  rem builder invocation: no -c code argument\r\n"
            f"  if defined {p}_LOG >>\"%{p}_LOG%\" echo BUILD %*\r\n"
            f"  if defined {p}_BUILD_EXIT exit %{p}_BUILD_EXIT%\r\n"
            "  exit 0\r\n"
            ")\r\n"
            "rem exact comparisons of the known launcher probe commands\r\n"
            "rem (echo|findstr of the code text is unsafe: the text carries\r\n"
            "rem parentheses and comparison operators)\r\n"
            f'if "%CODE%"=="import sys" exit /b %{p}_PROBE_EXIT%\r\n'
            f'if "%CODE%"=="import sys; sys.exit(0 if sys.version_info >= (3, 11) else 3)" exit /b %{p}_FLOOR_EXIT%\r\n'
            f'if "%CODE%"=="{VALIDATOR_CODE}" (\r\n'
            "  rem mirror the real validator's `none` rule in data space:\r\n"
            "  rem the value crosses only `set` output and a FIXED findstr\r\n"
            "  rem pattern - never this fake's parsed command text.\r\n"
            "  for /f \"delims=\" %%E in ('set 2^>nul ^| findstr /b \"DFL_SETUP_PYTHON=none\"') do exit 7\r\n"
            f"  exit %{p}_VALIDATE_EXIT%\r\n"
            ")\r\n"
            'if "%CODE%"=="import sys; print(sys.version.split()[0])" (\r\n'
            f"  if defined {p}_VER echo %{p}_VER%\r\n"
            "  exit 0\r\n"
            ")\r\n"
            "exit 0\r\n"
        ).encode("utf-8")
    )


def make_fake_host_dir(
    fakedir: Path,
    *,
    with_py: bool = True,
    with_python: bool = True,
) -> dict:
    """Create a directory holding fake `py` / `python` batch interpreters
    so automatic discovery resolves to them on a controlled PATH."""
    logs = {}
    if with_py:
        write_fake_host(fakedir / "py.bat", "FAKE_PY")
        logs["py"] = fakedir.parent / "fake_py.log"
    if with_python:
        write_fake_host(fakedir / "python.bat", "FAKE_PYTHON")
        logs["python"] = fakedir.parent / "fake_python.log"
    return logs


def auto_env(tmp_path: Path, **extra) -> tuple[dict, dict]:
    """Environment for automatic-discovery tests: PATH = fake dir +
    System32 only, no DFL_SETUP_PYTHON, no host Python reachable."""
    fakedir = tmp_path / "hosts"
    logs = make_fake_host_dir(fakedir)
    env = _env(HOSTILE_PARENT_ENV, **extra)
    env.pop("DFL_SETUP_PYTHON", None)
    env["PATH"] = f"{fakedir};C:\\Windows\\System32"
    env["FAKE_PY_LOG"] = str(fakedir.parent / "fake_py.log")
    env["FAKE_PYTHON_LOG"] = str(fakedir.parent / "fake_python.log")
    return env, logs


# ---------------------------------------------------------------------------
# static security / hygiene scans (all three tracked launcher bodies)
# ---------------------------------------------------------------------------


class TestLauncherStaticSecurity:
    FILES = ("dfl.bat", "dfl-envreport.bat", "dfl-setup-runtime.bat")

    @pytest.mark.parametrize("name", FILES)
    def test_bodies_are_ascii_only(self, name):
        text = _bat_text(LAUNCHERS_DIR / name)
        bad = [c for c in text if ord(c) > 127]
        assert not bad, f"{name} contains non-ASCII: {bad[:5]}"

    @pytest.mark.parametrize("name", FILES)
    def test_no_hardcoded_machine_paths_or_urls(self, name):
        text = _bat_text(LAUNCHERS_DIR / name)
        patterns = {
            "drive-letter path": re.compile(r"[A-Za-z]:\\", re.IGNORECASE),
            "Users path": re.compile(r"Users\\", re.IGNORECASE),
            "AppData": re.compile(r"AppData", re.IGNORECASE),
            "ipv4": re.compile(r"(?<!\d)\d{1,3}(?:\.\d{1,3}){3}(?!\d)"),
            "url": re.compile(r"https?://|ftp://"),
            "venv reference": re.compile(r"\B\.venv\b|\\venv\\|/venv/"),
        }
        for label, rx in patterns.items():
            m = rx.search(text)
            assert m is None, f"{name}: {label} hit at {m.group(0)!r}"

    @pytest.mark.parametrize("name", FILES)
    def test_no_deletion_or_network_commands(self, name):
        text = _bat_text(LAUNCHERS_DIR / name)
        deletions = {
            "rmdir": re.compile(r"(?i)\brmdir\b"),
            "rd": re.compile(r"(?i)(?:^|[|&(\s])rd\s"),
            "del": re.compile(r"(?i)(?:^|[|&(\s])del\s"),
            "erase": re.compile(r"(?i)\berase\b"),
            "remove-item": re.compile(r"Remove-Item", re.IGNORECASE),
        }
        network = {
            "curl": re.compile(r"(?i)\bcurl\b"),
            "bitsadmin": re.compile(r"(?i)\bbitsadmin\b"),
            "wget": re.compile(r"(?i)\bwget\b"),
            "invoke-webrequest": re.compile(r"Invoke-WebRequest"),
            "pip-download": re.compile(r"(?i)\bpip\s+(?:install|download)"),
        }
        for label, rx in {**deletions, **network}.items():
            m = rx.search(text)
            assert m is None, f"{name}: {label} command found: {m.group(0)!r}"

    @pytest.mark.parametrize("name", FILES)
    def test_no_system_python_reference_in_launchers(self, name):
        # The setup helper legitimately invokes a host CPython; the normal
        # launcher chain must not reference any host interpreter name.
        if name == "dfl-setup-runtime.bat":
            pytest.skip("setup helper is the documented host-Python path")
        text = _bat_text(LAUNCHERS_DIR / name)
        for token in ("python3", "py -3", "py.exe", "pythonw", "py launcher"):
            assert not re.search(re.escape(token), text, re.IGNORECASE), token

    def test_setup_helper_has_no_argument_string_accumulation(self):
        # Review finding 1 (round 1): caller-controlled arguments must never
        # be accumulated into one command string and re-expanded.
        text = _bat_text(LAUNCHERS_DIR / "dfl-setup-runtime.bat")
        assert "BFLAGS" not in text, "BFLAGS-style accumulation is forbidden"
        # each forwarded option is carried in its own dedicated variable
        for var in ("DFLS_ACT", "DFLS_OFF", "DFLS_OFFIN", "DFLS_ARTC", "DFLS_LOCKT"):
            assert var in text

    def test_setup_helper_has_no_call_reevaluation_of_caller_data(self):
        """Review findings 1/2 (rounds 2-3): the setup helper contains NO
        `call` anywhere in its executable body - neither `cmd /c call`
        (a second parse layer over caller data) nor a plain `call`. The
        automatic candidates are launcher-owned fixed literals (`py -3`,
        `python`) run through `cmd /c <fixed> ...`; the explicit override
        is prevalidated path data executed DIRECTLY as a quoted .exe path
        (never a CALL of caller text). Every line that carries
        caller-controlled state (a value variable, the variant, the root)
        executes directly with a single expansion."""
        text = _bat_text(LAUNCHERS_DIR / "dfl-setup-runtime.bat")
        body = [
            line
            for line in text.splitlines()
            if line.strip() and not line.lstrip().lower().startswith("rem")
        ]
        for line in body:
            assert not re.search(r"\bcall\b", line), (
                f"CALL found in the setup helper body: {line!r}"
            )
        caller_state = re.compile(r"%DFLS_(?!AUTOHOST)[A-Z_]+%|%VARIANT%|%ROOT%|%DFL_SETUP_PYTHON%")
        for line in body:
            if caller_state.search(line):
                assert "cmd /c" not in line, (
                    f"caller-controlled state on a child-cmd line: {line!r}"
                )

    def test_setup_helper_cmd_c_lines_carry_no_caller_data(self):
        """Every `cmd /c` line in the setup helper carries ONLY a
        launcher-owned host command (the fixed literals `py -3` /
        `python`, or the discovery variable %DFLS_AUTOHOST% which can
        only ever hold one of those two literals) and hardcoded
        probe/floor/capture/validator code - never a caller value
        variable, the override value, the variant, or a rebuilt command
        string. The override value crosses no child-cmd command line at
        all: the validator reads it from the environment."""
        text = _bat_text(LAUNCHERS_DIR / "dfl-setup-runtime.bat")
        # comment lines may quote the design prose (e.g. `cmd /c <host>`)
        body = "\n".join(
            line
            for line in text.splitlines()
            if line.strip() and not line.lstrip().lower().startswith("rem")
        )
        for m in re.finditer(r"cmd /c [^\r\n]*", body):
            frag = m.group(0)
            # caller-controlled state on a child-cmd command line is the
            # injection hole (rounds 2-3): it must not appear.
            assert not re.search(
                r"%DFLS_(?!AUTOHOST)[A-Z_]+%|%VARIANT%|%ROOT%|%DFL_SETUP_PYTHON%", frag
            ), f"caller data on a cmd /c line: {frag!r}"
            assert "build_runtime.py" not in frag, (
                f"builder invocation may not run through a child cmd: {frag!r}"
            )
            assert re.search(
                r"cmd /c (py -3|python|%DFLS_AUTOHOST%) -I -B -c \"import", frag
            ), (
                f"cmd /c line is not a fixed probe/floor/capture/validator: {frag!r}"
            )

    def test_setup_helper_builder_lines_are_direct_executions(self):
        """All sixteen builder-invocation lines execute the host DIRECTLY
        (no CALL, no child cmd): the eight automatic lines run
        %HOSTCMD% (which holds only the launcher-fixed literals `py -3`
        / `python`) and the eight override lines run "%DFLS_OVR%" (the
        prevalidated .exe path). All forward the values inside template
        double quotes and are immediately followed by
        `exit /b %ERRORLEVEL%`, so an in-process batch host reached
        through the automatic candidates is the helper's last command."""
        text = _bat_text(LAUNCHERS_DIR / "dfl-setup-runtime.bat")
        lines = text.splitlines()
        auto = [f":dfls-run-{i}" for i in range(8)]
        ovr = [f":dfls-run-ovr-{i}" for i in range(8)]
        for label, prefix in (
            [(l, '%HOSTCMD% -I -B "%ROOT%\\scripts\\build_runtime.py" build --variant "%VARIANT%"') for l in auto]
            + [(l, '"%DFLS_OVR%" -I -B "%ROOT%\\scripts\\build_runtime.py" build --variant "%VARIANT%"') for l in ovr]
        ):
            idx = lines.index(label)
            cmd = lines[idx + 1]
            assert cmd.startswith(prefix), (
                f"{label} builder line is not a direct host execution: {cmd!r}"
            )
            assert "call" not in cmd.lower(), f"{label} builder line uses CALL"
            assert lines[idx + 2] == "exit /b %ERRORLEVEL%", (
                f"{label} must be followed immediately by exit /b: {lines[idx + 2]!r}"
            )
            # every stored value variable appears only inside template quotes
            for var in ('--offline-inputs "%DFLS_OFFIN_VAL%"', '--artifact-cache "%DFLS_ARTC_VAL%"',
                        '--lock-timeout "%DFLS_LOCKT_VAL%"'):
                if var.split('"')[0] in cmd:
                    assert var in cmd, f"{label} value variable not quoted: {cmd!r}"

    def test_setup_helper_validator_code_matches_constant(self):
        """The setup bat's DFL_SETUP_PYTHON validator code must equal the
        VALIDATOR_CODE constant used by the fake-host template byte for
        byte (the fake host matches the code by exact comparison)."""
        text = _bat_text(LAUNCHERS_DIR / "dfl-setup-runtime.bat")
        m = re.search(
            r'cmd /c %DFLS_AUTOHOST% -I -B -c "([^"]*)"', text
        )
        assert m is not None, "validator cmd /c line not found in setup helper"
        assert m.group(1) == VALIDATOR_CODE, (
            "validator code in the bat differs from the test constant"
        )
        # and the override value variable must never appear on any
        # executable line except the single validated capture
        body = [
            line
            for line in text.splitlines()
            if line.strip() and not line.lstrip().lower().startswith("rem")
        ]
        captures = [line for line in body if 'set "DFLS_OVR=%DFL_SETUP_PYTHON%"' in line]
        assert len(captures) == 1, "the override value may be expanded exactly once"
        # and the raw value must not be expanded on any other
        # executable line - not even as one-character substrings: on
        # this cmd build (Windows 10.0.26200) a compound `if defined
        # VAR if "%VAR:~0,1%"==...` line aborts the whole batch with
        # a syntax error even when VAR is undefined, and a
        # quote-bearing value corrupts quote pairing far enough that a
        # trailing &/| command fragment can execute. Any expansion of
        # the value is therefore forbidden outside the single
        # validated capture; the `none` sentinel is recognized by the
        # data-space validator (rc 7), never by cmd parsing.
        for line in body:
            if line.strip() == 'set "DFLS_OVR=%DFL_SETUP_PYTHON%"':
                continue
            assert "%DFL_SETUP_PYTHON%" not in line, (
                f"raw override value expanded outside the validated capture: {line!r}"
            )
            assert "DFL_SETUP_PYTHON:~" not in line, (
                f"substring expansion of the raw override value is forbidden: {line!r}"
            )

    def test_normal_launcher_has_no_helper_file_mechanism(self):
        """Review finding 3 (round 2): the normal launcher reads the
        selector without ANY helper file - no TEMP file, no pattern file,
        no set /p over file content - so a normal launch mutates neither
        the repository tree nor the temp area."""
        text = _bat_text(LAUNCHERS_DIR / "dfl.bat")
        for token in ("%TEMP%", "%TMP%", "dfl_pct", ".tmp", "set /p"):
            assert token not in text, f"helper-file mechanism in dfl.bat: {token!r}"
        # the file-as-data gates: size, line count, character set -
        # all against the file, none of them expanding raw content
        assert r'for %%F in (runtime\active-runtime.txt) do set "DFL_SELSZ=%%~zF"' in text
        expected_gate = (
            'for /f "tokens=2 delims=:" %%N in (\''
            'find /c /v "" "runtime\\active-runtime.txt" 2^>nul\''
            ') do set "DFL_SELLINES=%%N"'
        )
        assert expected_gate in text
        assert r'findstr /R /C:"[^0-9abcdef]" "runtime\active-runtime.txt" >nul 2>nul' in text

    def test_selector_validates_file_as_data_before_any_read(self):
        """Review finding 1 (round 3): UNVALIDATED selector bytes must
        never be substituted into a parsed batch command (a quoted SET
        breaks on an embedded quote on this cmd build). The file is
        validated AS DATA (size gate, line-count gate, character-set
        gate - all operating on the file) and the line is captured ONLY
        AFTER the character-set gate proves it pure lowercase hex."""
        text = _bat_text(LAUNCHERS_DIR / "dfl.bat")
        lines = text.splitlines()

        def command_line(prefix):
            """Index of the first EXECUTABLE line (comments excluded)
            starting with the given prefix, so that rem-line quotes of a
            tool name can never be mistaken for the gate command itself."""
            for i, line in enumerate(lines):
                if line.lstrip().startswith(prefix):
                    return i
            raise AssertionError(f"gate command missing from dfl.bat: {prefix!r}")

        i_size = command_line('for %%F in (runtime\\active-runtime.txt) do set "DFL_SELSZ=')
        i_lines = command_line('for /f "tokens=2 delims=:"')
        i_charset = command_line('findstr /R /C:')
        i_capture = command_line('for /f "delims=" %%L in (runtime\\active-runtime.txt)')
        i_len63 = command_line('if "%DFL_ID:~63,1%"')
        i_len64 = command_line('if not "%DFL_ID:~64,1%"')

        # strict ordering: data gates first, capture only after them
        assert i_size < i_lines < i_charset < i_capture < i_len63 < i_len64
        # the old raw-capture mechanisms are gone
        assert 'set "DFL_LINE=%%A"' not in text, "raw first-line capture is forbidden"
        assert "findstr /n /r" not in text, "numbered-line pipeline is no longer used"
        # delimiter tokenization of the selector content is forbidden: the
        # ONLY for /f line using non-empty delims tokenizes the count
        # output of `find /c`, never the selector bytes; the selector
        # capture line must use an EMPTY delims list.
        for line in lines:
            if "delims=:" in line:
                assert "tokens=2" in line and "find /c /v" in line, (
                    f"delimiter tokenization outside the line-count gate: {line!r}"
                )
        assert lines[i_capture].startswith('for /f "delims="'), (
            "the selector capture must use an EMPTY delims list"
        )
        assert "tokens=1,*" not in text, "token splitting of the selector line is forbidden"
        # the captured value may only be expanded in inert forms
        assert r'set "DFL_ID=%%L"' in lines[i_capture]
        # the launch line uses the validated id
        assert r'"runtime\versions\%DFL_ID%\python.exe" -I -B "scripts\runtime_entry.py" %*' in text
    def test_envreport_alias_is_thin(self):
        text = _bat_text(LAUNCHERS_DIR / "dfl-envreport.bat")
        assert 'call "%~dp0dfl.bat" envreport' in text.replace("%%", "%") or (
            'call "%~dp0dfl.bat"' in text and "envreport" in text
        )

    def test_crlf_line_endings(self):
        for name in self.FILES:
            data = (LAUNCHERS_DIR / name).read_bytes()
            assert b"\r\n" in data, f"{name} has no CRLF lines"
            lf = data.count(b"\n")
            cr = data.count(b"\r\n")
            assert lf == cr, f"{name} has {lf - cr} bare LF line endings"

    def test_gitattributes_locks_bat_crlf(self):
        text = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
        assert re.search(r"^\*\.bat[ \t]+text[ \t]+eol=crlf", text, re.MULTILINE)


# ---------------------------------------------------------------------------
# normal launcher: selector handling
# ---------------------------------------------------------------------------


class TestLauncherSelector:
    VALID_64 = "ab" * 32  # exactly 64 lowercase hex

    def _run(self, tmp_path, selector, **kw):
        fx = make_fake_repo(
            tmp_path / "repo", selector=selector, with_python=kw.get("with_python", True)
        )
        log = tmp_path / "fakeint.log"
        env = _env(HOSTILE_PARENT_ENV, FAKEINT_LOG=str(log), **kw.get("env_extra", {}))
        root = fx["root"]
        p = run_bat(root / "launchers" / "dfl.bat", cwd=root / "launchers", env=env)
        return p, log, root, fx["runtime_id"]

    def test_missing_selector_fails_clearly(self, tmp_path):
        p, log, root, _ = self._run(tmp_path, selector=None)
        assert p.returncode == 2
        assert "active-runtime" in (p.stderr + p.stdout).lower()
        assert not log.exists(), "interpreter must not run without a selector"

    @pytest.mark.parametrize(
        "selector",
        [
            "",                                # blank file
            "ABCDEF0123456789" * 4,            # uppercase hex
            "ab" * 31 + "x",                   # non-hex char
            "ab" * 31,                         # 62 chars (too short)
            "ab" * 33,                         # 66 chars (too long)
            "ab" * 32 + " ",                   # trailing space on the line
            "..\\..\\evil",                    # traversal attempt
            "..\\evil",                        # short traversal attempt
            "ab\\..\\cd",                      # separator inside the id
            "ABCD" + "ab" * 30,                # mixed case
            "0" * 64 + "f",                    # 65 lowercase hex chars
            # colon-prefixed lines (review finding 2, round 2): the OLD
            # delimiter tokenization turned "::<valid-id>" into
            # "<valid-id>" and launched it. The new design reads the exact
            # raw bytes, so the colons survive into the value and the
            # charset/length gates reject the line - no normalization.
            ":" + "ab" * 31 + "a",             # :<63-hex>, 64 chars total
            ":" + VALID_64,                    # :<valid 64-hex id>, 65 chars
            "::" + "ab" * 31,                  # ::<62-hex>, 64 chars total
            "anything:" + "ab" * 27,           # prefix:<hex>, 64 chars total
            # round-3 review matrix: quote + metacharacter attacks
            "\"& mkdir selector-executed & rem",          # quote + ampersand (reviewer repro)
            "\"&set SELECTOR_EXECUTED=1 & echo PWNED > m-selset.txt&rem",  # quote + SET
            "\"| mkdir selector-pipe & rem",              # quote + pipe
            "\"^& mkdir selector-caret & rem",            # quote + caret + ampersand
            " " + "ab" * 31 + "a",                        # leading space + 64 hex
        ],
    )
    def test_malformed_selectors_fail_clearly_and_inertly(self, tmp_path, selector):
        p, log, root, _ = self._run(tmp_path, selector=selector)
        assert p.returncode == 2, f"selector {selector!r} must be rejected"
        assert not log.exists(), "interpreter must not run for a malformed selector"
        sel_file = root / "runtime" / "active-runtime.txt"
        # quote+metacharacter payloads must leave no side effects
        for marker in ("selector-executed", "selector-pipe", "selector-caret", "m-selset.txt"):
            assert not (root / marker).exists(), f"marker {marker!r} appeared: selector bytes executed"
        if selector is not None and sel_file.is_file():
            assert sel_file.read_bytes() == (selector + "\n").encode("utf-8")

    def test_valid_second_line_cannot_rescue_malformed_first_line(self, tmp_path):
        """Only the FIRST physical line is read: a valid id on a later line
        must never be selected when the first line is malformed."""
        selector = "junk-first-line\n" + self.VALID_64
        fx = make_fake_repo(tmp_path / "repo", selector=None)
        (fx["root"] / "runtime" / "active-runtime.txt").write_text(
            selector + "\n", encoding="utf-8", newline="\n"
        )
        log = tmp_path / "fakeint.log"
        env = _env(HOSTILE_PARENT_ENV, FAKEINT_LOG=str(log))
        p = run_bat(fx["root"] / "launchers" / "dfl.bat", cwd=fx["root"], env=env)
        assert p.returncode == 2
        assert not log.exists(), "a valid second line must never rescue a bad first line"

    def test_valid_first_line_plus_malicious_second_line(self, tmp_path):
        """The inverse of the rescue test: even when the FIRST line is a
        perfectly valid runtime id, a malicious second line makes the
        selector malformed (exactly one logical line is required) and
        must neither execute nor be launched."""
        selector = self.VALID_64 + "\n\"& mkdir selector-second & rem"
        fx = make_fake_repo(tmp_path / "repo", selector=None)
        (fx["root"] / "runtime" / "active-runtime.txt").write_text(
            selector + "\n", encoding="utf-8", newline="\n"
        )
        log = tmp_path / "fakeint.log"
        env = _env(HOSTILE_PARENT_ENV, FAKEINT_LOG=str(log))
        p = run_bat(fx["root"] / "launchers" / "dfl.bat", cwd=fx["root"], env=env)
        assert p.returncode == 2
        assert not log.exists(), "a malicious second line must never run"
        assert not (fx["root"] / "selector-second").exists()

    @pytest.mark.parametrize(
        "selector",
        [
            "%",                # bare percent
            "%SEL%",            # percent variable pair
            "%SEL, x",          # unbalanced percent pair
            '"quoted"',         # quote character
            'a"b',              # quote then ampersand-free text
            "a&b",              # ampersand
            "a&b & echo PWNED > m-metachar.txt",  # ampersand command chain
            "a|b",              # pipe
            "a^b",              # caret
            "(a)",              # parentheses
            "a b",              # space
            "a\tb",             # tab
            "<x>",              # redirection characters
            "a:b",              # colon
        ],
    )
    def test_selector_metacharacters_rejected_as_raw_content(self, tmp_path, selector):
        # The raw file bytes are authoritative: none of these characters may
        # be interpreted (as percent expansion, quoting, or command
        # metacharacters) - the line is rejected as malformed, and no
        # metacharacter in the line may execute during the validation.
        p, log, root, _ = self._run(tmp_path, selector=selector)
        assert p.returncode == 2, f"selector {selector!r} must be rejected"
        assert not log.exists(), "interpreter must not run for this selector"
        assert not (root / "m-metachar.txt").exists(), (
            "a metacharacter in the selector was interpreted as a command"
        )

    def test_percent_variable_expansion_attack_is_rejected(self, tmp_path):
        """Review finding 2: a selector line of the form %NAME% must be
        treated as raw malformed content even when NAME is defined in the
        environment to a VALID 64-hex runtime id - the id from the
        environment must never win, and the interpreter must not run.
        """
        rid = "cd" * 32
        fx = make_fake_repo(tmp_path / "repo", selector="%MALFORMED_SELECTOR_EXPANDS%", runtime_id=rid)
        log = tmp_path / "fakeint.log"
        env = _env(
            HOSTILE_PARENT_ENV,
            FAKEINT_LOG=str(log),
            MALFORMED_SELECTOR_EXPANDS=rid,  # env points at the ONLY existing runtime
        )
        p = run_bat(fx["root"] / "launchers" / "dfl.bat", cwd=fx["root"], env=env)
        assert p.returncode == 2, "the %NAME% selector must fail, not expand"
        assert not log.exists(), "interpreter must not run for a percent selector"

    def test_env_value_breakout_through_selector_capture_is_inert(self, tmp_path):
        """A hostile environment value (quote + & + redirect) referenced by
        a %VAR% selector line must expand into the captured text WITHOUT
        executing: the for /f capture body is a quoted set whose argument
        span (first template quote to last) contains any substituted or
        expanded text, so no command can start; the value then fails the
        grammar gates and the launcher exits 2."""
        rid = "cd" * 32
        fx = make_fake_repo(tmp_path / "repo", selector="%SELATTACK%", runtime_id=rid)
        log = tmp_path / "fakeint.log"
        env = _env(
            HOSTILE_PARENT_ENV,
            FAKEINT_LOG=str(log),
            SELATTACK='X & echo PWNED > m-breakout.txt',
        )
        p = run_bat(fx["root"] / "launchers" / "dfl.bat", cwd=fx["root"], env=env)
        assert p.returncode == 2
        assert not log.exists(), "interpreter must not run for this selector"
        assert not (fx["root"] / "m-breakout.txt").exists(), (
            "expanded env value was executed during selector validation"
        )

    def test_selector_file_unmodified_after_success(self, tmp_path):
        import hashlib

        rid = "cd" * 32
        fx = make_fake_repo(tmp_path / "repo", selector=rid, runtime_id=rid)
        sel = fx["root"] / "runtime" / "active-runtime.txt"
        before = hashlib.sha256(sel.read_bytes()).hexdigest()
        log = tmp_path / "fakeint.log"
        env = _env(HOSTILE_PARENT_ENV, FAKEINT_LOG=str(log))
        p = run_bat(fx["root"] / "launchers" / "dfl.bat", cwd=fx["root"], env=env)
        assert p.returncode == 0
        assert hashlib.sha256(sel.read_bytes()).hexdigest() == before

    def test_normal_launch_writes_no_temp_files(self, tmp_path):
        """Review finding 3 (round 2): a normal launch must not create,
        mutate, or delete ANY file in the temp area (the old design wrote
        a %TEMP%\\dfl_pct.tmp helper on every launch). Two cases: TEMP
        present (private empty dir) and TEMP absent entirely."""
        rid = "ab" * 32
        for case in ("temp-set", "temp-undefined"):
            fx = make_fake_repo(tmp_path / f"repo-{case}", selector=rid, runtime_id=rid)
            log = tmp_path / f"fakeint-{case}.log"
            env = _env(HOSTILE_PARENT_ENV, FAKEINT_LOG=str(log))
            if case == "temp-set":
                temp = tmp_path / f"private-temp-{case}"
                temp.mkdir()
                env["TEMP"] = str(temp)
                env["TMP"] = str(temp)
                before = tree_snapshot(temp)
            else:
                env.pop("TEMP", None)
                env.pop("TMP", None)
                before = None
            p = run_bat(fx["root"] / "launchers" / "dfl.bat", "--self-test", cwd=fx["root"], env=env)
            assert p.returncode == 0, p.stderr
            if case == "temp-set":
                assert tree_snapshot(temp) == before, "a normal launch wrote to the temp area"
            # and no helper pattern file may exist anywhere in the tree
            leftovers = [
                str(pth)
                for pth in fx["root"].rglob("*")
                if pth.is_file() and ("dfl_pct" in pth.name or pth.name.endswith(".tmp"))
            ]
            assert leftovers == [], f"helper file left behind: {leftovers}"

    def test_missing_runtime_directory(self, tmp_path):
        p, log, root, rid = self._run(tmp_path, selector="cd" * 32, with_python=True)
        # selector valid, but the version directory was not created for cd..
        # (fixture only creates it for the default id)
        assert p.returncode == 1
        assert "runtime" in p.stderr.lower()
        assert not log.exists()

    def test_runtime_dir_without_interpreter(self, tmp_path):
        fx = make_fake_repo(tmp_path / "repo", selector="cd" * 32, runtime_id="cd" * 32, with_python=False)
        env = _env(HOSTILE_PARENT_ENV, FAKEINT_LOG=str(tmp_path / "x.log"))
        p = run_bat(fx["root"] / "launchers" / "dfl.bat", cwd=fx["root"], env=env)
        assert p.returncode == 1
        assert "interpreter" in p.stderr.lower()

    def test_missing_runtime_entry_bootstrap(self, tmp_path):
        rid = "cd" * 32
        fx = make_fake_repo(tmp_path / "repo", selector=rid, runtime_id=rid, with_entry=False)
        (fx["root"] / "scripts").mkdir(parents=True, exist_ok=True)
        env = _env(HOSTILE_PARENT_ENV, FAKEINT_LOG=str(tmp_path / "x.log"))
        p = run_bat(fx["root"] / "launchers" / "dfl.bat", cwd=fx["root"], env=env)
        assert p.returncode == 1
        assert "runtime_entry" in p.stderr.lower()


# ---------------------------------------------------------------------------
# normal launcher: happy path, args, exit codes, CWD independence
# ---------------------------------------------------------------------------


class TestLauncherLaunch:
    @staticmethod
    def _happy(tmp_path, *args, cwd=None, **env_extra):
        rid = "ab" * 32
        fx = make_fake_repo(tmp_path / "repo", selector=rid, runtime_id=rid)
        log = tmp_path / "fakeint.log"
        env = _env(HOSTILE_PARENT_ENV, FAKEINT_LOG=str(log), **env_extra)
        root = fx["root"]
        p = run_bat(root / "launchers" / "dfl.bat", *args, cwd=cwd or root, env=env)
        return p, log, root

    def test_happy_path_launches_selected_interpreter(self, tmp_path):
        p, log, root = self._happy(tmp_path)
        assert p.returncode == 0
        info = read_fakeint_log(log)
        assert info["args"] is not None
        assert "runtime_entry.py" in info["args"]
        assert info["args"].startswith("-I -B")
        assert info["exec"] == f"runtime\\versions\\{'ab' * 32}\\python.exe"

    # Note: arguments are forwarded through raw cmd %%* semantics. Any
    # argument that is quoted on the command line (spaces/tabs/quotes)
    # is fully protected against cmd metacharacters; UNQUOTED arguments
    # containing cmd metacharacters (| & < > ^ %) cannot be preserved by
    # any Windows batch launcher, and such inputs are outside the
    # launcher contract (the same holds for every .bat-based launcher).
    @pytest.mark.parametrize(
        "args",
        [
            [],
            ["--self-test"],
            ["-m", "my model", "plain"],
            ["-a", "with & ampersand", "-b"],
            ["-s", "semi;colon"],
        ],
    )
    def test_argument_forwarding_preserved(self, tmp_path, args):
        p, log, _ = self._happy(tmp_path, *args)
        assert p.returncode == 0
        info = read_fakeint_log(log)
        assert info["args"] is not None
        tail = info["args"].split("runtime_entry.py", 1)[1].strip()
        for a in args:
            assert a in tail or a in info["args"], f"arg {a!r} lost in forwarding"

    @pytest.mark.parametrize("code", [0, 3, 7, 1])
    def test_exit_code_propagation(self, tmp_path, code):
        p, log, _ = self._happy(tmp_path, FAKEINT_EXIT=str(code))
        assert p.returncode == code

    def test_independent_of_caller_cwd(self, tmp_path):
        rid = "ab" * 32
        fx = make_fake_repo(tmp_path / "repo", selector=rid, runtime_id=rid)
        root = fx["root"]
        log = tmp_path / "fakeint.log"
        env = _env(HOSTILE_PARENT_ENV, FAKEINT_LOG=str(log))
        for cwd in (root, root.parent, tmp_path, REPO_ROOT):
            p = run_bat(root / "launchers" / "dfl.bat", "--self-test", cwd=cwd, env=env)
            assert p.returncode == 0, f"failed from cwd={cwd}"
            info = read_fakeint_log(log)
            assert info["exec"] == f"runtime\\versions\\{rid}\\python.exe"

    def test_spaces_in_repository_path(self, tmp_path):
        rid = "ab" * 32
        fx = make_fake_repo(tmp_path / "repo root", selector=rid, runtime_id=rid)
        log = tmp_path / "fakeint.log"
        env = _env(HOSTILE_PARENT_ENV, FAKEINT_LOG=str(log))
        p = run_bat(fx["root"] / "launchers" / "dfl.bat", "--self-test", cwd=REPO_ROOT, env=env)
        assert p.returncode == 0
        info = read_fakeint_log(log)
        assert info["exec"] == f"runtime\\versions\\{rid}\\python.exe"

    def test_no_host_python_on_path(self, tmp_path):
        # The launch chain must succeed even with a PATH that contains no
        # Python at all: the only interpreter reachable is the selected
        # runtime's bundled (fake) interpreter, executed via PATHEXT.
        rid = "ab" * 32
        fx = make_fake_repo(tmp_path / "repo", selector=rid, runtime_id=rid)
        log = tmp_path / "fakeint.log"
        env = _env(HOSTILE_PARENT_ENV, FAKEINT_LOG=str(log))
        env["PATH"] = r"C:\Windows\System32"
        p = run_bat(fx["root"] / "launchers" / "dfl.bat", cwd=fx["root"], env=env)
        assert p.returncode == 0
        info = read_fakeint_log(log)
        assert info["exec"] == f"runtime\\versions\\{rid}\\python.exe"

    def test_runtime_tree_unmodified_after_launch(self, tmp_path):
        rid = "ab" * 32
        fx = make_fake_repo(tmp_path / "repo", selector=rid, runtime_id=rid)
        root = fx["root"]
        log = tmp_path / "fakeint.log"
        env = _env(HOSTILE_PARENT_ENV, FAKEINT_LOG=str(log))
        runtime_before = tree_snapshot(root / "runtime")
        repo_before = tree_snapshot(root)
        p = run_bat(root / "launchers" / "dfl.bat", "--self-test", cwd=root, env=env)
        assert p.returncode == 0
        assert tree_snapshot(root / "runtime") == runtime_before
        assert tree_snapshot(root) == repo_before


# ---------------------------------------------------------------------------
# normal launcher: environment isolation (incl. NN case handling and
# injection safety)
# ---------------------------------------------------------------------------


class TestLauncherEnvironment:
    @staticmethod
    def _child_env(tmp_path):
        rid = "ab" * 32
        fx = make_fake_repo(tmp_path / "repo", selector=rid, runtime_id=rid)
        log = tmp_path / "fakeint.log"
        env = _env(HOSTILE_PARENT_ENV, FAKEINT_LOG=str(log))
        p = run_bat(fx["root"] / "launchers" / "dfl.bat", "--self-test", cwd=fx["root"], env=env)
        assert p.returncode == 0
        return read_fakeint_log(log)["env"], fx["root"]

    def test_host_python_env_cleared(self, tmp_path):
        env, _ = self._child_env(tmp_path)
        for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE"):
            assert name not in env, f"{name} must be cleared from the child environment"

    def test_isolation_markers_published(self, tmp_path):
        env, _ = self._child_env(tmp_path)
        assert env.get("PYTHONNOUSERSITE") == "1"
        assert env.get("PYTHONDONTWRITEBYTECODE") == "1"

    def test_stale_device_state_cleared_family_preserved(self, tmp_path):
        env, _ = self._child_env(tmp_path)
        for name in (
            "NN_DEVICES_INITIALIZED",
            "NN_DEVICES_COUNT",
            "NN_DEVICE_0",
            "NN_DEVICE_1",
            "NN_DEVICE_ABC",
            "Nn_Device_Mixed",
            "nn_device_lower",
        ):
            assert name not in env, f"{name} must be cleared (any letter case)"
        # NN_DEVICE (no trailing underscore) is not part of the family.
        assert env.get("NN_DEVICE") == "keepme"
        assert env.get("NN_DEVICES_X") == "keepme2"

    def test_cuda_vars_pass_through_to_runtime_entry(self, tmp_path):
        # CUDA sanitation is owned by runtime_entry.py (Commit-2 bootstrap
        # owner); the launcher must not add its own CUDA mask. (The
        # current-runtime integration test asserts the runtime_entry
        # clearance end to end.)
        env, _ = self._child_env(tmp_path)
        assert env.get("CUDA_PATH") == "stale-cuda-toolkit"
        assert env.get("CUDA_HOME") == "stale-cuda-home"
        assert env.get("CUDA_VISIBLE_DEVICES") == "0"

    def test_unrelated_variables_pass_through(self, tmp_path):
        env, _ = self._child_env(tmp_path)
        assert env.get("FAKESENTINEL_X") == "sent"

    def test_hostile_nn_values_never_executed(self, tmp_path):
        """Review finding 9: NN_* values containing cmd metacharacters must
        be cleared without any command execution (no CALL / delayed
        expansion over attacker-controlled values)."""
        env, root = self._child_env(tmp_path)
        # the hostile value must be gone from the child environment ...
        assert "NN_DEVICE_INJ" not in env
        # ... and must never have been evaluated as command text: the
        # value's `echo PWNED > marker-inj.txt` side effect never ran in
        # the launcher's working directory (the repository root).
        assert not (root / "marker-inj.txt").exists(), "hostile NN value was executed"
        # preserved variables remain intact through the hostile state.
        assert env.get("NN_DEVICE") == "keepme"
        assert env.get("NN_DEVICES_X") == "keepme2"


# ---------------------------------------------------------------------------
# dfl-envreport.bat alias
# ---------------------------------------------------------------------------


class TestEnvreportAlias:
    def test_alias_forwards_command_and_args(self, tmp_path):
        rid = "ab" * 32
        fx = make_fake_repo(tmp_path / "repo", selector=rid, runtime_id=rid)
        log = tmp_path / "fakeint.log"
        env = _env(HOSTILE_PARENT_ENV, FAKEINT_LOG=str(log))
        p = run_bat(
            fx["root"] / "launchers" / "dfl-envreport.bat",
            "--json",
            "--extra",
            cwd=fx["root"],
            env=env,
        )
        assert p.returncode == 0
        info = read_fakeint_log(log)
        assert "envreport" in info["args"]
        assert "--json" in info["args"]
        assert "--extra" in info["args"]
        assert "runtime_entry.py" in info["args"]

    def test_alias_propagates_exit_code(self, tmp_path):
        rid = "ab" * 32
        fx = make_fake_repo(tmp_path / "repo", selector=rid, runtime_id=rid)
        log = tmp_path / "fakeint.log"
        env = _env(HOSTILE_PARENT_ENV, FAKEINT_LOG=str(log), FAKEINT_EXIT="9")
        p = run_bat(fx["root"] / "launchers" / "dfl-envreport.bat", cwd=fx["root"], env=env)
        assert p.returncode == 9


# ---------------------------------------------------------------------------
# setup helper: usage, variants, option grammar, injection safety,
# host discovery / floor / fallback policy, delegation
# ---------------------------------------------------------------------------


class TestSetupHelper:
    @staticmethod
    def _setup(tmp_path, *args, **env_extra):
        """Run the setup helper in a fake repo where the explicit
        DFL_SETUP_PYTHON override is a lone copy of a real base CPython
        .exe (placed in a directory whose name contains a space) and the
        builder is the builtins-only stub logging its argv to build.log.

        The override host must be a real .exe on this cmd build: the
        helper's override probe/floor lines execute the host DIRECTLY
        in-process, where a batch file's `exit /b` terminates the whole
        process (it never returns to the helper). The fake .bat
        py/python hosts on a controlled PATH serve only as the override
        VALIDATOR (they always run in a child cmd, so in-process batch
        exit semantics never apply to them).
        """
        host = make_standalone_host(tmp_path / "sp ace host")
        fx = make_fake_repo(
            tmp_path / "repo",
            selector=None,
            with_python=False,
            with_builder_stub=True,
        )
        fakedir = tmp_path / "hosts"
        logs = make_fake_host_dir(fakedir)
        env = _env(
            DFL_SETUP_PYTHON=str(host),
            FAKE_PY_VALIDATE_EXIT="0",
            FAKE_PYTHON_VALIDATE_EXIT="0",
            FAKE_PY_LOG=str(fakedir.parent / "fake_py.log"),
            FAKE_PYTHON_LOG=str(fakedir.parent / "fake_python.log"),
            **env_extra,
        )
        env["PATH"] = f"{fakedir};C:\\Windows\\System32"
        p = run_bat(
            fx["root"] / "launchers" / "dfl-setup-runtime.bat", *args, cwd=fx["root"], env=env
        )
        stub_log = fx["root"] / "build.log"
        # only the builder-invocation lines (the stub also writes one
        # `EXE <interpreter>` line, consumed by the override-host tests
        # that read build.log directly)
        hlog = [
            line
            for line in (
                stub_log.read_text(encoding="utf-8", errors="replace").splitlines()
                if stub_log.is_file()
                else []
            )
            if line.startswith("BUILD ")
        ]
        return p, hlog, logs
    def test_missing_variant_usage_error(self, tmp_path):
        p, lines, _ = self._setup(tmp_path)
        assert p.returncode == 2
        assert "variant" in (p.stdout + p.stderr).lower()
        assert lines == [], "builder must not run on a usage error"

    @pytest.mark.parametrize(
        "variant", ["cuda", "CUDA-GUI", "cpu-guis", "cuda-nogui-x", "torch"]
    )
    def test_invalid_variant_rejected(self, tmp_path, variant):
        p, lines, _ = self._setup(tmp_path, variant)
        assert p.returncode == 2
        assert lines == []

    @pytest.mark.parametrize("variant", ["cuda-gui", "cuda-nogui", "cpu-gui", "cpu-nogui"])
    def test_all_four_variants_accepted_and_delegated(self, tmp_path, variant):
        p, lines, _ = self._setup(tmp_path, variant)
        assert p.returncode == 0, p.stderr
        assert len(lines) == 1
        line = lines[0]
        assert line.startswith("BUILD ")
        assert "build_runtime.py" in line
        assert "build" in line
        # the launcher passes the variant quoted; accept either rendering
        assert f'--variant "{variant}"' in line or f"--variant {variant}" in line

    @pytest.mark.parametrize(
        "extra",
        [
            ["--frobnicate"],
            ["--activate", "--frobnicate"],
            ["--offline-inputs"],            # valued option without a value
            ["--artifact-cache"],
            ["--lock-timeout"],
            ["--offline", "--lock-timeout"],
            # review finding 5 (round 2): a valued option FOLLOWED by
            # another supported option token must fail with rc 2 and the
            # builder must not run (a value must never be an option
            # token; no escape syntax is provided).
            ["--artifact-cache", "--activate"],
            ["--artifact-cache", "--offline"],
            ["--offline-inputs", "--activate"],
            ["--offline-inputs", "--offline"],
            ["--lock-timeout", "--activate"],
            ["--lock-timeout", "--offline"],
            # ... and an unknown option-looking token as a value.
            ["--artifact-cache", "--frobnicate"],
            ["--offline-inputs", "--frobnicate"],
            ["--lock-timeout", "--frobnicate"],
        ],
    )
    def test_unknown_or_incomplete_options_rejected(self, tmp_path, extra):
        p, lines, _ = self._setup(tmp_path, "cpu-nogui", *extra)
        assert p.returncode == 2, (p.stdout, p.stderr)
        assert lines == [], "builder must not run on an option error"

    @pytest.mark.parametrize(
        "extra,needle",
        [
            (["--activate"], "--activate"),
            (["--offline"], "--offline"),
            (["--offline", "--offline-inputs", "x y"], "x y"),
            (["--artifact-cache", "cache dir"], "cache dir"),
            (["--lock-timeout", "30"], "30"),
            (["--activate", "--offline", "--lock-timeout", "45"], "--activate"),
        ],
    )
    def test_builder_flags_forwarded_unchanged(self, tmp_path, extra, needle):
        p, lines, _ = self._setup(tmp_path, "cpu-nogui", *extra)
        assert p.returncode == 0, p.stderr
        assert len(lines) == 1
        assert needle in lines[0]
        assert "build_runtime.py" in lines[0]

    def test_quoted_ampersand_value_preserved_not_executed(self, tmp_path):
        """Review finding 1/2: a correctly quoted value containing & must be
        forwarded LITERALLY (the old string-accumulation design would have
        executed the extra command). Exactly one builder invocation, the
        literal value in its log, and no side-effect file anywhere."""
        value = "x & echo PWNED > marker-inject.txt"
        p, lines, _ = self._setup(tmp_path, "cpu-nogui", "--artifact-cache", value)
        assert p.returncode == 0, p.stderr
        assert len(lines) == 1, "the builder must be invoked exactly once"
        assert value in lines[0], f"literal value lost: {lines[0]!r}"
        assert not (tmp_path / "repo" / "marker-inject.txt").exists(), "injected command executed"

    def test_ampersand_value_in_offline_inputs(self, tmp_path):
        """Review finding 2 (benign preservation): --artifact-cache 'cache &
        artifacts' must reach the builder as the literal value."""
        p, lines, _ = self._setup(tmp_path, "cpu-nogui", "--artifact-cache", "cache & artifacts")
        assert p.returncode == 0, p.stderr
        assert len(lines) == 1
        assert 'cache & artifacts' in lines[0]
        assert "build_runtime.py" in lines[0]

    def test_parenthesis_semicolon_values_preserved(self, tmp_path):
        p, lines, _ = self._setup(
            tmp_path,
            "cpu-nogui",
            "--offline-inputs",
            "(inputs; set)",
            "--artifact-cache",
            "semi;colon dir",
        )
        assert p.returncode == 0, p.stderr
        assert len(lines) == 1
        assert "(inputs; set)" in lines[0]
        assert "semi;colon dir" in lines[0]

    @pytest.mark.parametrize("option", ["--artifact-cache", "--offline-inputs", "--lock-timeout"])
    def test_nested_percent_expansion_attack_is_inert(self, tmp_path, option):
        """Review finding 1 (round 2), the reviewer's exact proof: the
        caller sets P13_OUTER=%P13_INNER% (literal text) and
        P13_INNER to a quote+&+marker payload, and passes the literal
        argument text %P13_OUTER% as a valued option's value.

        The parent cmd expands %P13_OUTER% exactly once (the result
        %P13_INNER% is NOT re-scanned). The helper stores that text raw
        and substitutes it into the builder line exactly once - there is
        no CALL and no child cmd on the builder path, so no second parse
        layer exists anywhere: the %P13_INNER% text reaches the host as
        literal argument text and can neither expand (its value is a
        command fragment, not data) nor execute. Assertions: rc 0, exactly
        one BUILD line carrying the literal %P13_INNER% text, and NO
        marker file anywhere in the tree.
        """
        inner = '" & echo PWNED > marker-nested.txt"'
        p, lines, _ = self._setup(
            tmp_path,
            "cpu-nogui",
            option,
            "%P13_OUTER%",  # literal text on the command line
            P13_INNER=inner,
            P13_OUTER="%P13_INNER%",  # literal text: the parent expands ONCE
        )
        assert p.returncode == 0, (p.stdout, p.stderr)
        root = tmp_path / "repo"
        build_lines = [l for l in lines if l.startswith("BUILD ")]
        assert len(build_lines) == 1, f"builder must run exactly once: {lines!r}"
        assert "build_runtime.py" in build_lines[0]
        # The nested payload must have reached the host as INERT literal
        # text: the outer var expanded to the INNER var's NAME (not value)
        # and no layer ever expanded or executed the name further.
        assert "%P13_INNER%" in build_lines[0], f"literal nested text lost: {build_lines[0]!r}"
        assert inner not in build_lines[0], f"nested payload was expanded: {build_lines[0]!r}"
        assert not (root / "marker-nested.txt").exists(), "nested payload executed (root)"
        assert not (tmp_path / "marker-nested.txt").exists(), "nested payload executed (tmp)"

    def test_builder_exit_code_propagated(self, tmp_path):
        # The stub builder exits with its DFL_FAKE_BUILD_EXIT value; the
        # helper's builder line is its last command, so the code
        # propagates as the helper's exit code.
        p, lines, _ = self._setup(tmp_path, "cpu-nogui", DFL_FAKE_BUILD_EXIT="7")
        assert p.returncode == 7, p.stderr
        assert len(lines) == 1

    def test_override_host_takes_precedence(self, tmp_path):
        # A valid explicit override (a real .exe path - a lone copy of a
        # real python.exe, in a directory whose name contains a space)
        # must be THE host that runs the builder; the automatic fake
        # candidates must not run it.
        host = make_standalone_host(tmp_path / "sp ace host")
        fx = make_fake_repo(
            tmp_path / "repo",
            selector=None,
            with_python=False,
            with_builder_stub=True,
        )
        fakedir = tmp_path / "hosts"
        logs = make_fake_host_dir(fakedir)
        env = _env(
            DFL_SETUP_PYTHON=str(host),
            FAKE_PY_VALIDATE_EXIT="0",
            FAKE_PYTHON_VALIDATE_EXIT="0",
            FAKE_PY_LOG=str(fakedir.parent / "fake_py.log"),
            FAKE_PYTHON_LOG=str(fakedir.parent / "fake_python.log"),
        )
        env["PATH"] = f"{fakedir};C:\\Windows\\System32"
        p = run_bat(
            fx["root"] / "launchers" / "dfl-setup-runtime.bat", "cpu-nogui", cwd=fx["root"], env=env
        )
        assert p.returncode == 0, p.stderr
        stub_log = fx["root"] / "build.log"
        assert stub_log.is_file(), "the validated override host must run the builder"
        body = stub_log.read_text(encoding="utf-8", errors="replace").splitlines()
        exe_lines = [l for l in body if l.startswith("EXE ")]
        assert len(exe_lines) == 1, f"exactly one builder run: {body}"
        assert str(host).lower() in exe_lines[0].lower(), "the override exe must be the builder host"
        arg_lines = [l for l in body if l.startswith(("build ", '"'))]
        assert any("build" in l and "--variant" in l and "cpu-nogui" in l for l in body)
        for log in logs.values():
            if log.is_file():
                auto = log.read_text(encoding="utf-8", errors="replace").splitlines()
                assert not any(l.startswith("BUILD ") for l in auto), "no automatic fallback when a valid override is set"

    def test_override_builder_exit_code_propagated(self, tmp_path):
        # The validated .exe override host runs the builder line DIRECTLY
        # (spawned child for a real exe): the builder's exit code (here
        # carried by the stub's DFL_FAKE_BUILD_EXIT environment knob)
        # becomes the helper's exit code; the --lock-timeout value is a
        # legitimate forwarded option value on the same line.
        host = make_standalone_host(tmp_path / "sp ace host")
        fx = make_fake_repo(
            tmp_path / "repo",
            selector=None,
            with_python=False,
            with_builder_stub=True,
        )
        fakedir = tmp_path / "hosts"
        logs = make_fake_host_dir(fakedir)
        env = _env(
            DFL_SETUP_PYTHON=str(host),
            FAKE_PY_VALIDATE_EXIT="0",
            FAKE_PYTHON_VALIDATE_EXIT="0",
            FAKE_PY_LOG=str(fakedir.parent / "fake_py.log"),
            FAKE_PYTHON_LOG=str(fakedir.parent / "fake_python.log"),
            DFL_FAKE_BUILD_EXIT="7",
        )
        env["PATH"] = f"{fakedir};C:\\Windows\\System32"
        p = run_bat(
            fx["root"] / "launchers" / "dfl-setup-runtime.bat",
            "cpu-nogui",
            "--lock-timeout",
            "7",
            cwd=fx["root"],
            env=env,
        )
        assert p.returncode == 7, p.stderr
        stub_log = fx["root"] / "build.log"
        body = stub_log.read_text(encoding="utf-8", errors="replace").splitlines()
        assert any("--lock-timeout" in l and "7" in l for l in body)
    def test_setup_bat_contains_no_deletion_commands(self):
        text = _bat_text(LAUNCHERS_DIR / "dfl-setup-runtime.bat")
        for rx in (
            re.compile(r"(?i)\brmdir\b"),
            re.compile(r"(?i)(?:^|[|&(\s])rd\s"),
            re.compile(r"(?i)(?:^|[|&(\s])del\s"),
            re.compile(r"(?i)\berase\b"),
            re.compile(r"Remove-Item", re.IGNORECASE),
        ):
            assert rx.search(text) is None


class TestSetupHostPolicy:
    """Review findings 5/10/11: exact probe exit-code semantics and the
    deterministic host-discovery fallback policy."""

    @staticmethod
    def _auto(tmp_path, *args, **env_extra):
        fx = make_fake_repo(tmp_path / "repo", selector=None, with_python=False)
        env, logs = auto_env(tmp_path, **env_extra)
        p = run_bat(
            fx["root"] / "launchers" / "dfl-setup-runtime.bat", *args, cwd=fx["root"], env=env
        )
        lines = {}
        for name, log in logs.items():
            lines[name] = (
                log.read_text(encoding="utf-8", errors="replace").splitlines()
                if log.is_file()
                else []
            )
        return p, lines

    # The round-3 override contract: DFL_SETUP_PYTHON is PATH DATA (one
    # .exe interpreter path), never command text. These helpers build the
    # two validation scenarios: a fake automatic candidate validates the
    # value (knob-controlled), or no fallback interpreter exists at all.
    HOSTCMD_ATTACKS = [
        'x"=="x" mkdir hostcmd-executed & rem',  # reviewer reproduction
        '"& mkdir hostcmd-executed & rem',  # quote + ampersand
        '"&set HOSTCMD_EXECUTED=1 & echo PWNED > m-override.txt&rem',  # quote + SET
        '"| mkdir hostcmd-executed & rem',  # quote + pipe
        '"^& mkdir hostcmd-executed & rem',  # quote + caret + ampersand
        "%HOSTCMD_INJ%",  # environment expansion syntax
        "x & mkdir hostcmd-executed",  # ampersand command separator
        "x | mkdir hostcmd-executed",  # pipe
        "x(y)",  # parentheses
        "x (mkdir hostcmd-executed) y",  # quoted-command fragment
        "x > m-override.txt",  # redirection
        "x & y & z",  # multiple commands
        'x"|"y ^& echo PWNED > m-override.txt',  # combined attack
    ]

    @staticmethod
    def _override_with_validator(tmp_path, value, *args, knob="6", **env_extra):
        """Override value + fake automatic candidates on PATH whose
        validator code exits with `knob` (0 = value accepted, 6 = value
        rejected)."""
        fx = make_fake_repo(
            tmp_path / "repo",
            selector=None,
            with_python=False,
            with_builder_stub=True,
        )
        fakedir = tmp_path / "hosts"
        logs = make_fake_host_dir(fakedir)
        env = _env(
            HOSTILE_PARENT_ENV,
            DFL_SETUP_PYTHON=value,
            FAKE_PY_VALIDATE_EXIT=knob,
            FAKE_PYTHON_VALIDATE_EXIT=knob,
            FAKE_PY_LOG=str(fakedir.parent / "fake_py.log"),
            FAKE_PYTHON_LOG=str(fakedir.parent / "fake_python.log"),
            **env_extra,
        )
        env["PATH"] = f"{fakedir};C:\\Windows\\System32"
        p = run_bat(
            fx["root"] / "launchers" / "dfl-setup-runtime.bat", *args, cwd=fx["root"], env=env
        )
        return p, fx, logs

    @staticmethod
    def _override_no_fallback(tmp_path, value, *args, **env_extra):
        """Override value with NO fallback interpreter on PATH: the value
        cannot be validated and must be rejected without ever being
        expanded by the helper process."""
        fx = make_fake_repo(tmp_path / "repo", selector=None, with_python=False)
        env = _env(HOSTILE_PARENT_ENV, DFL_SETUP_PYTHON=value, **env_extra)
        env["PATH"] = r"C:\Windows\System32"
        p = run_bat(
            fx["root"] / "launchers" / "dfl-setup-runtime.bat", *args, cwd=fx["root"], env=env
        )
        return p, fx
    def test_probe_failure_continues_to_next_candidate(self, tmp_path):
        # (A) py probes fine but its floor probe fails with rc 7 (NOT a
        # below-floor verdict): the policy must not misclassify it as
        # "too old" and must fall through to the valid python candidate.
        p, lines = self._auto(tmp_path, "cpu-nogui", FAKE_PY_FLOOR_EXIT="7")
        assert p.returncode == 0, p.stderr
        assert len(lines["python"]) == 1 and lines["python"][0].startswith("BUILD ")
        assert lines["py"] == [], "py must not run the builder"

    def test_below_floor_auto_candidate_continues(self, tmp_path):
        # (B) py is below floor; python is valid. The Phase-13 plan does
        # not make below-floor terminal for automatic discovery, so the
        # next valid candidate is selected.
        p, lines = self._auto(tmp_path, "cpu-nogui", FAKE_PY_FLOOR_EXIT="3")
        assert p.returncode == 0, p.stderr
        assert len(lines["python"]) == 1 and lines["python"][0].startswith("BUILD ")
        assert lines["py"] == []

    def test_malformed_floor_probe_falls_through(self, tmp_path):
        # (C) py's floor probe is malformed (rc 2 = probe failure, not a
        # floor verdict): fall through to python.
        p, lines = self._auto(tmp_path, "cpu-nogui", FAKE_PY_FLOOR_EXIT="2")
        assert p.returncode == 0, p.stderr
        assert len(lines["python"]) == 1 and lines["python"][0].startswith("BUILD ")
        assert lines["py"] == []

    def test_probe_failure_then_valid_candidate(self, tmp_path):
        # py cannot execute at all (probe rc 1): python candidate is used.
        p, lines = self._auto(tmp_path, "cpu-nogui", FAKE_PY_PROBE_EXIT="1")
        assert p.returncode == 0, p.stderr
        assert len(lines["python"]) == 1 and lines["python"][0].startswith("BUILD ")

    def test_all_auto_candidates_failed(self, tmp_path):
        p, lines = self._auto(
            tmp_path, "cpu-nogui", FAKE_PY_PROBE_EXIT="1", FAKE_PYTHON_PROBE_EXIT="1"
        )
        assert p.returncode == 3
        assert lines["py"] == [] and lines["python"] == []
        assert "host" in (p.stdout + p.stderr).lower()

    def test_all_auto_candidates_below_floor_reports_rc4(self, tmp_path):
        p, lines = self._auto(
            tmp_path,
            "cpu-nogui",
            FAKE_PY_FLOOR_EXIT="3",
            FAKE_PY_VER="3.10.14",
            FAKE_PYTHON_FLOOR_EXIT="3",
            FAKE_PYTHON_VER="3.9.7",
        )
        assert p.returncode == 4
        assert lines["py"] == [] and lines["python"] == []
        out = p.stdout + p.stderr
        assert "3.11" in out

    def test_py_below_floor_python_probe_failed_reports_rc4(self, tmp_path):
        # py below floor, python unusable: the below-floor classification
        # wins (rc 4), not the generic no-host (rc 3).
        p, lines = self._auto(
            tmp_path,
            "cpu-nogui",
            FAKE_PY_FLOOR_EXIT="3",
            FAKE_PY_VER="3.10.14",
            FAKE_PYTHON_PROBE_EXIT="1",
        )
        assert p.returncode == 4
        assert "3.10.14" in (p.stdout + p.stderr)

    @pytest.mark.parametrize("value", HOSTCMD_ATTACKS)
    def test_override_invalid_values_rejected_before_execution(self, tmp_path, value):
        # (D) hostile override values: the validator rejects them (rc 6),
        # so the override fails TERMINALLY with rc 3 before any part of
        # the value is executed: no marker, no builder (override or
        # automatic fallback), no probe of the value.
        p, fx, logs = self._override_with_validator(tmp_path, value, "cpu-nogui")
        assert p.returncode == 3, f"override {value!r} must be rejected: {p.stderr}"
        out = (p.stdout + p.stderr).lower()
        assert "override" in out or "dfl_setup_python" in out
        for marker in ("hostcmd-executed", "m-override.txt"):
            assert not (fx["root"] / marker).exists(), f"marker {marker!r} appeared: the value executed"
        for log in logs.values():
            if log.is_file():
                body = log.read_text(encoding="utf-8", errors="replace").splitlines()
                assert not any(l.startswith("BUILD ") for l in body), "builder must not run for an invalid override"

    @pytest.mark.parametrize("value", [HOSTCMD_ATTACKS[0], HOSTCMD_ATTACKS[5], HOSTCMD_ATTACKS[12]])
    def test_override_unvalidatable_without_fallback_interpreter(self, tmp_path, value):
        # (E) the reviewer's exact scenario: a crafted DFL_SETUP_PYTHON on
        # a machine where no fallback interpreter (py/python) exists. The
        # override cannot be validated and is rejected (rc 3) WITHOUT ANY
        # part of the value ever being expanded in the helper process.
        p, fx = self._override_no_fallback(tmp_path, value, "cpu-nogui")
        assert p.returncode == 3, p.stderr
        out = (p.stdout + p.stderr).lower()
        assert "dfl_setup_python" in out
        assert not (fx["root"] / "hostcmd-executed").exists(), "the value executed: command injection"
        assert not (fx["root"] / "m-override.txt").exists()

    def test_override_probe_failure_is_terminal(self, tmp_path):
        # (F) a value that IS valid path data (reg.exe: existing .exe, no
        # metacharacters) but is not a Python interpreter: the probe
        # fails and the override fails terminally (rc 3) with no fallback
        # to automatic discovery. (cmd.exe is NOT usable as this host on
        # this cmd build: it accepts dash-prefixed junk switches, falls
        # back to interactive mode and exits 0 on stdin EOF, so its
        # probe would spuriously pass.)
        p, fx, logs = self._override_with_validator(
            tmp_path, r"C:\Windows\System32\reg.exe", "cpu-nogui", knob="0"
        )
        assert p.returncode == 3, p.stderr
        assert "override" in (p.stdout + p.stderr).lower()
        for log in logs.values():
            if log.is_file():
                body = log.read_text(encoding="utf-8", errors="replace").splitlines()
                assert not any(l.startswith("BUILD ") for l in body), "no automatic fallback for an explicit override"

    def test_none_sentinel_fails_as_no_host(self, tmp_path):
        # (H) the documented `none` sentinel: recognized by the
        # data-space validator (rc 7), mapped to the no-host-Python
        # failure (rc 3); no builder, no fallback, nothing of the
        # value executed (the value never crosses this bat's parsed
        # command text - the old parse-space sentinel line is gone).
        p, fx, logs = self._override_with_validator(tmp_path, "none", "cpu-nogui")
        assert p.returncode == 3, p.stderr
        out = (p.stdout + p.stderr).lower()
        assert "no usable host" in out
        for log in logs.values():
            if log.is_file():
                body = log.read_text(encoding="utf-8", errors="replace").splitlines()
                assert not any(l.startswith("BUILD ") for l in body), "builder must not run for `none`"
        assert not (fx["root"] / "hostcmd-executed").exists()

    def test_none_sentinel_is_case_sensitive(self, tmp_path):
        # `NONE` is NOT the sentinel: the validator rejects it as an
        # invalid interpreter path (rc 6 -> rc 3 override-invalid);
        # it is never routed to the no-host sentinel path.
        p, fx, logs = self._override_with_validator(
            tmp_path, "NONE", "cpu-nogui", knob="6"
        )
        assert p.returncode == 3, p.stderr
        out = (p.stdout + p.stderr).lower()
        assert "not an accepted host interpreter path" in out

    def test_explicit_override_below_floor_is_terminal(self, tmp_path):
        # (G) an override that validates and probes fine but reports a
        # version below the 3.11 floor must fail terminally (rc 4). This
        # machine has no CPython below the floor (only 3.12/3.13 are
        # available), so a real below-floor .exe host cannot be built; the
        # terminal floor policy is the same exact-exit-code branch as the
        # tested probe-failure terminal path above.
        pytest.skip(
            "no CPython below the 3.11 floor exists on this machine; "
            "the override floor branch shares the exact rc policy with the "
            "tested probe-failure terminal path"
        )


# ---------------------------------------------------------------------------
# runtime_entry CUDA sanitation (Commit-2 bootstrap owner, Phase-13 policy)
# ---------------------------------------------------------------------------


def _load_runtime_entry_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("dfl_runtime_entry", str(REAL_RUNTIME_ENTRY))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRuntimeEntryCudaSanitation:
    """The EXACT round-2 policy:
    (1) capture names/status (never values);
    (2) clear CUDA_PATH / CUDA_HOME / CUDA_VISIBLE_DEVICES (case-insensitive
        names);
    (3) from the CHILD PATH remove ONLY
        (a) <root>\\bin, (b) <root>\\lib\\x64 of each pre-removal
        CUDA_PATH / CUDA_HOME root, and (c) canonical
        ...\\NVIDIA GPU Computing Toolkit\\CUDA\\v*\\bin entries
        (case-insensitive, even with both variables absent);
    (4) preserve ALL other PATH entries - the root itself, docs, extras,
        non-CUDA toolkits - and never perform broad substring deletion.
    """

    def test_derived_bin_of_declared_root_pruned(self):
        # (a) <root>\bin
        mod = _load_runtime_entry_module()
        env = {"CUDA_PATH": "stale-root", "PATH": "stale-root\\bin;keep"}
        report = mod.sanitize_cuda_environment(env)
        assert report["pruned_path_entries"] == ["stale-root\\bin"]
        assert env["PATH"] == "keep"
        assert sorted(k.casefold() for k in report["cleared"]) == ["cuda_path"]

    def test_derived_lib_x64_of_declared_root_pruned(self):
        # (b) <root>\lib\x64
        mod = _load_runtime_entry_module()
        env = {"CUDA_PATH": "stale-root", "PATH": "stale-root\\lib\\x64;keep"}
        report = mod.sanitize_cuda_environment(env)
        assert report["pruned_path_entries"] == ["stale-root\\lib\\x64"]
        assert env["PATH"] == "keep"

    def test_cuda_home_roots_pruned_same_policy(self):
        # CUDA_HOME declares roots exactly like CUDA_PATH does.
        mod = _load_runtime_entry_module()
        env = {
            "CUDA_HOME": "stale-home",
            "PATH": "stale-home\\bin;stale-home\\lib\\x64;keep",
        }
        report = mod.sanitize_cuda_environment(env)
        assert set(report["pruned_path_entries"]) == {
            "stale-home\\bin",
            "stale-home\\lib\\x64",
        }
        assert env["PATH"] == "keep"

    def test_other_descendants_of_root_preserved(self):
        # docs / extras / any other descendant of a declared root stay.
        mod = _load_runtime_entry_module()
        env = {
            "CUDA_PATH": "stale-root",
            "PATH": "stale-root\\bin;stale-root\\docs;stale-root\\extras;keep",
        }
        report = mod.sanitize_cuda_environment(env)
        assert report["pruned_path_entries"] == ["stale-root\\bin"]
        assert env["PATH"] == "stale-root\\docs;stale-root\\extras;keep"

    def test_root_itself_preserved(self):
        # A bare root entry is NOT under <root>\bin or <root>\lib\x64 and
        # is preserved (the old broader policy pruned it - fixed).
        mod = _load_runtime_entry_module()
        env = {"CUDA_PATH": "stale-root", "PATH": "stale-root;stale-root\\bin;keep"}
        report = mod.sanitize_cuda_environment(env)
        assert report["pruned_path_entries"] == ["stale-root\\bin"]
        assert env["PATH"] == "stale-root;keep"

    def test_canonical_nvidia_toolkit_bin_pruned_without_cuda_vars(self):
        # (c) canonical NVIDIA toolkit bin entries are pruned even when
        # neither CUDA_PATH nor CUDA_HOME is set.
        mod = _load_runtime_entry_module()
        env = {
            "PATH": (
                "nv\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin;"
                "nv\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin;keep"
            ),
        }
        report = mod.sanitize_cuda_environment(env)
        assert set(report["pruned_path_entries"]) == {
            "nv\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin",
            "nv\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin",
        }
        assert env["PATH"] == "keep"
        assert report["cleared"] == []
        assert report["declared_roots"] == []

    def test_canonical_nvidia_toolkit_bin_forward_slash(self):
        mod = _load_runtime_entry_module()
        env = {"PATH": "c:/nv/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin;keep"}
        report = mod.sanitize_cuda_environment(env)
        assert report["pruned_path_entries"] == ["c:/nv/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin"]
        assert env["PATH"] == "keep"

    def test_non_canonical_cuda_named_entry_preserved(self):
        # An entry whose name merely contains 'cuda' (but is neither a
        # declared root's bin/lib\\x64 nor the canonical NVIDIA pattern)
        # is preserved - no CUDA-like entry is claimed safe or unsafe
        # by name.
        mod = _load_runtime_entry_module()
        env = {"CUDA_PATH": "stale-root", "PATH": "my-cuda-utils\\bin;keep"}
        report = mod.sanitize_cuda_environment(env)
        assert report["pruned_path_entries"] == []
        assert env["PATH"] == "my-cuda-utils\\bin;keep"

    def test_unrelated_toolkit_entries_preserved(self):
        mod = _load_runtime_entry_module()
        env = {"CUDA_PATH": "stale-root", "PATH": "ffmpeg\\bin;keep-dir"}
        report = mod.sanitize_cuda_environment(env)
        assert report["pruned_path_entries"] == []
        assert env["PATH"] == "ffmpeg\\bin;keep-dir"

    def test_duplicate_declared_entries_all_pruned_deterministically(self):
        mod = _load_runtime_entry_module()
        env = {"CUDA_PATH": "stale-root", "PATH": "stale-root\\bin;keep;stale-root\\bin"}
        report = mod.sanitize_cuda_environment(env)
        assert report["pruned_path_entries"] == ["stale-root\\bin", "stale-root\\bin"]
        assert env["PATH"] == "keep"

    def test_mixed_case_root_and_path_entries_pruned(self):
        # Windows path semantics: matching is case-insensitive on both the
        # declared root value and the PATH entry text.
        mod = _load_runtime_entry_module()
        env = {
            "CUDA_PATH": "Stale-Root",
            "PATH": "STALE-ROOT\\LIB\\X64;Stale-Root\\BIN;keep",
        }
        report = mod.sanitize_cuda_environment(env)
        assert set(report["pruned_path_entries"]) == {
            "STALE-ROOT\\LIB\\X64",
            "Stale-Root\\BIN",
        }
        assert env["PATH"] == "keep"

    def test_mixed_case_variable_names_are_cleared(self):
        # Windows environment names are case-insensitive.
        mod = _load_runtime_entry_module()
        env = {"cuda_path": "stale-root", "Cuda_Visible_Devices": "1"}
        report = mod.sanitize_cuda_environment(env)
        assert {k.casefold() for k in report["cleared"]} == {
            "cuda_path",
            "cuda_visible_devices",
        }
        assert env == {}

    def test_no_broad_substring_deletion(self):
        # A path that merely CONTAINS the stale root as a substring is not
        # under the root and must be preserved (structured prefix matching).
        mod = _load_runtime_entry_module()
        env = {"CUDA_PATH": "toolkit", "PATH": "my-toolkit-dir;toolkit\\bin;keep"}
        report = mod.sanitize_cuda_environment(env)
        assert report["pruned_path_entries"] == ["toolkit\\bin"]
        assert env["PATH"] == "my-toolkit-dir;keep"

    def test_path_untouched_without_stale_roots(self):
        mod = _load_runtime_entry_module()
        env = {"CUDA_VISIBLE_DEVICES": "0", "PATH": "a;b"}
        report = mod.sanitize_cuda_environment(env)
        assert report["cleared"] == ["CUDA_VISIBLE_DEVICES"]
        assert report["pruned_path_entries"] == []
        assert report["declared_roots"] == []
        assert env["PATH"] == "a;b"

    def test_report_captures_only_names_and_derived_locations(self):
        # The report exposes variable NAMES (as spelled) and the derived
        # locations, never the sensitive root values themselves.
        mod = _load_runtime_entry_module()
        env = {"CUDA_PATH": "stale-root", "CUDA_HOME": "stale-home"}
        report = mod.sanitize_cuda_environment(env)
        assert sorted(k.casefold() for k in report["cleared"]) == ["cuda_home", "cuda_path"]
        joined = str(report)
        assert "stale-root" not in joined.replace(str(report["declared_roots"]), "")
        assert "stale-home" not in joined.replace(str(report["declared_roots"]), "")

    def test_os_environ_case_insensitivity(self, tmp_path):
        # Through the real os.environ (case-insensitive on Windows): a
        # lower-case spelling of CUDA_PATH must still be cleared.
        mod = _load_runtime_entry_module()
        probe = (
            "import sys; sys.path.insert(0, 'scripts')\n"
            "from runtime_entry import sanitize_cuda_environment\n"
            "import os\n"
            "os.environ['cuda_path'] = 'stale-x'\n"
            "os.environ['CUDA_VISIBLE_DEVICES'] = '0'\n"
            "r = sanitize_cuda_environment(os.environ)\n"
            "assert sorted(k.casefold() for k in r['cleared']) == ['cuda_path', 'cuda_visible_devices'], r\n"
            "assert 'cuda_path' not in {k.casefold() for k in os.environ}\n"
            "assert 'CUDA_PATH' not in os.environ\n"
            "print('CI-OK')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        assert "CI-OK" in proc.stdout


# ---------------------------------------------------------------------------
# current-runtime integration: real launcher + real interpreter (the
# builder CLI selects a runtime whose recorded bootstrap SHA matches the
# CURRENT scripts/runtime_entry.py, i.e. `verify-runtime` passes)
# ---------------------------------------------------------------------------


def _builder_python() -> str:
    if VENV_PYTHON.is_file():
        return str(VENV_PYTHON)
    return sys.executable


def _builder_cli(*args, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_builder_python(), str(REPO_ROOT / "scripts" / "build_runtime.py"), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def current_runtimes():
    """Map variant -> runtime id for every runtime under runtime/versions
    whose builder `verify-runtime` check passes (recorded bootstrap SHA
    equals the CURRENT scripts/runtime_entry.py SHA). Runtimes built under
    an older bootstrap SHA fail this check and are STALE: they are never
    used as current evidence here."""
    out = {kind: None for kind in ("cpu-nogui", "cuda-nogui")}
    versions = REPO_ROOT / "runtime" / "versions"
    if not versions.is_dir():
        return out
    candidates = [
        d
        for d in sorted(versions.iterdir())
        if d.is_dir() and RUNTIME_ID_RE.match(d.name) and (d / "python.exe").is_file()
    ]
    for kind in out:
        for d in candidates:
            try:
                p = _builder_cli("verify-runtime", "--runtime-id", d.name, "--variant", kind)
            except subprocess.TimeoutExpired:
                continue
            if p.returncode == 0:
                out[kind] = d.name
                break
    return out


@pytest.mark.parametrize("kind", ["cpu-nogui", "cuda-nogui"])
def test_real_launcher_against_current_runtime(current_runtimes, kind):
    """CURRENT_MACHINE_CONTRACT_VALIDATION (no training; --self-test only).

    Selects a CURRENT runtime for the variant (builder CLI verify-runtime
    against the current bootstrap SHA), activates it (the Commit-2
    selector-swap primitive, byte-exactly restored afterwards), and runs
    the tracked launchers\\dfl.bat --self-test end to end: the real bundled
    interpreter executes the real scripts/runtime_entry.py under a hostile
    environment (stale CUDA variables + toolkit/canonical PATH entries,
    stale NN device state). The launcher layer passes CUDA through to the
    runtime_entry owner, which sanitizes it before any application import;
    the self-test JSON reports the sanitation. No file in the runtime
    versions tree is modified.
    """
    runtime_id = current_runtimes.get(kind)
    if runtime_id is None:
        pytest.skip(
            f"no CURRENT {kind} runtime on this machine (runtimes present are stale "
            "under the current bootstrap SHA); rebuild via "
            "launchers\\dfl-setup-runtime.bat <variant> --activate first"
        )

    selector = REPO_ROOT / "runtime" / "active-runtime.txt"
    selector_before = selector.read_bytes() if selector.is_file() else None

    try:
        # Fast selector swap under the setup/activation lock (no build).
        p = _builder_cli("activate", "--runtime-id", runtime_id, "--variant", kind)
        assert p.returncode == 0, f"activate failed: {p.stdout} {p.stderr}"

        env = _env(HOSTILE_PARENT_ENV)
        # stale toolkit PATH entries (declared via the CUDA_PATH/CUDA_HOME
        # values above) plus a canonical NVIDIA toolkit bin entry that
        # runtime_entry must prune even though no variable names it, and
        # an unrelated entry that must survive.
        env["PATH"] = (
            env.get("PATH", "")
            + r";stale-cuda-toolkit\bin;stale-cuda-home\bin"
            r";nv\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;keepme-toolkit-dir"
        )
        versions_before = tree_snapshot(REPO_ROOT / "runtime" / "versions")
        try:
            proc = run_bat(
                REPO_ROOT / "launchers" / "dfl.bat",
                "--self-test",
                cwd=REPO_ROOT,
                env=env,
                timeout=300,
            )
        finally:
            versions_after = tree_snapshot(REPO_ROOT / "runtime" / "versions")
    finally:
        # byte-exact selector restore: the test must leave the repository
        # in the state it found it
        if selector_before is None:
            if selector.is_file():
                selector.unlink()
        else:
            selector.write_bytes(selector_before)

    assert proc.returncode == 0, f"rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["ok"] is True
    assert report["isolated"] is True
    # CUDA sanitation (runtime_entry authority): every injected CUDA
    # variable is cleared from the child environment ...
    cleared = {name.casefold() for name in report.get("cuda_cleared_variables", [])}
    assert {"cuda_path", "cuda_home", "cuda_visible_devices"} <= cleared, report
    # ... the declared-root toolkit PATH components are pruned ...
    pruned = report.get("cuda_pruned_path_entries", [])
    pruned_norm = {x.casefold().replace("/", "\\") for x in pruned}
    assert "stale-cuda-toolkit\\bin" in pruned_norm, pruned
    assert "stale-cuda-home\\bin" in pruned_norm, pruned
    # ... and the canonical NVIDIA toolkit bin entry is pruned even
    # without a naming variable ...
    assert "nv\\nvidia gpu computing toolkit\\cuda\\v13.0\\bin" in pruned_norm, pruned
    # ... while the unrelated PATH entry is preserved (not pruned).
    assert all("keepme-toolkit-dir" not in x for x in pruned), pruned
    # launcher-layer NN sanitation reached the child through the chain.
    assert report.get("host_env_purged") is True
    assert versions_after == versions_before, "runtime versions tree was modified by the launcher chain"
