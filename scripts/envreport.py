"""Phase-13 ``envreport`` command: resilient, privacy-safe environment report.

Contract: private ``docs/PHASE13_STATE.md`` sections 20-23 (P13-ENVREPORT-VERIFY).

What this module provides
-------------------------
* A base environment report (source provenance, OS, running interpreter,
  runtime/selector state, installed key packages, and seven isolated
  capability probes) that ALWAYS renders even when individual probes fail -
  probe failure is the report's purpose, not its poison.
* A ``--verify`` section that re-uses the Commit-1 builder authority
  (``scripts/build_runtime.py``: ``verify_runtime`` + ``load_inputs``) plus
  three envreport-native checks (selector agreement, running-interpreter
  identity, Torch contract) and the base probe statuses as NONFATAL
  informational entries.
* One internal data model feeding both the text and the JSON serializers, so
  the two outputs can never drift apart.
* A fail-closed privacy gate: every string field is sanitized at collection
  time and the fully rendered output (text OR JSON, whichever is emitted) is
  re-scanned with the builder's reference privacy detector before anything
  reaches stdout. On any violation the report is suppressed and the process
  exits non-zero with a clean one-line message on stderr.

Stdlib-first dispatch
---------------------
The top level of this module imports stdlib only. ``scripts.build_runtime``
(the stdlib-only builder module) is imported lazily, inside the functions
that need its authority or its privacy detector, so the failure-safe path in
``main.py`` never depends on a heavy import having succeeded.

Runtime isolation
-----------------
Every risky probe (Torch import, CUDA initialization, ``nvidia-smi``,
``ffmpeg``/``ffprobe``, ONNX, ONNX Runtime) executes in a child process of
the SAME interpreter that produced the report::

    <python> -I -B scripts/envreport_probe.py <probe-name>

The parent enforces a bounded timeout (``TIMEOUT``), bounded captured output,
``shell=False``, and a stable status vocabulary. One probe failure never
blocks any other field of the report.
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import importlib.metadata as _imetadata
import json as _json
import platform as _platform
import re as _re
import shutil as _shutil
import subprocess as _subprocess
import sys as _sys
import threading as _threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Stable vocabulary
# ---------------------------------------------------------------------------

SCHEMA = "dfl-envreport-v1"

STATUS_OK = "OK"
STATUS_MISSING = "MISSING"
STATUS_IMPORT_ERROR = "IMPORT_ERROR"
STATUS_INIT_ERROR = "INIT_ERROR"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_MISMATCH = "MISMATCH"

STATUS_VOCABULARY = frozenset(
    {
        STATUS_OK,
        STATUS_MISSING,
        STATUS_IMPORT_ERROR,
        STATUS_INIT_ERROR,
        STATUS_TIMEOUT,
        STATUS_UNAVAILABLE,
        STATUS_MISMATCH,
    }
)

MODE_PACKAGED = "packaged"
MODE_DEVELOPMENT = "development"

# Probes, in report order. Each runs in an isolated child process.
PROBES = (
    "torch",
    "cuda",
    "nvidia_smi",
    "ffmpeg",
    "ffprobe",
    "onnx",
    "onnxruntime",
)

# Bounded per-probe timeouts (seconds), enforced by the PARENT process.
PROBE_TIMEOUTS = {
    "torch": 60.0,
    "cuda": 60.0,
    "nvidia_smi": 20.0,
    "ffmpeg": 20.0,
    "ffprobe": 20.0,
    "onnx": 30.0,
    "onnxruntime": 30.0,
}

# Bounded captured output from the probe child (bytes).
_PROBE_OUT_LIMIT = 65536

# Whitelisted field keys per probe. Anything the child worker reports outside
# this whitelist is dropped, so a buggy or hostile package cannot widen the
# report's surface. ``gpus`` is a list of dicts with its own whitelist.
PROBE_FIELD_WHITELIST = {
    "torch": ("version", "cuda", "local_tag"),
    "cuda": ("available", "device_count", "device_name", "vram_gb", "compute_capability"),
    "nvidia_smi": ("driver_version", "gpu_count", "gpus"),
    "ffmpeg": ("version",),
    "ffprobe": ("version",),
    "onnx": ("version",),
    "onnxruntime": ("version",),
}
_GPU_FIELD_WHITELIST = ("name", "vram_mb", "compute_capability")

# Key packages of the environment, always reported (distribution names, as
# stored in the runtime locks / importable metadata).
KEY_PACKAGES = (
    "colorama",
    "ffmpeg-python",
    "filelock",
    "flatbuffers",
    "fsspec",
    "future",
    "jinja2",
    "markupsafe",
    "ml-dtypes",
    "mpmath",
    "networkx",
    "numexpr",
    "numpy",
    "onnx",
    "onnxruntime",
    "opencv-python",
    "packaging",
    "pip",
    "protobuf",
    "scipy",
    "setuptools",
    "sympy",
    "tqdm",
    "typing-extensions",
    "torch",
    "PyQt5",
    "pyqt5-qt5",
    "pyqt5-sip",
)

# Engine check codes (scripts/build_runtime.verify_runtime) mapped onto the
# stable report vocabulary. Anything unexpected fails closed to MISMATCH.
_ENGINE_CODE_MAP = {
    "PASS": STATUS_OK,
    "MISMATCH": STATUS_MISMATCH,
    "SCHEMA_MISMATCH": STATUS_MISMATCH,
    "SOURCE_MISMATCH": STATUS_MISMATCH,
    "HASH_MISMATCH": STATUS_MISMATCH,
    "EXTRA_ENTRY": STATUS_MISMATCH,
    "PRESENT": STATUS_MISMATCH,  # prohibited artifact present = contract violation
    "MISSING_MANIFEST": STATUS_MISSING,
    "MISSING_INTERPRETER": STATUS_MISSING,
    "MISSING_ENTRY": STATUS_MISSING,
}

_USAGE = "usage: dfl.bat envreport [--json] [--verify]"

# ---------------------------------------------------------------------------
# Data model (single source of truth for text AND JSON serialization)
# ---------------------------------------------------------------------------


@dataclass
class SourceInfo:
    status: str
    commit: Optional[str]
    branch: Optional[str]
    note: str = ""


@dataclass
class OsInfo:
    status: str
    platform: Optional[str]
    version: Optional[str]
    build: Optional[str]
    note: str = ""


@dataclass
class PythonInfo:
    status: str
    implementation: str
    version: str
    architecture: str
    note: str = ""


@dataclass
class SelectorInfo:
    present: bool
    valid: bool
    runtime_id: Optional[str]
    note: str = ""


@dataclass
class RuntimeInfo:
    mode: str
    runtime_id: Optional[str]
    variant: Optional[str]
    note: str = ""
    selector: SelectorInfo = field(default_factory=SelectorInfo)


@dataclass
class PackageInfo:
    name: str
    status: str
    version: Optional[str]
    note: str = ""


@dataclass
class ProbeInfo:
    name: str
    status: str
    note: str = ""
    fields: dict = field(default_factory=dict)


@dataclass
class CheckInfo:
    id: str
    status: str
    note: str = ""
    required: bool = True


@dataclass
class VerifyInfo:
    available: bool
    target_runtime_id: Optional[str]
    variant: Optional[str]
    checks: list = field(default_factory=list)
    passed: int = 0
    total: int = 0
    required_failed: int = 0
    summary: str = ""


@dataclass
class EnvReport:
    schema: str
    command: str
    json_output: bool
    verify: bool
    generated_utc: str
    source: SourceInfo
    os: OsInfo
    python: PythonInfo
    runtime: RuntimeInfo
    packages: list = field(default_factory=list)
    probes: list = field(default_factory=list)
    verify_info: Optional[VerifyInfo] = None


# ---------------------------------------------------------------------------
# Lazy builder-module access (stdlib-only import closure)
# ---------------------------------------------------------------------------

_BR_CACHE: dict = {"module": None}


def _builder(repo_root: Path):
    """Import scripts.build_runtime once; ensure the repo root is importable."""
    module = _BR_CACHE["module"]
    if module is None:
        root = str(repo_root)
        if root not in _sys.path:
            _sys.path.insert(0, root)
        from scripts import build_runtime as _br  # noqa: PLC0415 (stdlib-only)

        _BR_CACHE["module"] = _br
        module = _br
    return module


# ---------------------------------------------------------------------------
# Privacy gate
# ---------------------------------------------------------------------------

_REDACTED = "<redacted>"

# Supplemental patterns applied on top of the builder's PRIVACY_RULES table.
# The shared table catches literal path segments ("C:\Users\...", "appdata\"),
# but cannot match environment-variable reference forms such as
# "%APPDATA%\Roaming\..." because a closing percent sign sits between the
# variable name and the separator.  This rule catches every ``%VAR%``-style
# reference followed by a path separator.
_EXTRA_PRIVACY_PATTERNS = (
    _re.compile(r"%[A-Za-z_][A-Za-z0-9_]*%[\\/]"),
)


# Fixed marker appended when the shared privacy scanner cannot run at all.
# It is deliberately a constant (no exception type or message) and is only
# ever consumed by the gates below, never rendered into a report.
_PRIVACY_SCANNER_FAILED = "privacy scanner failure"


def _privacy_violations(text: str, repo_root: Path) -> list:
    """Run the builder's reference privacy detector over *text*.

    On top of the builder's rules the envreport-local supplemental patterns
    above are applied.

    Fail closed: if the scanner cannot run (builder import failure or a
    raised detector), a fixed marker is reported instead of swallowing the
    error, so an unusable scanner is treated like a violation rather than a
    pass — every string field is redacted and the rendered payload is
    suppressed by the final gate.  The exception type and message are never
    surfaced.  An empty result means the value is treated as clean.
    """
    if not text:
        return []
    violations: list = []
    try:
        br = _builder(repo_root)
        violations.extend(br.privacy_violations(text))
    except Exception:  # noqa: BLE001 - fail closed: marker, never the exception
        violations.append(_PRIVACY_SCANNER_FAILED)
    for pattern in _EXTRA_PRIVACY_PATTERNS:
        if pattern.search(text):
            violations.append(f"supplemental: {pattern.pattern}")
    return violations


def _clean(value, repo_root: Path):
    """Gate one string value through the privacy detector (fail closed)."""
    if not isinstance(value, str):
        return value
    if _privacy_violations(value, repo_root):
        return _REDACTED
    return value


def _clean_fields(fields: dict, repo_root: Path) -> dict:
    """Gate a whitelisted probe-field dict: strings sanitized, shapes kept."""
    cleaned = {}
    for key, value in fields.items():
        if value is None or isinstance(value, (bool, int, float)):
            cleaned[key] = value
        elif isinstance(value, str):
            cleaned[key] = _clean(value, repo_root)
        elif isinstance(value, list):
            cleaned[key] = [
                _clean(item, repo_root)
                if isinstance(item, str)
                else {_key: _clean(value, repo_root) for _key, value in item.items()}
                if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------


def _repo_root_default() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_flags(argv: list[str]) -> tuple[bool, bool]:
    """Parse the envreport flags; raise ValueError on any unknown token.

    The exception message is for internal diagnostics only and must NEVER
    be written to a user-visible stream: unknown tokens may carry paths,
    credentials, or arbitrary user input (see the fixed wording in run()).
    """
    seen = set()
    for token in argv:
        if token in ("--json", "--verify"):
            if token in seen:
                raise ValueError(f"duplicate flag: {token}")
            seen.add(token)
        else:
            raise ValueError(f"unknown argument: {token!r}")
    return ("--json" in seen, "--verify" in seen)


def _git_source(repo_root: Path) -> SourceInfo:
    """Provenance from the working tree's .git/HEAD (never a path is reported)."""
    git_dir = repo_root / ".git"
    head = git_dir / "HEAD"
    if not git_dir.exists():
        return SourceInfo(STATUS_UNAVAILABLE, None, None, "not a git working tree")
    if not head.is_file():
        return SourceInfo(STATUS_UNAVAILABLE, None, None, "git HEAD not readable")
    try:
        text = head.read_text(encoding="utf-8").strip()
    except OSError:
        return SourceInfo(STATUS_UNAVAILABLE, None, None, "git HEAD not readable")
    if text.startswith("gitdir:"):
        # worktree: follow the pointer once and read its HEAD.
        target = text[len("gitdir:") :].strip()
        head2 = (git_dir / target).resolve() / "HEAD"
        if not head2.is_file():
            return SourceInfo(STATUS_UNAVAILABLE, None, None, "git worktree HEAD not readable")
        try:
            text = head2.read_text(encoding="utf-8").strip()
        except OSError:
            return SourceInfo(STATUS_UNAVAILABLE, None, None, "git worktree HEAD not readable")
    if text.startswith("ref:"):
        ref = text[len("ref:") :].strip()
        branch = ref[len("refs/heads/") :] if ref.startswith("refs/heads/") else ref
        commit = None
        loose = git_dir / "refs" / ref[len("refs/") :]
        try:
            if loose.is_file():
                commit = loose.read_text(encoding="utf-8").strip()
        except OSError:
            commit = None
        note = "" if commit else "branch head packed (commit not read)"
        return SourceInfo(STATUS_OK, commit, branch, note)
    if _re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", text):
        return SourceInfo(STATUS_OK, text, "detached", "")
    return SourceInfo(STATUS_UNAVAILABLE, None, None, "unrecognized git HEAD format")


def _os_info() -> OsInfo:
    if _platform.system() != "Windows":
        return OsInfo(
            STATUS_UNAVAILABLE,
            _platform.system() or None,
            None,
            None,
            "contract is defined for Windows hosts",
        )
    # Python < 3.13 reports (major, minor, build, platform, service_pack);
    # 3.13 changed the layout (minor may carry a composite release string).
    # Extract a stable (major, build) pair from the numeric tokens instead.
    parts = _platform.win32_ver()
    numeric = [t for t in _re.split(r"\D+", " ".join(str(p) for p in parts[:3])) if t]
    build = max(numeric, key=int) if numeric else None
    major = "11" if numeric and numeric[0] == "11" else ("10" if numeric else None)
    version = f"{major}.{build}" if (major and build) else (build or "unknown")
    return OsInfo(STATUS_OK, "Windows", version, build, "")


def _python_info() -> PythonInfo:
    return PythonInfo(
        STATUS_OK,
        _platform.python_implementation(),
        _platform.python_version(),
        _platform.machine(),
        "",
    )


def _selector_info(repo_root: Path, br) -> SelectorInfo:
    """Parse the active-runtime selector exactly as the launcher does:
    one logical line of 64 lowercase hex digits after one trailing CRLF/LF."""
    selector_path = repo_root / "runtime" / br.ACTIVE_POINTER_NAME
    if not selector_path.is_file():
        return SelectorInfo(False, False, None, "no active selector file")
    try:
        content = selector_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return SelectorInfo(True, False, None, "selector not readable")
    body = content
    if body.endswith("\r\n"):
        body = body[:-2]
    elif body.endswith("\n") or body.endswith("\r"):
        body = body[:-1]
    if not body or not br.RUNTIME_ID_RE.fullmatch(body):
        return SelectorInfo(True, False, None, "selector malformed")
    return SelectorInfo(True, True, body, "")


def _manifest_variant(runtime_dir: Optional[Path]) -> Optional[str]:
    """Read canonical.variant from a runtime manifest (or None)."""
    if runtime_dir is None:
        return None
    manifest = runtime_dir / "runtime-manifest.json"
    if not manifest.is_file():
        return None
    try:
        data = _json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    variant = (data.get("canonical") or {}).get("variant")
    return variant if isinstance(variant, str) and variant else None


def _runtime_info(repo_root: Path, br) -> RuntimeInfo:
    # The selector is reported in BOTH modes (packaged and development),
    # so it must be computed before the mode branch, not inside one arm.
    selector = _selector_info(repo_root, br)
    exe_dir = Path(_sys.executable).resolve().parent
    runtime_id = None
    mode = MODE_DEVELOPMENT
    if (
        br.RUNTIME_ID_RE.fullmatch(exe_dir.name)
        and exe_dir.parent.name == br.VERSIONS_DIRNAME
        and exe_dir.parent.parent.name == "runtime"
    ):
        mode = MODE_PACKAGED
        runtime_id = exe_dir.name
        runtime_dir = exe_dir
    else:
        if selector.valid and selector.runtime_id:
            runtime_dir = repo_root / "runtime" / br.VERSIONS_DIRNAME / selector.runtime_id
        else:
            runtime_dir = None
    variant = _manifest_variant(runtime_dir)
    if mode == MODE_PACKAGED:
        note = "selected runtime (packaged interpreter)"
    else:
        note = "developer interpreter (host environment)"
    if variant is None and mode == MODE_PACKAGED:
        note += "; runtime variant undecidable (manifest missing or invalid)"
    return RuntimeInfo(mode, runtime_id, variant, note, selector)


def _package_names(repo_root: Path, runtime_info: RuntimeInfo, br) -> tuple:
    names = list(KEY_PACKAGES)
    variant = runtime_info.variant
    if variant in br.VARIANTS:
        try:
            from scripts import build_runtime as _br  # noqa: PLC0415

            manifest = None
            if runtime_info.runtime_id:
                candidate = repo_root / "runtime" / br.VERSIONS_DIRNAME / runtime_info.runtime_id
                if candidate.is_dir():
                    manifest = candidate / br.MANIFEST_NAME
            elif runtime_info.selector.valid and runtime_info.selector.runtime_id:
                candidate = repo_root / "runtime" / br.VERSIONS_DIRNAME / runtime_info.selector.runtime_id
                if candidate.is_dir():
                    manifest = candidate / br.MANIFEST_NAME
            if manifest is not None and manifest.is_file():
                data = _json.loads(manifest.read_text(encoding="utf-8"))
                package_set = (data.get("canonical") or {}).get("package_set") or {}
                for name in package_set:
                    if isinstance(name, str) and name not in names:
                        names.append(name)
        except Exception:  # noqa: BLE001 - package list degradation is nonfatal
            pass
    return tuple(names)


def _probe_packages(names, repo_root: Path) -> list[PackageInfo]:
    results = []
    for name in names:
        try:
            version = _imetadata.version(name)
            results.append(PackageInfo(name, STATUS_OK, str(version), ""))
        except _imetadata.PackageNotFoundError:
            results.append(PackageInfo(name, STATUS_MISSING, None, "not installed in this environment"))
        except Exception as exc:  # noqa: BLE001
            results.append(PackageInfo(name, STATUS_UNAVAILABLE, None, f"metadata error: {type(exc).__name__}"))
    return results


# ---------------------------------------------------------------------------
# Probe execution (isolated child processes)
# ---------------------------------------------------------------------------


def _validate_status(status) -> str:
    if isinstance(status, str) and status in STATUS_VOCABULARY:
        return status
    return STATUS_UNAVAILABLE


def _whitelist_fields(name: str, raw_fields) -> dict:
    """Keep only whitelisted probe fields with sane types; gate the strings."""
    if not isinstance(raw_fields, dict):
        return {}
    allowed = PROBE_FIELD_WHITELIST.get(name, ())
    fields = {}
    for key in allowed:
        if key not in raw_fields:
            continue
        value = raw_fields[key]
        if name == "nvidia_smi" and key == "gpus":
            if not isinstance(value, list):
                continue
            gpus = []
            for item in value[:16]:
                if not isinstance(item, dict):
                    continue
                gpu = {}
                for gkey in _GPU_FIELD_WHITELIST:
                    if gkey in item:
                        gpu[gkey] = item[gkey]
                gpus.append(gpu)
            fields[key] = gpus
            continue
        if value is None or isinstance(value, (bool, int, float, str)):
            fields[key] = value
    return fields


def _parse_worker_output(stdout_bytes: bytes) -> Optional[dict]:
    """Parse the worker's single JSON protocol line (bounded, last-line first)."""
    if not stdout_bytes:
        return None
    text = stdout_bytes[:_PROBE_OUT_LIMIT].decode("utf-8", "replace")
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            data = _json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _drain_capped(proc, buf: bytearray) -> None:
    """Drain the child's stdout into *buf*, stopping at the capture limit.

    The reader runs in a daemon thread so the parent can always enforce the
    timeout against the process itself.  Once the limit is reached the
    thread stops reading; a verbose or hostile child then blocks on a full
    pipe and is killed by the timeout instead of pinning unbounded parent
    memory.
    """
    try:
        while len(buf) < _PROBE_OUT_LIMIT:
            chunk = proc.stdout.read(min(8192, _PROBE_OUT_LIMIT - len(buf)))
            if not chunk:
                break
            buf.extend(chunk)
    except Exception:  # noqa: BLE001 - pipe errors resolve via the timeout path
        return


def _make_default_probe_runner(repo_root: Path, command: Optional[list[str]] = None) -> Callable:
    """Build the production probe runner: isolated subprocess per probe."""
    probe_script = repo_root / "scripts" / "envreport_probe.py"

    def runner(name: str, timeout: Optional[float] = None) -> ProbeInfo:
        bounded = float(timeout if timeout is not None else PROBE_TIMEOUTS.get(name, 60.0))
        if command is not None:
            cmd = list(command)
        else:
            cmd = [_sys.executable, "-I", "-B", str(probe_script), name]
        try:
            proc = _subprocess.Popen(
                cmd,
                stdout=_subprocess.PIPE,
                stderr=_subprocess.DEVNULL,
                shell=False,
                cwd=str(repo_root),
            )
        except OSError as exc:
            return ProbeInfo(name, STATUS_INIT_ERROR, f"probe could not start: {type(exc).__name__}", {})
        buf = bytearray()
        reader = _threading.Thread(target=_drain_capped, args=(proc, buf), daemon=True)
        reader.start()
        try:
            try:
                returncode = proc.wait(timeout=bounded)
            except _subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except _subprocess.TimeoutExpired:
                    pass
                reader.join(5)
                return ProbeInfo(name, STATUS_TIMEOUT, "probe timed out", {})
        finally:
            reader.join(5)
            try:
                proc.stdout.close()
            except Exception:  # noqa: BLE001
                pass
        data = _parse_worker_output(bytes(buf))
        if data is None:
            if returncode == 0:
                return ProbeInfo(name, STATUS_UNAVAILABLE, "probe worker returned no protocol line", {})
            return ProbeInfo(name, STATUS_INIT_ERROR, f"probe worker failed (exit {returncode})", {})
        status = _validate_status(data.get("status"))
        note = data.get("note", "")
        if not isinstance(note, str):
            note = ""
        fields = _whitelist_fields(name, data.get("fields"))
        return ProbeInfo(name, status, note, fields)

    return runner


def _collect_probes(
    repo_root: Path,
    probe_runner: Optional[Callable] = None,
) -> list[ProbeInfo]:
    runner = probe_runner if probe_runner is not None else _make_default_probe_runner(repo_root)
    probes = []
    for name in PROBES:
        try:
            info = runner(name)
        except Exception as exc:  # noqa: BLE001 - one probe can never block the report
            info = ProbeInfo(name, STATUS_INIT_ERROR, f"probe runner error: {type(exc).__name__}", {})
        if not isinstance(info, ProbeInfo):
            info = ProbeInfo(name, STATUS_UNAVAILABLE, "probe runner returned an invalid result", {})
        probes.append(
            ProbeInfo(
                info.name,
                _validate_status(info.status),
                _clean(info.note, repo_root),
                _clean_fields(info.fields, repo_root),
            )
        )
    return probes


# ---------------------------------------------------------------------------
# --verify section
# ---------------------------------------------------------------------------


def _engine_checks(
    repo_root: Path,
    target_dir: Path,
    lock,
    identity,
    inventory,
    bootstrap_sha: str,
) -> list[CheckInfo]:
    """Run the builder's 19-check engine and map its codes onto the
    stable vocabulary. The engine is authoritative and static (no
    subprocess, no network)."""
    br = _builder(repo_root)
    try:
        engine_checks = br.verify_runtime(target_dir, lock, identity, inventory, bootstrap_sha)
    except br.BuilderError as exc:
        # e.g. the manifest privacy scan raising: the code is a stable
        # keyword; the message is never surfaced (it may quote paths).
        return [CheckInfo("engine", STATUS_MISMATCH, f"engine stopped: {exc.code}", True)]
    except Exception as exc:  # noqa: BLE001
        return [CheckInfo("engine", STATUS_INIT_ERROR, f"engine error: {type(exc).__name__}", True)]
    return [
        CheckInfo(
            check.name,
            _ENGINE_CODE_MAP.get(check.code, STATUS_MISMATCH),
            _clean_note(check.note, repo_root),
            True,
        )
        for check in engine_checks
    ]


def _clean_note(note, repo_root: Path) -> str:
    if not isinstance(note, str) or not note:
        return ""
    gated = _clean(note, repo_root)
    return gated if gated else ""


def _selector_agreement_check(selector: SelectorInfo, target_dir: Optional[Path]) -> CheckInfo:
    if not selector.present:
        return CheckInfo("selector_agreement", STATUS_MISSING, "no active selector file", True)
    if not selector.valid:
        return CheckInfo("selector_agreement", STATUS_MISMATCH, "selector malformed", True)
    if target_dir is None:
        return CheckInfo("selector_agreement", STATUS_UNAVAILABLE, "no target runtime to agree with", True)
    if selector.runtime_id == target_dir.name:
        return CheckInfo("selector_agreement", STATUS_OK, "", True)
    return CheckInfo("selector_agreement", STATUS_MISMATCH, "selector points to a different runtime", True)


def _running_python_check(runtime_info: RuntimeInfo, identity) -> CheckInfo:
    if runtime_info.mode != MODE_PACKAGED:
        return CheckInfo(
            "running_python",
            STATUS_UNAVAILABLE,
            "report runs from the host interpreter, not from the selected runtime",
            False,
        )
    actual_version = _platform.python_version()
    expected_version = str(identity.version)
    if actual_version != expected_version:
        return CheckInfo(
            "running_python",
            STATUS_MISMATCH,
            f"interpreter version {actual_version} deviates from the pinned {expected_version}",
            True,
        )
    return CheckInfo("running_python", STATUS_OK, f"interpreter {actual_version} matches the pinned asset", True)


def _local_tag(version: str) -> str:
    if not isinstance(version, str) or "+" not in version:
        return "stable"
    tail = version.rsplit("+", 1)[1].strip().lower()
    if tail.startswith("cpu"):
        return "cpu"
    m = _re.match(r"^cu(\d+)$", tail)
    if m:
        return f"cu{m.group(1)}"
    return tail or "custom"


def _expected_cuda_line(backend: str, repo_root: Path):
    """Expected ``torch.version.cuda`` for a torch backend, re-using the
    builder's authoritative table (cpu -> None, cu130 -> "13.0")."""
    br = _builder(repo_root)
    table = getattr(br, "EXPECTED_CUDA_LINE", {"cpu": None, "cu130": "13.0"})
    return table.get(backend)


def _torch_contract_check(
    repo_root: Path, variant: str, lock, torch_probe: Optional[ProbeInfo]
) -> CheckInfo:
    """Torch version / local tag / CUDA line against the locked contract."""
    if torch_probe is None or torch_probe.status != STATUS_OK:
        status = torch_probe.status if torch_probe is not None else STATUS_UNAVAILABLE
        note = torch_probe.note if torch_probe is not None else "torch probe absent"
        return CheckInfo("torch_contract", status, note or "torch probe did not complete", True)
    torch_version = str(getattr(lock, "torch_version", "") or "")
    if not torch_version:
        package_set = getattr(lock, "package_set", None)
        torch_version = (package_set or {}).get("torch", "") if isinstance(package_set, dict) else ""
    if not torch_version:
        return CheckInfo("torch_contract", STATUS_UNAVAILABLE, "lock carries no torch version", True)
    backend = str(getattr(lock, "torch_backend", "") or "")
    expected_cuda = _expected_cuda_line(backend, repo_root)
    actual_version = str(torch_probe.fields.get("version", "") or "")
    actual_tag = str(torch_probe.fields.get("local_tag", "") or "")
    raw_cuda = torch_probe.fields.get("cuda")
    actual_cuda = None if raw_cuda is None else str(raw_cuda)
    problems = []
    if actual_version != torch_version:
        problems.append(f"version {actual_version or 'unknown'} != locked {torch_version}")
    if actual_tag != _local_tag(torch_version):
        problems.append(f"local tag {actual_tag or 'unknown'} != locked {_local_tag(torch_version)}")
    if expected_cuda is None:
        if actual_cuda not in (None, ""):
            problems.append(f"cuda line {actual_cuda!r} present on a cpu backend")
    else:
        if actual_cuda != expected_cuda:
            problems.append(f"cuda line {actual_cuda!r} != expected {expected_cuda!r}")
    if problems:
        return CheckInfo("torch_contract", STATUS_MISMATCH, "; ".join(problems), True)
    return CheckInfo("torch_contract", STATUS_OK, f"torch {actual_version} matches the {variant} lock", True)


def _verify_target(repo_root: Path, runtime_info: RuntimeInfo, br) -> Optional[Path]:
    if runtime_info.mode == MODE_PACKAGED:
        return Path(_sys.executable).resolve().parent
    if runtime_info.selector.valid and runtime_info.selector.runtime_id:
        return repo_root / "runtime" / br.VERSIONS_DIRNAME / runtime_info.selector.runtime_id
    return None


def _build_verify(
    repo_root: Path,
    runtime_info: RuntimeInfo,
    probe_by_name: dict,
    resolve_inputs: Callable,
) -> VerifyInfo:
    br = _builder(repo_root)
    target_dir = _verify_target(repo_root, runtime_info, br)
    variant = runtime_info.variant
    target_id = target_dir.name if target_dir is not None else None

    if target_dir is None:
        check = CheckInfo("target", STATUS_MISSING, "no runnable runtime (no valid selector)", True)
        checks = [check] + [_probe_check(name, probe_by_name) for name in PROBES]
        return _verify_summary(False, None, None, checks, "no runnable runtime identified")

    if variant is None:
        check = CheckInfo(
            "variant",
            STATUS_UNAVAILABLE,
            "cannot determine variant (runtime manifest missing or invalid)",
            True,
        )
        checks = [check, _selector_agreement_check(runtime_info.selector, target_dir)]
        checks += [_probe_check(name, probe_by_name) for name in PROBES]
        return _verify_summary(True, target_id, None, checks, "variant undecidable; engine not run")

    try:
        identity, inventory, lock, bootstrap_sha = resolve_inputs(repo_root, variant)
    except Exception as exc:  # noqa: BLE001 - stable code/type only, never the message
        code = getattr(exc, "code", None)
        note = f"inputs: {code}" if code else f"inputs: {type(exc).__name__}"
        checks = [CheckInfo("inputs", STATUS_UNAVAILABLE, note, True)]
        checks.append(_selector_agreement_check(runtime_info.selector, target_dir))
        checks += [_probe_check(name, probe_by_name) for name in PROBES]
        return _verify_summary(True, target_id, variant, checks, "inputs unresolved; engine not run")

    checks: list[CheckInfo] = [CheckInfo("inputs", STATUS_OK, "", True)]
    checks += _engine_checks(repo_root, target_dir, lock, identity, inventory, bootstrap_sha)
    checks.append(_selector_agreement_check(runtime_info.selector, target_dir))
    checks.append(_running_python_check(runtime_info, identity))
    checks.append(_torch_contract_check(repo_root, variant, lock, probe_by_name.get("torch")))
    checks += [_probe_check(name, probe_by_name) for name in PROBES]
    return _verify_summary(True, target_id, variant, checks, None)


def _probe_check(name: str, probe_by_name: dict) -> CheckInfo:
    probe = probe_by_name.get(name)
    if probe is None:
        return CheckInfo(f"probe_{name}", STATUS_UNAVAILABLE, "probe not run", False)
    return CheckInfo(f"probe_{name}", probe.status, probe.note, False)


def _verify_summary(
    available: bool,
    target_id: Optional[str],
    variant: Optional[str],
    checks: list,
    forced_summary: Optional[str],
) -> VerifyInfo:
    passed = sum(1 for c in checks if c.status == STATUS_OK)
    total = len(checks)
    required_failed = sum(1 for c in checks if c.required and c.status != STATUS_OK)
    if forced_summary is not None:
        summary = forced_summary
    elif not available:
        summary = "verification unavailable"
    elif required_failed == 0:
        summary = f"PASS ({passed}/{total} checks OK; all required checks OK)"
    else:
        summary = f"FAIL ({required_failed} required checks not OK; {passed}/{total} checks OK)"
    return VerifyInfo(
        available=available,
        target_runtime_id=target_id,
        variant=variant,
        checks=checks,
        passed=passed,
        total=total,
        required_failed=required_failed,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Serialization (one model -> text and JSON)
# ---------------------------------------------------------------------------


def model_to_dict(report: EnvReport) -> dict:
    data = {
        "schema": report.schema,
        "command": report.command,
        "json": report.json_output,
        "verify": report.verify,
        "generated_utc": report.generated_utc,
        "source": dataclasses.asdict(report.source),
        "os": dataclasses.asdict(report.os),
        "python": dataclasses.asdict(report.python),
        "runtime": dataclasses.asdict(report.runtime),
        "packages": [dataclasses.asdict(p) for p in report.packages],
        "probes": [dataclasses.asdict(p) for p in report.probes],
        "verify_section": None,
    }
    if report.verify_info is not None:
        v = report.verify_info
        data["verify_section"] = {
            "available": v.available,
            "target_runtime_id": v.target_runtime_id,
            "variant": v.variant,
            "passed": v.passed,
            "total": v.total,
            "required_failed": v.required_failed,
            "summary": v.summary,
            "checks": [dataclasses.asdict(c) for c in v.checks],
        }
    return data


def render_json(report: EnvReport) -> str:
    return _json.dumps(model_to_dict(report), indent=2, ensure_ascii=True, sort_keys=False) + "\n"


def _kv(lines: list, key: str, value) -> None:
    text = "" if value is None else str(value)
    lines.append(f"  {key:<10} {text}")


def render_text(report: EnvReport) -> str:
    d = model_to_dict(report)
    lines: list[str] = []
    lines.append("DeepFaceLab environment report")
    lines.append(f"schema: {d['schema']}")
    lines.append(f"generated: {d['generated_utc']}")
    flags = "envreport"
    if d["json"]:
        flags += " --json"
    if d["verify"]:
        flags += " --verify"
    lines.append(f"command: dfl.bat {flags}")
    lines.append("")

    source = d["source"]
    lines.append("[Source]")
    _kv(lines, "status", source["status"])
    _kv(lines, "commit", source["commit"])
    _kv(lines, "branch", source["branch"])
    if source["note"]:
        _kv(lines, "note", source["note"])
    lines.append("")

    os_info = d["os"]
    lines.append("[OS]")
    _kv(lines, "status", os_info["status"])
    _kv(lines, "platform", os_info["platform"])
    _kv(lines, "version", os_info["version"])
    _kv(lines, "build", os_info["build"])
    if os_info["note"]:
        _kv(lines, "note", os_info["note"])
    lines.append("")

    py = d["python"]
    lines.append("[Python]")
    _kv(lines, "status", py["status"])
    _kv(lines, "implementation", py["implementation"])
    _kv(lines, "version", py["version"])
    _kv(lines, "architecture", py["architecture"])
    lines.append("")

    rt = d["runtime"]
    lines.append("[Runtime]")
    _kv(lines, "mode", rt["mode"])
    _kv(lines, "runtime_id", rt["runtime_id"])
    _kv(lines, "variant", rt["variant"])
    sel = rt["selector"]
    sel_state = "absent" if not sel["present"] else ("valid" if sel["valid"] else "malformed")
    _kv(lines, "selector", f"{sel_state} ({sel['runtime_id'] or 'n/a'})")
    if rt["note"]:
        _kv(lines, "note", rt["note"])
    if sel["note"]:
        _kv(lines, "selector_note", sel["note"])
    lines.append("")

    packages = d["packages"]
    lines.append(f"[Packages] ({len(packages)} listed)")
    width = max((len(p["name"]) for p in packages), default=0)
    for p in packages:
        version = p["version"] if p["version"] is not None else "-"
        line = f"  {p['name']:<{width}}  {p['status']:<13} {version}"
        if p["note"]:
            line += f"  ({p['note']})"
        lines.append(line)
    lines.append("")

    lines.append("[Probes]")
    for probe in d["probes"]:
        line = _format_probe_line_from_dict(probe)
        lines.append(line)
    lines.append("")

    vs = d["verify_section"]
    if vs is not None:
        lines.append("[Verify]")
        _kv(lines, "available", str(vs["available"]).lower())
        _kv(lines, "target", vs["target_runtime_id"])
        _kv(lines, "variant", vs["variant"])
        _kv(lines, "result", vs["summary"])
        for c in vs["checks"]:
            marker = " " if c["required"] else "-"
            line = f"  {marker}{c['id']:<22} {c['status']:<13}"
            if c["note"]:
                line += f" ({c['note']})"
            lines.append(line)
        lines.append("  (required checks are space-prefixed; '-' marks informational entries)")
    return "\n".join(lines) + "\n"


def _format_probe_line_from_dict(probe: dict) -> str:
    parts = []
    fields = probe.get("fields") or {}
    for key, value in fields.items():
        if key == "gpus":
            for index, gpu in enumerate(value or []):
                parts.append(
                    f"gpu[{index}]={gpu.get('name', '?')} vram={gpu.get('vram_mb', '?')}MiB cc={gpu.get('compute_capability', '?')}"
                )
            continue
        parts.append(f"{key}={value}")
    suffix = " ".join(parts)
    line = f"  {probe['name']:<12} {probe['status']:<13}"
    if suffix:
        line += f" {suffix}"
    if probe.get("note"):
        line += f"  ({probe['note']})"
    return line


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_report(
    repo_root: Path,
    probe_runner: Optional[Callable] = None,
) -> EnvReport:
    """Collect the base report (never raises for individual field failures)."""
    br = _builder(repo_root)
    now = _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    runtime_info = _runtime_info(repo_root, br)
    package_names = _package_names(repo_root, runtime_info, br)
    packages = _probe_packages(package_names, repo_root)
    probes = _collect_probes(repo_root, probe_runner)
    return EnvReport(
        schema=SCHEMA,
        command="envreport",
        json_output=False,
        verify=False,
        generated_utc=now,
        source=_clean_source(_git_source(repo_root), repo_root),
        os=_os_info(),
        python=_python_info(),
        runtime=runtime_info,
        packages=packages,
        probes=probes,
    )


def _clean_source(source: SourceInfo, repo_root: Path) -> SourceInfo:
    return SourceInfo(
        source.status,
        source.commit,  # a hash: safe
        _clean(source.branch, repo_root) if source.branch else source.branch,
        _clean(source.note, repo_root) if source.note else "",
    )


def run(
    argv: list[str],
    repo_root: Optional[Path] = None,
    probe_runner: Optional[Callable] = None,
    resolve_inputs: Optional[Callable] = None,
    out=None,
    err=None,
) -> int:
    """Execute ``envreport``. Returns the process exit code.

    Exit codes:
      0  report rendered (base) / all required verify checks OK
      1  base construction failure, privacy gate tripped, or verify failure
      2  usage error (unknown flag)
    """
    out = out if out is not None else _sys.stdout
    err = err if err is not None else _sys.stderr
    root = Path(repo_root) if repo_root is not None else _repo_root_default()

    flags_argv = list(argv)
    if flags_argv and flags_argv[0] == "envreport":
        flags_argv = flags_argv[1:]  # tolerate the command word from main.py

    try:
        json_flag, verify_flag = _parse_flags(flags_argv)
    except ValueError:
        # Fixed wording on purpose: the offending token is never echoed back
        # (it may carry a path, credential, or arbitrary user input) and
        # neither is the internal parse-error detail.
        err.write(f"envreport: invalid argument\n{_USAGE}\n")
        return 2

    try:
        report = build_report(root, probe_runner)
    except Exception as exc:  # noqa: BLE001 - base construction failure
        err.write(f"envreport: {type(exc).__name__}\n")
        return 1

    report.json_output = json_flag
    report.verify = verify_flag

    if verify_flag:
        try:
            if resolve_inputs is None:

                def resolve_inputs(_root, variant):  # noqa: F811
                    br = _builder(_root)
                    return br.load_inputs(_root, variant)

            probe_by_name = {p.name: p for p in report.probes}
            report.verify_info = _build_verify(root, report.runtime, probe_by_name, resolve_inputs)
            for check in report.verify_info.checks:
                check.note = _clean_note(check.note, root)
        except Exception as exc:  # noqa: BLE001 - verify must never crash the report
            report.verify_info = _verify_summary(
                False,
                None,
                None,
                [CheckInfo("verify", STATUS_INIT_ERROR, f"verify section error: {type(exc).__name__}", True)],
                "verification unavailable",
            )

    payload = render_json(report) if json_flag else render_text(report)
    violations = _privacy_violations(payload, root)
    if violations:
        # Fail closed: suppress the report entirely; one clean line on stderr.
        err.write("envreport: report output suppressed by the privacy gate\n")
        return 1

    if not json_flag:
        # Text mode emits raw model strings; keep the console write robust
        # under any legacy code page (JSON mode is ASCII-only and immune).
        reconfigure = getattr(out, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace")
            except Exception:  # noqa: BLE001
                pass
    out.write(payload)
    if not verify_flag:
        return 0
    required_failed = report.verify_info.required_failed if report.verify_info is not None else 1
    return 0 if required_failed == 0 else 1
