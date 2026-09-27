"""Environment-gated REAL integration test for the Phase-13 builder (Commit 2).

Unlike tests/test_build_runtime.py (fully faked, always run), this test drives
the builder CLI as a subprocess against the real pinned CPython 3.12.11
standalone archive, the real locked wheels, and the assembled runtime's own
pip. It is skipped unless explicitly enabled:

    DFL_REAL_RUNTIME_BUILD=1                enable this test (default: skip)
    DFL_REAL_RUNTIME_VARIANT=cpu-nogui      variant to build (default: cpu-nogui;
                                            cuda-nogui needs a ~2 GB wheel)
    DFL_REAL_BUILD_TIMEOUT_MIN=90           per-build subprocess timeout

What it proves (and records for the commit report):
  * the pinned Python archive downloads, verifies by SHA-256, and extracts
    into staging (safe-extraction path on the real archive);
  * the recursive Python license inventory verifies against the extracted
    tree;
  * the exact locked wheels install into the runtime's own Python with
    ``pip install --no-index --no-deps --require-hashes
    --only-binary=:all:`` (no resolution, no PyPI fallback);
  * the installed package set matches the selected lock; the variant
    contract holds (torch local version tag; PyQt5 family only for GUI);
  * basic imports (numpy/scipy/numexpr/cv2/onnx/onnxruntime/torch/ffmpeg,
    +PyQt5 for GUI variants) succeed under the ASSEMBLED Python with
    sys.executable inside the runtime;
  * scripts/runtime_entry.py isolated-bootstrap self-test passes;
  * the torch probe reports package identity; CUDA hardware state, if
    probed, is reported as PASS or ENVIRONMENTALLY_LIMITED (never a build
    failure);
  * the runtime promotes atomically to runtime/versions/<id>/, the manifest
    is privacy-clean, and ``verify-runtime`` passes on the promoted tree;
  * REPRODUCIBILITY: a second build of the same variant into a separate
    runtime root yields the same runtime ID and byte-identical canonical
    manifest (artifacts served from the shared hash-verified cache).

The run is left in place under .cache/real-runtime-<variant>/ (developer
state, gitignored) as evidence; the builder itself never deletes verified
runtimes.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import build_runtime as br
from scripts import generate_runtime_locks as locklib

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_runtime.py"

# Absolute Windows/POSIX paths in retained logs must never be kept: known
# locations are replaced by logical labels, anything else by <abs>.
_ABS_PATH_RE = re.compile(r"(?i)(?:[A-Za-z]):[\\/][^\s\"'`<>|()]*")


def _sanitize(text: str, known: dict[str, str]) -> str:
    """Privacy-safe rendering for retained evidence logs.

    The real build legitimately uses absolute paths internally; the RETAINED
    logs must not expose them (no absolute interpreter/repository/cache
    paths, no usernames or hostnames). Known locations become logical
    labels; every remaining absolute (drive-qualified) path becomes <abs>."""
    for path, label in sorted(known.items(), key=lambda kv: len(kv[0]), reverse=True):
        if path:
            text = text.replace(path, label)
    text = re.sub(r"file:///(?:[A-Za-z]):[\\/][^\s\"']*", "file://<abs>", text)
    text = _ABS_PATH_RE.sub("<abs>", text)
    return text


def _enabled() -> bool:
    return os.environ.get("DFL_REAL_RUNTIME_BUILD") == "1"


def _variant() -> str:
    variant = os.environ.get("DFL_REAL_RUNTIME_VARIANT", "cpu-nogui")
    if variant not in br.VARIANTS:
        raise ValueError(f"DFL_REAL_RUNTIME_VARIANT must be one of {br.VARIANTS}")
    return variant


def _timeout_seconds() -> int:
    return int(os.environ.get("DFL_REAL_BUILD_TIMEOUT_MIN", "90")) * 60


def _subprocess_env() -> dict[str, str]:
    """Builder subprocess environment: no host Python/pip influence, and the
    real-build switches carried through."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not (key.startswith("PYTHON") or key.startswith("PIP"))
    }
    for key in ("DFL_REAL_RUNTIME_BUILD", "DFL_REAL_RUNTIME_VARIANT", "DFL_REAL_BUILD_TIMEOUT_MIN"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _run_cli(
    args: list[str],
    timeout: int,
    log_path: Path | None = None,
    known: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-I", "-B", str(BUILDER), *args]
    proc = subprocess.run(
        command,
        env=_subprocess_env(),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if log_path is not None:
        labels = known or {}
        display = " ".join(_sanitize(part, labels) for part in command)
        log_path.write_text(
            "$ " + display + "\n"
            + "----- stdout -----\n" + _sanitize(proc.stdout, labels)
            + "----- stderr -----\n" + _sanitize(proc.stderr, labels),
            encoding="utf-8",
        )
    return proc


def _build_runtime_id(stdout: str, label: str) -> str:
    for line in stdout.splitlines():
        if line.startswith(("build: BUILT runtime ", "build: REUSED runtime ")):
            # "build: BUILT runtime <id> (variant <v>)" -> the id is index 3.
            return line.split(" ")[3]
    raise AssertionError(f"{label}: builder did not report a runtime id:\n{stdout}")


@pytest.mark.skipif(not _enabled(), reason="set DFL_REAL_RUNTIME_BUILD=1 to run the real assembly")
def test_real_runtime_assembly_verification_and_reproducibility():
    variant = _variant()
    timeout = _timeout_seconds()
    scratch = ROOT / ".cache" / f"real-runtime-{variant}"
    # Fresh scratch so each enabled run is self-contained (developer state).
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    cache = ROOT / ".cache" / "portable-runtime"
    root_a = scratch / "root-a"
    root_b = scratch / "root-b"
    # Logical labels used to redact absolute paths from retained logs.
    # (Longest path wins, so the harness interpreter — which lives inside
    # the repository's developer venv — is labeled before the repo root.)
    labels = {
        str(ROOT): "<repo>",
        str(cache): "<cache>",
        str(scratch): "<scratch>",
        str(root_a): "<runtime-root-a>",
        str(root_b): "<runtime-root-b>",
        str(sys.executable): "<harness-python>",
        Path(sys.executable).as_posix(): "<harness-python>",
    }

    # -- Build A: full online assembly (hash-verified cache).
    proc_a = _run_cli(
        [
            "build", "--variant", variant,
            "--runtime-root", str(root_a),
            "--artifact-cache", str(cache),
        ],
        timeout,
        scratch / "build-a.log",
        known=labels,
    )
    assert proc_a.returncode == 0, f"build A failed:\n{proc_a.stdout}\n{proc_a.stderr}"
    runtime_id_a = _build_runtime_id(proc_a.stdout, "build A")
    assert br.RUNTIME_ID_RE.fullmatch(runtime_id_a)
    version_a = root_a / "versions" / runtime_id_a
    assert (version_a / "python.exe").is_file(), "assembled runtime lacks its interpreter"
    manifest_a_path = version_a / br.MANIFEST_NAME
    assert manifest_a_path.is_file(), "assembled runtime lacks its manifest"

    # -- verify-runtime on the promoted tree (manifest-driven static checks).
    verify_a = _run_cli(
        [
            "verify-runtime",
            "--runtime-id", runtime_id_a,
            "--runtime-root", str(root_a),
            "--variant", variant,
        ],
        1800,
        scratch / "verify-a.log",
        known=labels,
    )
    assert verify_a.returncode == 0, f"verify-runtime failed:\n{verify_a.stdout}\n{verify_a.stderr}"
    assert "verify-runtime: PASS" in verify_a.stdout

    manifest_a = json.loads(manifest_a_path.read_text(encoding="utf-8"))
    canonical_a = manifest_a["canonical"]

    # -- Contract: canonical identity matches the Commit-1 inputs exactly.
    assert canonical_a["runtime_id"] == runtime_id_a
    assert canonical_a["variant"] == variant
    assert canonical_a["python"]["sha256"] == br.PINNED_PYTHON["sha256"]
    assert canonical_a["python"]["filename"] == br.PINNED_PYTHON["filename"]
    assert br.privacy_violations(manifest_a_path.read_text(encoding="utf-8")) == []

    expected_versions = locklib.expected_versions(locklib.load_json(ROOT / "runtime-lock.json"), variant)
    assert canonical_a["package_set"] == expected_versions, (
        f"package set deviates from the lock:\n"
        f"  manifest: {canonical_a['package_set']}\n  expected: {expected_versions}"
    )
    assert canonical_a["interpreter_baseline"], "interpreter baseline is empty"

    # -- Torch boundary: the CANONICAL verification records only the
    #    deterministic identity outcome (VERIFIED / NOT_APPLICABLE). The
    #    physical CUDA availability state (PASS / ENVIRONMENTALLY_LIMITED)
    #    and device observations are OPERATIONAL evidence for this machine
    #    only (CURRENT_MACHINE_CONTRACT_VALIDATION) and must not alter the
    #    canonical identity.
    torch_code = canonical_a["verification"]["torch_probe"]
    if variant.startswith("cuda"):
        assert torch_code == "VERIFIED", torch_code
        hw = manifest_a["operational"].get("torch_hardware", {})
        assert hw.get("code") in ("PASS", "ENVIRONMENTALLY_LIMITED"), hw
        if hw.get("code") == "PASS":
            assert hw.get("cuda_available") is True
            assert hw.get("device"), "device observation expected with cuda_available=true"
        else:
            assert hw.get("cuda_available") is False
    else:
        assert torch_code == "NOT_APPLICABLE", torch_code
        hw = manifest_a["operational"].get("torch_hardware", {})
        assert hw.get("code") == "NOT_APPLICABLE", hw
    for check in ("imports", "package_set", "variant_contract", "python_identity", "runtime_entry_self_test"):
        assert canonical_a["verification"][check] == "PASS", (check, canonical_a["verification"])

    # -- Build B: reproducibility from the same canonical inputs, separate
    #    runtime root; artifacts come from the hash-verified cache.
    proc_b = _run_cli(
        [
            "build", "--variant", variant,
            "--runtime-root", str(root_b),
            "--artifact-cache", str(cache),
        ],
        timeout,
        scratch / "build-b.log",
        known=labels,
    )
    assert proc_b.returncode == 0, f"build B failed:\n{proc_b.stdout}\n{proc_b.stderr}"
    runtime_id_b = _build_runtime_id(proc_b.stdout, "build B")

    assert runtime_id_b == runtime_id_a, (
        f"reproducibility violated: build A {runtime_id_a} != build B {runtime_id_b}"
    )
    manifest_b = json.loads((root_b / "versions" / runtime_id_b / br.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest_b["canonical"] == canonical_a, "canonical manifest differs between separate builds"
    assert br.privacy_violations((root_b / "versions" / runtime_id_b / br.MANIFEST_NAME).read_text(encoding="utf-8")) == []

    verify_b = _run_cli(
        [
            "verify-runtime",
            "--runtime-id", runtime_id_b,
            "--runtime-root", str(root_b),
            "--variant", variant,
        ],
        1800,
        scratch / "verify-b.log",
        known=labels,
    )
    assert verify_b.returncode == 0, f"verify-runtime B failed:\n{verify_b.stdout}\n{verify_b.stderr}"
    assert "verify-runtime: PASS" in verify_b.stdout

    # -- Retained evidence logs must be privacy-clean: no absolute
    #    interpreter/repository/runtime/cache paths, no usernames, hostnames
    #    or private IPs. (The build itself may use absolute paths internally;
    #    only what is retained for review is redacted.)
    for log_file in sorted(scratch.glob("*.log")):
        text = log_file.read_text(encoding="utf-8")
        violations = br.privacy_violations(text)
        assert violations == [], (
            f"retained log {log_file.name} is not privacy-clean: {violations[:3]}"
        )
