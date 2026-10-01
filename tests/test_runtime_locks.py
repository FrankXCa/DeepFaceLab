"""Phase-13 Commit-1 runtime input-lock contract tests."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import generate_runtime_locks as locks


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "runtime-lock.json").read_text(encoding="utf-8"))
VARIANTS = tuple(sorted(CONFIG["variants"]))
GUI_PACKAGES = {"pyqt5", "pyqt5-qt5", "pyqt5-sip"}
# Dev/test-only packages that must never enter a packaged runtime lock.
# tqdm is NOT in this set: it is a common production runtime dependency
# (imported unconditionally at module level by core/interact/interact.py)
# since P13-RUNTIME-DEP-CLOSURE, exactly as in the generator's validator.
# pillow is NOT in this set either: it is a common production runtime
# dependency (imported via core/imagelib/__init__.py by every production
# entry module, all variants) since the Pillow closure pass, exactly as
# colorama and tqdm.
DEV_ONLY = {"ipython", "matplotlib", "pytest"}
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def lock_path(variant: str) -> Path:
    return ROOT / f"requirements-lock-{variant}.txt"


def entries(variant: str) -> list[dict]:
    return locks.parse_lock(lock_path(variant))[1]


def package_map(variant: str) -> dict[str, dict]:
    return {entry["name"]: entry for entry in entries(variant)}


def common_requirement_pins() -> dict[str, str]:
    """Exact pins declared in requirements-runtime-common.txt (normalized)."""
    pins: dict[str, str] = {}
    for raw in (ROOT / "requirements-runtime-common.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = locks.PIN_RE.fullmatch(line)
        if match:
            pins[locks.normalized_name(match.group(1))] = match.group(2)
    return pins


def crlf_bytes(path: Path) -> bytes:
    canonical = locks.canonical_text_bytes(path.read_bytes())
    return canonical.replace(b"\n", b"\r\n")


def test_canonical_text_hash_normalizes_line_endings_only(tmp_path):
    lf = b"a\nb\n"
    expected = locks.sha256_text_canonical(lf)
    assert locks.sha256_text_canonical(b"a\r\nb\r\n") == expected
    assert locks.sha256_text_canonical(b"a\rb\r") == expected
    assert locks.sha256_text_canonical(b"a \n") != locks.sha256_text_canonical(b"a\n")
    assert locks.sha256_text_canonical(b"a") != locks.sha256_text_canonical(b"a\n")

    binary = tmp_path / "artifact.bin"
    binary.write_bytes(b"a\r\nb\r\n")
    assert locks.sha256_file_raw(binary) == locks.sha256_bytes(binary.read_bytes())
    assert locks.sha256_file_raw(binary) != locks.sha256_file_text_canonical(binary)


@pytest.mark.parametrize(
    "relative",
    [
        "requirements-runtime-common.txt",
        "runtime-lock.json",
        "scripts/generate_runtime_locks.py",
    ],
)
def test_real_text_input_hash_is_crlf_independent(relative, tmp_path):
    source = ROOT / relative
    alternate = tmp_path / source.name
    alternate.write_bytes(crlf_bytes(source))
    assert locks.sha256_file_text_canonical(alternate) == (
        locks.sha256_file_text_canonical(source)
    )


def test_exactly_four_explicit_variant_locks_replace_generic_lock():
    assert {path.name for path in ROOT.glob("requirements-lock-*.txt")} == {
        f"requirements-lock-{variant}.txt" for variant in VARIANTS
    }
    assert not (ROOT / "requirements-lock.txt").exists()


@pytest.mark.parametrize("variant", VARIANTS)
def test_reviewed_package_set_identity(variant):
    actual = {entry["name"]: entry["version"] for entry in entries(variant)}
    assert actual == locks.expected_versions(CONFIG, variant)


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_direct_pin_is_classified_direct(variant):
    direct, _ = locks.read_requirement_inputs(CONFIG["variants"][variant]["inputs"])
    actual = package_map(variant)
    assert direct.keys() <= actual.keys()
    assert all(actual[name]["version"] == version for name, version in direct.items())
    assert all(actual[name]["classification"] == "direct" for name in direct)


@pytest.mark.parametrize("variant", VARIANTS)
def test_gui_family_only_in_gui_variants(variant):
    actual = package_map(variant)
    assert (GUI_PACKAGES <= actual.keys()) is variant.endswith("-gui")
    assert GUI_PACKAGES.isdisjoint(actual) is variant.endswith("-nogui")


@pytest.mark.parametrize("variant", VARIANTS)
def test_exact_torch_artifact_matches_variant(variant):
    torch = package_map(variant)["torch"]
    backend = CONFIG["variants"][variant]["torch_backend"]
    expected_local = "cu130" if backend == "cu130" else "cpu"
    assert torch["version"] == f"2.14.0+{expected_local}"
    assert torch["filename"] == (
        f"torch-2.14.0+{expected_local}-cp312-cp312-win_amd64.whl"
    )
    assert torch["source"] == f"https://download.pytorch.org/whl/{expected_local}"
    assert urllib_hostname(torch["url"]) in {
        "download.pytorch.org",
        "download-r2.pytorch.org",
    }


def urllib_hostname(url: str) -> str | None:
    from urllib.parse import urlsplit

    return urlsplit(url).hostname


@pytest.mark.parametrize("variant", VARIANTS)
def test_no_dev_only_or_orphan_packages(variant):
    names = package_map(variant).keys()
    assert DEV_ONLY.isdisjoint(names)
    assert "psutil" not in names


@pytest.mark.parametrize("variant", VARIANTS)
def test_pillow_is_a_common_production_runtime_dependency(variant):
    """Regression: Pillow is imported via core/imagelib/__init__.py by every
    production entry module (extractor, sorter, trainer, export, merge,
    faceset, xseg), so it must be a DIRECT common dependency present in all
    four variant locks, not a dev/test-only package.

    The expected version is anchored to the reviewed Pillow compatibility
    contract in scripts/generate_runtime_locks.py
    (PILLOW_REVIEWED_VERSION / PILLOW_COMPATIBILITY_REVIEW_KEY), NOT to
    whatever runtime-lock.json currently declares: a Pillow version change --
    especially across a major version -- is only accepted after an explicit
    source-compatibility review and runtime validation on all four variants
    plus a deliberate update of that contract. Regenerating the locks alone
    (e.g. bumping the pin in runtime-lock.json and the common requirements)
    must therefore fail these tests until the reviewed contract is updated.
    Pillow 9.5.0 is the reviewed compatibility pin for the current baseline:
    the production source uses Pillow APIs including Image.ADAPTIVE via
    core/imagelib/reduce_colors.py on the conditional merge color-degradation
    path in merger/MergeMasked.py. It is an honest compatibility baseline,
    not an assertion that it is the newest possible compatible version."""
    contract_version = locks.PILLOW_REVIEWED_VERSION
    assert re.fullmatch(r"\d+\.\d+\.\d+", contract_version)
    assert locks.PILLOW_COMPATIBILITY_REVIEW_KEY.strip()
    # The reviewed pin must be declared in the common production inputs
    # (removing or de-classifying pillow from production inputs fails here).
    assert common_requirement_pins().get("pillow") == contract_version
    # The reviewed pin must be declared in the authoritative lock config.
    assert CONFIG["resolved_versions"]["common"]["pillow"] == contract_version
    # Pillow must not be classified as dev/test-only or forbidden.
    assert "pillow" not in DEV_ONLY
    assert "pillow" not in locks.FORBIDDEN_RUNTIME_PACKAGES
    # Pillow must be locked in every variant at the reviewed version.
    assert "pillow" in package_map(variant)
    pillow = package_map(variant)["pillow"]
    assert pillow["version"] == contract_version
    assert pillow["classification"] == "direct"
    assert pillow["filename"].endswith("-cp312-cp312-win_amd64.whl")
    assert pillow["source"] == "https://pypi.org/simple"


@pytest.mark.parametrize("variant", VARIANTS)
def test_ffmpeg_python_and_future_are_correct(variant):
    actual = package_map(variant)
    assert actual["ffmpeg-python"]["version"] == "0.2.0"
    assert actual["ffmpeg-python"]["classification"] == "direct"
    assert actual["future"]["version"] == "1.0.0"
    assert actual["future"]["classification"] == "transitive"
    assert "ffmpeg" not in actual


@pytest.mark.parametrize("variant", VARIANTS)
def test_artifact_records_are_complete_wheel_only_and_targeted(variant):
    required = {
        "classification",
        "compatibility_tags",
        "filename",
        "input_hashes",
        "input_identity_sha256",
        "name",
        "resolver",
        "resolver_config_sha256",
        "sha256",
        "source",
        "target_platform",
        "target_python",
        "url",
        "variant",
        "version",
    }
    for entry in entries(variant):
        assert set(entry) == required
        assert entry["version"]
        assert entry["filename"].endswith(".whl")
        assert locks.wheel_score(entry["filename"]) is not None
        assert entry["compatibility_tags"] == locks.compatibility_tags(entry["filename"])
        assert entry["source"].startswith("https://")
        assert entry["url"].startswith("https://")
        assert HASH_RE.fullmatch(entry["sha256"])
        assert entry["target_python"] == "3.12.11"
        assert entry["target_platform"] == "x86_64-pc-windows-msvc"
        assert entry["resolver"] == "uv 0.12.18"


@pytest.mark.parametrize("variant", VARIANTS)
def test_input_and_resolver_hashes_are_current(variant):
    headers, artifact_entries = locks.parse_lock(lock_path(variant))
    _, paths = locks.read_requirement_inputs(CONFIG["variants"][variant]["inputs"])
    expected_inputs = locks.input_hashes(paths)
    expected_aggregate = locks.sha256_bytes(
        locks.canonical_json(expected_inputs).encode("ascii")
    )
    assert headers["resolver"] == "uv 0.12.18"
    assert headers["resolver-config-sha256"] == locks.sha256_file_text_canonical(
        ROOT / "runtime-lock.json"
    )
    assert headers["generator-sha256"] == locks.sha256_file_text_canonical(
        ROOT / "scripts" / "generate_runtime_locks.py"
    )
    assert all(entry["input_hashes"] == expected_inputs for entry in artifact_entries)
    assert all(
        entry["input_identity_sha256"] == expected_aggregate
        for entry in artifact_entries
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_future_install_is_exact_hash_locked_and_no_resolution(variant):
    text = lock_path(variant).read_text(encoding="utf-8")
    assert "--only-binary=:all:" in text
    assert "--require-hashes" in text
    assert "# future-install: --only-binary=:all: --require-hashes --no-deps" in text
    assert text.count(" --hash=sha256:") == len(entries(variant))


@pytest.mark.parametrize("variant", VARIANTS)
def test_lock_rendering_is_deterministic_from_approved_inputs(variant):
    artifact_entries = entries(variant)
    artifacts = {
        entry["name"]: {
            "filename": entry["filename"],
            "sha256": entry["sha256"],
            "source": entry["source"],
            "url": entry["url"],
        }
        for entry in artifact_entries
    }
    versions = {entry["name"]: entry["version"] for entry in artifact_entries}
    first = locks.render_lock(CONFIG, variant, versions, artifacts)
    second = locks.render_lock(CONFIG, variant, versions, artifacts)
    assert first == second == lock_path(variant).read_text(encoding="utf-8")


@pytest.mark.parametrize("variant", VARIANTS)
def test_lock_rendering_is_identical_from_crlf_checkout(variant, tmp_path, monkeypatch):
    artifact_entries = entries(variant)
    artifacts = {
        entry["name"]: {
            "filename": entry["filename"],
            "sha256": entry["sha256"],
            "source": entry["source"],
            "url": entry["url"],
        }
        for entry in artifact_entries
    }
    versions = {entry["name"]: entry["version"] for entry in artifact_entries}
    _, requirement_paths = locks.read_requirement_inputs(
        CONFIG["variants"][variant]["inputs"]
    )
    relative_paths = {
        path.relative_to(ROOT).as_posix() for path in requirement_paths
    } | {"runtime-lock.json", "scripts/generate_runtime_locks.py"}
    for relative in relative_paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(crlf_bytes(ROOT / relative))

    monkeypatch.setattr(locks, "ROOT", tmp_path)
    monkeypatch.setattr(locks, "CONFIG_PATH", tmp_path / "runtime-lock.json")
    monkeypatch.setattr(
        locks, "GENERATOR_PATH", tmp_path / "scripts" / "generate_runtime_locks.py"
    )
    rendered = locks.render_lock(CONFIG, variant, versions, artifacts)
    assert rendered == lock_path(variant).read_text(encoding="utf-8")


def test_standalone_python_asset_identity_is_exact():
    identity = json.loads((ROOT / "runtime-python.json").read_text(encoding="utf-8"))
    archive = identity["archive"]
    assert identity["implementation"] == "CPython"
    assert identity["version"] == "3.12.11"
    assert identity["platform"] == "Windows x64 / AMD64"
    assert archive["release"] == "20250828"
    assert archive["target_triple"] == "x86_64-pc-windows-msvc"
    assert archive["cpu_baseline"] == "x86_64"
    assert archive["flavor"] == "install_only_stripped"
    assert archive["filename"] == (
        "cpython-3.12.11+20250828-x86_64-pc-windows-msvc-"
        "install_only_stripped.tar.gz"
    )
    assert archive["sha256"] == (
        "0b8fab064f852d3ccb7cf30f58195452c1d70f494a556b4c003889aa2f93038f"
    )


def test_recursive_license_inventory_is_nonempty_and_deterministic():
    inventory = json.loads(
        (ROOT / "runtime-python-licenses.json").read_text(encoding="utf-8")
    )
    entries = inventory["entries"]
    assert inventory["discovery"]["recursive"] is True
    assert inventory["discovery"]["case_sensitive"] is False
    assert entries
    assert [entry["path"] for entry in entries] == sorted(
        (entry["path"] for entry in entries), key=str.casefold
    )
    digest = hashlib.sha256(locks.canonical_json(entries).encode("ascii")).hexdigest()
    assert inventory["inventory_sha256"] == digest
    assert all(HASH_RE.fullmatch(entry["sha256"]) for entry in entries)


def test_nested_pip_and_tcl_licenses_are_discovered():
    inventory = json.loads(
        (ROOT / "runtime-python-licenses.json").read_text(encoding="utf-8")
    )
    paths = {entry["path"] for entry in inventory["entries"]}
    assert "python/Lib/site-packages/pip-24.3.1.dist-info/LICENSE.txt" in paths
    assert "python/tcl/tk8.6/demos/license.terms" in paths
    assert "python/tcl/tk8.6/license.terms" in paths


def test_generated_outputs_contain_no_private_paths_hosts_ips_or_credentials():
    candidates = [
        ROOT / "requirements-cpu.txt",
        ROOT / "requirements-cuda.txt",
        ROOT / "requirements-runtime-common.txt",
        ROOT / "requirements-runtime-gui.txt",
        ROOT / "runtime-lock.json",
        ROOT / "runtime-python.json",
        ROOT / "runtime-python-licenses.json",
        ROOT / "scripts" / "generate_runtime_locks.py",
        ROOT / "tests" / "test_runtime_locks.py",
        *(lock_path(variant) for variant in VARIANTS),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in candidates)
    forbidden = [
        r"(?i)[a-z]:[\\/](?:users|codes|personal|downloads|desktop)[\\/]",
        r"(?i)appdata[\\/]",
        r"(?i)(?:^|[\\/])\.venv(?:[\\/]|$)",
        r"\b10(?:\.\d{1,3}){3}\b",
        r"\b192\.168(?:\.\d{1,3}){2}\b",
        r"\b172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}\b",
        r"\b(?:AKIA|ghp_|github_pat_|xox[baprs]-)[A-Za-z0-9_-]+",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    ]
    for pattern in forbidden:
        assert re.search(pattern, combined, re.MULTILINE) is None


def test_offline_validator_accepts_all_tracked_outputs():
    completed = subprocess.run(
        [sys.executable, "scripts/generate_runtime_locks.py", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Phase-13 runtime input locks: PASS" in completed.stdout


def test_offline_validator_accepts_crlf_textual_identity_inputs(tmp_path):
    text_inputs = {
        "requirements-cpu.txt",
        "requirements-cuda.txt",
        "requirements-runtime-common.txt",
        "requirements-runtime-gui.txt",
        "runtime-lock.json",
        "scripts/generate_runtime_locks.py",
    }
    required = text_inputs | {
        "runtime-python.json",
        "runtime-python-licenses.json",
        *(f"requirements-lock-{variant}.txt" for variant in VARIANTS),
    }
    for relative in required:
        source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative in text_inputs:
            destination.write_bytes(crlf_bytes(source))
        else:
            shutil.copy2(source, destination)

    completed = subprocess.run(
        [sys.executable, "scripts/generate_runtime_locks.py", "--check"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Phase-13 runtime input locks: PASS" in completed.stdout


def _copy_lock_tree(tmp_path) -> None:
    """Copy the full checked-in lock/validator tree into tmp_path (byte
    copy, never mutating the checked-in files) so a case can mutate its
    private copy and rerun the offline validator against it."""
    required = {
        "requirements-cpu.txt",
        "requirements-cuda.txt",
        "requirements-runtime-common.txt",
        "requirements-runtime-gui.txt",
        "runtime-lock.json",
        "runtime-python.json",
        "runtime-python-licenses.json",
        "scripts/generate_runtime_locks.py",
        *(f"requirements-lock-{variant}.txt" for variant in VARIANTS),
    }
    for relative in sorted(required):
        source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _run_check(tmp_path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/generate_runtime_locks.py", "--check"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )


def _bump_locked_pillow(tmp_path, major_minor_patch: str) -> None:
    for variant in VARIANTS:
        lock = tmp_path / f"requirements-lock-{variant}.txt"
        text = lock.read_text(encoding="utf-8")
        old = f'"variant":"{variant}","version":"9.5.0"'
        new = f'"variant":"{variant}","version":"{major_minor_patch}"'
        assert old in text
        lock.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def _remove_pillow_from_one_variant_lock(tmp_path, variant: str) -> None:
    lock = tmp_path / f"requirements-lock-{variant}.txt"
    kept = []
    for line in lock.read_text(encoding="utf-8").splitlines():
        if line.startswith("# artifact:") and '"name":"pillow"' in line:
            continue
        if line.startswith("pillow @ "):
            continue
        kept.append(line)
    lock.write_text("\n".join(kept) + "\n", encoding="utf-8", newline="\n")


def _mark_generator_changed(tmp_path) -> None:
    """Refresh the generator-sha256 header of every tmp lock so a mutation
    of the tmp generator itself is not masked as a stale-generator failure."""
    new_hash = locks.sha256_file_text_canonical(tmp_path / "scripts" / "generate_runtime_locks.py")
    for variant in VARIANTS:
        lock = tmp_path / f"requirements-lock-{variant}.txt"
        text = lock.read_text(encoding="utf-8")
        text = re.sub(r"^# generator-sha256: [0-9a-f]{64}\r?$",
                      f"# generator-sha256: {new_hash}", text,
                      count=1, flags=re.MULTILINE)
        lock.write_text(text, encoding="utf-8", newline="\n")


@pytest.mark.parametrize("case", [
    "A-current-contract",
    "B-lock-only-major-bump",
    "C-removed-from-common-production",
    "D-classified-dev-only",
    "E-omitted-from-one-variant",
])
def test_pillow_contract_acceptance_cases(tmp_path, case):
    """Acceptance cases A-E for the reviewed Pillow compatibility contract,
    exercised on private tmp copies of the tree (the checked-in locks and
    requirements are never mutated). A current contract state must pass;
    every unreviewed Pillow version/classification change must make the
    offline validator fail, so a Pillow major-version change can never pass
    lock regeneration silently without updating the contract in
    scripts/generate_runtime_locks.py."""
    _copy_lock_tree(tmp_path)
    if case == "A-current-contract":
        completed = _run_check(tmp_path)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "Phase-13 runtime input locks: PASS" in completed.stdout
    elif case == "B-lock-only-major-bump":
        # Locks alone moved to a newer major version; config/contract stay.
        _bump_locked_pillow(tmp_path, "10.0.0")
        completed = _run_check(tmp_path)
        assert completed.returncode != 0, "B: lock-only major bump must FAIL"
    elif case == "C-removed-from-common-production":
        common = tmp_path / "requirements-runtime-common.txt"
        lines = common.read_text(encoding="utf-8").splitlines()
        assert "pillow==9.5.0" in lines
        common.write_text(
            "\n".join(l for l in lines if l.strip() != "pillow==9.5.0") + "\n",
            encoding="utf-8", newline="\n",
        )
        completed = _run_check(tmp_path)
        assert completed.returncode != 0, "C: removal from production inputs must FAIL"
    elif case == "D-classified-dev-only":
        gen = tmp_path / "scripts" / "generate_runtime_locks.py"
        text = gen.read_text(encoding="utf-8")
        old = '"ipython", "matplotlib", "psutil", "pytest", "ffmpeg"'
        assert old in text
        gen.write_text(
            text.replace(old,
                         '"ipython", "matplotlib", "pillow", "psutil", "pytest", "ffmpeg"'),
            encoding="utf-8", newline="\n",
        )
        _mark_generator_changed(tmp_path)
        completed = _run_check(tmp_path)
        assert completed.returncode != 0, "D: dev-only reclassification must FAIL"
        assert "pillow" in (completed.stdout + completed.stderr)
    elif case == "E-omitted-from-one-variant":
        _remove_pillow_from_one_variant_lock(tmp_path, "cuda-gui")
        completed = _run_check(tmp_path)
        assert completed.returncode != 0, "E: omission from one variant must FAIL"
    else:
        raise AssertionError(f"unknown acceptance case: {case}")
