"""Commit-2 builder unit tests (Phase 13).

These tests exercise scripts/build_runtime.py without network access:
artifact fetchers are faked (in-memory streams or prepared directories), the
staged interpreter is a scripted double, and the pip installer is a recorder
that materializes dist-info metadata. The real builder logic (safe
extraction, license-tree verification, runtime identity, staging/promotion,
lock semantics, privacy scanning, manifest round-trips) runs unmodified.

The one real integration build (real Python archive, real wheels, real pip)
lives in tests/test_real_runtime_build.py and is environment-gated.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import types
from pathlib import Path

import pytest

from scripts import build_runtime as br

ROOT = Path(__file__).resolve().parents[1]

BOOTSTRAP_SHA = "ab" * 32
INVENTORY_SHA = "cd" * 32
LOCK_SHA = "ef" * 32
RESOLVER_CONFIG_SHA = "11" * 32
INPUT_HASH_SHA = "33" * 32
INPUT_IDENTITY_SHA = "44" * 32
# Stand-in installed-tree integrity digest (the real value is computed from
# the tree's actual bytes; these unit tests only need a stable stand-in).
TREE_SHA = "55" * 32

LICENSE_FIXTURES = (
    ("python/Lib/site-packages/pip-24.3.1.dist-info/LICENSE.txt", b"PIP-LICENSE"),
    ("python/LICENSE.txt", b"ROOT-LICENSE"),
    ("python/tcl/tk8.6/demos/license.terms", b"TK-TERMS"),
    ("python/tcl/tk8.6/license.terms", b"TK-TERMS"),
)

CHILD_HOLD_CODE = r"""
import sys, time, os
repo, lock_path, ready_path, hold_seconds = sys.argv[1:5]
sys.path.insert(0, repo)
h = os.open(lock_path, os.O_RDWR | os.O_CREAT)
if os.name == "nt":
    import msvcrt
    msvcrt.locking(h, msvcrt.LK_NBLCK, 1)
else:
    import fcntl
    fcntl.flock(h, fcntl.LOCK_EX)
with open(ready_path, "w") as f:
    f.write("ready")
time.sleep(float(hold_seconds))
"""


# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_b64(data: bytes) -> str:
    """RECORD-style sha256 hash: urlsafe-base64 without padding."""
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")


def make_python_archive(path: Path, extra_members: dict[str, bytes] | None = None) -> Path:
    members = {"python/python.exe": b"FAKE-PYTHON-EXE"}
    members.update(dict(LICENSE_FIXTURES))
    # The pinned standalone archive ships pip as a baseline distribution:
    # include its dist-info (METADATA + RECORD) so the builder's
    # distribution re-enumeration and RECORD integrity checks have the
    # baseline pip to account for.
    pip_metadata = b"Metadata-Version: 2.1\nName: pip\nVersion: 24.3.1\n"
    pip_license = b"PIP-LICENSE"
    pip_record = (
        f"pip-24.3.1.dist-info/LICENSE.txt,sha256={record_b64(pip_license)},{len(pip_license)}\n"
        f"pip-24.3.1.dist-info/METADATA,sha256={record_b64(pip_metadata)},{len(pip_metadata)}\n"
        "pip-24.3.1.dist-info/RECORD,,\n"
    ).encode("ascii")
    members["python/Lib/site-packages/pip-24.3.1.dist-info/METADATA"] = pip_metadata
    members["python/Lib/site-packages/pip-24.3.1.dist-info/RECORD"] = pip_record
    if extra_members:
        members.update(extra_members)
    with tarfile.open(path, "w:gz") as package:
        dir_info = tarfile.TarInfo("python")
        dir_info.type = tarfile.DIRTYPE
        dir_info.mode = 0o755
        dir_info.mtime = 1700000000
        package.addfile(dir_info)
        for name in sorted(members):
            data = members[name]
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 1700000001
            package.addfile(info, io.BytesIO(data))
    return path


def make_hostile_archive(path: Path, kind: str) -> Path:
    with tarfile.open(path, "w:gz") as package:
        if kind == "traversal":
            info = tarfile.TarInfo("python/../evil.txt")
            info.size = 5
            package.addfile(info, io.BytesIO(b"evil!"))
        elif kind == "absolute":
            info = tarfile.TarInfo("/etc/passwd")
            info.size = 5
            package.addfile(info, io.BytesIO(b"evil!"))
        elif kind == "outside":
            info = tarfile.TarInfo("other/escape.txt")
            info.size = 5
            package.addfile(info, io.BytesIO(b"evil!"))
        elif kind == "symlink":
            info = tarfile.TarInfo("python/link.exe")
            info.type = tarfile.SYMTYPE
            info.linkname = "python.exe"
            package.addfile(info)
        elif kind == "hardlink":
            info = tarfile.TarInfo("python/hard.exe")
            info.type = tarfile.LNKTYPE
            info.linkname = "python.exe"
            package.addfile(info)
        elif kind == "drive":
            # Windows drive-qualified member (absolute or drive-relative).
            info = tarfile.TarInfo("C:/Windows/evil.dll")
            info.size = 5
            package.addfile(info, io.BytesIO(b"evil!"))
        elif kind == "ads":
            # Alternate-data-stream-style colon in the member name.
            info = tarfile.TarInfo("python/data.txt:hidden")
            info.size = 5
            package.addfile(info, io.BytesIO(b"evil!"))
        elif kind == "devicename":
            # Windows reserved device name (NUL) as a file member.
            info = tarfile.TarInfo("python/Lib/NUL.txt")
            info.size = 5
            package.addfile(info, io.BytesIO(b"evil!"))
        elif kind == "devicename-middle":
            # Reserved device name in a NON-leaf path component: the leaf is
            # harmless, the intermediate component is the hazard.
            info = tarfile.TarInfo("python/NUL/innocent.txt")
            info.size = 5
            package.addfile(info, io.BytesIO(b"evil!"))
        elif kind == "duplicate-root":
            # A SECOND archive-root member: the stripped `python` root maps
            # to the empty relative path, so duplicate detection must track
            # the RAW name or the second root member sails through.
            a = tarfile.TarInfo("python")
            a.size = 3
            package.addfile(a, io.BytesIO(b"AAA"))
            b = tarfile.TarInfo("python")
            b.size = 3
            package.addfile(b, io.BytesIO(b"BBB"))
        elif kind == "backslash":
            # Backslash in a member name is never a valid POSIX tar path.
            info = tarfile.TarInfo("python/ev\\il.txt")
            info.size = 5
            package.addfile(info, io.BytesIO(b"evil!"))
        elif kind == "case":
            # Case-insensitive collision: two members differing only by case.
            a = tarfile.TarInfo("python/Module.py")
            a.size = 5
            package.addfile(a, io.BytesIO(b"one!!"))
            b = tarfile.TarInfo("python/module.py")
            b.size = 5
            package.addfile(b, io.BytesIO(b"two!!"))
        elif kind == "duplicate":
            # The same relative path appears twice in the archive.
            a = tarfile.TarInfo("python/dup.txt")
            a.size = 3
            package.addfile(a, io.BytesIO(b"AAA"))
            b = tarfile.TarInfo("python/dup.txt")
            b.size = 3
            package.addfile(b, io.BytesIO(b"BBB"))
        else:
            raise ValueError(kind)
    return path


def fake_identity(archive_path: Path) -> br.PythonAsset:
    return br.PythonAsset(
        project="test/fake-python",
        implementation="CPython",
        version="3.12.11",
        release="20250828",
        target_triple="x86_64-pc-windows-msvc",
        cpu_baseline="x86_64",
        flavor="install_only_stripped",
        filename=archive_path.name,
        source_url="https://fake.invalid/python/fake-python.tar.gz",
        sha256=sha256_file(archive_path),
    )


def fake_inventory(identity: br.PythonAsset) -> br.PythonLicenseInventory:
    return br.PythonLicenseInventory(
        schema=br.LICENSE_INVENTORY_SCHEMA,
        inventory_sha256=INVENTORY_SHA,
        entries=tuple(
            br.LicenseEntry(
                archive_path=archive_path,
                installed_path=archive_path[len("python/") :],
                sha256=sha256_bytes(data),
                source_archive_filename=identity.filename,
                source_archive_sha256=identity.sha256,
            )
            for archive_path, data in LICENSE_FIXTURES
        ),
    )


def make_fake_wheel(path: Path, name: str, version: str) -> Path:
    data = f"FAKE-WHEEL {name} {version}".encode("ascii")
    path.write_bytes(data)
    return path


def wheels_for_variant(
    directory: Path,
    variant: str,
    torch_backend: str,
    *,
    make_artifact=None,
) -> tuple[list[br.WheelArtifact], dict[str, Path]]:
    """Create fake wheel files and their artifacts for one synthetic variant."""
    specs = [("alpha", "1.0.0", "https://fake.invalid/wheels/alpha-1.0.0-py3-none-any.whl")]
    if variant.endswith("-gui"):
        specs += [
            ("pyqt5", "5.15.11", "https://fake.invalid/wheels/PyQt5-5.15.11-py3-none-win_amd64.whl"),
            ("pyqt5-qt5", "5.15.11", "https://fake.invalid/wheels/PyQt5-Qt5-5.15.11-py3-none-win_amd64.whl"),
            ("pyqt5-sip", "12.15.11", "https://fake.invalid/wheels/PyQt5_sip-12.15.11-cp312-cp312-win_amd64.whl"),
        ]
    torch_version = "2.14.0+cpu" if torch_backend == "cpu" else f"2.14.0+{torch_backend}"
    specs.append(
        (
            "torch",
            torch_version,
            f"https://fake.invalid/wheels/torch-{torch_version}-cp312-cp312-win_amd64.whl",
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = []
    paths = {}
    for name, version, url in specs:
        filename = f"torch-{torch_version}-cp312-cp312-win_amd64.whl" if name == "torch" else f"{name}-{version}-fake.whl"
        wheel_path = directory / filename
        make_fake_wheel(wheel_path, name, version)
        paths[url] = wheel_path
        artifacts.append(
            br.WheelArtifact(
                name=name,
                version=version,
                classification="direct" if name == "torch" else "transitive",
                filename=filename,
                url=url,
                sha256=sha256_file(wheel_path),
                source="https://fake.invalid/simple",
                compatibility_tags=("cp312", "win_amd64"),
                input_identity_sha256=INPUT_IDENTITY_SHA,
                resolver="uv 0.12.18",
                resolver_config_sha256=RESOLVER_CONFIG_SHA,
            )
        )
    return artifacts, paths


def make_lock(
    variant: str,
    torch_backend: str,
    wheels: tuple[br.WheelArtifact, ...],
    lock_sha256: str = LOCK_SHA,
) -> br.Lock:
    return br.Lock(
        variant=variant,
        schema=br.LOCK_SCHEMA,
        target_python="3.12.11",
        target_platform="Windows",
        wheel_platform="win_amd64",
        resolver="uv 0.12.18",
        resolver_config_sha256=RESOLVER_CONFIG_SHA,
        generator_sha256="22" * 32,
        lock_filename=f"requirements-lock-{variant}.txt",
        lock_sha256=lock_sha256,
        input_hashes={"requirements-runtime-common.txt": INPUT_HASH_SHA},
        input_identity_sha256=INPUT_IDENTITY_SHA,
        torch_backend=torch_backend,
        gui=variant.endswith("-gui"),
        wheels=tuple(wheels),
    )


class FakeOpener:
    """Stands in for urllib: url -> in-memory byte stream; records calls."""

    def __init__(self, mapping: dict[str, bytes], fail_urls: set[str] | None = None) -> None:
        self.mapping = mapping
        self.fail_urls = fail_urls or set()
        self.calls: list[str] = []

    def __call__(self, url: str) -> io.BytesIO:
        self.calls.append(url)
        if url in self.fail_urls:
            raise OSError("network down (fake)")
        return io.BytesIO(self.mapping[url])


class LyingFetcher:
    """Fetcher that skips its own hash check (defense-in-depth target)."""

    network_enabled = True

    def __init__(self, python_path: Path, wheel_paths: dict[str, Path]) -> None:
        self.python_path = python_path
        self.wheel_paths = wheel_paths

    def fetch(self, spec: br.ArtifactSpec) -> Path:
        if spec.kind == "python":
            return self.python_path
        return self.wheel_paths[spec.url]


class FakeInstaller:
    """Records the no-resolution install call and materializes a
    RECORD-faithful installed tree.

    Per wheel: package Python source, a native ``.pyd`` and ``.dll``
    payload, a ``Scripts/`` console launcher (non-deterministic in reality;
    the builder excludes launcher lines from the canonical RECORD identity
    but byte-verifies them via the on-disk RECORD), the dist-info
    (METADATA with Name+Version, RECORD, direct_url.json provenance
    pointer), and — for the exact reviewed scipy 1.18.1 defect — the
    zero-byte self-copy file at the site-packages root. Every RECORD entry
    carries a real urlsafe-base64 SHA-256 and size, so the builder's
    byte-level RECORD integrity check is meaningful against this tree.
    """

    def __init__(
        self,
        wheels: list[br.WheelArtifact],
        plant_prohibited: str | None = None,
        plant_self_copy: bool = False,
    ) -> None:
        self.wheels = wheels
        self.plant_prohibited = plant_prohibited
        self.plant_self_copy = plant_self_copy
        self.calls: list[tuple[str, Path, Path, Path]] = []
        self.requirements_texts: list[str] = []

    def install(self, python_exe: Path, requirements_file: Path, cwd: Path, cache_dir: Path) -> None:
        self.calls.append((str(python_exe), Path(requirements_file), cwd, cache_dir))
        self.requirements_texts.append(Path(requirements_file).read_text(encoding="utf-8"))
        if self.plant_prohibited:
            plant = cwd / self.plant_prohibited
            plant.parent.mkdir(parents=True, exist_ok=True)
            plant.write_bytes(b"prohibited")
            return
        site = cwd / "Lib" / "site-packages"
        scripts = cwd / "Scripts"
        site.mkdir(parents=True, exist_ok=True)
        scripts.mkdir(parents=True, exist_ok=True)
        for wheel in self.wheels:
            dist_name = f"{wheel.name.replace('-', '_')}-{wheel.version}"
            dist_dir = site / f"{dist_name}.dist-info"
            dist_dir.mkdir(parents=True, exist_ok=True)
            pkg_dir = site / wheel.name
            pkg_dir.mkdir(parents=True, exist_ok=True)
            source = f"# fake package {wheel.name}\n".encode("ascii")
            pyd = f"FAKE-PYD {wheel.name}\n".encode("ascii")
            dll = f"FAKE-DLL {wheel.name}\n".encode("ascii")
            metadata = (
                f"Metadata-Version: 2.1\nName: {wheel.name}\nVersion: {wheel.version}\n"
            ).encode("ascii")
            launcher_name = f"{wheel.name}.exe"
            launcher_data = f"FAKE-LAUNCHER {wheel.name}\n".encode("ascii")
            (pkg_dir / "__init__.py").write_bytes(source)
            (pkg_dir / f"{wheel.name}_core.pyd").write_bytes(pyd)
            (pkg_dir / f"{wheel.name}_native.dll").write_bytes(dll)
            (dist_dir / "METADATA").write_bytes(metadata)
            (scripts / launcher_name).write_bytes(launcher_data)
            direct_url_data = (
                f'{{"url": "file:///cache/wheels/{wheel.filename}"}}\n'
            ).encode("ascii")
            (dist_dir / "direct_url.json").write_bytes(direct_url_data)
            record_lines = [
                f"{wheel.name}/__init__.py,sha256={record_b64(source)},{len(source)}",
                f"{wheel.name}/{wheel.name}_core.pyd,sha256={record_b64(pyd)},{len(pyd)}",
                f"{wheel.name}/{wheel.name}_native.dll,sha256={record_b64(dll)},{len(dll)}",
                f"{dist_dir.name}/METADATA,sha256={record_b64(metadata)},{len(metadata)}",
                f"{dist_dir.name}/direct_url.json,sha256={record_b64(direct_url_data)},{len(direct_url_data)}",
                f"../../Scripts/{launcher_name},sha256={record_b64(launcher_data)},{len(launcher_data)}",
                f"{dist_dir.name}/RECORD,,",
            ]
            if self.plant_self_copy and wheel.filename == br.SCIPY_SELF_COPY["filename"]:
                # The exact upstream defect: the scipy 1.18.1 win_amd64 wheel
                # bundles a zero-byte file named after the wheel itself and
                # pip installs it at the site-packages root. Recorded in the
                # RECORD; recognized (not deleted) by the builder only when
                # every SCIPY_SELF_COPY invariant holds.
                (site / wheel.filename).write_bytes(b"")
                record_lines.append(
                    f"{wheel.filename},sha256={record_b64(b'')},0"
                )
            (dist_dir / "RECORD").write_bytes(("\n".join(record_lines) + "\n").encode("ascii"))


class FakeInterpreter:
    """Scripted stand-in for the assembled portable Python probe runner."""

    def __init__(
        self,
        lock: br.Lock,
        *,
        installed_override: dict[str, list[str]] | None = None,
        imports_failure: dict[str, str] | None = None,
        executable_override: str | None = None,
        isolated: bool = True,
        dont_write_bytecode: bool = True,
        selftest_ok: bool | None = None,
        selftest_executable_override: str | None = None,
        torch: dict | None = None,
        torch_missing: bool = False,
    ) -> None:
        self.lock = lock
        self.installed_override = installed_override
        self.imports_failure = imports_failure or {}
        self.executable_override = executable_override
        self.isolated = isolated
        self.dont_write_bytecode = dont_write_bytecode
        self.selftest_ok = True if selftest_ok is None else selftest_ok
        self.selftest_executable_override = selftest_executable_override
        self.torch = torch
        self.torch_missing = torch_missing
        self.calls: list[tuple[str, tuple, object]] = []
        self._distribution_calls = 0

    def run(self, python_exe, *args, cwd=None, timeout=None):
        self.calls.append((str(python_exe), tuple(args), cwd))
        if len(args) >= 2 and args[0] == "-c":
            script = args[1]
            if "importlib.metadata" in script:
                return self._distributions(str(python_exe), cwd)
            if "torch.version.cuda" in script:
                return self._torch()
            if script.startswith("import json, sys"):
                return self._imports(str(python_exe), script, cwd)
        if args and str(args[0]).endswith("runtime_entry.py") and "--self-test" in args:
            return self._selftest(str(python_exe))
        raise AssertionError(f"unexpected interpreter invocation: {args!r}")

    def _distributions(self, python_exe: str, cwd) -> types.SimpleNamespace:
        self._distribution_calls += 1
        if self._distribution_calls == 1:
            # Baseline probe: the pinned archive bundles pip; mirror the
            # real probe, which also reports each distribution's dist-info
            # location (the builder uses it to capture the pre-install
            # baseline authority from the verified archive's extraction).
            raw = {"pip": ["24.3.1"]}
            if cwd is not None:
                pip_dist = Path(cwd) / "Lib" / "site-packages" / "pip-24.3.1.dist-info"
                raw["pip"] = ["24.3.1", "@@DISTINFO@@" + str(pip_dist)]
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(raw))
        if self.installed_override is not None:
            raw = self.installed_override
        else:
            raw = {"pip": ["24.3.1"]}
            for wheel in self.lock.wheels:
                location = cwd / "Lib" / "site-packages" / f"{wheel.name.replace('-', '_')}-{wheel.version}.dist-info"
                raw[wheel.name] = [wheel.version, "@@DISTINFO@@" + str(location)]
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(raw))

    def _imports(self, python_exe: str, script: str, cwd) -> types.SimpleNamespace:
        modules = ast.literal_eval(re.search(r"modules = (\[[^\]]*\])", script).group(1))
        report = {
            "executable": self.executable_override or python_exe,
            "version": "3.12.11",
            "isolated": self.isolated,
            "dont_write_bytecode": self.dont_write_bytecode,
            "modules": {m: self.imports_failure.get(m, "ok") for m in modules},
        }
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(report))

    def _selftest(self, python_exe: str) -> types.SimpleNamespace:
        report = {
            "mode": "self-test",
            "ok": self.selftest_ok,
            "python_version": "3.12.11",
            "implementation": "cpython",
            "executable": self.selftest_executable_override or python_exe,
            "isolated": self.isolated,
            "dont_write_bytecode": self.dont_write_bytecode,
        }
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(report))

    def _torch(self) -> types.SimpleNamespace:
        if self.torch_missing:
            return types.SimpleNamespace(returncode=0, stdout=json.dumps({"imported": False, "import_error": "ModuleNotFoundError: torch"}))
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(self.torch))


@pytest.fixture
def env(tmp_path):
    """Synthetic build environment: fake Python archive + fake wheels."""
    archive = make_python_archive(tmp_path / "fake-python.tar.gz")
    identity = fake_identity(archive)
    wheels_dir = tmp_path / "wheel-store"
    return types.SimpleNamespace(
        tmp_path=tmp_path,
        archive=archive,
        identity=identity,
        inventory=fake_inventory(identity),
        wheels_dir=wheels_dir,
    )


def build_artifacts(env, variant: str, torch_backend: str) -> tuple[list[br.WheelArtifact], dict[str, Path], br.Lock]:
    artifacts, paths = wheels_for_variant(env.wheels_dir, variant, torch_backend)
    lock = make_lock(variant, torch_backend, tuple(artifacts))
    return artifacts, paths, lock


def online_fetcher(env, variant: str, paths: dict[str, Path]) -> br.OnlineFetcher:
    mapping = {env.identity.source_url: env.archive.read_bytes()}
    for url, wheel_path in paths.items():
        mapping[url] = wheel_path.read_bytes()
    return br.OnlineFetcher(env.tmp_path / "cache", variant, opener=FakeOpener(mapping))


def fake_contract_bytes(env, lock: br.Lock) -> dict[str, bytes]:
    """The authoritative contract bytes for the fake Commit-1 world: the
    on-disk fake contract files (materialized by make_fake_lock_files).
    Offline builds pass these so the layout is anchored to them instead of
    (and never to) the layout's own copies."""
    make_fake_lock_files(env, lock.variant, lock)
    contract = env.tmp_path / "contract-files"
    return {name: (contract / name).read_bytes() for name in br._offline_lock_filenames(lock.variant)}


def make_variant_loader(mapping: dict[str, br.Lock]):
    """Test double for the variant -> current Commit-1 lock loader that
    perform_activation uses to fully verify the target AND the retained
    runtime (which may be a different variant)."""

    def load(variant: str) -> br.Lock:
        if variant not in mapping:
            raise br.BuilderError(
                "state", f"lock loader (test double) has no current lock for variant {variant!r}"
            )
        return mapping[variant]

    return load


def make_context(
    env,
    lock: br.Lock,
    *,
    runtime_root,
    fetcher,
    installer,
    interpreter,
    activate: bool = False,
    offline_contract_bytes: dict[str, bytes] | None = None,
    lock_loader=None,
) -> br.BuildContext:
    if lock_loader is None:
        lock_loader = make_variant_loader({lock.variant: lock})
    return br.BuildContext(
        repo_root=ROOT,
        runtime_root=runtime_root,
        identity=env.identity,
        inventory=env.inventory,
        lock=lock,
        bootstrap_sha256=BOOTSTRAP_SHA,
        fetcher=fetcher,
        installer=installer,
        interpreter=interpreter,
        activate=activate,
        lock_timeout=30.0,
        log=lambda _message: None,
        offline_contract_bytes=offline_contract_bytes,
        lock_loader=lock_loader,
    )


def scipy_wheel_artifact(directory: Path) -> tuple[br.WheelArtifact, Path]:
    """A fake scipy 1.18.1 wheel carrying the EXACT upstream filename — the
    wheel that bundles the zero-byte self-copy entry pip installs into
    Lib/site-packages/."""
    directory.mkdir(parents=True, exist_ok=True)
    wheel_path = directory / br.SCIPY_SELF_COPY["filename"]
    make_fake_wheel(wheel_path, "scipy", "1.18.1")
    artifact = br.WheelArtifact(
        name="scipy",
        version="1.18.1",
        classification="transitive",
        filename=br.SCIPY_SELF_COPY["filename"],
        url=f"https://fake.invalid/wheels/{br.SCIPY_SELF_COPY['filename']}",
        sha256=sha256_file(wheel_path),
        source="https://fake.invalid/simple",
        compatibility_tags=("cp312", "win_amd64"),
        input_identity_sha256=INPUT_IDENTITY_SHA,
        resolver="uv 0.12.18",
        resolver_config_sha256=RESOLVER_CONFIG_SHA,
    )
    return artifact, wheel_path


def make_fake_lock_files(env, variant: str, lock: br.Lock) -> list[Path]:
    """Materialize the Commit-1 contract files on disk. The variant lock
    file's canonical SHA-256 is what the test lock records as
    ``lock_sha256`` (locks are built from these files), so the strict
    offline contract's exact-hash lock check is satisfiable."""
    contract = env.tmp_path / "contract-files"
    contract.mkdir(parents=True, exist_ok=True)
    (contract / "runtime-lock.json").write_bytes(b"{}\n")
    (contract / "runtime-python.json").write_bytes(b"{}\n")
    (contract / "runtime-python-licenses.json").write_bytes(b"{}\n")
    req = contract / lock.lock_filename
    req.write_text(f"# fake offline lock for {variant}\n", encoding="utf-8")
    return [
        contract / "runtime-lock.json",
        contract / "runtime-python.json",
        contract / "runtime-python-licenses.json",
        req,
    ]


def offline_build_artifacts(
    env, variant: str, torch_backend: str
) -> tuple[list[br.WheelArtifact], dict[str, Path], br.Lock]:
    """Like build_artifacts, but the lock's SHA-256 equals the canonical
    hash of the on-disk fake lock file, so a strict offline-inputs layout
    (which must carry a byte-exact lock copy) validates."""
    artifacts, paths = wheels_for_variant(env.wheels_dir, variant, torch_backend)
    lock = make_lock(variant, torch_backend, tuple(artifacts))
    lock_files = make_fake_lock_files(env, variant, lock)
    lock = br.Lock(
        **{**lock.__dict__, "lock_sha256": br.locklib.sha256_file_text_canonical(lock_files[-1])}
    )
    return artifacts, paths, lock


def prepared_offline_inputs(env, variant: str, lock: br.Lock, paths: dict[str, Path]) -> Path:
    """A COMPLETE, strict offline-inputs layout — exactly what
    fetch-offline-inputs produces (python/, wheels/<variant>/, licenses/,
    locks/, artifact-manifest.json), generated and self-validated by the
    builder's own writer."""
    offline = env.tmp_path / "offline-inputs" / variant
    if offline.exists():
        shutil.rmtree(offline)
    lock_files = make_fake_lock_files(env, variant, lock)
    fetcher = LyingFetcher(env.archive, paths)
    br.write_offline_inputs(
        offline, variant, env.identity, env.inventory, lock, fetcher, lock_files,
        log=lambda _message: None,
    )
    return offline


# ---------------------------------------------------------------------------
# Runtime identity
# ---------------------------------------------------------------------------


def test_runtime_id_grammar_and_determinism(env):
    _, _, lock = build_artifacts(env, "cpu-nogui", "cpu")
    runtime_id = br.compute_runtime_id(BOOTSTRAP_SHA, env.identity, env.inventory, lock, TREE_SHA)
    assert re.fullmatch(br.RUNTIME_ID_RE.pattern, runtime_id)
    _, _, lock_again = build_artifacts(env, "cpu-nogui", "cpu")
    assert br.compute_runtime_id(BOOTSTRAP_SHA, env.identity, env.inventory, lock_again, TREE_SHA) == runtime_id


@pytest.mark.parametrize(
    "mutate",
    ["variant", "python_sha", "inventory_sha", "bootstrap_sha", "lock_sha", "lock_inputs", "tree_sha"],
)
def test_runtime_id_sensitive_to_canonical_inputs(env, mutate):
    _, _, lock = build_artifacts(env, "cpu-nogui", "cpu")
    baseline = br.compute_runtime_id(BOOTSTRAP_SHA, env.identity, env.inventory, lock, TREE_SHA)
    if mutate == "variant":
        _, _, other = build_artifacts(env, "cuda-nogui", "cu130")
        assert br.compute_runtime_id(BOOTSTRAP_SHA, env.identity, env.inventory, other, TREE_SHA) != baseline
    elif mutate == "python_sha":
        identity = br.PythonAsset(**{**env.identity.__dict__, "sha256": "00" * 32})
        assert br.compute_runtime_id(BOOTSTRAP_SHA, identity, env.inventory, lock, TREE_SHA) != baseline
    elif mutate == "inventory_sha":
        inventory = br.PythonLicenseInventory(
            schema=env.inventory.schema,
            inventory_sha256="99" * 32,
            entries=env.inventory.entries,
        )
        assert br.compute_runtime_id(BOOTSTRAP_SHA, env.identity, inventory, lock, TREE_SHA) != baseline
    elif mutate == "bootstrap_sha":
        assert br.compute_runtime_id("77" * 32, env.identity, env.inventory, lock, TREE_SHA) != baseline
    elif mutate == "lock_sha":
        other = br.Lock(**{**lock.__dict__, "lock_sha256": "55" * 32})
        assert br.compute_runtime_id(BOOTSTRAP_SHA, env.identity, env.inventory, other, TREE_SHA) != baseline
    elif mutate == "lock_inputs":
        other = br.Lock(
            **{
                **lock.__dict__,
                "input_hashes": {**lock.input_hashes, "requirements-extra.txt": "66" * 32},
            }
        )
        assert br.compute_runtime_id(BOOTSTRAP_SHA, env.identity, env.inventory, other, TREE_SHA) != baseline
    elif mutate == "tree_sha":
        # The runtime ID binds the installed-tree integrity digest: a
        # different tree (different actual bytes) yields a different ID.
        assert br.compute_runtime_id(BOOTSTRAP_SHA, env.identity, env.inventory, lock, "77" * 32) != baseline


def test_runtime_id_recomputes_from_manifest(env, tmp_path):
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    runtime_root = tmp_path / "runtime"
    context = make_context(
        env, lock,
        runtime_root=runtime_root,
        fetcher=online_fetcher(env, "cpu-nogui", paths),
        installer=FakeInstaller(artifacts),
        interpreter=FakeInterpreter(lock),
    )
    result = br.run_build(context)
    manifest = br.read_manifest(result.runtime_dir / br.MANIFEST_NAME)
    assert br.recompute_runtime_id_from_manifest(manifest["canonical"]) == result.runtime_id
    assert br.recompute_runtime_id_from_manifest(manifest["canonical"]) == manifest["canonical"]["runtime_id"]


# ---------------------------------------------------------------------------
# Commit-1 contract loading (real repository files, staleness detection)
# ---------------------------------------------------------------------------


def _copy_real_contract(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in (
        "runtime-python.json",
        "runtime-python-licenses.json",
        "runtime-lock.json",
        "requirements-lock-cpu-nogui.txt",
        "requirements-cpu.txt",
        "requirements-runtime-common.txt",
    ):
        shutil.copy2(ROOT / name, repo / name)
    shutil.copy2(ROOT / "scripts" / "generate_runtime_locks.py", repo / "scripts" / "generate_runtime_locks.py")
    shutil.copy2(ROOT / "scripts" / "runtime_entry.py", repo / "scripts" / "runtime_entry.py")
    return repo


def _point_generator_at(monkeypatch: pytest.MonkeyPatch, repo: Path) -> None:
    """The generator resolves requirement inputs against its own module ROOT
    (the real repository). For a copied contract in a temp repo, retarget it
    so input-hash freshness is measured against the copy under test."""
    monkeypatch.setattr(br.locklib, "ROOT", repo)


def test_real_contract_loads_and_matches_pins(tmp_path, monkeypatch):
    repo = _copy_real_contract(tmp_path)
    _point_generator_at(monkeypatch, repo)
    identity, inventory, lock, bootstrap = br.load_inputs(repo, "cpu-nogui")
    assert identity == br.PythonAsset(**br.PINNED_PYTHON)
    assert lock.variant == "cpu-nogui"
    assert lock.torch_backend == "cpu"
    assert lock.gui is False
    assert len(lock.wheels) > 10
    assert bootstrap == br.locklib.sha256_file_text_canonical(repo / "scripts" / "runtime_entry.py")


def test_stale_direct_input_is_rejected(tmp_path, monkeypatch):
    repo = _copy_real_contract(tmp_path)
    _point_generator_at(monkeypatch, repo)
    input_file = repo / "requirements-cpu.txt"
    text = input_file.read_text(encoding="utf-8")
    input_file.write_text(text + "# tampered\n", encoding="utf-8")
    with pytest.raises(br.BuilderError) as excinfo:
        br.load_inputs(repo, "cpu-nogui")
    assert excinfo.value.code == "lock"
    assert "stale direct input" in excinfo.value.message


def test_tampered_python_identity_is_rejected(tmp_path):
    repo = _copy_real_contract(tmp_path)
    data = json.loads((repo / "runtime-python.json").read_text(encoding="utf-8"))
    data["archive"]["sha256"] = "00" * 32
    (repo / "runtime-python.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(br.BuilderError) as excinfo:
        br.load_python_identity(repo)
    assert excinfo.value.code == "identity"


def test_tampered_license_inventory_source_is_rejected(tmp_path):
    repo = _copy_real_contract(tmp_path)
    data = json.loads((repo / "runtime-python-licenses.json").read_text(encoding="utf-8"))
    data["source_archive"]["release"] = "19990101"
    (repo / "runtime-python-licenses.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    identity = br.load_python_identity(repo)
    with pytest.raises(br.BuilderError) as excinfo:
        br.load_license_inventory(repo, identity)
    assert excinfo.value.code == "identity"


# ---------------------------------------------------------------------------
# Safe extraction
# ---------------------------------------------------------------------------


def test_safe_extraction_happy_path(env, tmp_path):
    staging = tmp_path / "stage"
    staging.mkdir()
    br.extract_python_archive(env.archive, staging)
    assert (staging / "python.exe").read_bytes() == b"FAKE-PYTHON-EXE"
    assert (staging / "LICENSE.txt").read_bytes() == b"ROOT-LICENSE"
    assert (staging / "Lib" / "site-packages" / "pip-24.3.1.dist-info" / "LICENSE.txt").read_bytes() == b"PIP-LICENSE"
    assert (staging / "tcl" / "tk8.6" / "license.terms").read_bytes() == b"TK-TERMS"
    assert (staging / "tcl" / "tk8.6" / "demos" / "license.terms").read_bytes() == b"TK-TERMS"
    assert int(os.stat(staging / "LICENSE.txt").st_mtime) == 1700000001


@pytest.mark.parametrize(
    ("kind", "needle"),
    [
        ("traversal", "path-traversal"),
        ("absolute", "absolute archive member"),
        ("outside", "outside"),
        ("symlink", "link members"),
        ("hardlink", "link members"),
        ("drive", "drive-qualified"),
        ("ads", "ADS-style colon"),
        ("devicename", "device-name hazard"),
        ("devicename-middle", "device-name hazard"),
        ("backslash", "backslash"),
        ("case", "case-insensitive"),
        ("duplicate", "duplicate archive member"),
        ("duplicate-root", "duplicate archive member"),
    ],
)
def test_safe_extraction_rejects_hostile_members(env, tmp_path, kind, needle):
    archive = make_hostile_archive(tmp_path / "hostile.tar.gz", kind)
    staging = tmp_path / "stage"
    staging.mkdir()
    with pytest.raises(br.BuilderError) as excinfo:
        br.extract_python_archive(archive, staging)
    assert excinfo.value.code == "archive-unsafe"
    assert needle in excinfo.value.message
    # Nothing may have been written: validation precedes any write.
    assert not (staging / "python.exe").exists()
    assert not (staging / "python").exists()


def test_safe_extraction_rejects_relative_destination(env):
    with pytest.raises(br.BuilderError) as excinfo:
        br.extract_python_archive(env.archive, Path("relative-stage"))
    assert excinfo.value.code == "archive-unsafe"


def test_safe_extraction_requires_existing_destination(env, tmp_path):
    with pytest.raises(br.BuilderError) as excinfo:
        br.extract_python_archive(env.archive, tmp_path / "never-created")
    assert excinfo.value.code == "archive-unsafe"
    assert "existing directory" in excinfo.value.message


def test_safe_extraction_requires_freshly_owned_empty_destination(env, tmp_path):
    staging = tmp_path / "stage"
    staging.mkdir()
    (staging / "leftover.bin").write_bytes(b"stale")
    with pytest.raises(br.BuilderError) as excinfo:
        br.extract_python_archive(env.archive, staging)
    assert excinfo.value.code == "archive-unsafe"
    assert "freshly owned empty" in excinfo.value.message
    # the pre-existing file is untouched (no overwrite, no partial write)
    assert (staging / "leftover.bin").read_bytes() == b"stale"
    assert not (staging / "python.exe").exists()


# ---------------------------------------------------------------------------
# Python license tree verification
# ---------------------------------------------------------------------------


def _extracted_staging(env, tmp_path) -> Path:
    staging = tmp_path / "stage"
    staging.mkdir()
    br.extract_python_archive(env.archive, staging)
    return staging


def test_license_tree_pass(env, tmp_path):
    staging = _extracted_staging(env, tmp_path)
    checks = br.check_python_license_tree(staging, env.inventory, env.identity)
    assert checks[0].code == "PASS"
    assert str(len(env.inventory.entries)) in checks[0].note


def test_license_tree_missing_entry(env, tmp_path):
    staging = _extracted_staging(env, tmp_path)
    (staging / "LICENSE.txt").unlink()
    checks = br.check_python_license_tree(staging, env.inventory, env.identity)
    assert checks[0].code == "MISSING_ENTRY"


def test_license_tree_hash_mismatch(env, tmp_path):
    staging = _extracted_staging(env, tmp_path)
    (staging / "LICENSE.txt").write_bytes(b"corrupted")
    checks = br.check_python_license_tree(staging, env.inventory, env.identity)
    assert checks[0].code == "HASH_MISMATCH"


def test_license_tree_extra_entry_rejected(env, tmp_path):
    staging = _extracted_staging(env, tmp_path)
    (staging / "LICENSE-extra.txt").write_bytes(b"extra")
    checks = br.check_python_license_tree(staging, env.inventory, env.identity)
    assert checks[0].code == "EXTRA_ENTRY"


def test_license_tree_source_mismatch(env, tmp_path):
    staging = _extracted_staging(env, tmp_path)
    inventory = br.PythonLicenseInventory(
        schema=env.inventory.schema,
        inventory_sha256=INVENTORY_SHA,
        entries=tuple(
            br.LicenseEntry(
                archive_path=entry.archive_path,
                installed_path=entry.installed_path,
                sha256=entry.sha256,
                source_archive_filename=entry.source_archive_filename,
                source_archive_sha256="99" * 32,
            )
            for entry in env.inventory.entries
        ),
    )
    checks = br.check_python_license_tree(staging, inventory, env.identity)
    assert checks[0].code == "SOURCE_MISMATCH"


# ---------------------------------------------------------------------------
# Artifact fetchers (online cache / offline directory)
# ---------------------------------------------------------------------------


def test_online_fetcher_downloads_verifies_and_caches(env):
    opener = FakeOpener({env.identity.source_url: env.archive.read_bytes()})
    fetcher = br.OnlineFetcher(env.tmp_path / "cache", "cpu-nogui", opener=opener)
    spec = br.python_spec(env.identity)
    first = fetcher.fetch(spec)
    second = fetcher.fetch(spec)
    assert first == second
    assert opener.calls == [spec.url]  # second fetch served from verified cache
    assert br.locklib.sha256_file_raw(first) == spec.sha256


def test_online_fetcher_rejects_bad_download_hash(env):
    bad = env.archive.read_bytes() + b"tamper"
    fetcher = br.OnlineFetcher(env.tmp_path / "cache", "cpu-nogui", opener=FakeOpener({env.identity.source_url: bad}))
    with pytest.raises(br.BuilderError) as excinfo:
        fetcher.fetch(br.python_spec(env.identity))
    assert excinfo.value.code == "artifact-hash"
    assert not fetcher.cache_path(br.python_spec(env.identity)).exists()


def test_online_fetcher_refetches_corrupt_cache(env):
    opener = FakeOpener({env.identity.source_url: env.archive.read_bytes()})
    fetcher = br.OnlineFetcher(env.tmp_path / "cache", "cpu-nogui", opener=opener)
    spec = br.python_spec(env.identity)
    target = fetcher.cache_path(spec)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"corrupt-bytes")
    result = fetcher.fetch(spec)
    assert result == target
    assert opener.calls == [spec.url]
    assert br.locklib.sha256_file_raw(result) == spec.sha256


def test_online_fetcher_reports_network_failure_and_leaves_no_part(env):
    opener = FakeOpener({env.identity.source_url: env.archive.read_bytes()}, fail_urls={env.identity.source_url})
    fetcher = br.OnlineFetcher(env.tmp_path / "cache", "cpu-nogui", opener=opener)
    with pytest.raises(br.BuilderError) as excinfo:
        fetcher.fetch(br.python_spec(env.identity))
    assert excinfo.value.code == "artifact-fetch"
    parts = list((env.tmp_path / "cache").rglob("*.part"))
    assert parts == []


def test_offline_fetcher_serves_verified_local_artifacts(env):
    offline = env.tmp_path / "offline"
    (offline / "python").mkdir(parents=True)
    shutil.copy2(env.archive, offline / "python" / env.identity.filename)
    fetcher = br.OfflineFetcher(offline, "cpu-nogui")
    assert fetcher.network_enabled is False
    spec = br.python_spec(env.identity)
    result = fetcher.fetch(spec)
    assert result == offline / "python" / env.identity.filename


def test_offline_fetcher_missing_artifact_fails(env):
    offline = env.tmp_path / "offline"
    offline.mkdir(parents=True)
    fetcher = br.OfflineFetcher(offline, "cpu-nogui")
    with pytest.raises(br.BuilderError) as excinfo:
        fetcher.fetch(br.python_spec(env.identity))
    assert excinfo.value.code == "artifact-missing"


def test_offline_fetcher_hash_mismatch_fails(env):
    offline = env.tmp_path / "offline"
    (offline / "python").mkdir(parents=True)
    (offline / "python" / env.identity.filename).write_bytes(b"wrong-bytes")
    fetcher = br.OfflineFetcher(offline, "cpu-nogui")
    with pytest.raises(br.BuilderError) as excinfo:
        fetcher.fetch(br.python_spec(env.identity))
    assert excinfo.value.code == "artifact-hash"


def test_offline_build_never_reaches_network(env, tmp_path):
    """Offline build completes with a fetcher that has no network capability;
    the strict offline layout is validated before staging exists."""
    artifacts, paths, lock = offline_build_artifacts(env, "cpu-nogui", "cpu")
    offline = prepared_offline_inputs(env, "cpu-nogui", lock, paths)
    runtime_root = tmp_path / "runtime"
    context = make_context(
        env, lock,
        runtime_root=runtime_root,
        fetcher=br.OfflineFetcher(offline, "cpu-nogui"),
        installer=FakeInstaller(artifacts),
        interpreter=FakeInterpreter(lock),
        offline_contract_bytes=fake_contract_bytes(env, lock),
    )
    result = br.run_build(context)
    assert result.state == "BUILT"
    assert context.fetcher.network_enabled is False
    assert not (runtime_root / br.ACTIVE_POINTER_NAME).exists()


# ---------------------------------------------------------------------------
# Post-install verification helpers
# ---------------------------------------------------------------------------


def test_verify_package_set_exact_and_mismatched(env):
    _, _, lock = build_artifacts(env, "cpu-nogui", "cpu")
    baseline = {"pip": {"24.3.1"}}
    installed = dict(baseline)
    installed.update({w.name: {w.version} for w in lock.wheels})
    checks = br.verify_package_set(installed, baseline, lock)
    assert checks[0].code == "PASS"

    missing = {k: v for k, v in installed.items() if k != "alpha"}
    checks = br.verify_package_set(missing, baseline, lock)
    assert checks[0].code == "MISMATCH"
    assert "alpha" in checks[0].note

    wrong = {**installed, "alpha": {"9.9.9"}}
    assert br.verify_package_set(wrong, baseline, lock)[0].code == "MISMATCH"

    extra = {**installed, "rogue": {"1.0"}}
    assert br.verify_package_set(extra, baseline, lock)[0].code == "MISMATCH"


def test_variant_contract_rules(env):
    _, _, cpu_lock = build_artifacts(env, "cpu-nogui", "cpu")
    installed = {w.name: {w.version} for w in cpu_lock.wheels}
    assert br.verify_variant_contract(installed, cpu_lock)[0].code == "PASS"

    gui_artifacts, _, gui_lock = build_artifacts(env, "cpu-gui", "cpu")
    gui_installed = {w.name: {w.version} for w in gui_artifacts}
    assert br.verify_variant_contract(gui_installed, gui_lock)[0].code == "PASS"
    no_pyqt = {k: v for k, v in gui_installed.items() if k != "pyqt5"}
    checks = br.verify_variant_contract(no_pyqt, gui_lock)
    assert checks[0].code == "MISMATCH"
    assert "pyqt5" in checks[0].note

    nogui_with_pyqt = {**installed, "pyqt5": {"5.15.11"}}
    checks = br.verify_variant_contract(nogui_with_pyqt, cpu_lock)
    assert checks[0].code == "MISMATCH"

    wrong_backend = {**installed, "torch": {"2.14.0+cu130"}}
    checks = br.verify_variant_contract(wrong_backend, cpu_lock)
    assert checks[0].code == "MISMATCH"
    assert "+cpu" in checks[0].note


def test_bytecode_scrub_and_prohibited_scan(tmp_path):
    tree = tmp_path / "rt"
    (tree / "Lib" / "__pycache__").mkdir(parents=True)
    (tree / "Lib" / "__pycache__" / "mod.pyc").write_bytes(b"x")
    (tree / "stray.pyc").write_bytes(b"x")
    (tree / "NVSTCache").mkdir()
    (tree / "Lib" / "site-packages").mkdir()
    (tree / "Lib" / "site-packages" / "leftover.whl").write_bytes(b"x")
    (tree / "Lib" / "site-packages" / "bundle.tar.gz").write_bytes(b"x")
    assert br.scan_prohibited(tree)
    removed = br.scrub_generated_bytecode(tree)
    # One .pyc inside __pycache__, the __pycache__ dir itself, and the stray .pyc.
    assert removed == 3
    assert list(tree.rglob("*.pyc")) == []
    assert not (tree / "Lib" / "__pycache__").exists()
    remaining = br.scan_prohibited(tree)
    assert "leftover.whl" in remaining
    assert "bundle.tar.gz" in remaining
    # case-insensitive directory match: NVSTCache (mixed case) is prohibited
    assert any(v.lower() == "directory nvstcache" for v in remaining)


# ---------------------------------------------------------------------------
# Privacy scanning
# ---------------------------------------------------------------------------


def test_privacy_scan_accepts_manifest_like_text():
    text = (
        "runtime id 0123456789abcdef\n"
        "url https://files.pythonhosted.org/packages/abc/wheel.whl\n"
        "url https://download-r2.pytorch.org/whl/cpu/torch-2.14.0%2Bcpu-cp312-cp312-win_amd64.whl\n"
        "version 2.14.0+cpu\n"
        "public endpoint 8.8.8.8\n"
    )
    assert br.privacy_violations(text) == []


@pytest.mark.parametrize(
    "text,label",
    [
        (r"built on C:\Users\dev\machine", "drive-letter path"),
        (r"cache under C:/Users/dev", "drive-letter path"),
        ("data in AppData\\Local\\cache", "AppData path"),
        ("user home /home/alice/project", "user home path"),
        ("files in /Users/bob/Downloads/x", "user home path"),
        ("saved to C:\\Users\\bob\\desktop\\x", "personal folder path"),
        ("venv at repo/.venv/Lib/site-packages", "venv path"),
        (r"share \\fileserver\share", "UNC share"),
        ("password=supersecret", "credential-like value"),
        ("-----BEGIN RSA PRIVATE KEY-----", "private key material"),
        ("bound to 10.0.0.5", "private IP"),
        ("bound to 192.168.1.20", "private IP"),
        ("bound to 172.20.10.4", "private IP"),
        ("loopback 127.0.0.1", "private IP"),
    ],
)
def test_privacy_scan_flags_violations(text, label):
    violations = br.privacy_violations(text)
    # IP labels carry the concrete address ("private IP 10.0.0.5");
    # all other labels are exact.
    assert any(v == label or (label == "private IP" and v.startswith("private IP ")) for v in violations)


def test_write_manifest_rejects_privacy_poison(tmp_path):
    manifest = {
        "canonical": {"variant": "cpu-nogui"},
        "operational": {"notes": [r"built on C:\Users\jane\dev machine 10.1.2.3"]},
        "schema": br.MANIFEST_SCHEMA,
    }
    with pytest.raises(br.BuilderError) as excinfo:
        br.write_manifest(tmp_path / "runtime-manifest.json", manifest)
    assert excinfo.value.code == "privacy"


def test_write_manifest_round_trips_clean_text(tmp_path):
    manifest = {
        "canonical": {"variant": "cpu-nogui", "runtime_id": "01" * 32},
        "operational": {"notes": ["clean note"], "built_utc": "2026-01-01T00:00:00Z"},
        "schema": br.MANIFEST_SCHEMA,
    }
    path = tmp_path / "runtime-manifest.json"
    br.write_manifest(path, manifest)
    assert br.read_manifest(path) == manifest


# ---------------------------------------------------------------------------
# Setup/activation lock and pointer
# ---------------------------------------------------------------------------


def test_setup_activation_lock_serializes_against_other_process(tmp_path):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(parents=True)
    ready = tmp_path / "ready"
    child = subprocess.Popen(
        [
            sys.executable, "-c", CHILD_HOLD_CODE,
            str(ROOT),
            str(runtime_root / br.ACTIVATION_LOCK_NAME),
            str(ready),
            "4",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists(), "lock-holder child did not signal ready"
        with pytest.raises(br.BuilderError) as excinfo:
            br.SetupActivationLock(runtime_root, 1.0).acquire()
        assert excinfo.value.code == "lock-busy"
    finally:
        child.wait(timeout=15)


def test_setup_activation_lock_acquire_release_cycle(tmp_path):
    runtime_root = tmp_path / "runtime"
    lock = br.SetupActivationLock(runtime_root, 5.0).acquire()
    # While the first lock is held, a second owner cannot acquire (same
    # semantics as the cross-process test on Windows).
    second = br.SetupActivationLock(runtime_root, 0.5)
    if os.name == "nt":
        with pytest.raises(br.BuilderError) as excinfo:
            second.acquire()
        assert excinfo.value.code == "lock-busy"
    lock.release()
    lock.acquire()
    lock.release()


def test_active_pointer_atomic_swap(tmp_path):
    first = "ab" * 32
    second = "cd" * 32
    previous = br.set_active_pointer(tmp_path, first)
    assert previous is None
    pointer = tmp_path / br.ACTIVE_POINTER_NAME
    assert pointer.read_text(encoding="utf-8").strip() == first
    previous = br.set_active_pointer(tmp_path, second)
    assert previous == first
    assert pointer.read_text(encoding="utf-8").strip() == second
    assert not (tmp_path / (br.ACTIVE_POINTER_NAME + ".tmp")).exists()
    with pytest.raises(br.BuilderError):
        br.set_active_pointer(tmp_path, "not-a-runtime-id")


# ---------------------------------------------------------------------------
# Full build orchestration (faked python/pip, real builder logic)
# ---------------------------------------------------------------------------


def _build(env, tmp_path, variant="cpu-nogui", torch_backend="cpu", **overrides):
    artifacts, paths, lock = build_artifacts(env, variant, torch_backend)
    runtime_root = tmp_path / "runtime"
    defaults = dict(
        runtime_root=runtime_root,
        fetcher=online_fetcher(env, variant, paths),
        installer=FakeInstaller(artifacts),
        interpreter=FakeInterpreter(lock),
        activate=False,
    )
    defaults.update(overrides)
    context = make_context(env, lock, **defaults)
    return br.run_build(context), context, lock, artifacts


def test_build_promotes_verified_runtime_and_cleans_staging(env, tmp_path):
    result, context, lock, artifacts = _build(env, tmp_path)
    assert result.state == "BUILT"
    # The runtime ID is the canonical hash of the CURRENT Commit-1 inputs
    # plus the installed-tree integrity digest recomputed from the PROMOTED
    # tree's actual bytes (not from the manifest).
    tree_digest = br.compute_installed_tree_digest(result.runtime_dir)
    assert result.runtime_id == br.compute_runtime_id(
        BOOTSTRAP_SHA, env.identity, env.inventory, lock, tree_digest
    )
    runtime_root = tmp_path / "runtime"
    version_dir = runtime_root / "versions" / result.runtime_id
    assert version_dir.is_dir()
    assert (version_dir / "python.exe").is_file()
    assert (version_dir / "LICENSE.txt").is_file()
    manifest = br.read_manifest(version_dir / br.MANIFEST_NAME)
    assert manifest["schema"] == br.MANIFEST_SCHEMA
    assert manifest["canonical"]["runtime_id"] == result.runtime_id
    assert manifest["canonical"]["variant"] == "cpu-nogui"
    assert manifest["canonical"]["lock"]["sha256"] == LOCK_SHA
    assert manifest["canonical"]["package_set"]["torch"] == "2.14.0+cpu"
    assert manifest["canonical"]["verification"]["imports"] == "PASS"
    assert manifest["canonical"]["verification"]["package_set"] == "PASS"
    # staging is clean after promotion; no pointer created without --activate
    staging = runtime_root / br.STAGING_DIRNAME
    assert list(staging.iterdir()) == []
    assert not (runtime_root / br.ACTIVE_POINTER_NAME).exists()
    # installer used the staged interpreter and the exact no-resolution flags
    python_exe, requirements_file, cwd, cache_dir = context.installer.calls[0]
    assert python_exe.endswith("python.exe")
    # The candidate is staged under a temporary name (the final runtime ID
    # is only known after the tree exists, because it binds the tree digest).
    assert cwd.name.startswith("candidate-")
    text = context.installer.requirements_texts[0]
    for wheel in lock.wheels:
        assert f"{wheel.name} @ " in text
        assert f"--hash=sha256:{wheel.sha256}" in text
    assert len(text.strip().splitlines()) == 2 * len(lock.wheels)
    # interpreter probes all ran under the staged python
    assert all(call[0].endswith("python.exe") for call in context.interpreter.calls)
    # operational namespace is separate from canonical
    assert "built_utc" in manifest["operational"]
    assert "built_utc" not in json.dumps(manifest["canonical"])


def test_rebuild_of_identical_runtime_reuses_verified_version(env, tmp_path):
    result, _, lock, _ = _build(env, tmp_path)
    version_dir = tmp_path / "runtime" / "versions" / result.runtime_id
    manifest_before = (version_dir / br.MANIFEST_NAME).read_bytes()
    result2, _, _, _ = _build(env, tmp_path)
    assert result2.state == "REUSED"
    assert result2.runtime_id == result.runtime_id
    assert (version_dir / br.MANIFEST_NAME).read_bytes() == manifest_before


def test_rebuild_of_mismatched_runtime_fails_without_modification(env, tmp_path):
    result, _, lock, _ = _build(env, tmp_path)
    version_dir = tmp_path / "runtime" / "versions" / result.runtime_id
    (version_dir / "python.exe").unlink()  # corrupt the retained runtime
    with pytest.raises(br.BuilderError) as excinfo:
        _build(env, tmp_path)
    assert excinfo.value.code == "state"
    assert version_dir.is_dir()  # never deleted or mutated


def test_build_with_stale_staging_cleans_it_before_build(env, tmp_path):
    runtime_root = tmp_path / "runtime"
    stale = runtime_root / br.STAGING_DIRNAME / "stale-candidate"
    stale.mkdir(parents=True)
    (stale / "junk.txt").write_bytes(b"stale")
    result, _, _, _ = _build(env, tmp_path)
    assert result.state == "BUILT"
    assert not stale.exists()


def test_build_activates_pointer_on_request(env, tmp_path):
    result, _, _, _ = _build(env, tmp_path, activate=True)
    pointer = tmp_path / "runtime" / br.ACTIVE_POINTER_NAME
    assert pointer.read_text(encoding="utf-8").strip() == result.runtime_id
    assert result.activated is True


def test_offline_build_matches_online_canonical_identity(env, tmp_path):
    artifacts, paths, lock = offline_build_artifacts(env, "cpu-nogui", "cpu")
    runtime_a = tmp_path / "runtime-a"
    online = br.run_build(
        make_context(
            env, lock,
            runtime_root=runtime_a,
            fetcher=online_fetcher(env, "cpu-nogui", paths),
            installer=FakeInstaller(artifacts),
            interpreter=FakeInterpreter(lock),
        )
    )
    offline = prepared_offline_inputs(env, "cpu-nogui", lock, paths)
    runtime_b = tmp_path / "runtime-b"
    offline_result = br.run_build(
        make_context(
            env, lock,
            runtime_root=runtime_b,
            fetcher=br.OfflineFetcher(offline, "cpu-nogui"),
            installer=FakeInstaller(artifacts),
            interpreter=FakeInterpreter(lock),
            offline_contract_bytes=fake_contract_bytes(env, lock),
        )
    )
    assert offline_result.runtime_id == online.runtime_id
    canonical_a = br.read_manifest(runtime_a / "versions" / online.runtime_id / br.MANIFEST_NAME)["canonical"]
    canonical_b = br.read_manifest(runtime_b / "versions" / offline_result.runtime_id / br.MANIFEST_NAME)["canonical"]
    assert canonical_a == canonical_b


def test_offline_missing_wheel_fails_before_promotion(env, tmp_path):
    artifacts, paths, lock = offline_build_artifacts(env, "cpu-nogui", "cpu")
    offline = prepared_offline_inputs(env, "cpu-nogui", lock, paths)
    victims = list((offline / "wheels" / "cpu-nogui").iterdir())
    victims[0].unlink()
    runtime_root = tmp_path / "runtime"
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=runtime_root,
                fetcher=br.OfflineFetcher(offline, "cpu-nogui"),
                installer=FakeInstaller(artifacts),
                interpreter=FakeInterpreter(lock),
                offline_contract_bytes=fake_contract_bytes(env, lock),
            )
        )
    assert excinfo.value.code == "artifact-missing"
    assert not (runtime_root / "versions").exists()


def test_offline_hash_mismatch_fails(env, tmp_path):
    artifacts, paths, lock = offline_build_artifacts(env, "cpu-nogui", "cpu")
    offline = prepared_offline_inputs(env, "cpu-nogui", lock, paths)
    victims = sorted((offline / "wheels" / "cpu-nogui").iterdir())
    victims[0].write_bytes(b"corrupted-wheel")
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=tmp_path / "runtime",
                fetcher=br.OfflineFetcher(offline, "cpu-nogui"),
                installer=FakeInstaller(artifacts),
                interpreter=FakeInterpreter(lock),
                offline_contract_bytes=fake_contract_bytes(env, lock),
            )
        )
    assert excinfo.value.code == "artifact-hash"


def test_pre_extraction_hash_reverification_catches_lying_fetcher(env, tmp_path):
    """Even if a fetcher lies, the builder re-verifies before extraction."""
    _, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    bad_archive = env.tmp_path / "corrupt-python.tar.gz"
    bad_archive.write_bytes(b"not the real archive")
    wheel_paths = {url: p for url, p in paths.items()}
    fetcher = LyingFetcher(bad_archive, wheel_paths)
    runtime_root = tmp_path / "runtime"
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=runtime_root,
                fetcher=fetcher,
                installer=FakeInstaller([]),
                interpreter=FakeInterpreter(lock),
            )
        )
    assert excinfo.value.code == "artifact-hash"
    assert not (runtime_root / "versions").exists()


def test_build_fails_on_package_set_mismatch(env, tmp_path):
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    runtime_root = tmp_path / "runtime"
    interpreter = FakeInterpreter(
        lock,
        installed_override={
            "pip": ["24.3.1"],
            "torch": ["2.14.0+cpu"],
        },
    )
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=runtime_root,
                fetcher=online_fetcher(env, "cpu-nogui", paths),
                installer=FakeInstaller(artifacts),
                interpreter=interpreter,
            )
        )
    assert excinfo.value.code == "package_set"
    assert not (runtime_root / "versions").exists()


def test_build_fails_on_module_import_failure(env, tmp_path):
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    interpreter = FakeInterpreter(lock, imports_failure={"cv2": "ImportError: fake failure"})
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=tmp_path / "runtime",
                fetcher=online_fetcher(env, "cpu-nogui", paths),
                installer=FakeInstaller(artifacts),
                interpreter=interpreter,
            )
        )
    assert excinfo.value.code == "imports"


def test_build_fails_when_executable_escapes_runtime(env, tmp_path):
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    interpreter = FakeInterpreter(lock, executable_override=str(env.tmp_path / "host-python.exe"))
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=tmp_path / "runtime",
                fetcher=online_fetcher(env, "cpu-nogui", paths),
                installer=FakeInstaller(artifacts),
                interpreter=interpreter,
            )
        )
    assert excinfo.value.code == "python_identity"


def test_build_fails_when_interpreter_not_isolated(env, tmp_path):
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    interpreter = FakeInterpreter(lock, isolated=False)
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=tmp_path / "runtime",
                fetcher=online_fetcher(env, "cpu-nogui", paths),
                installer=FakeInstaller(artifacts),
                interpreter=interpreter,
            )
        )
    assert excinfo.value.code == "python_identity"


def test_build_fails_on_bootstrap_selftest_failure(env, tmp_path):
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    interpreter = FakeInterpreter(lock, selftest_ok=False)
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=tmp_path / "runtime",
                fetcher=online_fetcher(env, "cpu-nogui", paths),
                installer=FakeInstaller(artifacts),
                interpreter=interpreter,
            )
        )
    assert excinfo.value.code == "bootstrap"


def test_build_fails_when_bootstrap_uses_other_interpreter(env, tmp_path):
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    interpreter = FakeInterpreter(lock, selftest_executable_override=str(env.tmp_path / "other-python.exe"))
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=tmp_path / "runtime",
                fetcher=online_fetcher(env, "cpu-nogui", paths),
                installer=FakeInstaller(artifacts),
                interpreter=interpreter,
            )
        )
    assert excinfo.value.code == "bootstrap"


def test_build_fails_on_prohibited_cache_leftovers(env, tmp_path):
    # A prohibited cache DIRECTORY (not a file the artifact scrub removes):
    # stray archives are auto-scrubbed as install byproducts, while cache
    # classes fail the build fail-closed.
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    installer = FakeInstaller(artifacts, plant_prohibited="Lib/site-packages/NVSTCache/nvlog.bin")
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=tmp_path / "runtime",
                fetcher=online_fetcher(env, "cpu-nogui", paths),
                installer=installer,
                interpreter=FakeInterpreter(lock),
            )
        )
    assert excinfo.value.code == "prohibited_cache"


def test_scipy_self_copy_recognized_by_exact_invariant(env, tmp_path):
    """The exact reviewed scipy 1.18.1 self-copy defect is recognized (NOT
    bulk-deleted), retained, recorded in the operational manifest, and
    remains fully covered by RECORD byte verification; direct_url.json
    provenance pointers are removed together with their RECORD lines; the
    interpreter's own ensurepip wheel is distribution content and survives."""
    archive = make_python_archive(
        env.tmp_path / "fake-python-ensurepip.tar.gz",
        extra_members={
            "python/Lib/ensurepip/_bundled/pip-9.9.9-py3-none-any.whl": b"FAKE-ENSUREPIP-WHEEL"
        },
    )
    identity = fake_identity(archive)
    inventory = fake_inventory(identity)
    artifacts, paths = wheels_for_variant(env.wheels_dir, "cpu-nogui", "cpu")
    scipy, scipy_path = scipy_wheel_artifact(env.tmp_path / "scipy-store")
    all_wheels = (*artifacts, scipy)
    all_paths = {**paths, scipy.url: scipy_path}
    lock = make_lock("cpu-nogui", "cpu", all_wheels)
    mapping = {identity.source_url: archive.read_bytes()}
    mapping.update({url: p.read_bytes() for url, p in all_paths.items()})
    fetcher = br.OnlineFetcher(env.tmp_path / "cache-ep", "cpu-nogui", opener=FakeOpener(mapping))
    result = br.run_build(
        br.BuildContext(
            repo_root=ROOT,
            runtime_root=env.tmp_path / "runtime-ep",
            identity=identity,
            inventory=inventory,
            lock=lock,
            bootstrap_sha256=BOOTSTRAP_SHA,
            fetcher=fetcher,
            installer=FakeInstaller(list(all_wheels), plant_self_copy=True),
            interpreter=FakeInterpreter(lock),
            activate=False,
            lock_timeout=30.0,
            log=lambda _message: None,
        )
    )
    assert result.state == "BUILT"
    tree = result.runtime_dir
    site_pkgs = tree / "Lib" / "site-packages"
    # The exact defect artifact is RETAINED (zero bytes), never deleted:
    self_copy = tree / br.SCIPY_SELF_COPY["relative_path"]
    assert self_copy.is_file()
    assert self_copy.stat().st_size == 0
    assert hashlib.sha256(self_copy.read_bytes()).hexdigest() == br.SCIPY_SELF_COPY["record_sha256_hex"]
    # The interpreter's own ensurepip wheel is pinned distribution content:
    assert (tree / "Lib" / "ensurepip" / "_bundled" / "pip-9.9.9-py3-none-any.whl").is_file()
    # Build-machine direct_url.json provenance is gone ...
    assert [p for p in site_pkgs.rglob("direct_url.json")] == []
    # ... and so are its RECORD lines (RECORD stays self-consistent):
    for record in site_pkgs.glob("*.dist-info/RECORD"):
        text = record.read_text(encoding="utf-8")
        assert "direct_url.json" not in text
    # The full corrected verifier (re-enumeration + RECORD byte integrity)
    # passes on the retained-but-recognized artifact:
    existing = br.verify_runtime(tree, lock, identity, inventory, BOOTSTRAP_SHA)
    assert all(item.code == "PASS" for item in existing), [c for c in existing if c.code != "PASS"]
    manifest = br.read_manifest(tree / br.MANIFEST_NAME)
    normalizations = manifest["operational"].get("normalizations", [])
    kinds = {n.get("type") for n in normalizations}
    assert "scipy-self-copy" in kinds
    assert "direct-url" in kinds
    scipy_record = next(n for n in normalizations if n["type"] == "scipy-self-copy")
    assert scipy_record["path"] == br.SCIPY_SELF_COPY["relative_path"]
    assert scipy_record["sha256"] == br.SCIPY_SELF_COPY["record_sha256_hex"]
    assert any("scipy 1.18.1 self-copy" in note for note in manifest["operational"]["notes"])


def test_scipy_path_with_wrong_invariants_fails_closed(env, tmp_path):
    """A file at the exact scipy self-copy path that violates ANY invariant
    (here: non-zero content) must NOT be normalized away: the build fails
    closed and the file is left in place."""
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    scipy, scipy_path = scipy_wheel_artifact(env.tmp_path / "scipy-store")
    lock = make_lock("cpu-nogui", "cpu", (*artifacts, scipy))
    all_paths = {**paths, scipy.url: scipy_path}
    installer = FakeInstaller(
        lock.wheels,
        plant_prohibited=br.SCIPY_SELF_COPY["relative_path"],  # b"prohibited" != b""
    )
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=tmp_path / "runtime",
                fetcher=online_fetcher(env, "cpu-nogui", all_paths),
                installer=installer,
                interpreter=FakeInterpreter(lock),
            )
        )
    assert excinfo.value.code == "prohibited-cache" or excinfo.value.code == "prohibited_cache"
    assert not (tmp_path / "runtime" / "versions").exists()


def test_unexpected_archive_file_fails_closed(env, tmp_path):
    """Any archive file that is NOT the exact reviewed scipy self-copy is
    never silently deleted: it fails the prohibited-cache scan."""
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    installer = FakeInstaller(lock.wheels, plant_prohibited="Lib/site-packages/rogue-cache.whl")
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=tmp_path / "runtime",
                fetcher=online_fetcher(env, "cpu-nogui", paths),
                installer=installer,
                interpreter=FakeInterpreter(lock),
            )
        )
    assert excinfo.value.code == "prohibited_cache"
    assert not (tmp_path / "runtime" / "versions").exists()


def test_gui_build_verifies_pyqt5_family(env, tmp_path):
    result, _, lock, _ = _build(env, tmp_path, variant="cpu-gui", torch_backend="cpu")
    assert result.state == "BUILT"
    manifest = br.read_manifest(result.runtime_dir / br.MANIFEST_NAME)
    package_set = manifest["canonical"]["package_set"]
    for name in br.GUI_PACKAGE_FAMILY:
        assert name in package_set
    assert manifest["canonical"]["verification"]["variant_contract"] == "PASS"


def test_cuda_build_records_environmentally_limited_torch_probe(env, tmp_path):
    artifacts, paths, lock = build_artifacts(env, "cuda-nogui", "cu130")
    torch_probe = {
        "imported": True,
        "version": "2.14.0+cu130",
        "cuda_built": "13.0",
        "cuda_available": False,
    }
    interpreter = FakeInterpreter(lock, torch=torch_probe)
    result = br.run_build(
        make_context(
            env, lock,
            runtime_root=tmp_path / "runtime",
            fetcher=online_fetcher(env, "cuda-nogui", paths),
            installer=FakeInstaller(artifacts),
            interpreter=interpreter,
        )
    )
    assert result.state == "BUILT"
    codes = {check.name: check.code for check in result.checks}
    assert codes["torch_probe"] == "ENVIRONMENTALLY_LIMITED"
    manifest = br.read_manifest(result.runtime_dir / br.MANIFEST_NAME)
    # Canonical records the DETERMINISTIC identity outcome only:
    assert manifest["canonical"]["verification"]["torch_probe"] == "VERIFIED"
    # The hardware state is OPERATIONAL evidence, never canonical identity:
    hardware = manifest["operational"]["torch_hardware"]
    assert hardware["code"] == "ENVIRONMENTALLY_LIMITED"
    assert hardware["cuda_available"] is False
    assert hardware["device"] is None
    assert any("is_available(): false" in note for note in manifest["operational"]["notes"])


def test_cuda_build_passes_torch_probe_with_physical_device(env, tmp_path):
    artifacts, paths, lock = build_artifacts(env, "cuda-nogui", "cu130")
    torch_probe = {
        "imported": True,
        "version": "2.14.0+cu130",
        "cuda_built": "13.0",
        "cuda_available": True,
        "device": "Fake CUDA Device",
    }
    result = br.run_build(
        make_context(
            env, lock,
            runtime_root=tmp_path / "runtime",
            fetcher=online_fetcher(env, "cuda-nogui", paths),
            installer=FakeInstaller(artifacts),
            interpreter=FakeInterpreter(lock, torch=torch_probe),
        )
    )
    codes = {check.name: check.code for check in result.checks}
    assert codes["torch_probe"] == "PASS"
    manifest = br.read_manifest(result.runtime_dir / br.MANIFEST_NAME)
    assert manifest["canonical"]["verification"]["torch_probe"] == "VERIFIED"
    hardware = manifest["operational"]["torch_hardware"]
    assert hardware["code"] == "PASS"
    assert hardware["cuda_available"] is True
    assert hardware["device"] == "Fake CUDA Device"


def test_cuda_canonical_manifest_independent_of_physical_cuda(env, tmp_path):
    """Same canonical inputs on a CUDA-capable and a non-CUDA-capable host
    must yield the SAME runtime ID and the SAME canonical manifest: hardware
    state (device presence, observations) is operational evidence only."""
    artifacts, paths, lock = build_artifacts(env, "cuda-nogui", "cu130")
    with_device = {
        "imported": True,
        "version": "2.14.0+cu130",
        "cuda_built": "13.0",
        "cuda_available": True,
        "device": "Fake CUDA Device",
    }
    without_device = {
        "imported": True,
        "version": "2.14.0+cu130",
        "cuda_built": "13.0",
        "cuda_available": False,
    }
    result_gpu = br.run_build(
        make_context(
            env, lock,
            runtime_root=tmp_path / "runtime-gpu",
            fetcher=online_fetcher(env, "cuda-nogui", paths),
            installer=FakeInstaller(artifacts),
            interpreter=FakeInterpreter(lock, torch=with_device),
        )
    )
    result_nogpu = br.run_build(
        make_context(
            env, lock,
            runtime_root=tmp_path / "runtime-nogpu",
            fetcher=online_fetcher(env, "cuda-nogui", paths),
            installer=FakeInstaller(artifacts),
            interpreter=FakeInterpreter(lock, torch=without_device),
        )
    )
    assert result_gpu.state == "BUILT"
    assert result_nogpu.state == "BUILT"
    assert result_gpu.runtime_id == result_nogpu.runtime_id
    manifest_gpu = br.read_manifest(result_gpu.runtime_dir / br.MANIFEST_NAME)
    manifest_nogpu = br.read_manifest(result_nogpu.runtime_dir / br.MANIFEST_NAME)
    assert manifest_gpu["canonical"] == manifest_nogpu["canonical"]
    assert manifest_gpu["operational"]["torch_hardware"]["code"] == "PASS"
    assert manifest_nogpu["operational"]["torch_hardware"]["code"] == "ENVIRONMENTALLY_LIMITED"


@pytest.mark.parametrize("probe", [
    {"imported": False, "import_error": "ModuleNotFoundError"},
    {"imported": True, "version": "2.14.0+cu128", "cuda_built": "12.8", "cuda_available": False},
    {"imported": True, "version": "2.14.0+cu130", "cuda_built": "12.8", "cuda_available": False},
])
def test_cuda_build_fails_on_torch_identity_violation(env, tmp_path, probe):
    artifacts, paths, lock = build_artifacts(env, "cuda-nogui", "cu130")
    interpreter = FakeInterpreter(lock, torch=probe) if probe.get("imported") else FakeInterpreter(lock, torch_missing=True)
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=tmp_path / "runtime",
                fetcher=online_fetcher(env, "cuda-nogui", paths),
                installer=FakeInstaller(artifacts),
                interpreter=interpreter,
            )
        )
    assert excinfo.value.code == "torch_probe"


def test_build_fails_on_distributions_probe_failure(env, tmp_path):
    class BrokenInterpreter(FakeInterpreter):
        def _distributions(self, python_exe, cwd):
            self._distribution_calls += 1
            return types.SimpleNamespace(returncode=1, stdout="boom")

    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=tmp_path / "runtime",
                fetcher=online_fetcher(env, "cpu-nogui", paths),
                installer=FakeInstaller(artifacts),
                interpreter=BrokenInterpreter(lock),
            )
        )
    assert excinfo.value.code == "interpreter"


# ---------------------------------------------------------------------------
# Runtime verification (manifest-driven, static)
# ---------------------------------------------------------------------------


def _fake_built_runtime(env, tmp_path, variant="cpu-nogui", torch_backend="cpu"):
    artifacts, paths, lock = build_artifacts(env, variant, torch_backend)
    runtime_root = tmp_path / "runtime"
    result = br.run_build(
        make_context(
            env, lock,
            runtime_root=runtime_root,
            fetcher=online_fetcher(env, variant, paths),
            installer=FakeInstaller(artifacts),
            interpreter=FakeInterpreter(lock),
        )
    )
    return runtime_root, result, lock


def test_verify_runtime_pass_on_built_runtime(env, tmp_path):
    runtime_root, result, lock = _fake_built_runtime(env, tmp_path)
    checks = br.verify_runtime(runtime_root / "versions" / result.runtime_id, lock, env.identity, env.inventory, BOOTSTRAP_SHA)
    assert all(check.code == "PASS" for check in checks), [c for c in checks if c.code != "PASS"]


def test_verify_runtime_reports_missing_manifest(env, tmp_path):
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    empty = runtime_root / "versions" / ("01" * 32)
    empty.mkdir(parents=True)
    checks = br.verify_runtime(empty, make_lock("cpu-nogui", "cpu", ()), env.identity, env.inventory, BOOTSTRAP_SHA)
    assert checks[0].name == "manifest"
    assert checks[0].code == "MISSING_MANIFEST"


def test_verify_runtime_detects_variant_mismatch(env, tmp_path):
    runtime_root, result, lock = _fake_built_runtime(env, tmp_path, variant="cpu-nogui")
    other_lock = make_lock("cuda-nogui", "cu130", lock.wheels)
    checks = br.verify_runtime(runtime_root / "versions" / result.runtime_id, other_lock, env.identity, env.inventory, BOOTSTRAP_SHA)
    assert checks[0].name == "variant"
    assert checks[0].code == "MISMATCH"


def test_verify_runtime_detects_python_identity_mismatch(env, tmp_path):
    runtime_root, result, lock = _fake_built_runtime(env, tmp_path)
    tampered = br.PythonAsset(**{**env.identity.__dict__, "sha256": "00" * 32})
    checks = br.verify_runtime(runtime_root / "versions" / result.runtime_id, lock, tampered, env.inventory, BOOTSTRAP_SHA)
    assert checks[0].name == "python"
    assert checks[0].code == "MISMATCH"


def test_verify_runtime_detects_missing_interpreter(env, tmp_path):
    runtime_root, result, lock = _fake_built_runtime(env, tmp_path)
    (runtime_root / "versions" / result.runtime_id / "python.exe").unlink()
    checks = br.verify_runtime(runtime_root / "versions" / result.runtime_id, lock, env.identity, env.inventory, BOOTSTRAP_SHA)
    python_check = next(c for c in checks if c.name == "python")
    assert python_check.code == "MISSING_INTERPRETER"


def test_verify_runtime_detects_package_set_drift(env, tmp_path):
    runtime_root, result, lock = _fake_built_runtime(env, tmp_path)
    drifted = make_lock("cpu-nogui", "cpu", lock.wheels)
    drifted = br.Lock(
        **{
            **drifted.__dict__,
            "lock_sha256": "55" * 32,
            "wheels": tuple(
                br.WheelArtifact(**{**w.__dict__, "version": "9.9.9"}) if w.name == "alpha" else w
                for w in lock.wheels
            ),
        }
    )
    checks = br.verify_runtime(runtime_root / "versions" / result.runtime_id, drifted, env.identity, env.inventory, BOOTSTRAP_SHA)
    # The manifest was honest; only the presented contract (package set)
    # deviates, so the first failing check is the package-set comparison.
    assert checks[0].name == "package_set"
    assert checks[0].code == "MISMATCH"


def test_verify_runtime_detects_manifest_runtime_id_tamper(env, tmp_path):
    """The manifest's runtime_id is evidence, not authority: the expected ID
    is derived from the CURRENT Commit-1 inputs plus the tree digest
    recomputed from the actual tree. A forged manifest runtime_id that the
    directory name does not match fails the runtime_id check."""
    runtime_root, result, lock = _fake_built_runtime(env, tmp_path)
    version_dir = runtime_root / "versions" / result.runtime_id
    manifest_path = version_dir / br.MANIFEST_NAME
    manifest = br.read_manifest(manifest_path)
    manifest["canonical"]["runtime_id"] = "77" * 32
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checks = br.verify_runtime(version_dir, lock, env.identity, env.inventory, BOOTSTRAP_SHA)
    id_check = _check_by_name(checks, "runtime_id")
    assert id_check.code == "MISMATCH"
    assert "manifest runtime_id" in id_check.note


def test_verify_runtime_detects_installed_tree_drift(env, tmp_path):
    runtime_root, result, lock = _fake_built_runtime(env, tmp_path)
    version_dir = runtime_root / "versions" / result.runtime_id
    (version_dir / "python.exe").write_bytes(b"tampered")
    checks = br.verify_runtime(version_dir, lock, env.identity, env.inventory, BOOTSTRAP_SHA)
    tree_check = next(c for c in checks if c.name == "installed_tree")
    assert tree_check.code == "MISMATCH"


def test_verify_runtime_detects_prohibited_artifacts(env, tmp_path):
    runtime_root, result, lock = _fake_built_runtime(env, tmp_path)
    version_dir = runtime_root / "versions" / result.runtime_id
    (version_dir / "rogue-cache.whl").write_bytes(b"x")
    checks = br.verify_runtime(version_dir, lock, env.identity, env.inventory, BOOTSTRAP_SHA)
    prohibited = next(c for c in checks if c.name == "prohibited_cache")
    assert prohibited.code == "PRESENT"


# ---------------------------------------------------------------------------
# CLI behavior (argument/validation paths only; no network)
# ---------------------------------------------------------------------------


def test_cli_rejects_unknown_variant(capsys):
    assert br.main(["build", "--variant", "bogus-variant"]) == 1
    assert "unknown variant" in capsys.readouterr().err


def test_cli_offline_requires_inputs_dir(capsys):
    assert br.main(["build", "--variant", "cpu-nogui", "--offline"]) == 2
    out = capsys.readouterr().err
    assert "--offline-inputs" in out


def test_cli_verify_runtime_rejects_bad_grammar(capsys):
    assert br.main(["verify-runtime", "--runtime-id", "xyz"]) == 2


def test_cli_verify_runtime_reports_missing_runtime(tmp_path, capsys):
    assert br.main(
        [
            "verify-runtime",
            "--runtime-id", "01" * 32,
            "--runtime-root", str(tmp_path / "nowhere"),
            "--variant", "cpu-nogui",
        ]
    ) == 1
    assert "runtime not found" in capsys.readouterr().err


def test_cli_rollback_rejects_missing_runtime(tmp_path, capsys):
    assert br.main(
        [
            "rollback",
            "--runtime-id", "02" * 32,
            "--runtime-root", str(tmp_path / "nowhere"),
        ]
    ) == 1
    assert "runtime not found" in capsys.readouterr().err


def test_cli_help_lists_subcommands(capsys):
    with pytest.raises(SystemExit) as excinfo:
        br.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for word in ("build", "verify-runtime", "activate", "rollback", "fetch-offline-inputs"):
        assert word in out


# ---------------------------------------------------------------------------
# Offline-inputs preparation (with faked downloads)
# ---------------------------------------------------------------------------


def test_write_offline_inputs_produces_strict_layout(env, tmp_path):
    """fetch-offline-inputs materializes EXACTLY the layout the strict
    offline build contract requires: python/, wheels/<variant>/, licenses/
    (Commit-1 inventory installed paths), locks/ (all contract files with a
    byte-exact selected lock) and artifact-manifest.json."""
    _, paths, lock = offline_build_artifacts(env, "cpu-nogui", "cpu")
    mapping = {env.identity.source_url: env.archive.read_bytes()}
    mapping.update({url: p.read_bytes() for url, p in paths.items()})
    fetcher = br.OnlineFetcher(env.tmp_path / "cache", "cpu-nogui", opener=FakeOpener(mapping))
    output = env.tmp_path / "offline-out"
    lock_files = make_fake_lock_files(env, "cpu-nogui", lock)
    br.write_offline_inputs(
        output, "cpu-nogui", env.identity, env.inventory, lock, fetcher, lock_files,
        log=lambda _m: None,
    )
    assert (output / "python" / env.identity.filename).is_file()
    assert br.locklib.sha256_file_raw(output / "python" / env.identity.filename) == env.identity.sha256
    for wheel in lock.wheels:
        assert (output / "wheels" / "cpu-nogui" / wheel.filename).is_file()
    assert (output / "artifact-manifest.json").is_file()
    manifest = json.loads((output / "artifact-manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == br.OFFLINE_INPUTS_SCHEMA
    assert manifest["variant"] == "cpu-nogui"
    assert {w["filename"] for w in manifest["wheels"]} == {w.filename for w in lock.wheels}
    assert manifest["python"]["sha256"] == env.identity.sha256
    license_files = {p.relative_to(output / "licenses").as_posix() for p in (output / "licenses").rglob("*") if p.is_file()}
    assert license_files == {entry.installed_path for entry in env.inventory.entries}
    lock_names = {p.name for p in (output / "locks").iterdir() if p.is_file()}
    assert lock_names == set(br._offline_lock_filenames("cpu-nogui"))
    # The generated layout passes the strict contract as-is:
    br.verify_offline_inputs(output, "cpu-nogui", env.identity, env.inventory, lock, {q.name: q.read_bytes() for q in lock_files})


def test_write_offline_inputs_rejects_unknown_variant(env):
    _, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    fetcher = br.OfflineFetcher(env.tmp_path / "offline", "cpu-nogui")
    with pytest.raises(br.BuilderError) as excinfo:
        br.write_offline_inputs(
            env.tmp_path / "x", "not-a-variant", env.identity, env.inventory, lock, fetcher, [],
            log=lambda _m: None,
        )
    assert excinfo.value.code == "variant"


# ---------------------------------------------------------------------------
# Strict offline-inputs contract: incomplete/malformed layouts are rejected
# ---------------------------------------------------------------------------


def _strict_offline(env, variant="cpu-nogui", backend="cpu"):
    artifacts, paths, lock = offline_build_artifacts(env, variant, backend)
    offline = prepared_offline_inputs(env, variant, lock, paths)
    return artifacts, paths, lock, offline


def test_offline_rejects_missing_artifact_manifest(env):
    _a, _p, lock, offline = _strict_offline(env)
    (offline / "artifact-manifest.json").unlink()
    with pytest.raises(br.BuilderError) as excinfo:
        br.verify_offline_inputs(offline, "cpu-nogui", env.identity, env.inventory, lock, fake_contract_bytes(env, lock))
    assert excinfo.value.code == "offline"


def test_offline_rejects_missing_license_file(env):
    _a, _p, lock, offline = _strict_offline(env)
    victim = (offline / "licenses" / "LICENSE.txt")
    victim.unlink()
    with pytest.raises(br.BuilderError) as excinfo:
        br.verify_offline_inputs(offline, "cpu-nogui", env.identity, env.inventory, lock, fake_contract_bytes(env, lock))
    assert excinfo.value.code == "offline"
    assert "licenses/" in excinfo.value.message


def test_offline_rejects_mismatched_license_inventory(env):
    _a, _p, lock, offline = _strict_offline(env)
    (offline / "licenses" / "LICENSE.txt").write_bytes(b"CORRUPTED-LICENSE")
    with pytest.raises(br.BuilderError) as excinfo:
        br.verify_offline_inputs(offline, "cpu-nogui", env.identity, env.inventory, lock, fake_contract_bytes(env, lock))
    assert excinfo.value.code == "offline"


def test_offline_rejects_wrong_lock_copy(env):
    _a, _p, lock, offline = _strict_offline(env)
    lock_file = offline / "locks" / lock.lock_filename
    lock_file.write_text("# tampered lock copy\n", encoding="utf-8")
    with pytest.raises(br.BuilderError) as excinfo:
        br.verify_offline_inputs(offline, "cpu-nogui", env.identity, env.inventory, lock, fake_contract_bytes(env, lock))
    assert excinfo.value.code == "offline"
    assert "lock" in excinfo.value.message


def test_offline_rejects_missing_wheel(env):
    _a, _p, lock, offline = _strict_offline(env)
    victims = sorted((offline / "wheels" / "cpu-nogui").iterdir())
    victims[0].unlink()
    with pytest.raises(br.BuilderError) as excinfo:
        br.verify_offline_inputs(offline, "cpu-nogui", env.identity, env.inventory, lock, fake_contract_bytes(env, lock))
    assert excinfo.value.code == "artifact-missing"


def test_offline_rejects_corrupt_wheel(env):
    _a, _p, lock, offline = _strict_offline(env)
    victims = sorted((offline / "wheels" / "cpu-nogui").iterdir())
    victims[0].write_bytes(b"corrupted")
    with pytest.raises(br.BuilderError) as excinfo:
        br.verify_offline_inputs(offline, "cpu-nogui", env.identity, env.inventory, lock, fake_contract_bytes(env, lock))
    assert excinfo.value.code == "artifact-hash"


def test_offline_rejects_corrupt_python_archive(env):
    _a, _p, lock, offline = _strict_offline(env)
    (offline / "python" / env.identity.filename).write_bytes(b"corrupt")
    with pytest.raises(br.BuilderError) as excinfo:
        br.verify_offline_inputs(offline, "cpu-nogui", env.identity, env.inventory, lock, fake_contract_bytes(env, lock))
    assert excinfo.value.code == "artifact-hash"


def test_offline_rejects_unexpected_extra_file(env):
    _a, _p, lock, offline = _strict_offline(env)
    (offline / "stray-notes.txt").write_bytes(b"extra")
    with pytest.raises(br.BuilderError) as excinfo:
        br.verify_offline_inputs(offline, "cpu-nogui", env.identity, env.inventory, lock, fake_contract_bytes(env, lock))
    assert excinfo.value.code == "offline"
    assert "unexpected file" in excinfo.value.message


def test_offline_rejects_other_variant_wheel_dir(env):
    """A wheel directory for a different variant is outside the exact
    structure for THIS variant and must be rejected."""
    _a, _p, lock, offline = _strict_offline(env)
    other = offline / "wheels" / "cpu-gui"
    other.mkdir(parents=True)
    (other / "rogue-1.0.0-fake.whl").write_bytes(b"x")
    with pytest.raises(br.BuilderError) as excinfo:
        br.verify_offline_inputs(offline, "cpu-nogui", env.identity, env.inventory, lock, fake_contract_bytes(env, lock))
    assert excinfo.value.code == "offline"


def test_offline_fetcher_has_no_network_path(env):
    """Structural guarantee: the offline fetcher cannot execute network
    access (no opener, no urlopen), so no network path can run in an
    offline build (pip runs with --no-index as well)."""
    assert br.OfflineFetcher.network_enabled is False
    fetcher = br.OfflineFetcher(env.tmp_path / "offline", "cpu-nogui")
    assert not hasattr(fetcher, "opener")
    with pytest.raises(br.BuilderError) as excinfo:
        fetcher.fetch(br.ArtifactSpec("wheel", "nope-fake.whl", "https://fake.invalid/x", "00" * 32))
    assert excinfo.value.code == "artifact-missing"


# ---------------------------------------------------------------------------
# Installed-tree integrity: byte-level RECORD verification (re-review)
# ---------------------------------------------------------------------------


def _tampered_verify(env, tmp_path, tamper, variant="cpu-nogui", backend="cpu"):
    """Build a fake runtime, apply `tamper` to the promoted tree, and run
    the corrected verifier. Returns (version_dir, checks)."""
    runtime_root, result, lock = _fake_built_runtime(env, tmp_path, variant=variant, torch_backend=backend)
    version_dir = runtime_root / "versions" / result.runtime_id
    tamper(version_dir)
    checks = br.verify_runtime(version_dir, lock, env.identity, env.inventory, BOOTSTRAP_SHA)
    return version_dir, checks


def _check_by_name(checks, name):
    return next(c for c in checks if c.name == name)


def test_verify_runtime_detects_payload_source_tamper(env, tmp_path):
    def tamper(tree):
        (tree / "Lib" / "site-packages" / "alpha" / "__init__.py").write_bytes(b"malicious payload\n")

    _version_dir, checks = _tampered_verify(env, tmp_path, tamper)
    integrity = _check_by_name(checks, "record_integrity")
    assert integrity.code == "MISMATCH"
    assert "alpha/__init__.py" in integrity.note


def test_verify_runtime_detects_nested_pyd_tamper(env, tmp_path):
    def tamper(tree):
        (tree / "Lib" / "site-packages" / "torch" / "torch_core.pyd").write_bytes(b"evil pyd bytes")

    _version_dir, checks = _tampered_verify(env, tmp_path, tamper)
    integrity = _check_by_name(checks, "record_integrity")
    assert integrity.code == "MISMATCH"
    assert "torch_core.pyd" in integrity.note


def test_verify_runtime_detects_nested_dll_tamper(env, tmp_path):
    def tamper(tree):
        (tree / "Lib" / "site-packages" / "alpha" / "alpha_native.dll").write_bytes(b"evil dll bytes")

    _version_dir, checks = _tampered_verify(env, tmp_path, tamper)
    integrity = _check_by_name(checks, "record_integrity")
    assert integrity.code == "MISMATCH"
    assert "alpha_native.dll" in integrity.note


def test_planted_scripts_launcher_fails_closed(env, tmp_path):
    """Launcher contract: the builder removes the pip console launcher
    binaries (not byte-reproducible); a promoted runtime therefore carries
    NO files under Scripts/. Planting a launcher in a promoted tree is an
    integrity violation: the runtime-ID tree digest changes (directory name
    no longer matches the expected ID) AND the prohibited-cache scan flags
    the Scripts/ tree. The RECORD no longer references launchers, so the
    record_integrity layer has no launcher entry to be fooled by."""

    def tamper(tree):
        scripts = tree / "Scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        (scripts / "alpha.exe").write_bytes(b"evil launcher")

    _version_dir, checks = _tampered_verify(env, tmp_path, tamper)
    assert _check_by_name(checks, "runtime_id").code == "MISMATCH"
    assert _check_by_name(checks, "prohibited_cache").code == "PRESENT"
    assert _check_by_name(checks, "record_integrity").code == "PASS"


def test_planted_launcher_with_record_line_fails(env, tmp_path):
    """The demonstrated bypass: plant a Scripts/*.exe launcher AND re-add a
    matching RECORD entry (editing the RECORD's digest/size to match the
    planted bytes). The on-disk RECORD can no longer hide it: the
    installed-tree digest hashes the ACTUAL bytes of every file (including
    the planted launcher and the edited RECORD), so the expected runtime ID
    derived from the current Commit-1 inputs no longer matches the
    directory name, and the prohibited-cache scan flags the Scripts/ tree.
    """

    def tamper(tree):
        import hashlib as _hl

        import base64 as _b64

        scripts = tree / "Scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        evil = b"evil launcher"
        (scripts / "alpha.exe").write_bytes(evil)
        digest = _b64.urlsafe_b64encode(_hl.sha256(evil).digest()).decode("ascii").rstrip("=")
        record = tree / "Lib" / "site-packages" / "alpha-1.0.0.dist-info" / "RECORD"
        record.write_text(
            record.read_text(encoding="utf-8").rstrip("\n")
            + f"\n../../Scripts/alpha.exe,sha256={digest},{len(evil)}\n",
            encoding="utf-8",
            newline="\n",
        )

    _version_dir, checks = _tampered_verify(env, tmp_path, tamper)
    assert _check_by_name(checks, "runtime_id").code == "MISMATCH"
    assert _check_by_name(checks, "prohibited_cache").code == "PRESENT"


def test_verify_runtime_detects_metadata_tamper(env, tmp_path):
    def tamper(tree):
        (
            tree / "Lib" / "site-packages" / "alpha-1.0.0.dist-info" / "METADATA"
        ).write_bytes(b"Metadata-Version: 2.1\nName: alpha\nVersion: 9.9.9\n")

    _version_dir, checks = _tampered_verify(env, tmp_path, tamper)
    assert _check_by_name(checks, "installed_tree").code == "MISMATCH"
    assert _check_by_name(checks, "installed_distributions").code == "MISMATCH"
    assert _check_by_name(checks, "record_integrity").code == "MISMATCH"


def test_verify_runtime_detects_recorded_file_deletion(env, tmp_path):
    def tamper(tree):
        (tree / "Lib" / "site-packages" / "torch" / "torch_core.pyd").unlink()

    _version_dir, checks = _tampered_verify(env, tmp_path, tamper)
    integrity = _check_by_name(checks, "record_integrity")
    assert integrity.code == "MISMATCH"
    assert "missing RECORD file" in integrity.note


class BytecodeRecordingInstaller(FakeInstaller):
    """Like FakeInstaller, but one wheel additionally ships pre-compiled
    bytecode: a .pyc file installed under the package AND a matching RECORD
    entry (as pip records bundled bytecode)."""

    def install(self, python_exe: Path, requirements_file: Path, cwd: Path, cache_dir: Path) -> None:
        super().install(python_exe, requirements_file, cwd, cache_dir)
        site = cwd / "Lib" / "site-packages"
        pyc_dir = site / "alpha" / "__pycache__"
        pyc_dir.mkdir(parents=True, exist_ok=True)
        data = b"FAKE-PYC"
        (pyc_dir / "__init__.cpython-312.pyc").write_bytes(data)
        record = site / "alpha-1.0.0.dist-info" / "RECORD"
        line = (
            f"alpha/__pycache__/__init__.cpython-312.pyc,"
            f"sha256={record_b64(data)},{len(data)}"
        )
        record.write_text(
            record.read_text(encoding="utf-8").rstrip("\n") + "\n" + line + "\n",
            encoding="utf-8",
        )


def test_scrubbed_bytecode_records_stay_consistent(env, tmp_path):
    """Generated bytecode is scrubbed from the tree and its RECORD entries
    are dropped so the on-disk RECORD stays self-consistent; the corrected
    verifier then passes (no phantom 'missing RECORD file') and a .pyc
    re-appearing in the promoted tree is reported by the prohibited-cache
    scan, not by the integrity layer."""
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    result = br.run_build(
        make_context(
            env, lock,
            runtime_root=tmp_path / "runtime",
            fetcher=online_fetcher(env, "cpu-nogui", paths),
            installer=BytecodeRecordingInstaller(artifacts),
            interpreter=FakeInterpreter(lock),
        )
    )
    assert result.state == "BUILT"
    tree = result.runtime_dir
    # the scrubbed bytecode is gone from the promoted tree ...
    assert not (tree / "Lib" / "site-packages" / "alpha" / "__pycache__").exists()
    # ... and its RECORD line was dropped (RECORD self-consistent):
    record = (tree / "Lib" / "site-packages" / "alpha-1.0.0.dist-info" / "RECORD").read_text(encoding="utf-8")
    assert ".pyc" not in record
    assert "__pycache__" not in record
    checks = br.verify_runtime(tree, lock, env.identity, env.inventory, BOOTSTRAP_SHA)
    assert all(c.code == "PASS" for c in checks), [c for c in checks if c.code != "PASS"]
    manifest = br.read_manifest(tree / br.MANIFEST_NAME)
    assert any(
        n["type"] == "bytecode-records" for n in manifest["operational"].get("normalizations", [])
    )
    # If bytecode re-appears in the promoted tree, verification fails
    # closed via the prohibited-cache scan:
    pyc = tree / "Lib" / "site-packages" / "alpha" / "__pycache__" / "__init__.cpython-312.pyc"
    pyc.parent.mkdir(parents=True, exist_ok=True)
    pyc.write_bytes(b"FAKE-PYC")
    codes = {c.name: c.code for c in br.verify_runtime(tree, lock, env.identity, env.inventory, BOOTSTRAP_SHA)}
    assert codes["runtime_id"] == "MISMATCH"
    assert codes["prohibited_cache"] == "PRESENT"
    assert codes["record_integrity"] == "PASS"


# ---------------------------------------------------------------------------
# Distribution re-enumeration (manifest is never trusted alone)
# ---------------------------------------------------------------------------


def test_verify_runtime_detects_missing_distribution(env, tmp_path):
    def tamper(tree):
        shutil.rmtree(tree / "Lib" / "site-packages" / "alpha-1.0.0.dist-info")
        shutil.rmtree(tree / "Lib" / "site-packages" / "alpha")

    _version_dir, checks = _tampered_verify(env, tmp_path, tamper)
    dists = _check_by_name(checks, "installed_distributions")
    assert dists.code == "MISMATCH"
    assert "alpha" in dists.note


def test_verify_runtime_detects_extra_distribution(env, tmp_path):
    def tamper(tree):
        rogue = tree / "Lib" / "site-packages" / "rogue-1.0.dist-info"
        rogue.mkdir()
        (rogue / "METADATA").write_bytes(
            b"Metadata-Version: 2.1\nName: rogue\nVersion: 1.0\n"
        )

    _version_dir, checks = _tampered_verify(env, tmp_path, tamper)
    dists = _check_by_name(checks, "installed_distributions")
    assert dists.code == "MISMATCH"
    assert "rogue" in dists.note


def test_verify_runtime_detects_wrong_version(env, tmp_path):
    def tamper(tree):
        # Rewrite METADATA with a DIFFERENT version so the re-enumerated
        # distribution set (from the on-disk tree, not the manifest) reports
        # a version that does not match the locked/baseline expectation.
        (tree / "Lib" / "site-packages" / "alpha-1.0.0.dist-info" / "METADATA").write_bytes(
            b"Metadata-Version: 2.1\nName: alpha\nVersion: 9.9.9\n"
        )
        # also move the dist-info dir name so re-enumeration sees the new version
        old = tree / "Lib" / "site-packages" / "alpha-1.0.0.dist-info"
        new = tree / "Lib" / "site-packages" / "alpha-9.9.9.dist-info"
        old.rename(new)

    _version_dir, checks = _tampered_verify(env, tmp_path, tamper)
    dists = _check_by_name(checks, "installed_distributions")
    assert dists.code == "MISMATCH"
    assert "alpha" in dists.note


def test_verify_runtime_detects_wrong_torch_suffix(env, tmp_path):
    """A CPU runtime whose tree carries the CUDA torch build (or vice versa)
    is caught from the re-enumerated tree, not the manifest."""

    def tamper(tree):
        (tree / "Lib" / "site-packages" / "torch-2.14.0+cpu.dist-info" / "METADATA").write_bytes(
            b"Metadata-Version: 2.1\nName: torch\nVersion: 2.14.0+cu130\n"
        )

    _version_dir, checks = _tampered_verify(env, tmp_path, tamper)
    assert _check_by_name(checks, "installed_distributions").code == "MISMATCH"
    assert _check_by_name(checks, "variant_contract").code == "MISMATCH"


def test_verify_runtime_detects_gui_family_in_nogui_tree(env, tmp_path):
    def tamper(tree):
        pyqt = tree / "Lib" / "site-packages" / "pyqt5-5.15.11.dist-info"
        pyqt.mkdir()
        (pyqt / "METADATA").write_bytes(
            b"Metadata-Version: 2.1\nName: PyQt5\nVersion: 5.15.11\n"
        )

    _version_dir, checks = _tampered_verify(env, tmp_path, tamper)
    assert _check_by_name(checks, "variant_contract").code == "MISMATCH"
    assert "pyqt5" in _check_by_name(checks, "variant_contract").note
    assert _check_by_name(checks, "installed_distributions").code == "MISMATCH"


def test_verify_runtime_detects_missing_baseline_pip(env, tmp_path):
    def tamper(tree):
        shutil.rmtree(tree / "Lib" / "site-packages" / "pip-24.3.1.dist-info")

    _version_dir, checks = _tampered_verify(env, tmp_path, tamper)
    assert _check_by_name(checks, "interpreter_baseline").code == "MISMATCH"
    assert _check_by_name(checks, "installed_distributions").code == "MISMATCH"


# ---------------------------------------------------------------------------
# Four-variant coverage (all Commit-1 variants explicitly exercised)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("variant", "backend"),
    [
        ("cpu-nogui", "cpu"),
        ("cpu-gui", "cpu"),
        ("cuda-nogui", "cu130"),
        ("cuda-gui", "cu130"),
    ],
)
def test_all_four_variants_build_and_verify(env, tmp_path, variant, backend):
    torch_version = "2.14.0+cpu" if backend == "cpu" else "2.14.0+cu130"
    torch_probe = {
        "imported": True,
        "version": torch_version,
        "cuda_built": "13.0" if backend == "cu130" else None,
        "cuda_available": False,
    }
    artifacts, paths, lock = build_artifacts(env, variant, backend)
    result = br.run_build(
        make_context(
            env, lock,
            runtime_root=tmp_path / "runtime",
            fetcher=online_fetcher(env, variant, paths),
            installer=FakeInstaller(artifacts),
            interpreter=FakeInterpreter(lock, torch=torch_probe),
        )
    )
    assert result.state == "BUILT"
    version_dir = tmp_path / "runtime" / "versions" / result.runtime_id
    checks = br.verify_runtime(version_dir, lock, env.identity, env.inventory, BOOTSTRAP_SHA)
    assert all(c.code == "PASS" for c in checks), [c for c in checks if c.code != "PASS"]
    manifest = br.read_manifest(version_dir / br.MANIFEST_NAME)
    assert manifest["canonical"]["target"]["gui"] is variant.endswith("-gui")
    package_set = manifest["canonical"]["package_set"]
    if variant.endswith("-gui"):
        for name in br.GUI_PACKAGE_FAMILY:
            assert name in package_set
    else:
        assert not any(name in package_set for name in br.GUI_PACKAGE_FAMILY)
    assert package_set["torch"] == torch_version


# ---------------------------------------------------------------------------
# Activation / rollback lifecycle (re-review)
# ---------------------------------------------------------------------------


def _built_runtime_activated(
    env, tmp_path, root_name, variant="cpu-nogui", backend="cpu"
):
    torch_version = "2.14.0+cpu" if backend == "cpu" else "2.14.0+cu130"
    torch_probe = {
        "imported": True,
        "version": torch_version,
        "cuda_built": "13.0" if backend == "cu130" else None,
        "cuda_available": False,
    }
    artifacts, paths, lock = build_artifacts(env, variant, backend)
    runtime_root = tmp_path / root_name
    result = br.run_build(
        make_context(
            env, lock,
            runtime_root=runtime_root,
            fetcher=online_fetcher(env, variant, paths),
            installer=FakeInstaller(artifacts),
            interpreter=FakeInterpreter(lock, torch=torch_probe),
            activate=True,
        )
    )
    return runtime_root, result, lock


def test_build_activate_promotes_verifies_and_activates(env, tmp_path):
    runtime_root, result, lock = _built_runtime_activated(env, tmp_path, "runtime-a")
    assert result.state == "BUILT"
    assert result.activated is True
    pointer = (runtime_root / br.ACTIVE_POINTER_NAME).read_text(encoding="utf-8").strip()
    assert pointer == result.runtime_id
    # The activated runtime passes the corrected verifier end to end.
    checks = br.verify_runtime(
        runtime_root / br.VERSIONS_DIRNAME / result.runtime_id, lock, env.identity, env.inventory,
        BOOTSTRAP_SHA,
    )
    assert all(c.code == "PASS" for c in checks)


def test_activate_rejects_invalid_selector_grammar(env, tmp_path):
    runtime_root, result, lock = _built_runtime_activated(env, tmp_path, "runtime-a")
    (runtime_root / br.ACTIVE_POINTER_NAME).write_text("not-a-64-hex-id\n", encoding="utf-8")
    setup_lock = br.SetupActivationLock(runtime_root, 5.0).acquire()
    try:
        with pytest.raises(br.BuilderError) as excinfo:
            br.perform_activation(
                runtime_root, result.runtime_id,
                env.identity, env.inventory, BOOTSTRAP_SHA,
                make_variant_loader({lock.variant: lock}), setup_lock, "activate",
            )
        assert excinfo.value.code == "selector"
    finally:
        setup_lock.release()
    # The malformed selector is left untouched for the operator to inspect.
    assert (runtime_root / br.ACTIVE_POINTER_NAME).read_text(encoding="utf-8").strip() == "not-a-64-hex-id"


def test_activate_rejects_missing_selected_runtime(env, tmp_path):
    runtime_root, result, lock = _built_runtime_activated(env, tmp_path, "runtime-a")
    setup_lock = br.SetupActivationLock(runtime_root, 5.0).acquire()
    try:
        with pytest.raises(br.BuilderError) as excinfo:
            br.perform_activation(
                runtime_root, "00" * 32,
                env.identity, env.inventory, BOOTSTRAP_SHA,
                make_variant_loader({lock.variant: lock}), setup_lock, "activate",
            )
        assert excinfo.value.code == "state"
    finally:
        setup_lock.release()


def test_activate_rejects_corrupted_selected_runtime(env, tmp_path):
    runtime_root, result, lock = _built_runtime_activated(env, tmp_path, "runtime-a")
    version_dir = runtime_root / br.VERSIONS_DIRNAME / result.runtime_id
    (version_dir / "Lib" / "site-packages" / "alpha" / "__init__.py").write_bytes(b"tampered")
    setup_lock = br.SetupActivationLock(runtime_root, 5.0).acquire()
    try:
        with pytest.raises(br.BuilderError) as excinfo:
            br.perform_activation(
                runtime_root, result.runtime_id,
                env.identity, env.inventory, BOOTSTRAP_SHA,
                make_variant_loader({lock.variant: lock}), setup_lock, "activate",
            )
        assert excinfo.value.code == "state"
    finally:
        setup_lock.release()


def test_rollback_to_retained_runtime_and_refuses_corrupted_target(env, tmp_path):
    """promote -> verify -> activate (A, then B retaining A); rollback
    repoints only the selector; a corrupted target is refused and NO
    verified runtime is ever deleted."""
    runtime_root, result_a, lock_a = _built_runtime_activated(
        env, tmp_path, "runtime-ab", variant="cpu-nogui", backend="cpu"
    )
    # B is a different variant built into the SAME runtime root: the
    # selector moves to B while A is retained (verified with A's OWN
    # variant's current Commit-1 lock).
    artifacts_b, paths_b, lock_b = build_artifacts(env, "cpu-gui", "cpu")
    result_b = br.run_build(
        make_context(
            env, lock_b,
            runtime_root=runtime_root,
            fetcher=online_fetcher(env, "cpu-gui", paths_b),
            installer=FakeInstaller(artifacts_b),
            interpreter=FakeInterpreter(lock_b),
            activate=True,
            lock_loader=make_variant_loader({"cpu-nogui": lock_a, "cpu-gui": lock_b}),
        )
    )
    assert result_b.state == "BUILT"
    assert result_b.runtime_id != result_a.runtime_id
    pointer = (runtime_root / br.ACTIVE_POINTER_NAME).read_text(encoding="utf-8").strip()
    assert pointer == result_b.runtime_id

    # Rollback to the retained (verified) runtime A: only the selector moves.
    setup_lock = br.SetupActivationLock(runtime_root, 5.0).acquire()
    try:
        previous = br.perform_activation(
            runtime_root, result_a.runtime_id,
            env.identity, env.inventory, BOOTSTRAP_SHA,
            make_variant_loader({"cpu-nogui": lock_a, "cpu-gui": lock_b}),
            setup_lock, "rollback",
        )
        assert previous == result_b.runtime_id
    finally:
        setup_lock.release()
    assert (runtime_root / br.ACTIVE_POINTER_NAME).read_text(encoding="utf-8").strip() == result_a.runtime_id

    # Corrupt retained B, then try to activate it: refused, selector
    # unchanged, and B is NOT deleted.
    version_b = runtime_root / br.VERSIONS_DIRNAME / result_b.runtime_id
    (version_b / "Lib" / "site-packages" / "alpha" / "__init__.py").write_bytes(b"tampered")
    setup_lock = br.SetupActivationLock(runtime_root, 5.0).acquire()
    try:
        with pytest.raises(br.BuilderError) as excinfo:
            br.perform_activation(
                runtime_root, result_b.runtime_id,
                env.identity, env.inventory, BOOTSTRAP_SHA,
                make_variant_loader({"cpu-nogui": lock_a, "cpu-gui": lock_b}),
                setup_lock, "activate",
            )
        assert excinfo.value.code == "state"
    finally:
        setup_lock.release()
    assert (runtime_root / br.ACTIVE_POINTER_NAME).read_text(encoding="utf-8").strip() == result_a.runtime_id
    versions = sorted(p.name for p in (runtime_root / br.VERSIONS_DIRNAME).iterdir())
    assert result_a.runtime_id in versions
    assert result_b.runtime_id in versions  # verified runtimes are never deleted


# ---------------------------------------------------------------------------
# runtime_entry hostile-environment isolation (re-review)
# ---------------------------------------------------------------------------


def test_runtime_entry_resists_hostile_environment(tmp_path):
    """runtime_entry must prove assembled-interpreter ownership even when
    the inherited environment injects PYTHONHOME/PYTHONPATH/PYTHONSTARTUP
    and a hostile site-packages, from an unrelated CWD."""
    entry = ROOT / "scripts" / "runtime_entry.py"
    assert entry.is_file()
    hostile_root = tmp_path / "hostile"
    (hostile_root / "site-packages").mkdir(parents=True)
    marker = hostile_root / "site-packages" / "marker.txt"
    (hostile_root / "site-packages" / "sitecustomize.py").write_text(
        f"open(r{str(marker)!r}, 'w').write('hostile-site-ran')\n", encoding="utf-8"
    )
    (hostile_root / "startup.py").write_text("print('should-not-run')\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(
        {
            "PYTHONHOME": str(hostile_root),
            "PYTHONPATH": str(hostile_root),
            "PYTHONSTARTUP": str(hostile_root / "startup.py"),
        }
    )
    # As the real launcher contract invokes it (-I -B):
    proc = subprocess.run(
        [sys.executable, "-I", "-B", str(entry), "--self-test"],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout.strip())
    assert report["ok"] is True
    assert report["isolated"] is True
    assert report["dont_write_bytecode"] is True
    assert report["host_env_purged"] is True
    assert report["usersite_disabled"] is True
    assert report["bytecode_disabled_env"] is True
    assert report["executable"] == sys.executable
    assert not marker.exists()  # the hostile sitecustomize never executed
    # Without interpreter flags the host state can reach interpreter
    # startup, but runtime_entry still proves the in-process purge and the
    # exact executable it will run the app on:
    env_no_home = {k: v for k, v in env.items() if k != "PYTHONHOME"}
    proc2 = subprocess.run(
        [sys.executable, str(entry), "--self-test"],
        env=env_no_home,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )
    report2 = json.loads(proc2.stdout.strip())
    assert report2["host_env_purged"] is True
    assert report2["usersite_disabled"] is True
    assert report2["executable"] == sys.executable


# ---------------------------------------------------------------------------
# Reproducibility: independence from cache root, build root, CUDA state
# ---------------------------------------------------------------------------


def test_canonical_identity_independent_of_cache_and_build_root(env, tmp_path):
    """The runtime ID and the canonical manifest must not depend on the
    artifact cache location or the absolute runtime root: two builds that
    differ ONLY in those produce identical canonical identity."""
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    mapping = {env.identity.source_url: env.archive.read_bytes()}
    mapping.update({url: p.read_bytes() for url, p in paths.items()})
    root_a, root_b = tmp_path / "root-a", tmp_path / "root-b"
    fetcher_a = br.OnlineFetcher(tmp_path / "cache-a", "cpu-nogui", opener=FakeOpener(dict(mapping)))
    fetcher_b = br.OnlineFetcher(tmp_path / "cache-b", "cpu-nogui", opener=FakeOpener(dict(mapping)))
    result_a = br.run_build(
        make_context(
            env, lock,
            runtime_root=root_a,
            fetcher=fetcher_a,
            installer=FakeInstaller(artifacts),
            interpreter=FakeInterpreter(lock),
        )
    )
    result_b = br.run_build(
        make_context(
            env, lock,
            runtime_root=root_b,
            fetcher=fetcher_b,
            installer=FakeInstaller(artifacts),
            interpreter=FakeInterpreter(lock),
        )
    )
    assert result_a.state == "BUILT"
    assert result_b.state == "BUILT"
    assert result_a.runtime_id == result_b.runtime_id
    manifest_a = br.read_manifest(result_a.runtime_dir / br.MANIFEST_NAME)
    manifest_b = br.read_manifest(result_b.runtime_dir / br.MANIFEST_NAME)
    assert manifest_a["canonical"] == manifest_b["canonical"]
    # and both promoted trees verify under the corrected verifier:
    for result, lock_ in ((result_a, lock), (result_b, lock)):
        checks = br.verify_runtime(result.runtime_dir, lock_, env.identity, env.inventory, BOOTSTRAP_SHA)
        assert all(c.code == "PASS" for c in checks)


# ---------------------------------------------------------------------------
# Adversarial regressions: trust-boundary repair round (A-O)
# ---------------------------------------------------------------------------


def test_verify_fails_on_lock_identity_drift_same_versions(env, tmp_path):
    """Adversarial A: the presented CURRENT lock differs in identity
    (lock SHA-256) although every package version is unchanged. The
    expected runtime ID is derived from the current inputs, so verification
    must fail on lock_authority AND runtime_id (the manifest's self-reported
    identity is evidence, not authority)."""
    runtime_root, result, lock = _fake_built_runtime(env, tmp_path)
    drifted = br.Lock(**{**lock.__dict__, "lock_sha256": "77" * 32})
    checks = br.verify_runtime(
        runtime_root / br.VERSIONS_DIRNAME / result.runtime_id,
        drifted, env.identity, env.inventory, BOOTSTRAP_SHA,
    )
    codes = {c.name: c.code for c in checks}
    assert codes["lock_authority"] == "MISMATCH"
    assert codes["runtime_id"] == "MISMATCH"


def test_verify_fails_on_artifact_identity_drift_same_versions(env, tmp_path):
    """Adversarial A (artifact identity): the current lock's wheel record
    carries a different SHA-256/URL for the SAME package version. The
    manifest's artifact records must match the current lock's records
    EXACTLY (classification/filename/sha256/source/url), so verification
    fails on artifact_authority even though versions are unchanged and the
    tree itself is untouched."""
    runtime_root, result, lock = _fake_built_runtime(env, tmp_path)
    wheels2 = tuple(
        br.WheelArtifact(**{**w.__dict__, "sha256": "77" * 32, "url": "https://rogue.invalid/other"})
        if w.name == "alpha" else w
        for w in lock.wheels
    )
    drifted = br.Lock(**{**lock.__dict__, "wheels": wheels2})
    checks = br.verify_runtime(
        runtime_root / br.VERSIONS_DIRNAME / result.runtime_id,
        drifted, env.identity, env.inventory, BOOTSTRAP_SHA,
    )
    codes = {c.name: c.code for c in checks}
    assert codes["artifact_authority"] == "MISMATCH"
    assert "alpha" in _check_by_name(checks, "artifact_authority").note


def test_verify_fails_when_manifest_inventory_shrinks_and_payload_tampered(env, tmp_path):
    """Adversarial B: the attacker deletes a distribution from the
    manifest's installed_tree inventory AND tampers with that
    distribution's payload, hoping the verifier's scope follows the
    manifest. The verification scope is filesystem-derived and the tree
    digest is anchored in the runtime ID: the installed_tree consistency
    check, the RECORD integrity check and the runtime_id check all fail."""
    runtime_root, result, lock = _fake_built_runtime(env, tmp_path)
    version_dir = runtime_root / br.VERSIONS_DIRNAME / result.runtime_id
    manifest_path = version_dir / br.MANIFEST_NAME
    manifest = br.read_manifest(manifest_path)
    del manifest["canonical"]["installed_tree"]["distributions"]["alpha"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (version_dir / "Lib" / "site-packages" / "alpha" / "__init__.py").write_bytes(b"evil")
    checks = br.verify_runtime(version_dir, lock, env.identity, env.inventory, BOOTSTRAP_SHA)
    codes = {c.name: c.code for c in checks}
    assert codes["runtime_id"] == "MISMATCH"
    assert codes["installed_tree"] == "MISMATCH"
    assert codes["record_integrity"] == "MISMATCH"


def test_verify_fails_on_duplicate_distribution_name(env, tmp_path):
    """Adversarial C: a second distribution directory (here an egg-info
    twin) declares the same normalized package name (shadow inventory).
    Independent enumeration detects the duplicate and rejects it; the
    exact normalized key set can no longer be satisfied."""
    runtime_root, result, lock = _fake_built_runtime(env, tmp_path)
    version_dir = runtime_root / br.VERSIONS_DIRNAME / result.runtime_id
    site = version_dir / "Lib" / "site-packages"
    dupe = site / "alpha.egg-info"
    dupe.mkdir()
    (dupe / "METADATA").write_bytes(b"Metadata-Version: 2.1\nName: alpha\nVersion: 1.0.0\n")
    checks = br.verify_runtime(version_dir, lock, env.identity, env.inventory, BOOTSTRAP_SHA)
    dists = _check_by_name(checks, "installed_distributions")
    assert dists.code == "MISMATCH"
    assert "duplicate" in dists.note
    assert "alpha" in dists.note


@pytest.mark.parametrize(
    "name",
    ["runtime-python.json", "runtime-lock.json", "runtime-python-licenses.json"],
)
def test_offline_rejects_modified_common_contract_copy(env, tmp_path, name):
    """Adversarial F/G/H: a modified copy of ANY Commit-1 contract file in
    the offline layout (not just the variant requirements lock) is
    detected by the byte-exact comparison against the authoritative
    repository contract bytes - the layout can never be its own authority."""
    _a, _p, lock, offline = _strict_offline(env)
    copy = offline / "locks" / name
    copy.write_bytes(copy.read_bytes() + b"tampered\n")
    with pytest.raises(br.BuilderError) as excinfo:
        br.verify_offline_inputs(offline, "cpu-nogui", env.identity, env.inventory, lock, fake_contract_bytes(env, lock))
    assert excinfo.value.code == "offline"
    assert name in excinfo.value.message


def test_offline_rejects_empty_other_variant_wheel_dir(env, tmp_path):
    """Adversarial I: an EMPTY other-variant wheel directory carries no
    files, so the file walk alone cannot see it; the explicit directory
    topology check rejects it."""
    _a, _p, lock, offline = _strict_offline(env)
    (offline / "wheels" / "cuda-nogui").mkdir()
    with pytest.raises(br.BuilderError) as excinfo:
        br.verify_offline_inputs(offline, "cpu-nogui", env.identity, env.inventory, lock, fake_contract_bytes(env, lock))
    assert excinfo.value.code == "offline"
    assert "wheels" in excinfo.value.message


def test_offline_rejects_duplicate_wheel_record(env, tmp_path):
    """Adversarial J: a duplicated wheel record in the offline artifact
    manifest is rejected by the strict record check (no filename may
    appear twice)."""
    _a, _p, lock, offline = _strict_offline(env)
    manifest_path = offline / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["wheels"].append(dict(manifest["wheels"][0]))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(br.BuilderError) as excinfo:
        br.verify_offline_inputs(offline, "cpu-nogui", env.identity, env.inventory, lock, fake_contract_bytes(env, lock))
    assert "duplicate wheel record" in excinfo.value.message


def test_offline_rejects_manifest_url_repoint(env, tmp_path):
    """Adversarial K: the offline manifest keeps a wheel's filename and
    SHA-256 but re-points its URL/source at a rogue mirror. Filename+hash
    alone are not enough: the strict record comparison compares
    filename/sha256/url/source against the current lock and fails."""
    _a, _p, lock, offline = _strict_offline(env)
    manifest_path = offline / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = dict(manifest["wheels"][0])
    record["url"] = "https://rogue.invalid/" + record["filename"]
    manifest["wheels"][0] = record
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(br.BuilderError) as excinfo:
        br.verify_offline_inputs(offline, "cpu-nogui", env.identity, env.inventory, lock, fake_contract_bytes(env, lock))
    assert excinfo.value.code == "offline"
    assert "deviates from the lock" in excinfo.value.message


def test_post_promotion_verify_failure_blocks_built(monkeypatch, env, tmp_path):
    """Adversarial N: a FRESH candidate that passes the staging checks but
    fails the FULL verification against the promoted path is never
    reported BUILT and never activated; the candidate is left in place
    (non-destructive) and the active pointer is untouched."""
    def always_fail(runtime_dir, lock, identity, inventory, bootstrap_sha256):
        return [br.Check("runtime_id", "MISMATCH", "forced failure (adversarial N)")]

    monkeypatch.setattr(br, "verify_runtime", always_fail)
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    runtime_root = tmp_path / "runtime"
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=runtime_root,
                fetcher=online_fetcher(env, "cpu-nogui", paths),
                installer=FakeInstaller(artifacts),
                interpreter=FakeInterpreter(lock),
                activate=True,
            )
        )
    assert excinfo.value.code == "post-promotion-verify"
    # nothing was activated ...
    assert not (runtime_root / br.ACTIVE_POINTER_NAME).exists()
    # ... and the failing candidate is left in place (never deleted):
    versions = list((runtime_root / br.VERSIONS_DIRNAME).iterdir())
    assert len(versions) == 1
    candidate = versions[0]
    assert candidate.is_dir()
    # its ID is still the canonical one derived from current inputs plus
    # the (intact) tree digest:
    tree_digest = br.compute_installed_tree_digest(candidate)
    expected = br.compute_runtime_id(BOOTSTRAP_SHA, env.identity, env.inventory, lock, tree_digest)
    assert candidate.name == expected


def test_activation_refuses_corrupted_active_retained_runtime(env, tmp_path):
    """Adversarial O: the previously active (retained) runtime is corrupted
    while a different, healthy runtime exists. Swapping the selector to the
    healthy runtime is refused: the retained runtime must pass the FULL
    verification with its own variant's current Commit-1 lock. The selector
    stays put and nothing is deleted."""
    runtime_root, result_a, lock_a = _built_runtime_activated(env, tmp_path, "runtime-o")
    artifacts_b, paths_b, lock_b = build_artifacts(env, "cpu-gui", "cpu")
    result_b = br.run_build(
        make_context(
            env, lock_b,
            runtime_root=runtime_root,
            fetcher=online_fetcher(env, "cpu-gui", paths_b),
            installer=FakeInstaller(artifacts_b),
            interpreter=FakeInterpreter(lock_b),
            activate=False,
        )
    )
    assert result_b.state == "BUILT"
    pointer = (runtime_root / br.ACTIVE_POINTER_NAME).read_text(encoding="utf-8").strip()
    assert pointer == result_a.runtime_id
    # Corrupt the ACTIVE (and therefore retained-on-swap) runtime A.
    version_a = runtime_root / br.VERSIONS_DIRNAME / result_a.runtime_id
    (version_a / "Lib" / "site-packages" / "alpha" / "__init__.py").write_bytes(b"tampered")
    setup_lock = br.SetupActivationLock(runtime_root, 5.0).acquire()
    try:
        with pytest.raises(br.BuilderError) as excinfo:
            br.perform_activation(
                runtime_root, result_b.runtime_id,
                env.identity, env.inventory, BOOTSTRAP_SHA,
                make_variant_loader({"cpu-nogui": lock_a, "cpu-gui": lock_b}),
                setup_lock, "activate",
            )
        assert excinfo.value.code == "state"
        assert "retained" in excinfo.value.message
    finally:
        setup_lock.release()
    # selector unchanged; both runtimes still exist (nothing deleted).
    assert (runtime_root / br.ACTIVE_POINTER_NAME).read_text(encoding="utf-8").strip() == result_a.runtime_id
    versions = {p.name for p in (runtime_root / br.VERSIONS_DIRNAME).iterdir()}
    assert result_a.runtime_id in versions
    assert result_b.runtime_id in versions


# ---------------------------------------------------------------------------
# Baseline authority repair (final narrow re-review round)
# ---------------------------------------------------------------------------

DISTLIB_EXE_PATH = "pip/_vendor/distlib/t64.exe"
_STALE_B64 = base64.urlsafe_b64encode(bytes.fromhex("11" * 32)).decode().rstrip("=")


def _custom_env(tmp_path, extra_members):
    """Synthetic environment built on a python archive that carries
    EXACTLY the extra members given (they override the default archive
    members, e.g. the bundled pip RECORD), so tests can reproduce the
    pinned archive's known defects or absence thereof."""
    archive = make_python_archive(tmp_path / "fake-python.tar.gz", extra_members=extra_members)
    identity = fake_identity(archive)
    return types.SimpleNamespace(
        tmp_path=tmp_path,
        archive=archive,
        identity=identity,
        inventory=fake_inventory(identity),
        wheels_dir=tmp_path / "wheel-store",
    )


class BaselineTamperingInstaller(FakeInstaller):
    """Simulates an installer that modifies a bundled (baseline) file -
    here the pip METADATA - after materializing the locked wheels."""

    def install(self, python_exe, requirements_file, cwd, cache_dir):
        super().install(python_exe, requirements_file, cwd, cache_dir)
        (cwd / "Lib" / "site-packages" / "pip-24.3.1.dist-info" / "METADATA").write_bytes(
            b"Metadata-Version: 2.1\nName: pip\nVersion: 24.3.1\nEVIL\n"
        )


class NonBaselineCorruptingInstaller(FakeInstaller):
    """Simulates an installer that corrupts a NON-baseline (locked-wheel)
    payload without updating its RECORD: baseline reconciliation must not
    touch it, and the full post-promotion verification must reject it."""

    def install(self, python_exe, requirements_file, cwd, cache_dir):
        super().install(python_exe, requirements_file, cwd, cache_dir)
        (cwd / "Lib" / "site-packages" / "alpha" / "__init__.py").write_bytes(
            b"corrupted-by-installer\n"
        )


class BaselineRecordLineDeleter(FakeInstaller):
    """Simulates an installer deleting a bundled baseline RECORD entry (the
    pip METADATA line) while leaving the actual file untouched: the
    reviewer's record-only deletion attack. The exact post-install path
    membership check (captured pre-install path set) must reject it."""

    def install(self, python_exe, requirements_file, cwd, cache_dir):
        super().install(python_exe, requirements_file, cwd, cache_dir)
        record = cwd / "Lib" / "site-packages" / "pip-24.3.1.dist-info" / "RECORD"
        lines = [
            ln
            for ln in record.read_text(encoding="utf-8").splitlines()
            if ln and not ln.startswith("pip-24.3.1.dist-info/METADATA,")
        ]
        record.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


class BaselineRecordLineAdder(FakeInstaller):
    """Simulates an installer adding an unapproved baseline RECORD path
    (a new file plus its RECORD line, no approved normalization reason):
    the exact path membership check must reject the addition in the other
    direction."""

    def install(self, python_exe, requirements_file, cwd, cache_dir):
        super().install(python_exe, requirements_file, cwd, cache_dir)
        site = cwd / "Lib" / "site-packages"
        exe = site / "pip" / "_vendor" / "distlib" / "rogue.exe"
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_bytes(b"ROGUE")
        record = site / "pip-24.3.1.dist-info" / "RECORD"
        line = f"pip/_vendor/distlib/rogue.exe,sha256={record_b64(b'ROGUE')},{len(b'ROGUE')}\n"
        record.write_text(record.read_text(encoding="utf-8") + line, encoding="utf-8", newline="\n")


def test_installer_modified_baseline_metadata_fails_build(env, tmp_path):
    """A. Installer-modified baseline METADATA: the pre-install captured
    authority (the verified archive's bytes) is the sole baseline
    authority; a file the installer modified after installation is
    rejected fail-closed. No re-anchor, no BUILT, no activation."""
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    runtime_root = tmp_path / "runtime"
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=runtime_root,
                fetcher=online_fetcher(env, "cpu-nogui", paths),
                installer=BaselineTamperingInstaller(artifacts),
                interpreter=FakeInterpreter(lock),
                activate=True,
            )
        )
    assert excinfo.value.code == "baseline_authority"
    assert "METADATA" in excinfo.value.message
    # Never reported built, never activated, nothing published.
    assert not (runtime_root / br.ACTIVE_POINTER_NAME).exists()
    versions = runtime_root / br.VERSIONS_DIRNAME
    assert not (versions.is_dir() and any(versions.iterdir()))


def test_known_distlib_baseline_exception_accepted_end_to_end(tmp_path):
    """B. The actual known archive-normalized distlib executable case
    still succeeds: the archive ships the distlib exe with a STALE
    wheel-era RECORD hash; reconciliation re-anchors ONLY that entry's
    hash/size fields to the CAPTURED pre-install (archive) bytes - path
    unchanged, file bytes unchanged - and the promoted runtime verifies
    fully, with the rewrite recorded as an operational normalization."""
    exe_bytes = b"FAKE-DISTLIB-EXE"
    pip_metadata = b"Metadata-Version: 2.1\nName: pip\nVersion: 24.3.1\n"
    pip_license = b"PIP-LICENSE"
    record = (
        f"{DISTLIB_EXE_PATH},sha256={_STALE_B64},{len(exe_bytes)}\n"
        f"pip-24.3.1.dist-info/LICENSE.txt,sha256={record_b64(pip_license)},{len(pip_license)}\n"
        f"pip-24.3.1.dist-info/METADATA,sha256={record_b64(pip_metadata)},{len(pip_metadata)}\n"
        "pip-24.3.1.dist-info/RECORD,,\n"
    ).encode("ascii")
    env = _custom_env(
        tmp_path,
        {
            "python/Lib/site-packages/pip/_vendor/distlib/t64.exe": exe_bytes,
            "python/Lib/site-packages/pip-24.3.1.dist-info/RECORD": record,
        },
    )
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    runtime_root = tmp_path / "runtime"
    result = br.run_build(
        make_context(
            env, lock,
            runtime_root=runtime_root,
            fetcher=online_fetcher(env, "cpu-nogui", paths),
            installer=FakeInstaller(artifacts),
            interpreter=FakeInterpreter(lock),
        )
    )
    assert result.state == "BUILT"
    version_dir = runtime_root / br.VERSIONS_DIRNAME / result.runtime_id
    manifest = br.read_manifest(version_dir / br.MANIFEST_NAME)
    reanchor_entries = [
        n for n in manifest["operational"]["normalizations"]
        if n["type"] == "baseline-record-reanchor"
    ]
    assert reanchor_entries, "the reviewed rewrite must be recorded operationally"
    assert reanchor_entries[0]["distributions"] == {"pip": [DISTLIB_EXE_PATH]}
    # Only the RECORD's hash/size fields were updated, to the captured
    # (archive) values; the path and the file bytes are unchanged.
    lines = (version_dir / "Lib" / "site-packages" / "pip-24.3.1.dist-info" / "RECORD").read_text(
        encoding="utf-8"
    ).splitlines()
    actual_digest = "sha256=" + base64.urlsafe_b64encode(
        hashlib.sha256(exe_bytes).digest()
    ).decode().rstrip("=")
    assert f"{DISTLIB_EXE_PATH},{actual_digest},{len(exe_bytes)}" in lines
    assert (version_dir / "Lib" / "site-packages" / "pip" / "_vendor" / "distlib" / "t64.exe").read_bytes() == exe_bytes
    # The promoted runtime verifies fully against the current inputs.
    checks = br.verify_runtime(version_dir, lock, env.identity, env.inventory, BOOTSTRAP_SHA)
    assert all(c.code == "PASS" for c in checks), [c for c in checks if c.code != "PASS"]


def test_non_baseline_stale_record_fails_build(env, tmp_path):
    """C. A locked-wheel (NON-baseline) distribution whose RECORD no
    longer matches its payload: the baseline reconciliation logic never
    touches it, and the full post-promotion verification fails the build
    closed (post-promotion-verify, not baseline_authority)."""
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    runtime_root = tmp_path / "runtime"
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=runtime_root,
                fetcher=online_fetcher(env, "cpu-nogui", paths),
                installer=NonBaselineCorruptingInstaller(artifacts),
                interpreter=FakeInterpreter(lock),
            )
        )
    assert excinfo.value.code == "post-promotion-verify"
    assert "record_integrity" in excinfo.value.message
    assert not (runtime_root / br.ACTIVE_POINTER_NAME).exists()


def test_baseline_non_allowlisted_stale_record_fails_build(tmp_path):
    """D. A baseline distribution file whose RECORD entry is stale OUTSIDE
    the reviewed exception set: even though the file exists and is
    exactly the archive's bytes, the stale METADATA entry is NOT a
    recognized archive-normalization case, so reconciliation fails closed
    (re-anchor is no generic repair mechanism)."""
    pip_metadata = b"Metadata-Version: 2.1\nName: pip\nVersion: 24.3.1\n"
    pip_license = b"PIP-LICENSE"
    record = (
        f"pip-24.3.1.dist-info/METADATA,sha256={_STALE_B64},{len(pip_metadata)}\n"
        f"pip-24.3.1.dist-info/LICENSE.txt,sha256={record_b64(pip_license)},{len(pip_license)}\n"
        "pip-24.3.1.dist-info/RECORD,,\n"
    ).encode("ascii")
    env = _custom_env(
        tmp_path,
        {"python/Lib/site-packages/pip-24.3.1.dist-info/RECORD": record},
    )
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    runtime_root = tmp_path / "runtime"
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=runtime_root,
                fetcher=online_fetcher(env, "cpu-nogui", paths),
                installer=FakeInstaller(artifacts),
                interpreter=FakeInterpreter(lock),
            )
        )
    assert excinfo.value.code == "baseline_authority"
    assert "METADATA" in excinfo.value.message
    assert not (runtime_root / br.ACTIVE_POINTER_NAME).exists()
    versions = runtime_root / br.VERSIONS_DIRNAME
    assert not (versions.is_dir() and any(versions.iterdir()))


def test_baseline_authority_unaffected_by_later_mutation(tmp_path):
    """E. The captured baseline authority is an immutable snapshot of the
    pre-install (verified-archive) state: mutating the file, the RECORD,
    or both afterwards cannot alter the expected digests, and
    reconciliation rejects each mutation. The allowlisted distlib entry is
    the only entry ever rewritten, from the captured values."""
    root = tmp_path / "staging"
    site = root / "Lib" / "site-packages"
    dist = site / "pip-24.3.1.dist-info"
    dist.mkdir(parents=True)
    exe = site / "pip" / "_vendor" / "distlib" / "t64.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"EXE")
    metadata = b"Metadata-Version: 2.1\nName: pip\nVersion: 24.3.1\n"
    (dist / "METADATA").write_bytes(metadata)
    healthy_meta = f"pip-24.3.1.dist-info/METADATA,sha256={record_b64(metadata)},{len(metadata)}"
    stale_record = (
        f"{DISTLIB_EXE_PATH},sha256={_STALE_B64},3\n"
        f"{healthy_meta}\n"
        "pip-24.3.1.dist-info/RECORD,,\n"
    )
    (dist / "RECORD").write_text(stale_record, encoding="utf-8")
    baseline_dirs = {"pip": str(dist)}
    authority = br.capture_baseline_authority(root, baseline_dirs)
    # The capture holds the authoritative pre-install RECORD path set AND
    # the per-file digest inventory.
    assert DISTLIB_EXE_PATH in authority["pip"]["paths"]
    assert authority["pip"]["files"][DISTLIB_EXE_PATH] == (hashlib.sha256(b"EXE").hexdigest(), 3)
    assert authority["pip"]["files"]["pip-24.3.1.dist-info/METADATA"] == (
        hashlib.sha256(metadata).hexdigest(),
        len(metadata),
    )
    snapshot = {
        name: {"paths": meta["paths"], "files": dict(meta["files"])}
        for name, meta in authority.items()
    }

    def reconcile():
        return br.reconcile_baseline_records(root, authority, baseline_dirs)

    # (1) File-only mutation: rejected (the allowlisted entry is still
    # rewrite-eligible because its bytes remain the captured bytes).
    (dist / "METADATA").write_bytes(b"EVIL")
    _reanchored, violations = reconcile()
    assert any("METADATA" in v and "modified" in v for v in violations)
    (dist / "METADATA").write_bytes(metadata)

    # (2) RECORD-only mutation (stale hash on a non-allowlisted entry):
    # rejected.
    (dist / "RECORD").write_text(
        f"{DISTLIB_EXE_PATH},sha256={_STALE_B64},3\n"
        f"pip-24.3.1.dist-info/METADATA,sha256={_STALE_B64},{len(metadata)}\n"
        "pip-24.3.1.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    _reanchored, violations = reconcile()
    assert any("METADATA" in v for v in violations)

    # (3) File and RECORD both mutated: still rejected.
    (dist / "METADATA").write_bytes(b"EVIL2")
    _reanchored, violations = reconcile()
    assert any("METADATA" in v for v in violations)

    # Restored state: the allowlisted stale distlib entry is the only
    # issue and is legitimately re-anchored from the captured bytes.
    (dist / "METADATA").write_bytes(metadata)
    (dist / "RECORD").write_text(stale_record, encoding="utf-8")
    reanchored, violations = reconcile()
    assert violations == []
    assert reanchored == {"pip": [DISTLIB_EXE_PATH]}
    # The captured authority never moved through any of it.
    assert authority == snapshot


def test_baseline_record_path_deletion_fails_build(env, tmp_path):
    """A2 (reviewer reproduction). The installer deletes the bundled pip's
    METADATA line from pip-24.3.1.dist-info/RECORD while leaving the
    actual METADATA file unchanged. The exact post-install path
    membership check (captured pre-install path set minus only the
    approved droppable classes) rejects the deletion: fail closed - no
    BUILT, no promotion as a verified runtime, no activation, no
    weakened RECORD accepted."""
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    runtime_root = tmp_path / "runtime"
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=runtime_root,
                fetcher=online_fetcher(env, "cpu-nogui", paths),
                installer=BaselineRecordLineDeleter(artifacts),
                interpreter=FakeInterpreter(lock),
                activate=True,
            )
        )
    assert excinfo.value.code == "baseline_authority"
    assert "deleted" in excinfo.value.message
    assert "pip-24.3.1.dist-info/METADATA" in excinfo.value.message
    assert not (runtime_root / br.ACTIVE_POINTER_NAME).exists()
    versions = runtime_root / br.VERSIONS_DIRNAME
    assert not (versions.is_dir() and any(versions.iterdir()))


def test_baseline_record_path_addition_fails_build(env, tmp_path):
    """B2. An unexpected baseline RECORD path added by the installer
    (new file + RECORD line, no approved normalization reason) is
    rejected: path membership is exact in BOTH directions."""
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    runtime_root = tmp_path / "runtime"
    with pytest.raises(br.BuilderError) as excinfo:
        br.run_build(
            make_context(
                env, lock,
                runtime_root=runtime_root,
                fetcher=online_fetcher(env, "cpu-nogui", paths),
                installer=BaselineRecordLineAdder(artifacts),
                interpreter=FakeInterpreter(lock),
            )
        )
    assert excinfo.value.code == "baseline_authority"
    assert "added" in excinfo.value.message
    assert "pip/_vendor/distlib/rogue.exe" in excinfo.value.message
    assert not (runtime_root / br.ACTIVE_POINTER_NAME).exists()
    versions = runtime_root / br.VERSIONS_DIRNAME
    assert not (versions.is_dir() and any(versions.iterdir()))


def test_approved_direct_url_drop_still_accepted(tmp_path):
    """C2. The approved direct_url.json normalization (file removed,
    RECORD line dropped) must NOT trigger a false baseline-authority
    failure: such paths belong to the explicitly approved droppable
    class. The build succeeds and records the normalization."""
    direct_url = b'{"url": "file:///cache/wheels/pip"}\n'
    pip_metadata = b"Metadata-Version: 2.1\nName: pip\nVersion: 24.3.1\n"
    pip_license = b"PIP-LICENSE"
    record = (
        f"pip-24.3.1.dist-info/direct_url.json,sha256={record_b64(direct_url)},{len(direct_url)}\n"
        f"pip-24.3.1.dist-info/LICENSE.txt,sha256={record_b64(pip_license)},{len(pip_license)}\n"
        f"pip-24.3.1.dist-info/METADATA,sha256={record_b64(pip_metadata)},{len(pip_metadata)}\n"
        "pip-24.3.1.dist-info/RECORD,,\n"
    ).encode("ascii")
    env = _custom_env(
        tmp_path,
        {
            "python/Lib/site-packages/pip-24.3.1.dist-info/direct_url.json": direct_url,
            "python/Lib/site-packages/pip-24.3.1.dist-info/RECORD": record,
        },
    )
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    runtime_root = tmp_path / "runtime"
    result = br.run_build(
        make_context(
            env, lock,
            runtime_root=runtime_root,
            fetcher=online_fetcher(env, "cpu-nogui", paths),
            installer=FakeInstaller(artifacts),
            interpreter=FakeInterpreter(lock),
        )
    )
    assert result.state == "BUILT"
    version_dir = runtime_root / br.VERSIONS_DIRNAME / result.runtime_id
    # The approved removal happened: file gone, line dropped ...
    assert not (version_dir / "Lib" / "site-packages" / "pip-24.3.1.dist-info" / "direct_url.json").exists()
    rec_lines = (version_dir / "Lib" / "site-packages" / "pip-24.3.1.dist-info" / "RECORD").read_text(
        encoding="utf-8"
    ).splitlines()
    assert not any(l.startswith("pip-24.3.1.dist-info/direct_url.json,") for l in rec_lines)
    # ... and is recorded in the operational manifest.
    manifest = br.read_manifest(version_dir / br.MANIFEST_NAME)
    direct = [n for n in manifest["operational"]["normalizations"] if n["type"] == "direct-url"]
    assert direct, "the approved direct_url normalization must be recorded"
    assert "Lib/site-packages/pip-24.3.1.dist-info/direct_url.json" in direct[0]["paths"]
    # And the promoted runtime verifies fully.
    checks = br.verify_runtime(version_dir, lock, env.identity, env.inventory, BOOTSTRAP_SHA)
    assert all(c.code == "PASS" for c in checks), [c for c in checks if c.code != "PASS"]


def test_approved_bytecode_drop_still_accepted(tmp_path):
    """D2. The approved generated-bytecode normalization (pyc scrubbed,
    RECORD line dropped) must NOT trigger a false baseline-authority
    failure for a baseline distribution that shipped the bytecode: the
    build succeeds, the pyc is gone, and the normalization is recorded."""
    pyc = b"FAKE-PYC"
    pip_metadata = b"Metadata-Version: 2.1\nName: pip\nVersion: 24.3.1\n"
    pip_license = b"PIP-LICENSE"
    record = (
        f"pip/__pycache__/__init__.cpython-312.pyc,sha256={record_b64(pyc)},{len(pyc)}\n"
        f"pip-24.3.1.dist-info/LICENSE.txt,sha256={record_b64(pip_license)},{len(pip_license)}\n"
        f"pip-24.3.1.dist-info/METADATA,sha256={record_b64(pip_metadata)},{len(pip_metadata)}\n"
        "pip-24.3.1.dist-info/RECORD,,\n"
    ).encode("ascii")
    env = _custom_env(
        tmp_path,
        {
            "python/Lib/site-packages/pip/__pycache__/__init__.cpython-312.pyc": pyc,
            "python/Lib/site-packages/pip-24.3.1.dist-info/RECORD": record,
        },
    )
    artifacts, paths, lock = build_artifacts(env, "cpu-nogui", "cpu")
    runtime_root = tmp_path / "runtime"
    result = br.run_build(
        make_context(
            env, lock,
            runtime_root=runtime_root,
            fetcher=online_fetcher(env, "cpu-nogui", paths),
            installer=FakeInstaller(artifacts),
            interpreter=FakeInterpreter(lock),
        )
    )
    assert result.state == "BUILT"
    version_dir = runtime_root / br.VERSIONS_DIRNAME / result.runtime_id
    # The approved scrub happened: no pyc anywhere, line dropped ...
    assert not list(version_dir.rglob("*.pyc"))
    rec_lines = (version_dir / "Lib" / "site-packages" / "pip-24.3.1.dist-info" / "RECORD").read_text(
        encoding="utf-8"
    ).splitlines()
    assert not any(l.startswith("pip/__pycache__/") for l in rec_lines)
    # ... and is recorded in the operational manifest.
    manifest = br.read_manifest(version_dir / br.MANIFEST_NAME)
    bytecode = [n for n in manifest["operational"]["normalizations"] if n["type"] == "bytecode-records"]
    assert bytecode, "the approved bytecode normalization must be recorded"
    # And the promoted runtime verifies fully.
    checks = br.verify_runtime(version_dir, lock, env.identity, env.inventory, BOOTSTRAP_SHA)
    assert all(c.code == "PASS" for c in checks), [c for c in checks if c.code != "PASS"]


def test_manifest_console_scripts_tamper_fails_verify(env, tmp_path):
    """F. Manifest console_scripts tamper ONLY: the entry points are
    re-derived from the installed entry_points.txt metadata during
    verification, so a manifest field alone can never redefine them."""
    runtime_root, result, lock = _fake_built_runtime(env, tmp_path)
    version_dir = runtime_root / br.VERSIONS_DIRNAME / result.runtime_id
    manifest_path = version_dir / br.MANIFEST_NAME
    manifest = br.read_manifest(manifest_path)
    tampered = dict(manifest["canonical"].get("console_scripts", {}))
    tampered["rogue-script"] = "rogue.module:main"
    manifest["canonical"]["console_scripts"] = tampered
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checks = br.verify_runtime(version_dir, lock, env.identity, env.inventory, BOOTSTRAP_SHA)
    codes = {c.name: c.code for c in checks}
    assert codes["console_scripts"] == "MISMATCH"
    # A manifest-only edit does not move the tree digest: the runtime-ID
    # check still passes, which is exactly why the entry points are
    # re-derived independently instead of trusting the manifest field.
    assert codes["runtime_id"] == "PASS"


def test_installed_entry_points_tamper_fails_verify(env, tmp_path):
    """G. Installed entry_points.txt tamper: the re-derived entry points
    deviate from the canonical manifest (console_scripts MISMATCH) AND
    the tree digest moves (runtime_id MISMATCH) - double coverage."""
    runtime_root, result, lock = _fake_built_runtime(env, tmp_path)
    version_dir = runtime_root / br.VERSIONS_DIRNAME / result.runtime_id
    (
        version_dir / "Lib" / "site-packages" / "alpha-1.0.0.dist-info" / "entry_points.txt"
    ).write_text("[console_scripts]\nrogue-script = rogue.module:main\n", encoding="utf-8")
    checks = br.verify_runtime(version_dir, lock, env.identity, env.inventory, BOOTSTRAP_SHA)
    codes = {c.name: c.code for c in checks}
    assert codes["console_scripts"] == "MISMATCH"
    assert codes["runtime_id"] == "MISMATCH"
