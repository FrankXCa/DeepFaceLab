#!/usr/bin/env python3
"""Generate and validate Phase-13 Windows runtime input locks.

This helper resolves only the pinned dependency graph and selects artifacts. It
does not install, assemble, activate, or launch a runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import html.parser
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "runtime-lock.json"
PYTHON_IDENTITY_PATH = ROOT / "runtime-python.json"
LICENSE_INVENTORY_PATH = ROOT / "runtime-python-licenses.json"
GENERATOR_PATH = Path(__file__).resolve()
LOCK_SCHEMA = "dfl-runtime-artifact-lock-v1"
LICENSE_SCHEMA = "dfl-python-license-inventory-v1"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
PIN_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)$")
ENTRY_PREFIX = "# artifact: "
PRIVATE_PATH_RE = re.compile(
    r"(?i)(?:[a-z]:[\\/](?:users|codes|downloads|desktop|personal)[\\/]|"
    r"appdata[\\/]|(?:^|[\\/])\.venv(?:[\\/]|$))"
)


def fail(message: str) -> "NoReturn":
    raise SystemExit(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file_raw(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_text_bytes(data: bytes) -> bytes:
    """Return UTF-8 text with only CRLF and lone CR normalized to LF."""
    text = data.decode("utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def sha256_text_canonical(data: bytes) -> str:
    return sha256_bytes(canonical_text_bytes(data))


def sha256_file_text_canonical(path: Path) -> str:
    return sha256_text_canonical(path.read_bytes())


def normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def read_requirement_inputs(initial_paths: list[str]) -> tuple[dict[str, str], list[Path]]:
    pins: dict[str, str] = {}
    visited: set[Path] = set()

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in visited:
            return
        if not path.is_relative_to(ROOT):
            fail(f"requirement input escapes repository: {path}")
        visited.add(path)
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith(("-r ", "--requirement ")):
                include = line.split(None, 1)[1]
                visit(path.parent / include)
                continue
            if line.startswith("-"):
                continue
            match = PIN_RE.fullmatch(line)
            if not match:
                fail(f"direct requirement is not an exact pin in {path.relative_to(ROOT)}: {line}")
            name, version = normalized_name(match.group(1)), match.group(2)
            previous = pins.setdefault(name, version)
            if previous != version:
                fail(f"conflicting direct pins for {name}: {previous} and {version}")

    for relative in initial_paths:
        visit(ROOT / relative)
    return pins, sorted(visited, key=lambda item: item.relative_to(ROOT).as_posix())


def expected_versions(config: dict, variant: str) -> dict[str, str]:
    result = dict(config["resolved_versions"]["common"])
    variant_config = config["variants"][variant]
    if variant.endswith("-gui"):
        result.update(config["resolved_versions"]["gui"])
    backend = variant_config["torch_backend"]
    result["torch"] = config["resolved_versions"]["torch"][backend]
    return dict(sorted(result.items()))


def verify_uv(uv_executable: str, expected: str) -> None:
    completed = subprocess.run(
        [uv_executable, "--version"], check=True, capture_output=True, text=True
    )
    actual = completed.stdout.strip().split()
    if len(actual) < 2 or actual[:2] != ["uv", expected]:
        fail(f"required uv {expected}, got: {completed.stdout.strip()!r}")


def resolve_variant(config: dict, variant: str, uv_executable: str) -> dict[str, str]:
    variant_config = config["variants"][variant]
    expected = expected_versions(config, variant)
    with tempfile.TemporaryDirectory(prefix="dfl-lock-") as temporary:
        temporary_path = Path(temporary)
        constraints = temporary_path / "constraints.txt"
        output = temporary_path / "resolved.txt"
        constraints.write_text(
            "".join(f"{name}=={version}\n" for name, version in expected.items()),
            encoding="utf-8",
            newline="\n",
        )
        target = config["target"]
        resolver = config["resolver"]
        command = [
            uv_executable,
            "pip",
            "compile",
            *variant_config["inputs"],
            "--constraints",
            str(constraints),
            "--python-version",
            target["python"],
            "--python-platform",
            target["platform"],
            "--only-binary",
            resolver["only_binary"],
            "--torch-backend",
            variant_config["torch_backend"],
            "--index-strategy",
            resolver["index_strategy"],
            "--resolution",
            resolver["resolution"],
            "--format",
            "requirements.txt",
            "--output-file",
            str(output),
            "--no-header",
            "--no-annotate",
            "--no-progress",
            "--quiet",
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        resolved: dict[str, str] = {}
        for line in output.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = PIN_RE.fullmatch(line)
            if not match:
                fail(f"unexpected uv output for {variant}: {line}")
            resolved[normalized_name(match.group(1))] = match.group(2)
    if resolved != expected:
        fail(
            f"uv resolution for {variant} does not match reviewed graph:\n"
            f"expected={expected!r}\nactual={resolved!r}"
        )
    return resolved


def url_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "DeepFaceLab-lock-generator/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def wheel_parts(filename: str) -> tuple[str, str, str]:
    if not filename.lower().endswith(".whl"):
        fail(f"not a wheel: {filename}")
    parts = filename[:-4].rsplit("-", 3)
    if len(parts) != 4:
        fail(f"invalid wheel filename: {filename}")
    return parts[1], parts[2], parts[3]


def compatibility_tags(filename: str) -> list[str]:
    python_tags, abi_tags, platform_tags = wheel_parts(filename)
    return sorted(
        f"{python_tag}-{abi_tag}-{platform_tag}"
        for python_tag in python_tags.split(".")
        for abi_tag in abi_tags.split(".")
        for platform_tag in platform_tags.split(".")
    )


def wheel_score(filename: str) -> tuple[int, int, int] | None:
    python_tags, abi_tags, platform_tags = wheel_parts(filename)
    platforms = platform_tags.split(".")
    if "win_amd64" in platforms:
        platform_score = 2
    elif "any" in platforms:
        platform_score = 1
    else:
        return None

    best_python = -1
    for python_tag in python_tags.split("."):
        for abi_tag in abi_tags.split("."):
            if python_tag == "cp312" and abi_tag == "cp312":
                best_python = max(best_python, 1000)
            elif python_tag.startswith("cp") and python_tag[2:].isdigit() and abi_tag == "abi3":
                minor = int(python_tag[2:])
                if 32 <= minor <= 312:
                    best_python = max(best_python, 600 + minor)
            elif python_tag == "py3" and abi_tag == "none":
                best_python = max(best_python, 300)
            elif python_tag == "py2" and abi_tag == "none" and "py3" in python_tags.split("."):
                best_python = max(best_python, 290)
    if best_python < 0:
        return None
    # Prefer platform-specific wheels, then the strongest Python/ABI match,
    # then a filename without a build tag. Equal scores are rejected below.
    return platform_score, best_python, -filename.count("-")


def select_one(candidates: list[dict], package: str, version: str) -> dict:
    compatible: list[tuple[tuple[int, int, int], dict]] = []
    for candidate in candidates:
        filename = candidate["filename"]
        if not filename.lower().endswith(".whl"):
            continue
        score = wheel_score(filename)
        if score is not None:
            compatible.append((score, candidate))
    if not compatible:
        fail(f"no Windows cp312 wheel for {package}=={version}")
    compatible.sort(key=lambda item: (item[0], item[1]["filename"]), reverse=True)
    best_score = compatible[0][0]
    best = [candidate for score, candidate in compatible if score == best_score]
    if len(best) != 1:
        fail(
            f"ambiguous Windows cp312 wheel selection for {package}=={version}: "
            + ", ".join(item["filename"] for item in best)
        )
    return best[0]


def pypi_artifact(package: str, version: str) -> dict:
    metadata = url_json(
        f"https://pypi.org/pypi/{urllib.parse.quote(package)}/{urllib.parse.quote(version)}/json"
    )
    candidates = []
    for item in metadata.get("urls", []):
        digest = item.get("digests", {}).get("sha256")
        if item.get("packagetype") == "bdist_wheel" and digest and HASH_RE.fullmatch(digest):
            candidates.append(
                {"filename": item["filename"], "sha256": digest, "url": item["url"]}
            )
    selected = select_one(candidates, package, version)
    selected["source"] = "https://pypi.org/simple"
    return selected


def torch_artifact(config: dict, backend: str, version: str) -> dict:
    index = config["indexes"]["torch"]["cuda" if backend == "cu130" else "cpu"]
    page_url = index.rstrip("/") + "/torch/"
    request = urllib.request.Request(page_url, headers={"User-Agent": "DeepFaceLab-lock-generator/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        page = response.read().decode("utf-8")
    parser = LinkParser()
    parser.feed(page)
    candidates: list[dict] = []
    wanted = "torch-" + version
    for href in parser.links:
        absolute = urllib.parse.urljoin(page_url, href)
        parsed = urllib.parse.urlsplit(absolute)
        filename = urllib.parse.unquote(PurePosixPath(parsed.path).name)
        if not filename.startswith(wanted + "-") or not filename.endswith(".whl"):
            continue
        fragment = urllib.parse.parse_qs(parsed.fragment)
        hashes = fragment.get("sha256", [])
        if len(hashes) != 1 or not HASH_RE.fullmatch(hashes[0]):
            fail(f"missing SHA256 fragment for official torch artifact: {absolute}")
        clean_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))
        candidates.append({"filename": filename, "sha256": hashes[0], "url": clean_url})
    selected = select_one(candidates, "torch", version)
    selected["source"] = index
    return selected


def input_hashes(paths: list[Path]) -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256_file_text_canonical(path)
        for path in paths
    }


def render_lock(
    config: dict,
    variant: str,
    versions: dict[str, str],
    artifacts: dict[str, dict],
) -> str:
    variant_config = config["variants"][variant]
    direct, input_paths = read_requirement_inputs(variant_config["inputs"])
    hashes = input_hashes(input_paths)
    config_hash = sha256_file_text_canonical(CONFIG_PATH)
    generator_hash = sha256_file_text_canonical(GENERATOR_PATH)
    aggregate_input_hash = sha256_bytes(canonical_json(hashes).encode("ascii"))
    target = config["target"]
    resolver = config["resolver"]
    lines = [
        "# DeepFaceLab deterministic Windows runtime artifact lock",
        f"# schema: {LOCK_SCHEMA}",
        f"# variant: {variant}",
        f"# target-python: CPython {target['python']} (cp312)",
        f"# target-platform: {target['platform']} ({target['wheel_platform']})",
        f"# resolver: uv {resolver['version']}",
        f"# resolver-config-sha256: {config_hash}",
        f"# generator-sha256: {generator_hash}",
        "# generation-command: python scripts/generate_runtime_locks.py --generate "
        "--uv <UV_0_12_18> --python-archive <PYTHON_ARCHIVE>",
        "# semantics: PACKAGE_SET_IDENTITY=name+version; "
        "ARTIFACT_IDENTITY=filename+raw-byte-sha256; INPUT_IDENTITY=inputs+config+tool",
        "# text-input-sha256: UTF-8 bytes after CRLF/CR -> LF normalization only",
        "# future-install: --only-binary=:all: --require-hashes --no-deps",
    ]
    lines.extend(f"# input-sha256: {path}={digest}" for path, digest in hashes.items())
    lines.extend(["", "--only-binary=:all:", "--require-hashes", ""])
    for name, version in versions.items():
        artifact = artifacts[name]
        entry = {
            "classification": "direct" if name in direct else "transitive",
            "compatibility_tags": compatibility_tags(artifact["filename"]),
            "filename": artifact["filename"],
            "input_hashes": hashes,
            "input_identity_sha256": aggregate_input_hash,
            "name": name,
            "resolver": f"uv {resolver['version']}",
            "resolver_config_sha256": config_hash,
            "sha256": artifact["sha256"],
            "source": artifact["source"],
            "target_platform": target["platform"],
            "target_python": target["python"],
            "url": artifact["url"],
            "variant": variant,
            "version": version,
        }
        lines.append(ENTRY_PREFIX + canonical_json(entry))
        lines.append(f"{name} @ {artifact['url']} \\")
        lines.append(f"    --hash=sha256:{artifact['sha256']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def license_path_matches(path: PurePosixPath) -> bool:
    parts = [part.lower() for part in path.parts]
    basename = parts[-1]
    return (
        basename.startswith("license")
        or basename == "copying"
        or basename.startswith("copying.")
        or basename == "notice"
        or basename.startswith("notice.")
        or "licenses" in parts[:-1]
    )


def build_license_inventory(archive: Path, identity: dict) -> dict:
    archive_data = identity["archive"]
    actual_hash = sha256_file_raw(archive)
    if actual_hash != archive_data["sha256"]:
        fail(
            f"standalone Python SHA256 mismatch: expected {archive_data['sha256']}, "
            f"got {actual_hash}"
        )
    entries = []
    with tarfile.open(archive, mode="r:gz") as package:
        for member in package:
            if not member.isfile():
                continue
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                fail(f"unsafe archive member: {member.name}")
            if not license_path_matches(path):
                continue
            stream = package.extractfile(member)
            if stream is None:
                fail(f"could not read archive member: {member.name}")
            entries.append(
                {
                    "path": path.as_posix(),
                    "sha256": sha256_bytes(stream.read()),
                    "source_archive_filename": archive_data["filename"],
                    "source_archive_sha256": archive_data["sha256"],
                }
            )
    entries.sort(key=lambda item: item["path"].casefold())
    if not entries:
        fail("standalone Python license discovery was empty")
    paths = {item["path"].casefold() for item in entries}
    required = {
        "python/license.txt",
        "python/lib/site-packages/pip-24.3.1.dist-info/license.txt",
        "python/tcl/tk8.6/demos/license.terms",
        "python/tcl/tk8.6/license.terms",
    }
    missing = sorted(required - paths)
    if missing:
        fail("standalone Python license inventory is missing: " + ", ".join(missing))
    inventory_hash = sha256_bytes(canonical_json(entries).encode("ascii"))
    return {
        "discovery": {
            "case_sensitive": False,
            "recursive": True,
            "rules": ["LICENSE*", "COPYING", "COPYING.*", "NOTICE", "NOTICE.*", "licenses/"],
        },
        "entries": entries,
        "inventory_sha256": inventory_hash,
        "schema": LICENSE_SCHEMA,
        "source_archive": archive_data,
    }


def parse_lock(path: Path) -> tuple[dict[str, str], list[dict]]:
    headers: dict[str, str] = {}
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(ENTRY_PREFIX):
            entries.append(json.loads(line[len(ENTRY_PREFIX) :]))
        elif line.startswith("# ") and ": " in line:
            key, value = line[2:].split(": ", 1)
            headers[key] = value
    return headers, entries


def validate_license_inventory(identity: dict) -> None:
    inventory = load_json(LICENSE_INVENTORY_PATH)
    if inventory.get("schema") != LICENSE_SCHEMA:
        fail("invalid Python license inventory schema")
    if inventory.get("source_archive") != identity["archive"]:
        fail("Python license inventory source identity mismatch")
    entries = inventory.get("entries")
    if not isinstance(entries, list) or not entries:
        fail("Python license inventory is empty")
    expected_hash = sha256_bytes(canonical_json(entries).encode("ascii"))
    if inventory.get("inventory_sha256") != expected_hash:
        fail("Python license inventory content hash mismatch")
    previous = ""
    for entry in entries:
        path = entry.get("path", "")
        if path.casefold() < previous:
            fail("Python license inventory is not deterministically sorted")
        previous = path.casefold()
        if not HASH_RE.fullmatch(entry.get("sha256", "")):
            fail(f"invalid license SHA256: {path}")
        if entry.get("source_archive_filename") != identity["archive"]["filename"]:
            fail(f"license source filename mismatch: {path}")
        if entry.get("source_archive_sha256") != identity["archive"]["sha256"]:
            fail(f"license source SHA256 mismatch: {path}")
    required = {
        "python/license.txt",
        "python/lib/site-packages/pip-24.3.1.dist-info/license.txt",
        "python/tcl/tk8.6/demos/license.terms",
        "python/tcl/tk8.6/license.terms",
    }
    paths = {entry["path"].casefold() for entry in entries}
    if not required.issubset(paths):
        fail("Python license inventory lost required nested entries")


def validate_locks(config: dict) -> None:
    if (ROOT / "requirements-lock.txt").exists():
        fail("ambiguous generic requirements-lock.txt must not exist")
    config_hash = sha256_file_text_canonical(CONFIG_PATH)
    generator_hash = sha256_file_text_canonical(GENERATOR_PATH)
    # Dev/test-only packages that must never enter a packaged runtime lock.
    # tqdm is NOT in this set: it is a common production runtime dependency
    # (imported unconditionally by core/interact/interact.py at application
    # startup, all variants).
    forbidden = {
        "ipython", "matplotlib", "pillow", "psutil", "pytest", "ffmpeg"
    }
    for variant, variant_config in sorted(config["variants"].items()):
        lock_path = ROOT / f"requirements-lock-{variant}.txt"
        if not lock_path.is_file():
            fail(f"missing lock: {lock_path.name}")
        text = lock_path.read_text(encoding="utf-8")
        if "--only-binary=:all:" not in text or "--require-hashes" not in text:
            fail(f"install contract missing from {lock_path.name}")
        if PRIVATE_PATH_RE.search(text):
            fail(f"private/local path leaked into {lock_path.name}")
        headers, entries = parse_lock(lock_path)
        if headers.get("schema") != LOCK_SCHEMA or headers.get("variant") != variant:
            fail(f"schema/variant header mismatch in {lock_path.name}")
        if headers.get("resolver") != f"uv {config['resolver']['version']}":
            fail(f"resolver header mismatch in {lock_path.name}")
        if headers.get("resolver-config-sha256") != config_hash:
            fail(f"stale resolver config hash in {lock_path.name}")
        if headers.get("generator-sha256") != generator_hash:
            fail(f"stale generator hash in {lock_path.name}")
        direct, paths = read_requirement_inputs(variant_config["inputs"])
        hashes = input_hashes(paths)
        expected_input_identity = sha256_bytes(canonical_json(hashes).encode("ascii"))
        expected = expected_versions(config, variant)
        actual = {entry["name"]: entry["version"] for entry in entries}
        if actual != expected or list(actual) != sorted(actual):
            fail(f"package-set identity mismatch in {lock_path.name}")
        if forbidden.intersection(actual):
            fail(f"dev/orphan package in {lock_path.name}: {sorted(forbidden.intersection(actual))}")
        if "ffmpeg-python" not in actual or actual.get("future") != "1.0.0":
            fail(f"ffmpeg-python/future contract missing in {lock_path.name}")
        gui_names = {"pyqt5", "pyqt5-qt5", "pyqt5-sip"}
        if gui_names.issubset(actual) != variant.endswith("-gui"):
            fail(f"GUI package-set mismatch in {lock_path.name}")
        expected_torch = config["resolved_versions"]["torch"][variant_config["torch_backend"]]
        if actual.get("torch") != expected_torch:
            fail(f"torch variant mismatch in {lock_path.name}")
        for entry in entries:
            name = entry["name"]
            required_fields = {
                "classification", "compatibility_tags", "filename", "input_hashes",
                "input_identity_sha256", "name", "resolver", "resolver_config_sha256",
                "sha256", "source", "target_platform", "target_python", "url",
                "variant", "version",
            }
            if set(entry) != required_fields:
                fail(f"artifact schema mismatch for {name} in {lock_path.name}")
            if entry["classification"] != ("direct" if name in direct else "transitive"):
                fail(f"classification mismatch for {name} in {lock_path.name}")
            if entry["input_hashes"] != hashes or entry["input_identity_sha256"] != expected_input_identity:
                fail(f"input identity mismatch for {name} in {lock_path.name}")
            if entry["resolver_config_sha256"] != config_hash:
                fail(f"config identity mismatch for {name} in {lock_path.name}")
            if entry["variant"] != variant or entry["target_python"] != config["target"]["python"]:
                fail(f"target identity mismatch for {name} in {lock_path.name}")
            if not entry["filename"].endswith(".whl") or wheel_score(entry["filename"]) is None:
                fail(f"non-wheel/incompatible artifact for {name} in {lock_path.name}")
            if PurePosixPath(urllib.parse.urlsplit(entry["url"]).path).name != urllib.parse.quote(
                entry["filename"], safe="-_."
            ) and urllib.parse.unquote(PurePosixPath(urllib.parse.urlsplit(entry["url"]).path).name) != entry["filename"]:
                fail(f"artifact URL/filename mismatch for {name} in {lock_path.name}")
            if not HASH_RE.fullmatch(entry["sha256"]):
                fail(f"invalid artifact SHA256 for {name} in {lock_path.name}")
            host = urllib.parse.urlsplit(entry["url"]).hostname
            if name == "torch":
                if host not in {"download.pytorch.org", "download-r2.pytorch.org"}:
                    fail(f"unofficial torch source in {lock_path.name}")
            elif host != "files.pythonhosted.org" or entry["source"] != "https://pypi.org/simple":
                fail(f"non-PyPI ordinary artifact for {name} in {lock_path.name}")


def generate(config: dict, uv_executable: str, python_archive: Path, accept_changes: bool) -> None:
    verify_uv(uv_executable, config["resolver"]["version"])
    identity = load_json(PYTHON_IDENTITY_PATH)
    inventory = build_license_inventory(python_archive, identity)
    if LICENSE_INVENTORY_PATH.exists() and not accept_changes:
        previous = load_json(LICENSE_INVENTORY_PATH)
        if previous.get("entries") != inventory["entries"]:
            fail("Python license inventory changed; review and rerun with --accept-changes")
    write_json(LICENSE_INVENTORY_PATH, inventory)

    artifact_cache: dict[tuple[str, str, str], dict] = {}
    for variant, variant_config in sorted(config["variants"].items()):
        versions = resolve_variant(config, variant, uv_executable)
        artifacts: dict[str, dict] = {}
        for name, version in versions.items():
            source_key = variant_config["torch_backend"] if name == "torch" else "pypi"
            cache_key = (name, version, source_key)
            if cache_key not in artifact_cache:
                artifact_cache[cache_key] = (
                    torch_artifact(config, variant_config["torch_backend"], version)
                    if name == "torch"
                    else pypi_artifact(name, version)
                )
            artifacts[name] = artifact_cache[cache_key]
        rendered = render_lock(config, variant, versions, artifacts)
        lock_path = ROOT / f"requirements-lock-{variant}.txt"
        if lock_path.exists() and not accept_changes and lock_path.read_text(encoding="utf-8") != rendered:
            fail(f"artifact lock changed: {lock_path.name}; review and rerun with --accept-changes")
        lock_path.write_text(rendered, encoding="utf-8", newline="\n")
    validate_license_inventory(identity)
    validate_locks(config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--generate", action="store_true", help="resolve and generate all locks")
    action.add_argument("--check", action="store_true", help="validate tracked locks without network")
    parser.add_argument("--uv", help="path to the exact uv 0.12.18 executable")
    parser.add_argument("--python-archive", type=Path, help="path to the pinned standalone archive")
    parser.add_argument(
        "--accept-changes",
        action="store_true",
        help="accept reviewed changes to an existing inventory/artifact lock",
    )
    args = parser.parse_args()
    config = load_json(CONFIG_PATH)
    identity = load_json(PYTHON_IDENTITY_PATH)
    if args.generate:
        if not args.uv or not args.python_archive:
            parser.error("--generate requires --uv and --python-archive")
        generate(config, args.uv, args.python_archive.resolve(), args.accept_changes)
    else:
        validate_license_inventory(identity)
        validate_locks(config)
    print("Phase-13 runtime input locks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
