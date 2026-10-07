# -----------------------------------------------------------------------------
# Phase 13 - P13-CURATED-WORKFLOW-LAUNCHERS: curated workflow launcher tests.
#
# Covers the curated per-workflow launchers shipped in launchers/ (47 thin
# wrappers) plus the machine-readable surface manifest
# launchers/workflow_launchers.json (the machine-readable home of the 56
# reference-disposition ledger for test purposes; the full ledger with
# observed purpose, resource dependency and validation status lives in the
# private Phase-13 state doc).
#
# Every curated launcher is a thin wrapper around launchers/dfl.bat - the
# sole execution boundary. The launcher files never run an interpreter,
# never install/modify anything, never read or write the active runtime
# selector themselves, and propagate the child exit code unchanged.
#
# Method:
#   * Static tests assert the thin-wrapper integrity of every curated .bat
#     (ASCII, LF, @echo off, allowed line shapes only, exactly the shared
#     dfl.bat call lines, exit-code propagation, no second execution
#     surface, no absolute paths, no delayed expansion, no forbidden
#     batch tokens), the manifest/ledger completeness (47 curated +
#     9 excluded references = the full 56-reference ledger; no launcher
#     for any excluded reference; the launcher directory matches the
#     manifest exactly), and the exact CLI vocabulary of every mapped
#     command against the frozen main.py argparse surface.
#   * Behavioural tests run the real tracked .bat files (copied unchanged
#     into a temporary fake repository tree, same harness as
#     tests/test_launchers.py) through cmd.exe with a fake bundled
#     interpreter that logs argv/executor/environment: exact forwarded
#     command text (byte-for-byte, incl. quoted values), arbitrary-CWD
#     safety (repo root / repo parent / unrelated directory), hostile
#     parent-environment isolation, user-argument forwarding for the
#     single forwarding launcher (cut-video), chained-command failure
#     propagation for the result-video launchers, and no repository-tree
#     mutation across the whole curated set.
#
# These tests are Windows-only (cmd batch semantics) and skip elsewhere.
# -----------------------------------------------------------------------------

import json
import os
import re
from pathlib import Path

import pytest

import test_launchers as tl

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows-only launcher tests")

REPO_ROOT = tl.REPO_ROOT
LAUNCHERS_DIR = tl.LAUNCHERS_DIR
MAIN_PY = REPO_ROOT / "main.py"

SHARED_LAUNCHERS = ("dfl.bat", "dfl-envreport.bat", "dfl-setup-runtime.bat")
MANIFEST = LAUNCHERS_DIR / "workflow_launchers.json"

ALLOWED_DISPOSITIONS = {
    "REPRODUCE_BEHAVIOR",
    "REPLACE",
    "RESOURCE_GATED",
    "PHASE14_PENDING",
    "POST_BASELINE",
    "DROP",
}

# Subcommands of the audit families that must NOT be exposed by any curated
# launcher (hard exclusions of this feature).
FORBIDDEN_SUBCOMMANDS = ("editor",)
FORBIDDEN_TOKENS = (
    "FaceEnhancer",
    "model_generic_xseg",
    "pretrain_faces",
    "tensorflow",
    "EbSynth",
    "XnView",
)

# Fake-repo runtime selector shared by all behavioural tests here (same
# value test_launchers.py uses for its normal-launcher tests).
FAKE_RUNTIME_ID = "ab" * 32

# Representative launchers for the multi-CWD matrix (one per launcher shape:
# forwarding, wildcard arg, longest fixed arg, chain, xseg, util, sort).
CWD_MATRIX_FILES = (
    "cut-video.bat",
    "extract-frames-src.bat",
    "extract-faces-dst-manual-reextract-debug.bat",
    "train-amp.bat",
    "xseg-apply-trained-masks-src.bat",
    "faces-src-pack.bat",
    "faces-src-enhance.bat",
    "sort-faces-src.bat",
    "merge-amp.bat",
    "result-video-mp4.bat",
)

CHAIN_LAUNCHERS = (
    "result-video-avi.bat",
    "result-video-mp4.bat",
    "result-video-mp4-lossless.bat",
    "result-video-mov-lossless.bat",
)


# ---------------------------------------------------------------------------
# manifest / launcher inventory
# ---------------------------------------------------------------------------


def _manifest() -> dict:
    text = MANIFEST.read_text(encoding="utf-8")
    return json.loads(text)


def _curated_entries() -> list:
    entries = _manifest()["curated"]
    assert entries, "manifest lists no curated launchers"
    return entries


def _curated_names() -> list:
    return [e["file"] for e in _curated_entries()]


def _excluded_entries() -> list:
    entries = _manifest()["excluded_references"]
    assert entries, "manifest lists no excluded references"
    return entries


def _launcher_text(name: str) -> str:
    data = (LAUNCHERS_DIR / name).read_bytes()
    return data.decode("utf-8")


def _call_lines(text: str) -> list:
    prefix = 'call "%~dp0dfl.bat" '
    return [ln[len(prefix):] for ln in text.split("\n") if ln.startswith(prefix)]


def _call_args(text: str) -> str:
    calls = _call_lines(text)
    assert len(calls) == 1, f"expected exactly one call line, got {len(calls)}"
    return calls[0]


def _expected_args_line(name: str, extra: str = "") -> str:
    """Exact argv the fake interpreter must log for a curated launcher.

    dfl.bat executes:
        <runtime python> -I -B "scripts\\runtime_entry.py" <forwarded args>
    and the fake interpreter logs `ARGS %*` - the literal, un-reparsed
    text - so the expectation is the raw call-argument text.
    """
    prefix = '-I -B "scripts\\runtime_entry.py" '
    text = _launcher_text(name)
    calls = _call_lines(text)
    if name == "cut-video.bat":
        # The forwarding launcher appends its user arguments after the
        # fixed `--input-file` token. cmd drops the space before an empty
        # %* expansion, so with no user arguments the forwarded text ends
        # at the fixed token (the app then reports a clean argparse error).
        base = "videoed cut-video --input-file"
        assert calls[0] == base + " %*"
        if extra:
            return prefix + base + " " + extra
        return prefix + base
    if name in CHAIN_LAUNCHERS:
        # Both calls run on success; read_fakeint_log returns the LAST
        # ARGS line, i.e. the second call's forwarded command.
        assert len(calls) == 2
        return prefix + calls[-1]
    assert len(calls) == 1
    return prefix + calls[0]


# ---------------------------------------------------------------------------
# A. manifest and ledger completeness (static)
# ---------------------------------------------------------------------------


def test_manifest_is_well_formed():
    m = _manifest()
    for key in ("feature", "phase", "shared_launcher", "curated", "excluded_references"):
        assert key in m, f"manifest missing key: {key}"
    assert m["shared_launcher"] == "dfl.bat"

    names = _curated_names()
    assert len(names) == len(set(names)), "duplicate curated launcher in manifest"
    for e in _curated_entries():
        for key in ("file", "group", "purpose", "forwarding"):
            assert key in e, f"curated entry missing {key}: {e}"
        assert e["file"].endswith(".bat")
        assert isinstance(e["forwarding"], bool)
        assert (e["forwarding"] is True) == (e["file"] == "cut-video.bat"), (
            "exactly one curated launcher forwards user arguments: cut-video"
        )

    refs = [x["reference_launcher"] for x in _excluded_entries()]
    assert len(refs) == len(set(refs)), "duplicate excluded reference in manifest"
    for x in _excluded_entries():
        for key in ("reference_launcher", "disposition", "reason"):
            assert key in x, f"excluded entry missing {key}: {x}"
        assert x["disposition"] in ALLOWED_DISPOSITIONS, x
        assert x["reason"].strip(), x


def test_launcher_directory_matches_manifest():
    on_disk = {n for n in os.listdir(LAUNCHERS_DIR) if n.endswith(".bat")}
    on_disk -= set(SHARED_LAUNCHERS)
    in_manifest = set(_curated_names())
    assert on_disk == in_manifest, (
        f"launcher dir / manifest mismatch: "
        f"missing-from-manifest={sorted(on_disk - in_manifest)} "
        f"missing-from-dir={sorted(in_manifest - on_disk)}"
    )
    assert MANIFEST.is_file(), "tracked surface manifest missing"


def test_full_ledger_covers_fifty_six_references():
    # The complete reference-launcher ledger is 56 root .bat entries
    # (audit doc section 334: 56 root launchers, authoritative); this
    # feature's ledger disposes every one of them.
    #
    # Supersession (P13-GENERIC-XSEG-RESOURCE-POLICY): the two
    # "5.XSeg Generic) ... apply.bat" references previously excluded as
    # RESOURCE_GATED are now REPRODUCED by the curated
    # xseg-apply-generic-masks-src/dst launchers, which forward
    # --model-dir to the documented user-provided resource location
    # resources/xseg_generic_model (CONFIGURABLE_USER_SUPPLIED_PATH:
    # no bundled model bytes, no redistribution, no downloads).
    assert len(_curated_entries()) == 47
    assert len(_excluded_entries()) == 9
    assert len(_curated_entries()) + len(_excluded_entries()) == 56

    counts = {
        disposition: (
            (len(_curated_entries()) if disposition == "REPRODUCE_BEHAVIOR" else 0)
            + sum(x["disposition"] == disposition for x in _excluded_entries())
        )
        for disposition in ALLOWED_DISPOSITIONS
    }
    assert counts == {
        "REPRODUCE_BEHAVIOR": 49,
        "REPLACE": 1,
        "DROP": 1,
        "RESOURCE_GATED": 1,
        "PHASE14_PENDING": 0,
        "POST_BASELINE": 4,
    }


def test_excluded_reference_rows_are_present():
    # Explicit negative rows: the ledger must not silently shrink.
    refs = {x["reference_launcher"] for x in _excluded_entries()}
    for required in (
        "1) clear workspace.bat",
        "10.misc) make CPU only.bat",
        "10.misc) start EBSynth.bat",
        "4.1) data_src view aligned result.bat",
        "5.1) data_dst view aligned results.bat",
        "5.1) data_dst view aligned_debug results.bat",
        # Supersession (P13-GENERIC-XSEG-RESOURCE-POLICY): the two
        # "5.XSeg Generic) ... apply.bat" references are no longer
        # excluded - they are now reproduced by the curated
        # xseg-apply-generic-masks-src/dst launchers (CONFIGURABLE_
        # USER_SUPPLIED_PATH at the documented resources/xseg_generic_
        # model location; no bundled bytes, no redistribution, no
        # downloads).
        "5.XSeg) data_src mask - edit.bat",
        "5.XSeg) data_dst mask - edit.bat",
        "5.XSeg) train.bat",
    ):
        assert required in refs, f"required excluded row missing: {required}"


def test_no_launcher_created_for_excluded_dispositions():
    # No curated launcher may serve a RESOURCE_GATED / PHASE14_PENDING /
    # POST_BASELINE / DROP reference, and the hard-exclusion tokens must
    # not appear in any curated launcher body.
    groups = {e["group"] for e in _curated_entries()}
    for token in FORBIDDEN_TOKENS:
        for name in _curated_names():
            if token == "FaceEnhancer" and name == "faces-src-enhance.bat":
                continue
            assert token not in _launcher_text(name), f"{token} in {name}"
    for name in _curated_names():
        text = _launcher_text(name)
        assert "xseg editor" not in text, f"xseg editor exposed by {name}"
    assert groups <= {
        "videoed", "sort", "util", "facesettool", "extract", "xseg",
        "train", "exportdfm", "merge",
    }


def test_xseg_editor_references_are_reproduced_by_generic_cli():
    expected = {
        "5.XSeg) data_src mask - edit.bat": "workspace\\data_src\\aligned",
        "5.XSeg) data_dst mask - edit.bat": "workspace\\data_dst\\aligned",
    }
    rows = {
        row["reference_launcher"]: row
        for row in _excluded_entries()
        if row["reference_launcher"] in expected
    }
    assert set(rows) == set(expected)
    for reference, input_dir in expected.items():
        row = rows[reference]
        assert row["disposition"] == "REPRODUCE_BEHAVIOR"
        assert f"dfl.bat xseg editor --input-dir {input_dir}" in row["reason"]

    # These historical fixed-path wrappers are reproduced by the generic
    # packaged CLI, not by adding duplicate curated launchers.
    assert all("editor" not in name for name in _curated_names())

    main_text = MAIN_PY.read_text(encoding="utf-8")
    assert 'xseg_parser.add_parser( "editor", help="XSeg editor.")' in main_text
    editor_block = main_text.split('xseg_parser.add_parser( "editor"', 1)[1]
    editor_block = editor_block.split('xseg_parser.add_parser( "apply"', 1)[0]
    assert "--input-dir" in editor_block
    assert "XSegEditor.start" in editor_block


def test_faceenhancer_workflow_is_curated_and_pending_row_is_removed():
    reference = "4.2) data_src util faceset enhance.bat"
    rows = [x for x in _excluded_entries() if x["reference_launcher"] == reference]
    assert rows == []
    assert all(
        row["disposition"] != "PHASE14_PENDING"
        for row in _excluded_entries()
    )

    entries = {entry["file"]: entry for entry in _curated_entries()}
    entry = entries["faces-src-enhance.bat"]
    assert entry["group"] == "facesettool"
    assert entry["forwarding"] is False
    assert "data_src" in entry["purpose"]
    assert "FaceEnhancer" in entry["purpose"]
    assert (LAUNCHERS_DIR / "faces-src-enhance.bat").is_file()

    text = _launcher_text("faces-src-enhance.bat")
    assert _call_args(text) == (
        'facesettool enhance --input-dir "workspace\\data_src\\aligned"')
    assert "%*" not in text
    assert not re.search(r"\b(?:python|py|pip)\b", text, re.IGNORECASE)
    assert not re.search(r"[0-9a-f]{64}", text)
    assert not re.search(r"\b(?:install|download)\b", text, re.IGNORECASE)
    assert text.rstrip().endswith("exit /b %ERRORLEVEL%")

    main_text = MAIN_PY.read_text(encoding="utf-8")
    enhance_block = main_text.split(
        'facesettool_parser.add_parser ("enhance"', 1)[1]
    enhance_block = enhance_block.split(
        'facesettool_parser.add_parser ("resize"', 1)[0]
    assert "--input-dir" in enhance_block
    assert "--force-gpu-idxs" in enhance_block
    assert "type=parse_gpu_idxs" in enhance_block


def test_quick96_train_and_merge_are_curated_without_export():
    entries = {entry["file"]: entry for entry in _curated_entries()}
    expected = {
        "train-quick96.bat": ("train", "train the Quick96 model"),
        "merge-quick96.bat": ("merge", "merge aligned data_dst faces"),
    }
    for name, (group, purpose_fragment) in expected.items():
        assert (LAUNCHERS_DIR / name).is_file()
        assert entries[name]["group"] == group
        assert entries[name]["forwarding"] is False
        assert purpose_fragment in entries[name]["purpose"]

    excluded = {row["reference_launcher"] for row in _excluded_entries()}
    assert "6) train Quick96.bat" not in excluded
    assert "7) merge Quick96.bat" not in excluded
    assert "export-quick96-dfm.bat" not in entries
    assert not (LAUNCHERS_DIR / "export-quick96-dfm.bat").exists()
    assert {
        name for name in _curated_names()
        if "Quick96" in _launcher_text(name)
    } == set(expected)

    assert _call_args(_launcher_text("train-quick96.bat")) == (
        'train --training-data-src-dir "workspace\\data_src\\aligned" '
        '--training-data-dst-dir "workspace\\data_dst\\aligned" '
        '--model-dir "workspace\\model" --model Quick96 '
        '--no-preview --silent-start')
    assert _call_args(_launcher_text("merge-quick96.bat")) == (
        'merge --input-dir "workspace\\data_dst" '
        '--output-dir "workspace\\data_dst\\merged" '
        '--output-mask-dir "workspace\\data_dst\\merged_mask" '
        '--aligned-dir "workspace\\data_dst\\aligned" '
        '--model-dir "workspace\\model" --model Quick96')


# ---------------------------------------------------------------------------
# B. thin-wrapper integrity of every curated .bat (static)
# ---------------------------------------------------------------------------

CALL_PREFIX = 'call "%~dp0dfl.bat" '


def _classify(line: str):
    if line == "@echo off":
        return "head"
    if line == "":
        return "blank"
    if line == "rem" or line.startswith("rem "):
        return "rem"
    if line == "exit /b %ERRORLEVEL%":
        return "final-exit"
    if line == "if errorlevel 1 exit /b %ERRORLEVEL%":
        return "gate"
    if line.startswith(CALL_PREFIX):
        return "call"
    return None


@pytest.mark.parametrize("name", _curated_names())
def test_curated_launcher_is_ascii_lf_thin_wrapper(name):
    data = (LAUNCHERS_DIR / name).read_bytes()
    text = data.decode("utf-8")  # raises on invalid UTF-8 (ASCII is a subset)
    for ch in data:
        assert ch < 128, f"non-ASCII byte in {name}"
    assert "\r" not in text, f"CR byte in {name} (LF contract)"

    lines = text.split("\n")
    assert lines[-1] == "", "file must end with a trailing newline"
    lines = lines[:-1]
    assert lines[0] == "@echo off", f"{name}: first line must be @echo off"
    assert lines[-1] == "exit /b %ERRORLEVEL%", f"{name}: last line must propagate the child exit code"

    for ln in lines:
        assert _classify(ln) is not None, f"{name}: unrecognized line shape: {ln!r}"

    calls = _call_lines(text)
    is_chain = name in CHAIN_LAUNCHERS
    assert len(calls) == (2 if is_chain else 1), f"{name}: unexpected call count {len(calls)}"
    if is_chain:
        # call / gate / call, in order.
        shapes = [_classify(ln) for ln in lines]
        assert shapes.count("call") == 2
        assert shapes.count("gate") == 1
        i1, i2 = shapes.index("call"), shapes.index("call", shapes.index("call") + 1)
        assert shapes[i1 + 1] == "gate", f"{name}: gate must directly follow the first call"
        assert shapes.index("gate") == (i1 + 1) < i2

    # No second execution surface, no delayed expansion, no absolute paths
    # in the forwarded arguments.
    for args in calls:
        assert "%" not in args or name == "cut-video.bat", (
            f"{name}: unexplained % token in mapped args: {args!r}"
        )
        if name == "cut-video.bat":
            assert args == "videoed cut-video --input-file %*"
        else:
            assert "%" not in args
        assert "!" not in args, f"{name}: delayed-expansion character"
        for meta in "&|<>^":
            assert meta not in args, f"{name}: cmd metacharacter {meta!r} in mapped args"
        assert not re.search(r"[A-Za-z]:[\\/]", args), f"{name}: absolute path in mapped args"
        # No cmd-level command-execution tokens: substring scans false-
        # positive on CLI words (`model` contains `del `, `remove` contains
        # `move `), so scan the cmd tokenization instead: quoted tokens are
        # data (paths) and only unquoted bare tokens could execute a command.
        toks = _split_cli_args_quoted(args)
        for i, (tok, quoted) in enumerate(toks):
            if quoted:
                continue
            low = tok.lower()
            assert low not in BARE_CMD_FORBIDDEN, (
                f"{name}: forbidden bare command token {tok!r} in mapped args"
            )
            if low == "cmd" and i + 1 < len(toks):
                nxt = toks[i + 1][0].lower()
                assert nxt not in ("/c", "/k"), f"{name}: cmd /c|/k in mapped args"

    # No absolute/user path anywhere in the body (headers included).
    assert not re.search(r"[A-Za-z]:[\\/]", text), f"{name}: drive-letter path in body"
    for bad in ("%userprofile%", "%homedrive%", "%homepath%", "%appdata%", "%programfiles%"):
        assert bad not in text.lower(), f"{name}: user-env reference {bad}"


@pytest.mark.parametrize("name", _curated_names())
def test_mapped_command_header_matches_call_line(name):
    text = _launcher_text(name)
    header = [ln[3:].lstrip() for ln in text.split("\n") if ln.startswith("rem   ")]
    calls = _call_lines(text)
    if name == "cut-video.bat":
        # The human-readable mapped command is documented; the call line
        # carries the forwarding token instead.
        assert any("videoed cut-video" in h and "<dropped-file>" in h for h in header)
        assert calls == ["videoed cut-video --input-file %*"]
        return
    assert header == calls, (
        f"{name}: header documented commands {header} do not match call lines {calls}"
    )


# ---------------------------------------------------------------------------
# C. exact CLI vocabulary against the frozen main.py argparse surface
# ---------------------------------------------------------------------------


def _main_py_vocabulary():
    src = MAIN_PY.read_text(encoding="utf-8")
    subcommands = set(re.findall(r'add_parser\s*\(\s*["\']([A-Za-z0-9_-]+)["\']', src))
    options = set(re.findall(r"add_argument\(\s*['\"]--([A-Za-z0-9-]+)['\"]", src))
    return subcommands, options


def _split_cli_args(text: str):
    """Whitespace-split that honours double quotes (drops the quotes).

    Backslashes are kept verbatim - Windows path data in this project.
    """
    return [t for t, _ in _split_cli_args_quoted(text)]


def _split_cli_args_quoted(text: str):
    """Like _split_cli_args but returns (token, was_quoted) pairs.

    Quoted tokens are data (cmd cannot execute their content); only
    unquoted tokens are candidates for cmd-level command execution, so
    the forbidden-bare-command scan must inspect unquoted tokens only.
    """
    out, cur, inq, quoted = [], [], False, False
    for ch in text:
        if ch == '"':
            inq = not inq
            quoted = True
        elif ch.isspace() and not inq:
            if cur:
                out.append(("".join(cur), quoted))
                cur, quoted = [], False
        else:
            cur.append(ch)
    if cur:
        out.append(("".join(cur), quoted))
    return out


BARE_CMD_FORBIDDEN = frozenset({
    "setlocal", "pause", "python", "py", "pip", "conda", "curl",
    "winget", "mklink", "del", "rmdir", "copy", "move", "net",
    "powershell", "start",
})


@pytest.mark.parametrize("name", _curated_names())
def test_mapped_command_uses_only_declared_cli_surface(name):
    subcommands, options = _main_py_vocabulary()
    for args in _call_lines(_launcher_text(name)):
        if name == "cut-video.bat":
            args = "videoed cut-video --input-file <file>"
        tokens = _split_cli_args(args)
        assert tokens, f"{name}: empty mapped command"
        top = tokens[0]
        assert top in subcommands, f"{name}: unknown subcommand {top!r}"
        rest = tokens[1:]
        # Nested subcommand (videoed / xseg / facesettool).
        if rest and not rest[0].startswith("--") and top in ("videoed", "xseg", "facesettool"):
            nested = rest[0]
            assert nested in subcommands, f"{name}: unknown nested subcommand {nested!r}"
            rest = rest[1:]
        for tok in rest:
            if tok.startswith("--"):
                assert tok[2:] in options, f"{name}: option {tok!r} not declared in main.py"


# ---------------------------------------------------------------------------
# D. behavioural tests through the real launch chain (fake repository)
# ---------------------------------------------------------------------------


def make_curated_fake_repo(root: Path, **kw) -> dict:
    """test_launchers harness + the curated launcher files, byte-identical."""
    info = tl.make_fake_repo(root, selector=FAKE_RUNTIME_ID, **kw)
    for name in _curated_names():
        src = LAUNCHERS_DIR / name
        dst = root / "launchers" / name
        dst.write_bytes(src.read_bytes())
        assert dst.read_bytes() == src.read_bytes(), f"copy of {name} diverged"
    return info


def _run_curated(tmp_path: Path, name: str, *args, cwd=None, exit_code="0"):
    repo_root = tmp_path / "fake-dfl"
    make_curated_fake_repo(repo_root)
    log = tmp_path / "fakeint.log"
    env = tl._env(
        tl.HOSTILE_PARENT_ENV,
        FAKEINT_LOG=str(log),
        FAKEINT_EXIT=exit_code,
    )
    launcher = repo_root / "launchers" / name
    if cwd is None:
        cwd = repo_root
    proc = tl.run_bat(launcher, *args, cwd=cwd, env=env)
    loginfo = tl.read_fakeint_log(log)
    return {"proc": proc, "repo": repo_root, "log": log, "loginfo": loginfo, "cwd": cwd}


@pytest.mark.parametrize(
    "name", ("train-quick96.bat", "merge-quick96.bat"))
def test_quick96_launcher_is_spaced_install_safe(tmp_path, name):
    repo_root = tmp_path / "fake dfl with spaces"
    make_curated_fake_repo(repo_root)
    log = tmp_path / f"{name}.log"
    env = tl._env(
        tl.HOSTILE_PARENT_ENV,
        FAKEINT_LOG=str(log),
        FAKEINT_EXIT="0",
    )

    proc = tl.run_bat(
        repo_root / "launchers" / name,
        cwd=tmp_path,
        env=env,
    )

    assert proc.returncode == 0
    loginfo = tl.read_fakeint_log(log)
    assert loginfo["args"] == _expected_args_line(name)
    assert loginfo["exec"] == (
        f"runtime\\versions\\{FAKE_RUNTIME_ID}\\python.exe")
    assert '"workspace\\' in _call_args(_launcher_text(name))


def test_faceenhancer_launcher_is_spaced_install_safe(tmp_path):
    name = "faces-src-enhance.bat"
    repo_root = tmp_path / "fake dfl with spaces"
    make_curated_fake_repo(repo_root)
    log = tmp_path / "faceenhancer-spaced.log"
    env = tl._env(
        tl.HOSTILE_PARENT_ENV,
        FAKEINT_LOG=str(log),
        FAKEINT_EXIT="0",
    )

    proc = tl.run_bat(
        repo_root / "launchers" / name,
        cwd=tmp_path,
        env=env,
    )

    assert proc.returncode == 0
    loginfo = tl.read_fakeint_log(log)
    assert loginfo["args"] == (
        '-I -B "scripts\\runtime_entry.py" facesettool enhance '
        '--input-dir "workspace\\data_src\\aligned"')
    assert loginfo["exec"] == (
        f"runtime\\versions\\{FAKE_RUNTIME_ID}\\python.exe")


def _assert_isolated_child_env(env: dict):
    # Sanitized families must be empty in the child (set-to-empty shows up
    # as an empty value in the findstr dump).
    cleared = {
        "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE",
        "NN_DEVICES_INITIALIZED", "NN_DEVICES_COUNT",
        "NN_DEVICE_0", "NN_DEVICE_1", "NN_DEVICE_ABC", "NN_DEVICE_INJ",
    }
    cleared |= {k for k in tl.HOSTILE_PARENT_ENV if k.lower().startswith("nn_device_")}
    for k in cleared:
        assert env.get(k, "") in ("", None), f"{k} not isolated: {env.get(k)!r}"
    # dfl.bat pins the no-bytecode / no-user-site hygiene contract.
    assert env.get("PYTHONNOUSERSITE") == "1"
    assert env.get("PYTHONDONTWRITEBYTECODE") == "1"
    # Non-family state passes through untouched.
    for k, v in (
        ("NN_DEVICE", "keepme"),
        ("NN_DEVICES_X", "keepme2"),
        ("CUDA_PATH", "stale-cuda-toolkit"),
        ("CUDA_HOME", "stale-cuda-home"),
        ("CUDA_VISIBLE_DEVICES", "0"),
        ("FAKESENTINEL_X", "sent"),
    ):
        assert env.get(k) == v, f"{k} altered: {env.get(k)!r}"


@pytest.mark.parametrize("name", _curated_names())
def test_curated_launcher_forwards_exact_command_and_isolates_env(tmp_path, name):
    r = _run_curated(tmp_path, name)
    assert r["proc"].returncode == 0
    expected = _expected_args_line(name)
    assert r["loginfo"]["args"] == expected, (
        f"{name}: forwarded command mismatch\n"
        f"  expected: {expected!r}\n"
        f"  actual  : {r['loginfo']['args']!r}"
    )
    # The interpreter that ran is the fake BUNDLED one inside the fake
    # runtime: proof that dfl.bat (not any host/venv interpreter) is the
    # sole execution boundary. dfl.bat invokes it relative to the repo
    # root CWD, so the logged executor is a relative path.
    assert r["loginfo"]["exec"] == f"runtime\\versions\\{FAKE_RUNTIME_ID}\\python.exe"
    _assert_isolated_child_env(r["loginfo"]["env"])
    # The hostile value must not have been evaluated as command text.
    assert not (r["repo"] / "marker-inj.txt").exists()


@pytest.mark.parametrize("name", CWD_MATRIX_FILES)
@pytest.mark.parametrize("cwd_mode", ("repo", "repo-parent", "unrelated"))
def test_curated_launcher_is_arbitrary_cwd_safe(tmp_path, name, cwd_mode):
    repo_root = tmp_path / "fake-dfl"
    make_curated_fake_repo(repo_root)
    if cwd_mode == "repo":
        cwd = repo_root
    elif cwd_mode == "repo-parent":
        cwd = repo_root.parent
    else:
        cwd = tmp_path / "elsewhere"
        cwd.mkdir()
    log = tmp_path / "fakeint.log"
    env = tl._env(tl.HOSTILE_PARENT_ENV, FAKEINT_LOG=str(log), FAKEINT_EXIT="0")
    proc = tl.run_bat(repo_root / "launchers" / name, cwd=cwd, env=env)
    assert proc.returncode == 0
    loginfo = tl.read_fakeint_log(log)
    assert loginfo["args"] == _expected_args_line(name)


def test_cut_video_forwards_dropped_file_with_spaces(tmp_path):
    # Drop semantics: the launcher's remaining arguments become the
    # --input-file value, quoted end to end.
    r = _run_curated(
        tmp_path, "cut-video.bat",
        r"C:\fake path\clip with space.mp4",
    )
    assert r["proc"].returncode == 0
    assert r["loginfo"]["args"] == (
        '-I -B "scripts\\runtime_entry.py" videoed cut-video --input-file '
        + r'"C:\fake path\clip with space.mp4"'
    )
    assert not (r["repo"] / "marker-inj.txt").exists()


def test_cut_video_forwards_file_and_options_with_metacharacters(tmp_path):
    r = _run_curated(
        tmp_path, "cut-video.bat",
        r"C:\fake path\clip & (evil).mp4",
        "--from-time", "00:00:10.000",
    )
    assert r["proc"].returncode == 0
    assert r["loginfo"]["args"] == (
        '-I -B "scripts\\runtime_entry.py" videoed cut-video --input-file '
        + r'"C:\fake path\clip & (evil).mp4" --from-time 00:00:10.000'
    )


@pytest.mark.parametrize("name", CHAIN_LAUNCHERS)
@pytest.mark.parametrize("exit_code,expected_rc,expected_calls", (
    ("0", 0, 2),
    ("7", 7, 1),
    ("42", 42, 1),
))
def test_result_video_chain_propagates_exit_code(tmp_path, name, exit_code, expected_rc, expected_calls):
    r = _run_curated(tmp_path, name, exit_code=exit_code)
    assert r["proc"].returncode == expected_rc
    raw = r["log"].read_text(encoding="utf-8", errors="replace") if r["log"].is_file() else ""
    args_lines = [ln[len("ARGS "):] for ln in raw.splitlines() if ln.strip().startswith("ARGS ")]
    assert len(args_lines) == expected_calls, (
        f"{name}: exit {exit_code} should run {expected_calls} call(s), ran {len(args_lines)}"
    )
    if expected_calls == 2:
        calls = _call_lines(_launcher_text(name))
        assert args_lines[0] == '-I -B "scripts\\runtime_entry.py" ' + calls[0]
        assert args_lines[1] == '-I -B "scripts\\runtime_entry.py" ' + calls[1]


@pytest.mark.parametrize("exit_code", ("0", "1", "2", "255"))
def test_launcher_layer_propagates_child_exit_code_verbatim(tmp_path, exit_code):
    # The launcher must never remap app-layer exit codes (0 = success,
    # 1/2 = app codes, 255 = extreme code); dfl.bat's own launcher-layer
    # codes (missing runtime/selector) are covered by test_launchers.py.
    r = _run_curated(tmp_path, "extract-faces-src.bat", exit_code=exit_code)
    assert r["proc"].returncode == int(exit_code)
    if exit_code == "0":
        assert r["loginfo"]["args"] == _expected_args_line("extract-faces-src.bat")


@pytest.mark.parametrize("exit_code", ("7", "42", "255"))
def test_faceenhancer_launcher_propagates_child_failure(tmp_path, exit_code):
    r = _run_curated(
        tmp_path, "faces-src-enhance.bat", exit_code=exit_code)
    assert r["proc"].returncode == int(exit_code)
    assert r["loginfo"]["args"] == _expected_args_line(
        "faces-src-enhance.bat")


def test_curated_launchers_do_not_mutate_the_repository_tree(tmp_path):
    repo_root = tmp_path / "fake-dfl"
    make_curated_fake_repo(repo_root)
    before = tl.tree_snapshot(repo_root)
    log = tmp_path / "fakeint.log"  # outside the snapshot root on purpose
    for name in _curated_names():
        env = tl._env(tl.HOSTILE_PARENT_ENV, FAKEINT_LOG=str(log), FAKEINT_EXIT="0")
        proc = tl.run_bat(repo_root / "launchers" / name, cwd=repo_root, env=env)
        assert proc.returncode == 0, f"{name} failed in mutation sweep"
    after = tl.tree_snapshot(repo_root)
    assert before == after, (
        "curated launcher run mutated the repository tree: "
        f"added={sorted(set(after) - set(before))} removed={sorted(set(before) - set(after))}"
    )
    assert not (repo_root / "marker-inj.txt").exists()
