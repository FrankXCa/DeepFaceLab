# -----------------------------------------------------------------------------
# Phase 13 - P13-GENERIC-XSEG-RESOURCE-POLICY: focused unit tests.
#
# These tests pin the `xseg apply` resource-policy contract:
# missing or structurally invalid --input-dir / --model-dir resources
# must stop with an actionable diagnostic (SystemExit code 1, no raw
# traceback) BEFORE any interactive prompt, device selection, NN
# initialization, or model loading. Model CONTENTS failures (a loader
# rejection of a structurally-present model) are likewise classified:
# the loader boundary catches only the typed resource/content failures
# the strict loader can raise (CheckpointLoadError,
# pickle.UnpicklingError, EOFError); an unrelated typed constructor
# error (implementation / device bug) propagates unwrapped, never
# rewritten into a resource diagnostic. The configured-user-path flow
# makes no network acquisition calls and never falls back to any other
# model location; a configured path containing spaces resolves exactly
# as given. The documented generic-model location gets an additional
# USER-PROVIDED guidance note; no diagnostic ever names the historical
# pack or embeds private absolute paths.
#
# Synthetic directories only: no private model bytes, no GPU, no
# network, no real device prompts (all NN boundaries stubbed), and the
# interactive face-type prompt is stubbed out.
# -----------------------------------------------------------------------------

import pickle
from pathlib import Path

import pytest

from core.leras.checkpoint import CheckpointLoadError
from mainscripts import XSegUtil as xsegutil

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERIC_MODEL_DIR = REPO_ROOT / "resources" / "xseg_generic_model"


def _write_model_dir(model_dir: Path, extra=None) -> Path:
    """Create a structurally complete model directory (fake weight
    bytes: contents validation is what the loader tests target, so
    these bytes never reach the loader in these unit tests)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / xsegutil.XSEG_MODEL_REQUIRED_FILE).write_bytes(b"fake-weights")
    for name, payload in (extra or {}).items():
        target = model_dir / name
        if isinstance(payload, bytes):
            target.write_bytes(payload)
        else:
            target.write_bytes(pickle.dumps(payload))
    return model_dir


@pytest.fixture
def stub_prompt(monkeypatch):
    """The interactive face-type prompt must never run in unit tests."""
    monkeypatch.setattr(xsegutil.io, "input_str", lambda *a, **k: "same")


@pytest.fixture
def stub_nn(monkeypatch):
    """Stub the NN device boundary: record calls, never touch a device."""
    calls = {"device": 0, "init": 0}

    def fake_ask_choose_device(choose_only_one=True):
        calls["device"] += 1
        return None

    monkeypatch.setattr(xsegutil.nn.DeviceConfig, "ask_choose_device", staticmethod(fake_ask_choose_device))

    def fake_initialize(*a, **k):
        calls["init"] += 1

    monkeypatch.setattr(xsegutil.nn, "initialize", fake_initialize)
    return calls


def _fake_xsegnet(monkeypatch, ctor_behavior):
    """Replace the XSegNet class; ctor_behavior(kwargs) may raise to
    simulate a loader failure after the structural preflight."""
    calls = []

    class FakeXSegNet:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)
            ctor_behavior(kwargs)

        def get_resolution(self):
            return 256

        def extract(self, img):
            return img

    monkeypatch.setattr(xsegutil, "XSegNet", FakeXSegNet)
    return calls


# --- Structural preflight: missing / invalid resources ----------------------

def test_missing_input_dir_fails_before_any_nn(monkeypatch, tmp_path, capsys, stub_nn, stub_prompt):
    calls = _fake_xsegnet(monkeypatch, lambda k: None)
    with pytest.raises(SystemExit) as ei:
        xsegutil.apply_xseg(tmp_path / "missing_input", tmp_path / "missing_model")
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "input directory not found" in out
    assert "xseg apply" in out
    assert stub_nn["device"] == 0 and stub_nn["init"] == 0
    assert calls == []


def test_missing_model_dir_is_an_actionable_failure(monkeypatch, tmp_path, capsys, stub_nn, stub_prompt):
    in_dir = tmp_path / "in"; in_dir.mkdir()
    calls = _fake_xsegnet(monkeypatch, lambda k: None)
    with pytest.raises(SystemExit) as ei:
        xsegutil.apply_xseg(in_dir, tmp_path / "no_such_model")
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "XSeg model directory not found" in out
    assert "XSeg_256.npy" in out
    assert "bundles no XSeg model and downloads nothing" in out
    # Diagnostics must never name the historical pack or embed private paths.
    assert "model_generic_xseg" not in out
    assert "_internal" not in out
    assert stub_nn["device"] == 0 and stub_nn["init"] == 0
    assert calls == []


def test_model_path_that_is_a_file_is_an_actionable_failure(monkeypatch, tmp_path, capsys, stub_nn, stub_prompt):
    in_dir = tmp_path / "in"; in_dir.mkdir()
    model_file = tmp_path / "model"; model_file.write_bytes(b"x")
    calls = _fake_xsegnet(monkeypatch, lambda k: None)
    with pytest.raises(SystemExit) as ei:
        xsegutil.apply_xseg(in_dir, model_file)
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "is a file, not a directory" in out
    assert "XSeg_256.npy" in out
    assert calls == []


def test_structurally_incomplete_model_dir_is_an_actionable_failure(monkeypatch, tmp_path, capsys, stub_nn, stub_prompt):
    in_dir = tmp_path / "in"; in_dir.mkdir()
    model_dir = tmp_path / "model"; model_dir.mkdir()
    (model_dir / "XSeg_summary.txt").write_text("summary only, no weights")
    calls = _fake_xsegnet(monkeypatch, lambda k: None)
    with pytest.raises(SystemExit) as ei:
        xsegutil.apply_xseg(in_dir, model_dir)
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "structurally incomplete" in out
    assert "required file missing: XSeg_256.npy" in out
    assert "XSeg_summary.txt" in out
    assert calls == []


def test_structurally_valid_resources_pass_preflight(tmp_path):
    in_dir = tmp_path / "in"; in_dir.mkdir()
    model_dir = _write_model_dir(tmp_path / "model")
    xsegutil._validate_apply_resources(in_dir, model_dir)  # must not exit


# --- Documented generic-model location: USER-PROVIDED guidance ---------------

def test_documented_location_recognition_covers_relative_and_absolute_forms():
    assert xsegutil._is_documented_generic_location(Path("resources/xseg_generic_model"))
    assert xsegutil._is_documented_generic_location(Path("resources\\xseg_generic_model"))
    assert xsegutil._is_documented_generic_location(Path("resources/xseg_generic_model/"))
    assert xsegutil._is_documented_generic_location(Path("Resources/XSEG_Generic_Model"))
    assert xsegutil._is_documented_generic_location(GENERIC_MODEL_DIR)
    assert xsegutil._is_documented_generic_location(str(GENERIC_MODEL_DIR))
    assert not xsegutil._is_documented_generic_location(Path("resources/other_model"))
    assert not xsegutil._is_documented_generic_location(Path("workspace/model"))


def test_missing_documented_generic_location_adds_user_provided_note(monkeypatch, tmp_path, capsys, stub_nn, stub_prompt):
    if GENERIC_MODEL_DIR.exists():
        pytest.skip("generic model resource present on this host (matrix D in progress)")
    in_dir = tmp_path / "in"; in_dir.mkdir()
    calls = _fake_xsegnet(monkeypatch, lambda k: None)
    with pytest.raises(SystemExit) as ei:
        xsegutil.apply_xseg(in_dir, Path("resources/xseg_generic_model"))
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "XSeg model directory not found" in out
    assert "USER-PROVIDED" in out
    assert "untracked and never shipped" in out
    assert "downloads" in out
    assert calls == []


def test_missing_non_generic_model_location_has_no_generic_note(monkeypatch, tmp_path, capsys, stub_nn, stub_prompt):
    in_dir = tmp_path / "in"; in_dir.mkdir()
    calls = _fake_xsegnet(monkeypatch, lambda k: None)
    with pytest.raises(SystemExit) as ei:
        xsegutil.apply_xseg(in_dir, tmp_path / "my_own_model_dir")
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "XSeg model directory not found" in out
    assert "USER-PROVIDED" not in out
    assert calls == []


# --- Loader (contents) failures: classified, never raw -----------------------

def test_loader_checkpoint_error_is_classified(monkeypatch, tmp_path, capsys, stub_nn, stub_prompt):
    in_dir = tmp_path / "in"; in_dir.mkdir()
    model_dir = _write_model_dir(tmp_path / "model")

    def fail(kwargs):
        raise CheckpointLoadError("strict checkpoint validation failed: weight report mismatch")

    calls = _fake_xsegnet(monkeypatch, fail)
    with pytest.raises(SystemExit) as ei:
        xsegutil.apply_xseg(in_dir, model_dir)
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "FAILED strict weight validation" in out
    assert "bundles no XSeg model and downloads nothing" in out
    assert len(calls) == 1
    assert calls[0]["weights_file_root"] == model_dir
    assert calls[0]["raise_on_no_model_files"] is True


def test_loader_unpicklable_weight_bytes_are_classified(monkeypatch, tmp_path, capsys, stub_nn, stub_prompt):
    # Bytes that exist but cannot be decoded as a checkpoint at all
    # (truncated / garbage): the real loader surfaces this from pickle
    # as UnpicklingError; it must stay an actionable resource
    # diagnostic, not a raw traceback.
    in_dir = tmp_path / "in"; in_dir.mkdir()
    model_dir = _write_model_dir(tmp_path / "model")

    def fail(kwargs):
        raise pickle.UnpicklingError("invalid load key, 'n'.")

    calls = _fake_xsegnet(monkeypatch, fail)
    with pytest.raises(SystemExit) as ei:
        xsegutil.apply_xseg(in_dir, model_dir)
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "not a readable checkpoint" in out
    assert "bundles no XSeg model and downloads nothing" in out
    assert len(calls) == 1
    assert calls[0]["weights_file_root"] == model_dir


def test_loader_empty_weight_file_is_classified(monkeypatch, tmp_path, capsys, stub_nn, stub_prompt):
    # A zero-byte weight file (partial copy / interrupted write) exists,
    # so the pre-load state check passes, but pickle raises EOFError
    # ("Ran out of input") on it; that typed failure must be classified
    # too.
    in_dir = tmp_path / "in"; in_dir.mkdir()
    model_dir = _write_model_dir(tmp_path / "model")

    def fail(kwargs):
        raise EOFError("Ran out of input")

    calls = _fake_xsegnet(monkeypatch, fail)
    with pytest.raises(SystemExit) as ei:
        xsegutil.apply_xseg(in_dir, model_dir)
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "not a readable checkpoint" in out
    assert "bundles no XSeg model and downloads nothing" in out
    assert len(calls) == 1
    assert calls[0]["weights_file_root"] == model_dir


def test_weight_file_removed_after_preflight_is_actionable(monkeypatch, tmp_path, capsys, stub_nn, stub_prompt):
    # The required file is present at preflight and disappears before
    # load: the pre-load re-verification (a state check, not exception
    # string classification) must turn this known case into an
    # actionable diagnostic without constructing the model.
    in_dir = tmp_path / "in"; in_dir.mkdir()
    model_dir = _write_model_dir(tmp_path / "model")
    (model_dir / xsegutil.XSEG_MODEL_REQUIRED_FILE).unlink()
    # Waive the structural preflight to isolate the pre-load re-check.
    monkeypatch.setattr(xsegutil, "_validate_apply_resources", lambda *a: None)
    calls = _fake_xsegnet(monkeypatch, lambda k: None)
    with pytest.raises(SystemExit) as ei:
        xsegutil.apply_xseg(in_dir, model_dir)
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "weight file is missing" in out
    assert "XSeg_256.npy" in out
    assert calls == []


def test_unexpected_constructor_error_propagates_unwrapped(monkeypatch, tmp_path, capsys, stub_nn, stub_prompt):
    # An unrelated typed constructor failure (implementation / device
    # bug) must NOT be converted into a resource diagnostic: it
    # propagates with the app's normal full diagnostic context - no
    # sys.exit(1), no resource error message, no private bytes.
    in_dir = tmp_path / "in"; in_dir.mkdir()
    model_dir = _write_model_dir(tmp_path / "model")

    def fail(kwargs):
        raise RuntimeError("controlled unrelated constructor failure")

    calls = _fake_xsegnet(monkeypatch, fail)
    with pytest.raises(RuntimeError, match="controlled unrelated constructor failure"):
        xsegutil.apply_xseg(in_dir, model_dir)
    out = capsys.readouterr().out
    assert "could not be loaded" not in out
    assert "bundles no XSeg model" not in out
    assert "USER-PROVIDED" not in out
    assert "model_generic_xseg" not in out
    assert "_internal" not in out
    assert len(calls) == 1


# --- Positive control: valid structure + valid loader proceeds ----------------

def test_valid_model_with_unreadable_dat_warns_and_completes(monkeypatch, tmp_path, capsys, stub_nn, stub_prompt):
    # Empty input dir: the image loop processes zero files, so the run
    # ends right after the (stubbed) model construction. The corrupt
    # XSeg_data.dat must downgrade to the interactive prompt path (stubbed)
    # with a warning, not a raw pickle traceback.
    in_dir = tmp_path / "in"; in_dir.mkdir()
    model_dir = _write_model_dir(tmp_path / "model", extra={
        xsegutil.XSEG_MODEL_OPTIONAL_FILE: b"not a pickle",
    })
    calls = _fake_xsegnet(monkeypatch, lambda k: None)
    xsegutil.apply_xseg(in_dir, model_dir)  # must return normally
    out = capsys.readouterr().out
    assert "could not be read" in out
    assert "Applying XSeg model to in/ folder" in out
    assert stub_nn["device"] == 1 and stub_nn["init"] == 1
    assert len(calls) == 1
    assert calls[0]["weights_file_root"] == model_dir
    assert calls[0]["load_weights"] is True
    assert calls[0]["raise_on_no_model_files"] is True


def test_valid_model_with_valid_dat_face_type_is_used(monkeypatch, tmp_path, capsys, stub_nn, stub_prompt):
    # A well-formed XSeg_data.dat (face type WF) must bypass the prompt
    # entirely: the prompt stub is replaced by a recorder that fails the
    # test if the prompt fires.
    in_dir = tmp_path / "in"; in_dir.mkdir()
    model_dir = _write_model_dir(tmp_path / "model", extra={
        xsegutil.XSEG_MODEL_OPTIONAL_FILE: {"options": {"face_type": "wf"}},
    })
    prompts = []
    monkeypatch.setattr(xsegutil.io, "input_str",
                        lambda *a, **k: prompts.append(a) or "same")
    calls = _fake_xsegnet(monkeypatch, lambda k: None)
    xsegutil.apply_xseg(in_dir, model_dir)
    assert prompts == []
    assert len(calls) == 1


# --- Curated launcher / manifest binding --------------------------------------

def test_generic_launchers_bind_the_documented_location():
    import json
    import os

    launchers_dir = REPO_ROOT / "launchers"
    for name in ("xseg-apply-generic-masks-src.bat", "xseg-apply-generic-masks-dst.bat"):
        text = (launchers_dir / name).read_text(encoding="utf-8")
        raw = (launchers_dir / name).read_bytes()
        assert all(b < 128 for b in raw), f"{name}: non-ASCII"
        assert b"\r" not in raw, f"{name}: CRLF"
        lines = text.splitlines()
        call_lines = [ln for ln in lines if ln.startswith('call "%~dp0dfl.bat" ')]
        assert len(call_lines) == 1
        assert '--model-dir "resources\\xseg_generic_model"' in call_lines[0]
        assert "xseg apply" in call_lines[0]
        # No second execution surface, no forbidden pack token.
        assert "model_generic_xseg" not in text
    manifest = json.loads((launchers_dir / "workflow_launchers.json").read_text(encoding="utf-8"))
    curated = {e["file"] for e in manifest["curated"]}
    assert "xseg-apply-generic-masks-src.bat" in curated
    assert "xseg-apply-generic-masks-dst.bat" in curated
    for entry in manifest["curated"]:
        if "generic" in entry["file"]:
            assert entry["group"] == "xseg"
            assert entry["forwarding"] is False


# --- Configured-path robustness (P13 correction pass F6) ----------------------

def test_missing_model_dir_with_spaces_is_an_actionable_failure(monkeypatch, tmp_path, capsys, stub_nn, stub_prompt):
    # A configured model path containing spaces must resolve exactly as
    # given and its absence must still fail actionably, echoing the path
    # with the spaces intact (synthetic only; no private weights).
    in_dir = tmp_path / "in"; in_dir.mkdir()
    calls = _fake_xsegnet(monkeypatch, lambda k: None)
    with pytest.raises(SystemExit) as ei:
        xsegutil.apply_xseg(in_dir, tmp_path / "generic model with spaces")
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "generic model with spaces" in out
    assert "XSeg model directory not found" in out
    assert calls == []


def test_structurally_incomplete_model_dir_with_spaces_is_an_actionable_failure(monkeypatch, tmp_path, capsys, stub_nn, stub_prompt):
    in_dir = tmp_path / "in"; in_dir.mkdir()
    model_dir = tmp_path / "generic model with spaces"; model_dir.mkdir()
    (model_dir / "XSeg_summary.txt").write_text("summary only, no weights")
    calls = _fake_xsegnet(monkeypatch, lambda k: None)
    with pytest.raises(SystemExit) as ei:
        xsegutil.apply_xseg(in_dir, model_dir)
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "structurally incomplete" in out
    assert "generic model with spaces" in out
    assert "XSeg_summary.txt" in out
    assert calls == []


def test_complete_model_dir_with_spaces_passes_and_loads_from_that_path(monkeypatch, tmp_path, capsys, stub_nn, stub_prompt):
    # Positive control: a configured location whose path contains spaces
    # passes the preflight and the loader is invoked against exactly
    # that path (synthetic bytes; the loader is stubbed).
    in_dir = tmp_path / "in"; in_dir.mkdir()
    model_dir = _write_model_dir(tmp_path / "generic model with spaces")
    calls = _fake_xsegnet(monkeypatch, lambda k: None)
    xsegutil.apply_xseg(in_dir, model_dir)  # must return normally
    assert len(calls) == 1
    assert calls[0]["weights_file_root"] == model_dir
    assert calls[0]["raise_on_no_model_files"] is True


def test_normal_flow_makes_no_network_acquisition_calls(monkeypatch, tmp_path, capsys, stub_nn, stub_prompt):
    # Deterministic no-network sentinel: the normal configured-user-path
    # flow (here: the missing-model-location failure a user hits first)
    # must not touch any network acquisition entry point. If a future
    # change adds an implicit download to resource resolution or the
    # apply path, this test fails. No network dependency: every entry
    # point is stubbed to a recorder before the flow runs.
    import http.client
    import socket
    import urllib.request

    hits = []
    monkeypatch.setattr(socket.socket, "connect",
                        lambda self, *a, **k: hits.append(("socket", a)))
    monkeypatch.setattr(http.client.HTTPConnection, "connect",
                        lambda self, *a, **k: hits.append(("http", a)))
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: hits.append(("urlopen", a)))

    in_dir = tmp_path / "in"; in_dir.mkdir()
    calls = _fake_xsegnet(monkeypatch, lambda k: None)
    with pytest.raises(SystemExit) as ei:
        xsegutil.apply_xseg(in_dir, tmp_path / "no_such_model")
    assert ei.value.code == 1
    assert hits == []
    assert calls == []


def test_missing_model_dir_does_not_probe_any_fallback_location(monkeypatch, tmp_path, capsys, stub_nn, stub_prompt):
    # Behavioral no-legacy-fallback test: when the configured --model-dir
    # is missing, the app fails on exactly that path and never silently
    # searches a legacy / developer / other private model location. A
    # valid decoy model at another location is set up and must never be
    # constructed against (the loader boundary is stubbed to record
    # exactly which path it is given).
    in_dir = tmp_path / "in"; in_dir.mkdir()
    decoy = _write_model_dir(tmp_path / "some_other_model_dir")
    sentinel = b"decoy-weights-sentinel"
    (decoy / xsegutil.XSEG_MODEL_REQUIRED_FILE).write_bytes(sentinel)
    calls = _fake_xsegnet(monkeypatch, lambda k: None)
    with pytest.raises(SystemExit) as ei:
        xsegutil.apply_xseg(in_dir, tmp_path / "no_such_model")
    assert ei.value.code == 1
    assert calls == []  # no model was constructed against any location
    assert (decoy / xsegutil.XSEG_MODEL_REQUIRED_FILE).read_bytes() == sentinel


def test_loader_is_invoked_only_with_the_configured_model_dir(monkeypatch, tmp_path, capsys, stub_nn, stub_prompt):
    # The model path actually consumed by the loader is exactly the
    # configured --model-dir: one construction, and no other (valid)
    # location is ever loaded.
    in_dir = tmp_path / "in"; in_dir.mkdir()
    configured = _write_model_dir(tmp_path / "configured model dir")
    _write_model_dir(tmp_path / "some_other_model_dir")  # valid decoy
    calls = _fake_xsegnet(monkeypatch, lambda k: None)
    xsegutil.apply_xseg(in_dir, configured)  # must return normally
    assert len(calls) == 1
    assert calls[0]["weights_file_root"] == configured


# --- .gitignore narrowness (P13 correction pass F3) ---------------------------

def test_gitignore_narrowly_ignores_only_the_generic_model_location():
    # The Phase-13 .gitignore addition must hide ONLY the user-managed
    # generic-model location, never the whole resources/ tree (other
    # future resources/ content, metadata, or manifests must stay
    # visible to Git).
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    lines = [ln.strip() for ln in gitignore.splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    assert "/resources/xseg_generic_model/" in lines
    known_ok = {"/resources/xseg_generic_model/",
                "resources/xseg_generic_model/",
                "resources/xseg_generic_model"}
    for ln in lines:
        if ln in known_ok:
            continue
        if ln.rstrip("/").split("/")[-1] == "resources" and "xseg_generic_model" not in ln:
            pytest.fail(f"too-broad .gitignore rule hides the whole resources/ tree: {ln!r}")
