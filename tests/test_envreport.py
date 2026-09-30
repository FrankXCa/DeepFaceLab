"""Tests for P13-ENVREPORT-VERIFY (``scripts/envreport.py``).

Strategy (hermetic, deterministic):

* Probe workers are NOT executed against real heavy packages in pytest.
  An injected ``probe_runner`` returns canned ``ProbeInfo`` objects instead.
  The real subprocess protocol is exercised only where it stays fast and
  deterministic: an unknown probe name (the worker always emits
  ``UNAVAILABLE`` without importing anything heavy), a short-timeout
  command, a fast silent child (EOF before the timeout), a child that
  writes only to stderr (stderr suppression), and a worker child whose
  stdout far exceeds the (monkeypatched) capture limit — it emits
  100,000 bytes against a 1,024-byte cap, blocks on the full pipe, and
  must be killed by the timeout (TIMEOUT, no unbounded parent buffering).
  The worker's NESTED external-tool capture (nvidia-smi / ffmpeg /
  ffprobe) is exercised with controlled fake executables: a child that
  emits 132,000 bytes (2x the 65,536-byte cap, must be capped and
  discarded), a continuously-writing child (timeout + kill, no hang),
  a fast silent child, and a non-zero-exit child.
* ``--verify`` runs against a fake built runtime created with the
  ``test_build_runtime`` fixture machinery, and ``resolve_inputs`` is
  injected so the engine never reads the real repository's contract files.
* Privacy: poison strings matching the builder's ``PRIVACY_RULES`` are fed
  through injected probes and must be redacted per field; a forced final
  scan hit must suppress the whole report (fail closed).
* The stdlib-first dispatch boundary (``main.py envreport`` must import no
  third-party root) is proven in a subprocess with an import recorder.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_runtime as br
from scripts import envreport
from scripts import envreport_probe

import test_build_runtime as tb

ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = ROOT / "main.py"

# Guaranteed hits against scripts/build_runtime.PRIVACY_RULES.
POISON_DRIVE = "C:\\Users\\Alice\\AppData\\Roaming\\torch"
POISON_IP = "192.168.1.50"
POISON_UNC = "\\\\fileserver\\share"
POISON_CRED = "password=hunter2"
# Hits the envreport-local supplemental rule (env-var reference forms, which
# the shared PRIVACY_RULES table cannot match).
POISON_ENVVAR = "%APPDATA%\\Roaming\\torch"


# ---------------------------------------------------------------------------
# Fixtures (mirrors the module-local fixture machinery of test_build_runtime)
# ---------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path):
    """Synthetic build environment: fake Python archive + fake wheels."""
    archive = tb.make_python_archive(tmp_path / "fake-python.tar.gz")
    identity = tb.fake_identity(archive)
    wheels_dir = tmp_path / "wheel-store"
    return SimpleNamespace(
        tmp_path=tmp_path,
        archive=archive,
        identity=identity,
        inventory=tb.fake_inventory(identity),
        wheels_dir=wheels_dir,
    )


@pytest.fixture
def world(env, tmp_path):
    """A fake built cpu-nogui runtime with an active selector pointing at it."""
    runtime_root, result, lock = tb._fake_built_runtime(
        env, tmp_path, variant="cpu-nogui", torch_backend="cpu"
    )
    br.set_active_pointer(runtime_root, result.runtime_id)
    return SimpleNamespace(
        env=env,
        runtime_root=runtime_root,
        runtime_id=result.runtime_id,
        lock=lock,
    )


def make_resolvers(world):
    def resolve(root, variant):
        return (world.env.identity, world.env.inventory, world.lock, tb.BOOTSTRAP_SHA)

    return resolve


def cpu_probe_set(**overrides):
    """Canned probes consistent with the fake cpu-nogui lock (torch 2.14.0+cpu)."""
    probes = {
        "torch": envreport.ProbeInfo(
            "torch", "OK", "", {"version": "2.14.0+cpu", "cuda": None, "local_tag": "cpu"}
        ),
        "cuda": envreport.ProbeInfo("cuda", "MISSING", "no CUDA device visible", {}),
        "nvidia_smi": envreport.ProbeInfo("nvidia_smi", "MISSING", "nvidia-smi not found", {}),
        "ffmpeg": envreport.ProbeInfo("ffmpeg", "OK", "", {"version": "6.1"}),
        "ffprobe": envreport.ProbeInfo("ffprobe", "OK", "", {"version": "6.1"}),
        "onnx": envreport.ProbeInfo("onnx", "OK", "", {"version": "1.19.1"}),
        "onnxruntime": envreport.ProbeInfo("onnxruntime", "MISSING", "not importable", {}),
    }
    probes.update(overrides)
    return probes


def runner_for(probes):
    def runner(name, timeout=None):
        if name not in probes:
            raise AssertionError(f"unexpected probe requested: {name}")
        return probes[name]

    return runner


def run_envreport(argv, **kwargs):
    out = io.StringIO()
    err = io.StringIO()
    rc = envreport.run(argv, out=out, err=err, **kwargs)
    return rc, out.getvalue(), err.getvalue()


def probe_by_name(probes):
    return {p.name: p for p in probes}


def verify_checks(data):
    """(check dict by id, list of verify checks) from a parsed JSON report."""
    vs = data["verify_section"]
    return {c["id"]: c for c in vs["checks"]}, vs


def engine_checks(vs):
    non_probe = {c["id"] for c in vs["checks"] if c["id"].startswith("probe_")}
    reserved = {"inputs", "selector_agreement", "running_python", "torch_contract"}
    return [c for c in vs["checks"] if not c["id"].startswith("probe_") and c["id"] not in reserved]


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


class TestUsage:
    def test_unknown_flag_exits_2(self, tmp_path):
        rc, out, err = run_envreport(["envreport", "--bogus"], repo_root=tmp_path)
        assert rc == 2
        assert out == ""
        # Fixed wording: the offending token is never echoed back.
        assert "envreport: invalid argument" in err
        assert "usage:" in err
        assert "--bogus" not in err

    def test_duplicate_flag_exits_2(self, tmp_path):
        rc, out, err = run_envreport(["envreport", "--json", "--json"], repo_root=tmp_path)
        assert rc == 2
        assert out == ""
        assert "envreport: invalid argument" in err

    def test_leading_command_word_tolerated(self, world):
        probes = runner_for(cpu_probe_set())
        rc_a, out_a, err_a = run_envreport(
            ["envreport", "--json"], repo_root=world.env.tmp_path, probe_runner=probes
        )
        rc_b, out_b, err_b = run_envreport(
            ["--json"], repo_root=world.env.tmp_path, probe_runner=probes
        )
        assert rc_a == rc_b == 0
        da, db = json.loads(out_a), json.loads(out_b)
        da.pop("generated_utc")  # wall-clock timestamp: not comparable across calls
        db.pop("generated_utc")
        assert da == db


class TestInvalidArgumentPrivacy:
    """Unknown arguments are untrusted input: the fixed usage-error text
    must never echo the token (or any fragment of it) on stdout or stderr.
    All tokens below are synthetic — no real paths or credentials."""

    @pytest.mark.parametrize(
        "tokens",
        [
            ["C:\\Users\\someone\\secret"],  # absolute user path
            ["\\\\private-server\\share\\secret"],  # UNC share path
            ["ghp_SYNTHETIC_SECRET"],  # credential-shaped token
            ["Bearer", "SYNTHETIC-TOKEN"],  # two-token credential pair
            ["--does-not-exist"],  # ordinary unknown flag
        ],
    )
    def test_poison_arguments_are_never_echoed(self, tmp_path, tokens):
        rc, out, err = run_envreport(["envreport", *tokens], repo_root=tmp_path)
        combined = out + err
        assert rc == 2
        assert out == ""
        assert "envreport: invalid argument" in err
        assert "usage:" in err
        for token in tokens:
            assert token not in combined
            for fragment in re.split(r"[\\/:]", token):
                if len(fragment) > 2:  # skip trivial pieces like "C"
                    assert fragment not in combined


# ---------------------------------------------------------------------------
# Base report
# ---------------------------------------------------------------------------


class TestBaseReport:
    def test_json_shape(self, world):
        rc, payload, err = run_envreport(
            ["envreport", "--json"],
            repo_root=world.env.tmp_path,
            probe_runner=runner_for(cpu_probe_set()),
        )
        assert rc == 0
        assert err == ""
        data = json.loads(payload)
        assert data["schema"] == envreport.SCHEMA == "dfl-envreport-v1"
        assert data["command"] == "envreport"
        assert data["json"] is True
        assert data["verify"] is False
        assert data["verify_section"] is None
        assert data["source"]["status"] == "UNAVAILABLE"  # fake root is not a git tree
        assert data["os"]["status"] == "OK"
        assert data["python"]["status"] == "OK"
        runtime = data["runtime"]
        assert runtime["mode"] == envreport.MODE_DEVELOPMENT
        assert runtime["runtime_id"] is None
        assert runtime["variant"] == "cpu-nogui"
        selector = runtime["selector"]
        assert selector["present"] is True
        assert selector["valid"] is True
        assert selector["runtime_id"] == world.runtime_id
        names = [p["name"] for p in data["packages"]]
        assert "torch" in names
        assert "alpha" in names
        probes = data["probes"]
        assert [p["name"] for p in probes] == list(envreport.PROBES)
        assert probes[0]["status"] == "OK"
        assert probes[0]["fields"]["version"] == "2.14.0+cpu"
        # Contract-6 wire form: indent=2, ensure_ascii=True, insertion-order
        # keys, trailing newline; generated_utc has second precision.
        assert payload == json.dumps(data, indent=2, ensure_ascii=True, sort_keys=False) + "\n"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", data["generated_utc"])

    def test_text_sections(self, world):
        rc, payload, err = run_envreport(
            ["envreport"], repo_root=world.env.tmp_path, probe_runner=runner_for(cpu_probe_set())
        )
        assert rc == 0
        for section in ("[Source]", "[OS]", "[Python]", "[Runtime]", "[Packages]", "[Probes]"):
            assert section in payload
        assert world.runtime_id in payload
        assert "alpha" in payload
        for name in envreport.PROBES:
            assert name in payload
        assert "[Verify]" not in payload

    def test_text_json_parity_same_model(self, world):
        report = envreport.build_report(world.env.tmp_path, runner_for(cpu_probe_set()))
        data = envreport.model_to_dict(report)
        text = envreport.render_text(report)
        reparsed = json.loads(envreport.render_json(report))
        assert reparsed == data
        for probe in report.probes:
            assert probe.name in text
            assert probe.status in text
        for package in report.packages:
            assert package.name in text

    def test_text_write_uses_replace_errors_under_any_codepage(self, world):
        # Text mode must harden the output stream (errors="replace") before
        # writing so a legacy console code page can never abort the report
        # mid-write with UnicodeEncodeError.  JSON mode never reconfigures.
        class _RecordingOut(io.StringIO):
            def __init__(self):
                super().__init__()
                self.reconfigure_calls = []

            def reconfigure(self, **kwargs):
                self.reconfigure_calls.append(kwargs)

        out = _RecordingOut()
        err = io.StringIO()
        rc = envreport.run(
            ["envreport"],
            out=out,
            err=err,
            repo_root=world.env.tmp_path,
            probe_runner=runner_for(cpu_probe_set()),
        )
        assert rc == 0
        assert out.reconfigure_calls == [{"errors": "replace"}]
        assert "DeepFaceLab environment report" in out.getvalue()

        out_json = _RecordingOut()
        rc = envreport.run(
            ["envreport", "--json"],
            out=out_json,
            err=io.StringIO(),
            repo_root=world.env.tmp_path,
            probe_runner=runner_for(cpu_probe_set()),
        )
        assert rc == 0
        assert out_json.reconfigure_calls == []


# ---------------------------------------------------------------------------
# Probe resilience
# ---------------------------------------------------------------------------


class TestProbeResilience:
    def test_invalid_worker_status_becomes_unavailable(self, world):
        probes = cpu_probe_set()
        probes["torch"] = envreport.ProbeInfo("torch", "WEIRD", "", {})
        report = envreport.build_report(world.env.tmp_path, runner_for(probes))
        assert probe_by_name(report.probes)["torch"].status == envreport.STATUS_UNAVAILABLE

    def test_probe_runner_exception_is_isolated(self, world):
        probes = cpu_probe_set()

        def runner(name, timeout=None):
            if name == "torch":
                raise RuntimeError("boom")
            return probes[name]

        rc, payload, err = run_envreport(
            ["envreport"], repo_root=world.env.tmp_path, probe_runner=runner
        )
        assert rc == 0
        assert payload  # full report still rendered
        by_name = probe_by_name(envreport.build_report(world.env.tmp_path, runner).probes)
        torch_probe = by_name["torch"]
        assert torch_probe.status == envreport.STATUS_INIT_ERROR
        assert "RuntimeError" in torch_probe.note
        assert "boom" not in torch_probe.note  # type name only, never the message
        assert by_name["cuda"].status == "MISSING"  # siblings unaffected

    def test_probe_timeout_via_injected_command(self):
        real = envreport._make_default_probe_runner(
            ROOT, command=[sys.executable, "-c", "import time; time.sleep(30)"]
        )
        info = real("ffmpeg", timeout=0.4)
        assert info.name == "ffmpeg"
        assert info.status == envreport.STATUS_TIMEOUT
        assert "timed out" in info.note
        assert info.fields == {}

    def test_worker_unknown_probe_is_unavailable(self):
        real = envreport._make_default_probe_runner(ROOT)
        info = real("definitely_not_a_probe", timeout=20)
        assert info.status == envreport.STATUS_UNAVAILABLE
        assert "unknown probe" in info.note

    def test_fast_silent_child_is_unavailable(self):
        # Child exits 0 immediately without a protocol line (EOF before the
        # timeout): UNAVAILABLE, not TIMEOUT.
        real = envreport._make_default_probe_runner(
            ROOT, command=[sys.executable, "-I", "-B", "-c", "pass"]
        )
        info = real("torch", timeout=10)
        assert info.status == envreport.STATUS_UNAVAILABLE
        assert "no protocol line" in info.note
        assert info.fields == {}

    def test_child_stderr_never_surfaces(self):
        # The runner wires child stderr to DEVNULL: stderr output must be
        # impossible to reach the probe note or the rendered payload.
        real = envreport._make_default_probe_runner(
            ROOT,
            command=[
                sys.executable,
                "-I",
                "-B",
                "-c",
                "import sys; print('SECRET-STDERR traceback', file=sys.stderr)",
            ],
        )
        info = real("torch", timeout=10)
        assert info.status == envreport.STATUS_UNAVAILABLE
        assert "SECRET-STDERR" not in info.note
        assert "traceback" not in info.note
        assert info.fields == {}

    def test_over_cap_worker_stdout_child_times_out(self, monkeypatch):
        # The child writes exactly 100,000 bytes (about 0.1 MB, 100x the
        # monkeypatched 1,024-byte capture cap — this is NOT a 1 MB test)
        # and then sleeps 30 s.  The parent stops storing at the cap, the
        # child blocks on the full pipe, and the 0.6 s timeout must kill it
        # (TIMEOUT) instead of the parent buffering unbounded output.
        monkeypatch.setattr(envreport, "_PROBE_OUT_LIMIT", 1024)
        real = envreport._make_default_probe_runner(
            ROOT,
            command=[
                sys.executable,
                "-I",
                "-B",
                "-c",
                "import sys; sys.stdout.write('x' * 100000); "
                "sys.stdout.flush(); import time; time.sleep(30)",
            ],
        )
        info = real("cuda", timeout=0.6)
        assert info.status == envreport.STATUS_TIMEOUT
        assert "timed out" in info.note
        assert info.fields == {}

    @pytest.mark.parametrize(
        ("stdout", "expect"),
        [
            (None, None),
            (b"", None),
            (b"garbage\n", None),
            (
                b'{"name":"torch","status":"OK","note":"","fields":{"version":"1"}}\n',
                {"name": "torch", "status": "OK", "note": "", "fields": {"version": "1"}},
            ),
            (
                b"prefix line\n{'name':'torch'}\n"
                b'{"name": "cuda", "status": "MISSING", "note": "", "fields": {}}\n',
                {"name": "cuda", "status": "MISSING", "note": "", "fields": {}},
            ),
        ],
    )
    def test_worker_output_parsing(self, stdout, expect):
        assert envreport._parse_worker_output(stdout) == expect


class TestNestedToolCapture:
    """The worker's NESTED external-tool capture (nvidia-smi / ffmpeg /
    ffprobe) is bounded by execution, not by source inspection: the fake
    tools below are real external child processes (cmd interpreting a .bat)
    driven through the worker's ``_run_capped`` Popen + capped-drain runner.

    * ``fake_large.bat`` emits exactly 132,000 bytes (2,000 lines of 64
      'X' chars + CRLF) — 2x the 65,536-byte capture cap.
    * ``fake_hang.bat`` writes that line forever until it is killed.
    * ``fake_silent.bat`` exits 0 immediately with no output.
    * ``fake_fail.bat`` exits 3 immediately with no output.
    """

    pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="fake .bat tools")

    _X64 = "X" * 64

    @staticmethod
    def _bat(tmp_path, name, body):
        path = tmp_path / name
        path.write_text(body, encoding="ascii", newline="")
        return str(path)

    @pytest.fixture
    def fake_tools(self, tmp_path):
        x = self._X64
        return SimpleNamespace(
            large=self._bat(
                tmp_path, "fake_large.bat", "@echo off\r\nfor /L %%i in (1,1,2000) do @echo(" + x + "\r\n"
            ),
            hang=self._bat(
                tmp_path, "fake_hang.bat", "@echo off\r\n:loop\r\n@echo(" + x + "\r\ngoto loop\r\n"
            ),
            silent=self._bat(tmp_path, "fake_silent.bat", "@echo off\r\nexit /b 0\r\n"),
            failing=self._bat(tmp_path, "fake_fail.bat", "@echo off\r\nexit /b 3\r\n"),
        )

    @staticmethod
    def _point_which_at(monkeypatch, exe):
        # The fake executable stands in for whichever system tool the
        # handler resolves; arguments the handler appends are ignored by
        # the .bat.
        monkeypatch.setattr(envreport_probe.shutil, "which", lambda name: exe)

    def test_nested_large_stdout_is_capped_and_never_surfaced(self, fake_tools, monkeypatch):
        # 132,000 bytes of tool stdout (2x the 65,536-byte cap): the stored
        # prefix is discarded, no raw output reaches the protocol line, and
        # both handler families report the stable over-cap status.
        self._point_which_at(monkeypatch, fake_tools.large)
        cap = envreport_probe._CHILD_OUT_LIMIT
        obj = envreport_probe.probe_nvidia_smi("nvidia_smi")
        assert obj["status"] == envreport_probe.STATUS_UNAVAILABLE
        assert obj["note"] == f"nvidia-smi output exceeded the {cap}-byte capture limit"
        assert obj["fields"] == {}
        obj = envreport_probe.probe_ffmpeg("ffmpeg")
        assert obj["status"] == envreport_probe.STATUS_UNAVAILABLE
        assert obj["note"] == f"ffmpeg output exceeded the {cap}-byte capture limit"
        assert obj["fields"] == {}

    def test_nested_continuous_output_times_out_and_is_killed(self, fake_tools, monkeypatch):
        # The tool writes forever; the worker's bounded timeout must kill
        # and reap it (no hang, no orphan) and report TIMEOUT.
        monkeypatch.setattr(envreport_probe, "_CHILD_CMD_TIMEOUT", 1.0)
        self._point_which_at(monkeypatch, fake_tools.hang)
        start = time.monotonic()
        obj = envreport_probe.probe_ffprobe("ffprobe")
        elapsed = time.monotonic() - start
        assert obj["status"] == envreport_probe.STATUS_TIMEOUT
        assert obj["note"] == "ffprobe timed out"
        assert obj["fields"] == {}
        # sanity: the runner gave up long before any parent kill window
        assert elapsed < 10

    def test_nested_silent_child_preserves_existing_semantics(self, fake_tools, monkeypatch):
        # Fast silent child (rc 0, no output): pre-existing semantics are
        # unchanged — nvidia-smi reports no GPU rows, ffmpeg an unknown
        # version — without the new cap machinery ever triggering.
        self._point_which_at(monkeypatch, fake_tools.silent)
        obj = envreport_probe.probe_nvidia_smi("nvidia_smi")
        assert obj["status"] == envreport_probe.STATUS_UNAVAILABLE
        assert obj["note"] == "nvidia-smi returned no GPU rows"
        assert obj["fields"] == {"gpu_count": 0}
        obj = envreport_probe.probe_ffmpeg("ffmpeg")
        assert obj["status"] == envreport_probe.STATUS_OK
        assert obj["fields"] == {"version": "unknown"}

    def test_nested_nonzero_exit_maps_to_init_error(self, fake_tools, monkeypatch):
        self._point_which_at(monkeypatch, fake_tools.failing)
        obj = envreport_probe.probe_nvidia_smi("nvidia_smi")
        assert obj["status"] == envreport_probe.STATUS_INIT_ERROR
        assert obj["note"] == "nvidia-smi exited with code 3"
        obj = envreport_probe.probe_ffprobe("ffprobe")
        assert obj["status"] == envreport_probe.STATUS_INIT_ERROR
        assert obj["note"] == "ffprobe exited with code 3"


class TestProcessTreeContainment:
    """R2 regression: containment is established BEFORE the tool executes
    (spawn suspended via CPython's C CreateProcess -> verified KILL_ON_CLOSE
    job configuration -> checked job assignment -> resume), and a DESCENDANT
    that inherits the tool's stdout pipe and outlives the direct child
    must not keep ``_run_capped`` from returning.

    Each test builds its own synthetic tree (files under ``tmp_path``):
    the direct child (a ``python -I -B`` process) spawns a grandchild
    whose ``stdout=None`` inherits the runner's pipe, records BOTH pids
    in temp files, and then either sleeps (or exits 0 while the grandchild
    keeps the pipe open).  The grandchild is spawned IMMEDIATELY after
    start (an immediate-spawn descendant race, which the suspended launch
    cannot prevent -- it happens after resume -- but which job
    containment must still contain).  The grandchild lives ~30 s (or
    writes forever) — far beyond the runner's 0.6 s timeout — so a
    direct-``kill()``-only cleanup would leave it writing into an unread
    pipe and stall the reader join indefinitely (the exact P13 R2 defect:
    the old run took 8.08 s for a 0.5 s timeout).

    The runner must instead terminate the WHOLE tree, return well before
    the descendant's natural exit, and leave no descendant alive.  Liveness
    is verified with a real OS query (``GetExitCodeProcess``: live
    processes report ``STILL_ACTIVE`` == 259, dead ones a real exit code
    — the job kill records 65534); the tests never rely on the descendant
    terminating naturally, and a direct-kill-only implementation fails
    them because the descendant is still ``STILL_ACTIVE`` after return.

    The fail-closed tests additionally prove that when job creation,
    job configuration/verification, or job assignment (including a
    malformed ``JobObjectProcessIdList`` membership response) fails, the
    suspended child is terminated WITHOUT ever being resumed (its first
    instruction — writing a marker file — never executes), the tool
    never runs, and the probe reports a stable, non-sensitive
    UNAVAILABLE note: there is no third state.
    """

    pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows job-object containment")

    # winbase.h: GetExitCodeProcess reports 259 (STILL_ACTIVE) while the
    # process runs and its real exit code once it has terminated.  (The
    # old 0xFFFFFFFF assumption was disproven by the third-round kernel
    # probe; see the third-round record in docs/PHASE13_STATE.md.)
    STILL_ACTIVE = 259

    # argv: [1]=child pidfile [2]=desc pidfile [3]=grandcode file
    _CHILD_SLEEP = (
        "import os, subprocess, sys, time\n"
        "g = subprocess.Popen([sys.executable, '-I', '-B', sys.argv[3], sys.argv[2]], stdout=None)\n"
        "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
        "open(sys.argv[2], 'w').write(str(g.pid))\n"
        "time.sleep(30)\n"
    )
    _CHILD_EXIT_NOW = (
        "import os, subprocess, sys\n"
        "g = subprocess.Popen([sys.executable, '-I', '-B', sys.argv[3], sys.argv[2]], stdout=None)\n"
        "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
        "open(sys.argv[2], 'w').write(str(g.pid))\n"
    )
    _GRAND_SLEEP = (
        "import sys, time\n"
        "sys.stdout.write('desc-born\\n'); sys.stdout.flush()\n"
        "time.sleep(30)\n"
    )
    _GRAND_CONTINUOUS = (
        "import sys, time\n"
        "sys.stdout.write('desc-born\\n'); sys.stdout.flush()\n"
        "while True:\n"
        "    sys.stdout.write('w' * 64); sys.stdout.flush()\n"
        "    time.sleep(0.01)\n"
    )

    @staticmethod
    def _write(tmp_path, name, body):
        path = tmp_path / name
        path.write_text(body, encoding="ascii", newline="")
        return str(path)

    @staticmethod
    def _exit_state(pid):
        """Authoritative liveness query.

        Returns ``None`` if the process cannot be opened (gone),
        ``STILL_ACTIVE`` (259) while it is running, or its real
        exit code once it has exited.  Plain OpenProcess-based checks are
        deliberately NOT used: on current Windows builds a just-killed
        process stays openable for a long time (lingering process
        object), so only the exit code distinguishes dead from alive.
        """
        import ctypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        k32.GetExitCodeProcess.restype = ctypes.c_int
        k32.GetExitCodeProcess.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        handle = k32.OpenProcess(0x0400, False, pid)  # PROCESS_QUERY_INFORMATION
        if not handle:
            return None
        try:
            code = ctypes.c_ulong()
            k32.GetExitCodeProcess(handle, ctypes.byref(code))
        finally:
            k32.CloseHandle(handle)
        return code.value

    def _assert_tree_dead(self, *pids, grace=3.0):
        """Every pid must be dead (unopenable or a real exit code)."""
        end = time.monotonic() + grace
        pending = dict(zip(pids, [self._exit_state(p) for p in pids]))
        while True:
            still = {
                pid: st
                for pid, st in pending.items()
                if st == self.STILL_ACTIVE
            }
            if not still:
                return
            if time.monotonic() > end:
                self.fail(
                    "processes still active after the runner returned: "
                    f"{sorted(still)} (direct-kill-only cleanup would do this)"
                )
            time.sleep(0.2)
            pending.update({pid: self._exit_state(pid) for pid in still})

    def _run_tree(self, tmp_path, child_body, grand_body, timeout=0.6):
        pidfile_child = tmp_path / "child.pid"
        pidfile_desc = tmp_path / "desc.pid"
        child = self._write(tmp_path, "tree_child.py", child_body)
        grand = self._write(tmp_path, "tree_grand.py", grand_body)
        start = time.monotonic()
        result = envreport_probe._run_capped(
            [sys.executable, "-I", "-B", child, str(pidfile_child), str(pidfile_desc), grand],
            timeout=timeout,
        )
        elapsed = time.monotonic() - start
        child_pid = int(pidfile_child.read_text())
        desc_pid = int(pidfile_desc.read_text())
        return result, elapsed, child_pid, desc_pid

    def test_descendant_inheriting_stdout_cannot_hold_return(self, tmp_path):
        # The R2 repro itself: direct child times out while its grandchild
        # (inherited stdout) keeps the pipe open and would naturally live
        # ~30 s.  The whole tree must be terminated and the runner must
        # return well before that natural exit.
        (timed_out, rc, text, over), elapsed, child_pid, desc_pid = self._run_tree(
            tmp_path, self._CHILD_SLEEP, self._GRAND_SLEEP
        )
        assert timed_out is True
        assert rc is None
        assert text == ""
        assert elapsed < 4.0  # descendant's natural life is ~30 s
        self._assert_tree_dead(child_pid, desc_pid)

    def test_continuous_output_descendant_killed_and_bounded(self, tmp_path):
        # A descendant that keeps WRITING to the inherited pipe (not just
        # holding it) must not stall cleanup either: the reader reaches
        # EOF as soon as the tree dies and the return stays bounded.
        (timed_out, rc, text, over), elapsed, child_pid, desc_pid = self._run_tree(
            tmp_path, self._CHILD_SLEEP, self._GRAND_CONTINUOUS
        )
        assert timed_out is True
        assert rc is None
        assert elapsed < 4.0
        self._assert_tree_dead(child_pid, desc_pid)

    def test_descendant_dead_after_return_when_child_exits_zero(self, tmp_path):
        # The direct child exits 0 IMMEDIATELY, but its grandchild keeps
        # the inherited pipe open: the capture stalls, so the runner must
        # take the tree down (conservatively mapping to TIMEOUT) and the
        # descendant must be dead when it returns — independently
        # verified via its recorded pid.
        (timed_out, rc, text, over), elapsed, child_pid, desc_pid = self._run_tree(
            tmp_path, self._CHILD_EXIT_NOW, self._GRAND_SLEEP
        )
        assert timed_out is True
        assert rc is None
        assert elapsed < 5.0
        self._assert_tree_dead(child_pid, desc_pid)
        # The job kill is the primary mechanism: a direct-kill-only cleanup
        # would leave the descendant running (STILL_ACTIVE) here.
        state = self._exit_state(desc_pid)
        assert state is None or state != self.STILL_ACTIVE

    def test_assignment_failure_fails_closed_tool_never_runs(self, tmp_path, monkeypatch):
        # R2-C: when job assignment fails, the runner is FAIL-CLOSED:
        # the suspended child is terminated WITHOUT ever being resumed
        # (its first instruction — writing the marker file — never
        # executes, so the marker must not exist), the tool never runs,
        # and the probe reports a stable, non-sensitive UNAVAILABLE note.
        # The call stays bounded and never raises.
        marker = tmp_path / "marker.txt"
        child = self._write(
            tmp_path,
            "attach_child.py",
            "import time\n"
            f"open({str(marker)!r}, 'w').write('ran')\n"
            "time.sleep(30)\n",
        )
        monkeypatch.setattr(
            envreport_probe, "_win_job_attach", lambda api, job, ch: False
        )
        start = time.monotonic()
        with pytest.raises(envreport_probe._ContainmentSetupError):
            envreport_probe._run_capped(
                [sys.executable, "-I", "-B", child], timeout=0.6
            )
        elapsed = time.monotonic() - start
        assert elapsed < 6.0
        assert not marker.exists(), (
            "the child ran although it must never have been resumed"
        )
        # Handler level: the same failure maps to a stable, non-sensitive
        # UNAVAILABLE result (no WinError text, no traceback, no path).
        bat = tmp_path / "fake_tool.bat"
        bat.write_text("@echo off\r\nexit /b 0\r\n", encoding="ascii", newline="")
        monkeypatch.setattr(envreport_probe.shutil, "which", lambda name: str(bat))
        info = envreport_probe.probe_nvidia_smi("nvidia_smi")
        assert info["status"] == envreport_probe.STATUS_UNAVAILABLE
        assert info["note"] == (
            "nvidia-smi could not be executed in a contained process tree"
        )
        assert info["fields"] == {}
        assert "winerror" not in info["note"].lower()
        assert "traceback" not in info["note"].lower()

    def test_job_configuration_failure_fails_closed_tool_never_runs(self, tmp_path, monkeypatch):
        # R2-C: when KILL_ON_CLOSE cannot be configured AND verified on
        # this kernel, the tool must never be launched (the suspended
        # child is not even created): the probe reports the same stable,
        # non-sensitive UNAVAILABLE note and the call stays bounded.
        marker = tmp_path / "marker2.txt"
        child = self._write(
            tmp_path,
            "config_child.py",
            "import time\n"
            f"open({str(marker)!r}, 'w').write('ran')\n"
            "time.sleep(30)\n",
        )
        monkeypatch.setattr(
            envreport_probe,
            "_win_job_configure_kill_on_close",
            lambda api, job: False,
        )
        start = time.monotonic()
        with pytest.raises(envreport_probe._ContainmentSetupError):
            envreport_probe._run_capped(
                [sys.executable, "-I", "-B", child], timeout=0.6
            )
        elapsed = time.monotonic() - start
        assert elapsed < 6.0
        assert not marker.exists()
        bat = tmp_path / "fake_ffmpeg.bat"
        bat.write_text("@echo off\r\nexit /b 0\r\n", encoding="ascii", newline="")
        monkeypatch.setattr(envreport_probe.shutil, "which", lambda name: str(bat))
        info = envreport_probe.probe_system_tool("ffmpeg", "ffmpeg")
        assert info["status"] == envreport_probe.STATUS_UNAVAILABLE
        assert info["note"] == (
            "ffmpeg could not be executed in a contained process tree"
        )
        assert info["fields"] == {}

    def test_job_creation_failure_fails_closed_tool_never_runs(
        self, tmp_path, monkeypatch
    ):
        # R2-C: when CreateJobObjectW fails (falsy handle), nothing may
        # be spawned and nothing can ever be resumed: the same stable,
        # non-sensitive UNAVAILABLE contract as the other setup
        # failures, with a bounded call and no third state.
        api = envreport_probe._win_api()
        monkeypatch.setattr(api, "CreateJobObjectW", lambda attr, name: 0)
        marker = tmp_path / "marker3.txt"
        child = self._write(
            tmp_path,
            "create_child.py",
            "import time\n"
            f"open({str(marker)!r}, 'w').write('ran')\n"
            "time.sleep(30)\n",
        )
        start = time.monotonic()
        with pytest.raises(envreport_probe._ContainmentSetupError):
            envreport_probe._run_capped(
                [sys.executable, "-I", "-B", child], timeout=0.6
            )
        elapsed = time.monotonic() - start
        assert elapsed < 6.0
        assert not marker.exists()
        bat = tmp_path / "fake_ffprobe.bat"
        bat.write_text("@echo off\r\nexit /b 0\r\n", encoding="ascii", newline="")
        monkeypatch.setattr(envreport_probe.shutil, "which", lambda name: str(bat))
        info = envreport_probe.probe_system_tool("ffprobe", "ffprobe")
        assert info["status"] == envreport_probe.STATUS_UNAVAILABLE
        assert info["note"] == (
            "ffprobe could not be executed in a contained process tree"
        )
        assert info["fields"] == {}
        assert "winerror" not in info["note"].lower()
        assert "traceback" not in info["note"].lower()

    def test_malformed_class3_response_fails_closed_tool_never_runs(
        self, tmp_path, monkeypatch
    ):
        # R2-C: a malformed JobObjectProcessIdList response — the
        # reviewer's 12-byte repro: a two-DWORD header claiming one
        # listed PID plus only 4 bytes of it (the child's real PID in
        # the truncated slot) — must be rejected by the strict
        # documented-layout parser, so the suspended child is never
        # resumed (the tool never runs) and the probe reports the
        # stable UNAVAILABLE note.  The fake QueryInformationJobObject
        # returns exactly those 12 bytes for class 3 and delegates
        # everything else (the class-9 KILL_ON_CLOSE round trip) to the
        # real kernel function.
        import ctypes

        api = envreport_probe._win_api()
        spawned = []
        real_spawn = envreport_probe._win_spawn_suspended

        def spawn_spy(*args, **kwargs):
            child = real_spawn(*args, **kwargs)
            spawned.append(child)
            return child

        monkeypatch.setattr(envreport_probe, "_win_spawn_suspended", spawn_spy)
        real_query = api.QueryInformationJobObject
        query_type = ctypes.WINFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
        )

        def fake_query(job_h, cls, buf, size, retlen):
            if cls == envreport_probe._JOB_PROCESS_ID_LIST and spawned:
                data = (
                    (1).to_bytes(4, "little")
                    + (1).to_bytes(4, "little")
                    + (spawned[-1].pid & 0xFFFFFFFF).to_bytes(4, "little")
                )
                view = (ctypes.c_uint8 * 16).from_address(buf)
                for i, b in enumerate(data):
                    view[i] = b
                ctypes.cast(
                    retlen, ctypes.POINTER(ctypes.c_ulong)
                ).contents.value = 12
                return 1
            return real_query(
                job_h,
                cls,
                ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8)),
                size,
                ctypes.cast(retlen, ctypes.POINTER(ctypes.c_ulong)),
            )

        monkeypatch.setattr(
            api, "QueryInformationJobObject", query_type(fake_query)
        )
        marker = tmp_path / "marker4.txt"
        child = self._write(
            tmp_path,
            "malformed_child.py",
            "import time\n"
            f"open({str(marker)!r}, 'w').write('ran')\n"
            "time.sleep(30)\n",
        )
        start = time.monotonic()
        with pytest.raises(envreport_probe._ContainmentSetupError):
            envreport_probe._run_capped(
                [sys.executable, "-I", "-B", child], timeout=0.6
            )
        elapsed = time.monotonic() - start
        assert elapsed < 6.0
        assert not marker.exists(), (
            "the child was resumed although the membership response was malformed"
        )
        info = envreport_probe.probe_system_tool("ffmpeg", "ffmpeg")
        assert info["status"] == envreport_probe.STATUS_UNAVAILABLE
        assert info["note"] == (
            "ffmpeg could not be executed in a contained process tree"
        )
        assert info["fields"] == {}

    def test_zero_assigned_one_listed_response_fails_closed_tool_never_runs(
        self, tmp_path, monkeypatch
    ):
        # A class-3 response whose two-DWORD count header is
        # semantically impossible — ZERO assigned processes but ONE
        # listed PID (the spawned child's real PID in the single
        # pointer-width slot) — is malformed under the documented
        # JOBOBJECT_BASIC_PROCESS_ID_LIST count semantics
        # (NumberOfProcessIdsInList <= NumberOfAssignedProcesses) and
        # must be rejected BEFORE any membership test: the suspended
        # child is never resumed (the tool never runs) and the probe
        # reports the stable UNAVAILABLE note.  The fake
        # QueryInformationJobObject returns exactly those 16 bytes for
        # class 3 and delegates everything else (the class-9
        # KILL_ON_CLOSE round trip) to the real kernel function.
        import ctypes
        import struct

        api = envreport_probe._win_api()
        spawned = []
        real_spawn = envreport_probe._win_spawn_suspended

        def spawn_spy(*args, **kwargs):
            child = real_spawn(*args, **kwargs)
            spawned.append(child)
            return child

        monkeypatch.setattr(envreport_probe, "_win_spawn_suspended", spawn_spy)
        real_query = api.QueryInformationJobObject
        query_type = ctypes.WINFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
        )

        def fake_query(job_h, cls, buf, size, retlen):
            if cls == envreport_probe._JOB_PROCESS_ID_LIST and spawned:
                data = struct.pack("<IIQ", 0, 1, spawned[-1].pid)
                view = (ctypes.c_uint8 * 16).from_address(buf)
                for i, b in enumerate(data):
                    view[i] = b
                ctypes.cast(
                    retlen, ctypes.POINTER(ctypes.c_ulong)
                ).contents.value = 16
                return 1
            return real_query(
                job_h,
                cls,
                ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8)),
                size,
                ctypes.cast(retlen, ctypes.POINTER(ctypes.c_ulong)),
            )

        monkeypatch.setattr(
            api, "QueryInformationJobObject", query_type(fake_query)
        )
        marker = tmp_path / "marker5.txt"
        child = self._write(
            tmp_path,
            "zero_assigned_child.py",
            "import time\n"
            f"open({str(marker)!r}, 'w').write('ran')\n"
            "time.sleep(30)\n",
        )
        start = time.monotonic()
        with pytest.raises(envreport_probe._ContainmentSetupError):
            envreport_probe._run_capped(
                [sys.executable, "-I", "-B", child], timeout=0.6
            )
        elapsed = time.monotonic() - start
        assert elapsed < 6.0
        assert not marker.exists(), (
            "the child was resumed although the count header was malformed"
        )
        info = envreport_probe.probe_system_tool("ffmpeg", "ffmpeg")
        assert info["status"] == envreport_probe.STATUS_UNAVAILABLE
        assert info["note"] == (
            "ffmpeg could not be executed in a contained process tree"
        )
        assert info["fields"] == {}

    def test_kill_on_close_is_verified_and_proven_by_handle_close(self):
        # R2-A/R2-B: the production configure step must PROVE
        # KILL_ON_CLOSE on the kernel (Set -> Query round trip of the
        # flag at the verified layout) and closing ONLY the job handle
        # must terminate an attached, resumed child — no
        # TerminateJobObject and no PID kill involved.  This test fails
        # whenever the kernel-side kill-on-close bit is absent.
        import ctypes

        api = envreport_probe._win_api()
        job = envreport_probe._win_job_create(api)
        assert job
        child = None
        try:
            assert envreport_probe._win_job_configure_kill_on_close(api, job)
            # Independent kernel-side verification with the same typed ABI
            # (144-byte extended layout) the production configure step
            # writes: the KILL_ON_CLOSE bit must survive the round trip.
            back = envreport_probe._JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            back_ptr = ctypes.cast(
                ctypes.byref(back), ctypes.POINTER(ctypes.c_uint8)
            )
            written = ctypes.c_ulong(0)
            assert api.QueryInformationJobObject(
                job,
                envreport_probe._JOB_EXTENDED_LIMIT,
                back_ptr,
                ctypes.sizeof(back),
                ctypes.byref(written),
            )
            assert back.BasicLimitInformation.LimitFlags & envreport_probe._JOB_KILL_ON_CLOSE
            child = envreport_probe._win_spawn_suspended(
                api, [sys.executable, "-I", "-B", "-c", "import time; time.sleep(30)"]
            )
            assert child is not None
            assert envreport_probe._win_job_attach(api, job, child)
            envreport_probe._win_job_resume(api, child)
            time.sleep(0.3)  # let it start running
            assert envreport_probe._win_process_code(api, child.h_process) == 259
            # Close ONLY the job handle: kill-on-close must do the rest.
            envreport_probe._win_job_close(api, job)
            job = 0
            deadline = time.monotonic() + 5.0
            code = envreport_probe._STILL_ACTIVE
            while time.monotonic() < deadline:
                code = envreport_probe._win_process_code(api, child.h_process)
                if code != envreport_probe._STILL_ACTIVE:
                    break
                time.sleep(0.1)
            assert code != envreport_probe._STILL_ACTIVE, (
                "kill-on-close not in effect on this kernel"
            )
        finally:
            if child is not None:
                try:
                    if child.file is not None:
                        child.file.close()
                        child.file = None
                except Exception:  # noqa: BLE001
                    pass
                if child.h_process:
                    try:
                        if envreport_probe._win_process_code(
                            api, child.h_process
                        ) == envreport_probe._STILL_ACTIVE:
                            api.TerminateProcess(child.h_process, 0xFFFE)
                            api.WaitForSingleObject(child.h_process, 5000)
                    except Exception:  # noqa: BLE001
                        pass
                    api.CloseHandle(child.h_process)
                if child.h_thread:
                    api.CloseHandle(child.h_thread)
            if job:
                envreport_probe._win_job_close(api, job)

    def test_launch_order_is_create_configure_attach_then_resume(self, monkeypatch):
        # R2-B: the production sequence must be  create job -> configure
        # + verify kill-on-close -> create SUSPENDED -> assign (checked)
        # -> resume, with the resume strictly AFTER a successful assign.
        # The recorded call order of the real (wrapped) production seams
        # proves the ordering; the run is a real end-to-end integration
        # launch (the child actually runs and prints).
        calls = []

        def record(name, fn):
            def recorder(*args, **kwargs):
                calls.append(name)
                return fn(*args, **kwargs)

            return recorder

        for name in (
            "_win_job_create",
            "_win_job_configure_kill_on_close",
            "_win_spawn_suspended",
            "_win_job_attach",
            "_win_job_resume",
        ):
            monkeypatch.setattr(
                envreport_probe, name, record(name, getattr(envreport_probe, name))
            )

        timed_out, rc, text, over = envreport_probe._run_capped(
            [sys.executable, "-I", "-B", "-c", "print('ordertest')"], timeout=10.0
        )
        assert timed_out is False
        assert rc == 0
        assert "ordertest" in text
        assert calls == [
            "_win_job_create",
            "_win_job_configure_kill_on_close",
            "_win_spawn_suspended",
            "_win_job_attach",
            "_win_job_resume",
        ]

    def test_still_active_oracle_is_259_on_this_kernel(self):
        # R2-E: the liveness oracle must use STILL_ACTIVE == 259
        # (winbase.h), not 0xFFFFFFFF: the production query helper must
        # report 259 for a running process (even while suspended) and the
        # real exit code after TerminateProcess.
        assert envreport_probe._STILL_ACTIVE == 259
        api = envreport_probe._win_api()
        child = envreport_probe._win_spawn_suspended(
            api, [sys.executable, "-I", "-B", "-c", "pass"]
        )
        assert child is not None
        try:
            assert envreport_probe._win_process_code(api, child.h_process) == 259
            api.TerminateProcess(child.h_process, 7)
            api.WaitForSingleObject(child.h_process, 5000)
            assert envreport_probe._win_process_code(api, child.h_process) == 7
        finally:
            try:
                if child.file is not None:
                    child.file.close()
                    child.file = None
            except Exception:  # noqa: BLE001
                pass
            if child.h_process:
                api.CloseHandle(child.h_process)


# ---------------------------------------------------------------------------
# Fourth-round ABI / field / budget / hygiene pins
# ---------------------------------------------------------------------------


class TestWindowsJobObjectAbi:
    """Fifth-round ABI pin: the runner drives exactly ONE typed
    JOBOBJECT_*_LIMIT_INFORMATION ABI — the documented public SDK field
    spellings at native C alignment (IO_COUNTERS 48 B; the nine documented
    basic-limit fields totalling exactly 64 B with ``LimitFlags`` at
    offset 16; the 144-byte extended layout).  These tests fail if the
    struct spellings drift from the documented layout (sizes, field
    names, or offsets), if the class-7 cross-check returns, if the
    KILL_ON_CLOSE constant drifts, or if the kernel stops honouring the
    typed Set -> Query round trip.

    Kernel truths pinned by measurement (Windows build 26200): exactly
    144-byte extended buffers are accepted; 128/136/152/184/352 are
    rejected with ERROR_MORE_DATA.  On Query the kernel fills the
    documented fields with values it manages (working set, affinity, and
    the PriorityClass/SchedulingClass pair at offsets 56/60), so after
    the Set -> Query round trip proves the KILL_ON_CLOSE bit the runner
    reads only ``LimitFlags``.
    """

    pytestmark = pytest.mark.skipif(
        sys.platform != "win32", reason="Windows job-object ABI"
    )

    def test_struct_sizes_are_the_kernel_accepted_layout(self):
        import ctypes

        assert ctypes.sizeof(envreport_probe._IO_COUNTERS) == 48
        assert ctypes.sizeof(envreport_probe._JOBOBJECT_BASIC_LIMIT_INFORMATION) == 64
        assert ctypes.sizeof(envreport_probe._JOBOBJECT_EXTENDED_LIMIT_INFORMATION) == 144

    def test_all_documented_field_offsets_are_pinned(self):
        # Native C alignment of the documented fields produces exactly
        # these offsets (the gaps after LimitFlags and after
        # ActiveProcessLimit are ordinary C padding, not kernel
        # extensions).  A wrong field type reintroduced here moves an
        # offset and/or a size and fails this test.
        import ctypes

        basic = envreport_probe._JOBOBJECT_BASIC_LIMIT_INFORMATION
        ext = envreport_probe._JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        assert [name for name, _ in basic._fields_] == [
            "PerProcessUserTimeLimit",
            "PerJobUserTimeLimit",
            "LimitFlags",
            "MinimumWorkingSetSize",
            "MaximumWorkingSetSize",
            "ActiveProcessLimit",
            "Affinity",
            "PriorityClass",
            "SchedulingClass",
        ]
        assert basic.PerProcessUserTimeLimit.offset == 0
        assert basic.PerJobUserTimeLimit.offset == 8
        assert basic.LimitFlags.offset == 16
        assert basic.MinimumWorkingSetSize.offset == 24
        assert basic.MaximumWorkingSetSize.offset == 32
        assert basic.ActiveProcessLimit.offset == 40
        assert basic.Affinity.offset == 48
        assert basic.PriorityClass.offset == 56
        assert basic.SchedulingClass.offset == 60
        assert [name for name, _ in ext._fields_][2:] == [
            "ProcessMemoryLimit",
            "JobMemoryLimit",
            "PeakProcessMemoryUsed",
            "PeakJobMemoryUsed",
        ]
        assert ext.BasicLimitInformation.offset == 0
        assert ext.IoInfo.offset == 64
        assert ext.ProcessMemoryLimit.offset == 112
        assert ext.JobMemoryLimit.offset == 120
        assert ext.PeakProcessMemoryUsed.offset == 128
        assert ext.PeakJobMemoryUsed.offset == 136
        assert ctypes.sizeof(basic) == 64
        assert ctypes.sizeof(ext) == 144

    def test_kernel_constants_are_pinned(self):
        assert envreport_probe._JOB_EXTENDED_LIMIT == 9
        assert envreport_probe._JOB_PROCESS_ID_LIST == 3
        assert envreport_probe._JOB_KILL_ON_CLOSE == 0x2000
        assert envreport_probe._PID_LIST_QUERY_SIZE == 65536

    def test_class_seven_cross_check_is_gone(self):
        # Class 7 is JobObjectAssociateCompletionPortInformation (NOT a
        # process-count class); the attach cross-check must be the
        # documented JobObjectProcessIdList membership check, strictly
        # parsed through the dedicated parser.
        import inspect

        assert not hasattr(envreport_probe, "_JOB_ASSOCIATE_PROCESS")
        source = inspect.getsource(envreport_probe._win_job_attach)
        assert "_JOB_PROCESS_ID_LIST" in source
        assert "_parse_job_process_id_list" in source
        assert "ProcessCount" not in source

    def test_typed_set_query_round_trip_proves_flag_on_kernel(self):
        # The exact production verification, end to end: a fresh job
        # configured through the typed ABI must report KILL_ON_CLOSE
        # back through an independent typed query on this kernel.
        import ctypes

        api = envreport_probe._win_api()
        job = envreport_probe._win_job_create(api)
        assert job
        try:
            assert envreport_probe._win_job_configure_kill_on_close(api, job)
            back = envreport_probe._JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            back_ptr = ctypes.cast(
                ctypes.byref(back), ctypes.POINTER(ctypes.c_uint8)
            )
            written = ctypes.c_ulong(0)
            assert api.QueryInformationJobObject(
                job,
                envreport_probe._JOB_EXTENDED_LIMIT,
                back_ptr,
                ctypes.sizeof(back),
                ctypes.byref(written),
            )
            assert back.BasicLimitInformation.LimitFlags & envreport_probe._JOB_KILL_ON_CLOSE
        finally:
            if job:
                envreport_probe._win_job_close(api, job)

    def test_class_three_membership_is_what_the_attach_check_uses(self):
        # The attach cross-check is PID membership on
        # JobObjectProcessIdList (class 3), decoded from the documented
        # JOBOBJECT_BASIC_PROCESS_ID_LIST layout: a two-DWORD header
        # (NumberOfAssignedProcesses, NumberOfProcessIdsInList) followed
        # by one ULONG_PTR PID per listed process.  The test decodes the
        # raw kernel response with its OWN documented-layout reader
        # (independent of the production parser) AND feeds the same
        # bytes to the production parser: both must list the child's
        # PID.  A count check would be wrong — modern kernels list
        # extra bookkeeping entries alongside the real PIDs.
        import ctypes
        import struct

        api = envreport_probe._win_api()
        job = envreport_probe._win_job_create(api)
        assert job
        child = None
        try:
            assert envreport_probe._win_job_configure_kill_on_close(api, job)
            child = envreport_probe._win_spawn_suspended(
                api, [sys.executable, "-I", "-B", "-c", "import time; time.sleep(30)"]
            )
            assert child is not None
            assert envreport_probe._win_job_attach(api, job, child)
            # Re-query class 3 directly with the same bounded buffer
            # the production attach uses.
            qbuf = (ctypes.c_uint8 * envreport_probe._PID_LIST_QUERY_SIZE)()
            written = ctypes.c_ulong(0)
            assert api.QueryInformationJobObject(
                job,
                envreport_probe._JOB_PROCESS_ID_LIST,
                qbuf,
                envreport_probe._PID_LIST_QUERY_SIZE,
                ctypes.byref(written),
            )
            raw = bytes(qbuf[: written.value])
            # Test-owned documented-layout decode (oracle).
            n_assigned, n_listed = struct.unpack_from("<II", raw, 0)
            pid_w = ctypes.sizeof(ctypes.c_size_t)
            # Documented count semantics: the returned list may be
            # truncated by buffer space (listed < assigned), but can
            # never list more processes than are assigned.
            assert n_listed <= n_assigned
            assert len(raw) == 8 + n_listed * pid_w
            oracle = [
                int.from_bytes(
                    raw[8 + i * pid_w : 8 + (i + 1) * pid_w], "little"
                )
                for i in range(n_listed)
            ]
            assert child.pid in oracle
            # The production parser must agree on the same bytes.
            assert child.pid in envreport_probe._parse_job_process_id_list(raw)
        finally:
            # Closing the job (kill-on-close is configured) terminates
            # the attached child; child handles are cleaned up exactly
            # like the R2 regression tests do.
            if child is not None:
                try:
                    if child.file is not None:
                        child.file.close()
                        child.file = None
                except Exception:  # noqa: BLE001
                    pass
                if child.h_process:
                    try:
                        if envreport_probe._win_process_code(
                            api, child.h_process
                        ) == envreport_probe._STILL_ACTIVE:
                            api.TerminateProcess(child.h_process, 0xFFFE)
                            api.WaitForSingleObject(child.h_process, 5000)
                    except Exception:  # noqa: BLE001
                        pass
                    api.CloseHandle(child.h_process)
                if child.h_thread:
                    api.CloseHandle(child.h_thread)
            if job:
                envreport_probe._win_job_close(api, job)


class TestJobProcessIdListParser:
    """Fifth-round parser pin: the class-3 response is decoded ONLY as
    the documented JOBOBJECT_BASIC_PROCESS_ID_LIST layout — two DWORD
    counts followed by one ULONG_PTR (pointer-width) PID per listed
    process — and every deviation fails closed with ``_JobPidListError``
    (the attach treats that as "membership not verified" and the runner
    never resumes the suspended child).

    Fixtures are built here from the documented layout (an independent
    oracle, per the test-oracle rule) and fed to the PRODUCTION parser;
    the production parsing code is never copied into a test.
    """

    pytestmark = pytest.mark.skipif(
        sys.platform != "win32", reason="Windows job-object ABI"
    )

    @staticmethod
    def _fixture(n_assigned, n_listed, pids=(), pad=0):
        import ctypes
        import struct

        width = ctypes.sizeof(ctypes.c_size_t)
        body = b"".join(int(p).to_bytes(width, "little") for p in pids)
        return struct.pack("<II", n_assigned, n_listed) + body + b"\x00" * pad

    def test_pointer_width_is_the_native_size_t(self):
        import ctypes

        # x64 Windows: ULONG_PTR is 8 bytes; a hard-coded 4-byte PID
        # model (the previous defect) would parse the wrong bytes.
        assert ctypes.sizeof(ctypes.c_size_t) == 8

    def test_short_buffers_are_rejected(self):
        for data in (b"", b"\x00\x01\x00\x00", b"\x00\x00\x00\x00\x00\x00\x00\x00"[:7]):
            with pytest.raises(envreport_probe._JobPidListError):
                envreport_probe._parse_job_process_id_list(data)

    def test_header_claiming_pids_without_bytes_is_rejected(self):
        with pytest.raises(envreport_probe._JobPidListError):
            envreport_probe._parse_job_process_id_list(
                self._fixture(1, 1)  # claims one PID, supplies no PID bytes
            )

    def test_reviewer_twelve_byte_repro_is_rejected(self):
        # The reviewer's exact repro: header (1, 1) + only 4 bytes of
        # the "listed" PID (a 32-bit value, not the pointer-width value
        # the layout requires).  The old flat-DWORD parser accepted this
        # and resumed the child; the strict parser must reject it.
        data = (
            (1).to_bytes(4, "little")
            + (1).to_bytes(4, "little")
            + (0x01020304).to_bytes(4, "little")
        )
        assert len(data) == 12
        with pytest.raises(envreport_probe._JobPidListError):
            envreport_probe._parse_job_process_id_list(data)

    def test_listed_count_exceeding_assigned_is_rejected(self):
        # Documented count semantics: NumberOfProcessIdsInList is the
        # number of PID entries actually returned and can never exceed
        # NumberOfAssignedProcesses.  The buffer below holds exactly
        # those 5 pointer-width PID slots, so only the inverted-count
        # rule can reject it: a header claiming 5 listed processes out
        # of 2 assigned is impossible.
        with pytest.raises(envreport_probe._JobPidListError):
            envreport_probe._parse_job_process_id_list(
                self._fixture(2, 5, pids=(1, 2, 3, 4, 5))
            )

    def test_zero_assigned_one_listed_reviewer_repro_is_rejected(self):
        # The reviewer's exact count-semantics reproduction on x64: a
        # header claiming ZERO assigned processes but ONE listed PID,
        # with that one pointer-width PID in the buffer.  The old
        # (reversed) check accepted this; if the listed PID had been
        # the child's own PID, the attach would have resumed the
        # suspended child on a malformed header.  The corrected parser
        # must reject it.
        import struct

        target = 0xDEADBEEF
        data = struct.pack("<IIQ", 0, 1, target)
        assert len(data) == 16
        with pytest.raises(envreport_probe._JobPidListError):
            envreport_probe._parse_job_process_id_list(data)

    def test_impossible_listed_exceeding_assigned_counts_are_rejected(self):
        # Pins the corrected semantic relationship independently of the
        # length check: each buffer holds exactly its listed PID count,
        # so only the listed > assigned rule can reject it.
        for n_assigned, n_listed in ((0, 1), (1, 2), (2, 3)):
            with pytest.raises(envreport_probe._JobPidListError):
                envreport_probe._parse_job_process_id_list(
                    self._fixture(
                        n_assigned, n_listed, pids=range(1, n_listed + 1)
                    )
                )

    def test_valid_truncated_list_target_present_membership_succeeds(self):
        # The documented semantics accept a valid incomplete list
        # (assigned > listed) when the buffer holds exactly the listed
        # PID slots: the structure parses and the target PID, being
        # among the returned entries, lets membership verification
        # succeed on the truncated list (the fix must not overcorrect
        # and reject legitimate truncation).
        listed = envreport_probe._parse_job_process_id_list(
            self._fixture(3, 1, pids=(4242,))
        )
        assert listed == [4242]
        assert 4242 in listed

    def test_valid_truncated_list_target_absent_membership_fails_normally(self):
        # A truncated valid list whose returned entries do not contain
        # the target PID must parse cleanly (the structure is NOT
        # classified malformed) and simply fail membership.
        listed = envreport_probe._parse_job_process_id_list(
            self._fixture(3, 1, pids=(4242,))
        )
        assert 99999 not in listed

    def test_over_count_header_with_short_buffer_is_rejected(self):
        with pytest.raises(envreport_probe._JobPidListError):
            envreport_probe._parse_job_process_id_list(
                self._fixture(1, 5, pids=(1, 2))  # claims 5 PIDs, lists 2
            )

    def test_non_integral_trailing_bytes_are_rejected(self):
        with pytest.raises(envreport_probe._JobPidListError):
            envreport_probe._parse_job_process_id_list(
                self._fixture(1, 1, pids=(42,), pad=3)
            )

    def test_huge_list_count_is_rejected_without_allocation(self):
        # A corrupt/attacker-sized count must be rejected by the length
        # check before any allocation proportional to it.
        with pytest.raises(envreport_probe._JobPidListError):
            envreport_probe._parse_job_process_id_list(
                self._fixture(1, 2**31 - 1)
            )

    def test_valid_empty_and_single_pid_responses_parse(self):
        assert envreport_probe._parse_job_process_id_list(self._fixture(0, 0)) == []
        assert envreport_probe._parse_job_process_id_list(
            self._fixture(1, 1, pids=(12345,))
        ) == [12345]

    def test_valid_multi_pid_response_including_bookkeeping_entries_parses(self):
        # Modern kernels list extra bookkeeping entries alongside the
        # real PIDs; membership is a set test, so the full list must
        # come back in order, unfiltered.
        assert envreport_probe._parse_job_process_id_list(
            self._fixture(3, 3, pids=(777, 42, 7))
        ) == [777, 42, 7]

    def test_wide_pid_beyond_32_bits_parses_at_pointer_width(self):
        # A pointer-width value that does not fit in a DWORD parses as
        # itself (and would be mis-parsed by a 4-byte model).
        wide = 1 << 40
        assert envreport_probe._parse_job_process_id_list(
            self._fixture(1, 1, pids=(wide,))
        ) == [wide]

    def test_error_is_a_value_error(self):
        assert issubclass(envreport_probe._JobPidListError, ValueError)


class TestNvidiaSmiProbe:
    """Fourth-round contract for the nvidia-smi handler: the query must
    use the field name installed nvidia-smi versions accept
    (``compute_cap`` — the longer spelling ``compute_capability`` is
    rejected with exit code 2 by the installed tool), the parser must
    preserve the modeled ``compute_capability`` field name plus driver
    version, memory total and GPU identity, and a tool that rejects the
    old field name must make the probe fail — so the old query cannot
    regress silently.
    """

    _REAL_LINE = "NVIDIA GeForce RTX 4090, 591.86, 24564, 8.9"

    @staticmethod
    def _fake_which(monkeypatch, tmp_path):
        bat = tmp_path / "nvidia-smi.bat"
        bat.write_text("@echo off\r\nexit /b 0\r\n", encoding="ascii", newline="")
        monkeypatch.setattr(envreport_probe.shutil, "which", lambda name: str(bat))

    def test_query_tokens_are_the_accepted_field_set(self, tmp_path, monkeypatch):
        # The query string must request exactly the fields the installed
        # tool accepts, in order.
        self._fake_which(monkeypatch, tmp_path)
        seen = []

        def fake_run(cmd, timeout=None, with_stderr=False):
            seen.append(list(cmd))
            return (False, 0, self._REAL_LINE + "\r\n", False)

        monkeypatch.setattr(envreport_probe, "_run_capped", fake_run)
        info = envreport_probe.probe_nvidia_smi("nvidia_smi")
        assert info["status"] == envreport_probe.STATUS_OK
        assert len(seen) == 1
        query_args = [a for a in seen[0][1:] if a.startswith("--query-gpu=")]
        assert len(query_args) == 1
        tokens = query_args[0].split("=", 1)[1].split(",")
        assert tokens == ["name", "driver_version", "memory.total", "compute_cap"]

    def test_rejecting_tool_fails_on_the_old_field_name(self, tmp_path, monkeypatch):
        # A tool like the installed nvidia-smi rejects the unknown
        # field ``compute_capability`` with exit code 2 and accepts
        # ``compute_cap``.  Production code that still queries the old
        # field name gets rc 2 -> INIT_ERROR and this test fails; the
        # corrected query passes.
        self._fake_which(monkeypatch, tmp_path)

        def fake_run(cmd, timeout=None, with_stderr=False):
            query = next((a for a in cmd if a.startswith("--query-gpu=")), "")
            if "compute_capability" in query.split("=", 1)[1].split(","):
                return (False, 2, "", False)
            return (False, 0, self._REAL_LINE + "\r\n", False)

        monkeypatch.setattr(envreport_probe, "_run_capped", fake_run)
        info = envreport_probe.probe_nvidia_smi("nvidia_smi")
        assert info["status"] == envreport_probe.STATUS_OK

    def test_parser_keeps_modeled_field_names_and_identity(self, tmp_path, monkeypatch):
        # The parsed model keeps the field name ``compute_capability``
        # (stability), and preserves GPU identity, driver version and
        # memory total from the real output shape.
        self._fake_which(monkeypatch, tmp_path)
        monkeypatch.setattr(
            envreport_probe,
            "_run_capped",
            lambda *a, **k: (False, 0, self._REAL_LINE + "\r\n", False),
        )
        info = envreport_probe.probe_nvidia_smi("nvidia_smi")
        assert info["status"] == envreport_probe.STATUS_OK
        assert info["fields"]["driver_version"] == "591.86"
        assert info["fields"]["gpu_count"] == 1
        (gpu,) = info["fields"]["gpus"]
        assert gpu == {
            "name": "NVIDIA GeForce RTX 4090",
            "vram_mb": 24564,
            "compute_capability": "8.9",
        }

    def test_missing_executable_reports_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(envreport_probe.shutil, "which", lambda name: None)
        info = envreport_probe.probe_nvidia_smi("nvidia_smi")
        assert info["status"] == envreport_probe.STATUS_MISSING


class TestCleanupBudgetPin:
    """Fourth-round pin: the worst-case cleanup bound is
    ``timeout + 9 s + epsilon`` — one 5 s reap window plus one 2 s
    reader-settle window per drained pipe (two pipes on Windows, one on
    POSIX) — and ``_CLEANUP_BUDGET_MAX`` is the single pinned source of
    that bound.
    """

    def test_cleanup_budget_is_pinned_to_nine_seconds(self):
        assert envreport_probe._REAP_BUDGET == 5.0
        assert envreport_probe._SETTLE_BUDGET == 2.0
        assert envreport_probe._CLEANUP_BUDGET_MAX == 9.0
        assert envreport_probe._CLEANUP_BUDGET_MAX == (
            envreport_probe._REAP_BUDGET + 2 * envreport_probe._SETTLE_BUDGET
        )


class TestProhibitedBytecodeHygiene:
    """Fourth-round regression: a production-faithful probe-worker spawn
    (the runtime tree's own interpreter, ``-I -B``, exactly the command
    the launcher chain and ``envreport.py`` build) must leave the active
    runtime tree free of new ``__pycache__``/``*.pyc`` entries.  The
    third-round run left generated bytecode in ALL assembled runtimes:
    a direct, unguarded invocation of a tree interpreter writes
    ``encodings``/``_distutils_hack`` bytecode during interpreter
    STARTUP — before any in-process guard can run — and the guarded
    production chain (``-I -B`` launcher flags, runtime_entry's
    ``PYTHONDONTWRITEBYTECODE=1`` / ``sys.dont_write_bytecode``,
    ``-I -B`` worker spawns) never does.  The worker probe also
    exercises the nested contained run (nvidia-smi inside a job object).
    """

    @staticmethod
    def _active_runtime_tree():
        marker = ROOT / "runtime" / "active-runtime.txt"
        if not marker.is_file():
            return None
        runtime_id = marker.read_text(encoding="ascii", errors="replace").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", runtime_id):
            return None
        tree = ROOT / "runtime" / "versions" / runtime_id
        if not (tree / "python.exe").is_file():
            return None
        return tree

    @staticmethod
    def _count_prohibited(tree):
        dirs, files = 0, 0
        for path in tree.rglob("*"):
            if path.is_dir() and path.name == "__pycache__":
                dirs += 1
            elif path.is_file() and path.name.endswith(".pyc"):
                files += 1
        return (dirs, files)

    def test_worker_spawn_leaves_no_new_bytecode_in_the_runtime_tree(self):
        # CURRENT-MACHINE: needs an assembled runtime in this checkout.
        tree = self._active_runtime_tree()
        if tree is None:
            pytest.skip("no assembled runtime in this checkout")
        probe_script = ROOT / "scripts" / "envreport_probe.py"
        before = self._count_prohibited(tree)
        proc = subprocess.run(
            [str(tree / "python.exe"), "-I", "-B", str(probe_script), "nvidia_smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30.0,
            cwd=str(ROOT),
        )
        after = self._count_prohibited(tree)
        assert after == before, (
            "the production worker spawn wrote prohibited bytecode into "
            f"the runtime tree: before={before} after={after}"
        )
        assert proc.returncode == 0
        record = json.loads(proc.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
        assert record["name"] == "nvidia_smi"
        assert record["status"] == envreport_probe.STATUS_OK


# ---------------------------------------------------------------------------
# Privacy gate
# ---------------------------------------------------------------------------


class TestPrivacy:
    def test_poison_strings_are_redacted_in_both_serializers(self, world):
        probes = cpu_probe_set(
            cuda=envreport.ProbeInfo("cuda", "OK", POISON_DRIVE, {}),
            torch=envreport.ProbeInfo(
                "torch", "OK", POISON_ENVVAR, {"version": "2.14.0+cpu", "cuda": None, "local_tag": "cpu"}
            ),
            nvidia_smi=envreport.ProbeInfo(
                "nvidia_smi",
                "OK",
                "",
                {
                    "driver_version": POISON_CRED,
                    "gpus": [
                        {"name": POISON_UNC, "vram_mb": 24576, "compute_capability": "8.9"},
                        {"name": POISON_IP, "vram_mb": 1024, "compute_capability": "7.5"},
                    ],
                },
            ),
        )
        rc_json, payload_json, _ = run_envreport(
            ["envreport", "--json"], repo_root=world.env.tmp_path, probe_runner=runner_for(probes)
        )
        rc_txt, payload_txt, _ = run_envreport(
            ["envreport"], repo_root=world.env.tmp_path, probe_runner=runner_for(probes)
        )
        assert rc_json == 0 and rc_txt == 0
        for poison in (POISON_DRIVE, POISON_ENVVAR, POISON_IP, POISON_UNC, POISON_CRED, "Alice", "fileserver"):
            assert poison not in payload_json
            assert poison not in payload_txt
        assert "<redacted>" in payload_json
        assert "<redacted>" in payload_txt
        for payload in (payload_json, payload_txt):
            assert br.privacy_violations(payload) == []
        data = json.loads(payload_json)
        by_name = {p["name"]: p for p in data["probes"]}
        assert by_name["cuda"]["note"] == "<redacted>"
        assert by_name["torch"]["note"] == "<redacted>"
        assert by_name["nvidia_smi"]["fields"]["driver_version"] == "<redacted>"
        gpus = by_name["nvidia_smi"]["fields"]["gpus"]
        assert gpus[0]["name"] == "<redacted>"
        assert gpus[1]["name"] == "<redacted>"
        assert gpus[0]["vram_mb"] == 24576  # non-string shapes survive intact

    def test_final_gate_fail_closed(self, world, monkeypatch):
        monkeypatch.setattr(envreport, "_privacy_violations", lambda text, root: ["synthetic"])
        rc, out, err = run_envreport(
            ["envreport", "--json"],
            repo_root=world.env.tmp_path,
            probe_runner=runner_for(cpu_probe_set()),
        )
        assert rc == 1
        assert out == ""
        assert err == "envreport: report output suppressed by the privacy gate\n"

    def test_final_gate_fail_closed_verify_payload(self, world, monkeypatch):
        monkeypatch.setattr(envreport, "_privacy_violations", lambda text, root: ["synthetic"])
        rc, out, err = run_envreport(
            ["envreport", "--verify", "--json"],
            repo_root=world.env.tmp_path,
            probe_runner=runner_for(cpu_probe_set()),
            resolve_inputs=make_resolvers(world),
        )
        assert rc == 1
        assert out == ""
        assert "suppressed by the privacy gate" in err

    def test_final_gate_real_detector_via_uncleaned_package_version(self, world, monkeypatch):
        # packages[].version flows raw from importlib.metadata (no per-field
        # _clean); the final rendered-payload scan with the REAL detector
        # (builder PRIVACY_RULES + local supplemental rule) must trip and
        # suppress the report with the exact fail-closed line.
        monkeypatch.setattr(envreport._imetadata, "version", lambda name: POISON_DRIVE)
        rc, out, err = run_envreport(
            ["envreport", "--json"],
            repo_root=world.env.tmp_path,
            probe_runner=runner_for(cpu_probe_set()),
        )
        assert rc == 1
        assert out == ""
        assert err == "envreport: report output suppressed by the privacy gate\n"

    def test_scanner_failure_fails_closed_text(self, tmp_path, monkeypatch):
        # The shared scanner is forced to raise on EVERY input.  The run
        # must fail closed exactly like a real privacy violation (empty
        # stdout, one fixed stderr line, rc 1) and must never surface the
        # exception type or message.
        def boom(text):
            raise RuntimeError("scanner died on C:\\Secret\\TOKEN-XYZ")

        monkeypatch.setattr(br, "privacy_violations", boom)
        rc, out, err = run_envreport(
            ["envreport"],
            repo_root=tmp_path,
            probe_runner=runner_for(cpu_probe_set()),
        )
        assert rc == 1
        assert out == ""
        assert err == "envreport: report output suppressed by the privacy gate\n"
        assert "C:\\Secret" not in err
        assert "TOKEN-XYZ" not in err
        assert "RuntimeError" not in err
        assert "scanner died" not in err

    def test_scanner_failure_fails_closed_json(self, tmp_path, monkeypatch):
        # Same forced scanner failure through the JSON render path: the
        # rendered payload is still suppressed by the final gate.
        def boom(text):
            raise RuntimeError("scanner died on C:\\Secret\\TOKEN-XYZ")

        monkeypatch.setattr(br, "privacy_violations", boom)
        rc, out, err = run_envreport(
            ["envreport", "--json"],
            repo_root=tmp_path,
            probe_runner=runner_for(cpu_probe_set()),
        )
        assert rc == 1
        assert out == ""
        assert err == "envreport: report output suppressed by the privacy gate\n"
        assert "C:\\Secret" not in err
        assert "TOKEN-XYZ" not in err
        assert "RuntimeError" not in err

    def test_clean_strings_pass_through(self, world):
        clean = "torch 2.14.0+cpu (cuda 13.0)"
        assert envreport._clean(clean, world.env.tmp_path) == clean
        assert envreport._clean(123, world.env.tmp_path) == 123
        assert envreport._clean(None, world.env.tmp_path) is None

    def test_env_var_reference_forms_are_caught_by_local_rule(self, world):
        root = world.env.tmp_path
        # The shared PRIVACY_RULES table cannot match these (a closing '%'
        # sits between the variable name and the separator); the local
        # supplemental rule must.
        for poisoned in ("%APPDATA%\\x", "%LOCALAPPDATA%\\pip", "%appdata%\\x", "%TEMP%/cache"):
            assert envreport._privacy_violations(poisoned, root)
        # Clean strings stay clean (no false positives from the local rule).
        clean = ("torch 2.14.0+cpu", "https://github.com/x", "172.32.0.1", "%NOSEPARATOR% here")
        for value in clean:
            assert not envreport._privacy_violations(value, root)
            assert envreport._clean(value, root) == value
        assert envreport._clean(POISON_ENVVAR, root) == "<redacted>"


# ---------------------------------------------------------------------------
# --verify section
# ---------------------------------------------------------------------------


class TestVerify:
    def test_full_pass_json(self, world):
        rc, payload, err = run_envreport(
            ["envreport", "--verify", "--json"],
            repo_root=world.env.tmp_path,
            probe_runner=runner_for(cpu_probe_set()),
            resolve_inputs=make_resolvers(world),
        )
        assert rc == 0
        data = json.loads(payload)
        checks, vs = verify_checks(data)
        assert vs["available"] is True
        assert vs["target_runtime_id"] == world.runtime_id
        assert vs["variant"] == "cpu-nogui"
        assert vs["required_failed"] == 0
        assert vs["total"] == 30
        assert vs["passed"] == 26
        assert vs["summary"] == "PASS (26/30 checks OK; all required checks OK)"
        assert checks["inputs"]["status"] == "OK"
        assert checks["selector_agreement"]["status"] == "OK"
        assert checks["torch_contract"]["status"] == "OK"
        assert checks["running_python"]["status"] == "UNAVAILABLE"
        assert checks["running_python"]["required"] is False
        engine = engine_checks(vs)
        assert len(engine) == 19  # the builder engine's check list (incl. the duplicate id)
        assert all(c["status"] == "OK" for c in engine)
        for name in envreport.PROBES:
            assert checks[f"probe_{name}"]["required"] is False

    def test_full_pass_text_render(self, world):
        rc, payload, err = run_envreport(
            ["envreport", "--verify"],
            repo_root=world.env.tmp_path,
            probe_runner=runner_for(cpu_probe_set()),
            resolve_inputs=make_resolvers(world),
        )
        assert rc == 0
        assert "[Verify]" in payload
        assert "PASS (26/30 checks OK; all required checks OK)" in payload
        for check_id in ("inputs", "selector_agreement", "running_python", "torch_contract"):
            assert check_id in payload

    def test_torch_contract_mismatch_cuda_line(self, world):
        probes = cpu_probe_set(
            torch=envreport.ProbeInfo(
                "torch", "OK", "", {"version": "2.14.0+cpu", "cuda": "13.0", "local_tag": "cpu"}
            ),
        )
        rc, payload, err = run_envreport(
            ["envreport", "--verify", "--json"],
            repo_root=world.env.tmp_path,
            probe_runner=runner_for(probes),
            resolve_inputs=make_resolvers(world),
        )
        assert rc == 1
        data = json.loads(payload)
        checks, vs = verify_checks(data)
        assert vs["required_failed"] == 1
        assert vs["summary"] == "FAIL (1 required checks not OK; 25/30 checks OK)"
        assert checks["torch_contract"]["status"] == "MISMATCH"
        assert "cpu backend" in checks["torch_contract"]["note"]
        # the full report is still rendered: engine checks all present and OK
        assert all(c["status"] == "OK" for c in engine_checks(vs))

    def test_torch_contract_takes_probe_status_when_probe_fails(self, world):
        probes = cpu_probe_set(
            torch=envreport.ProbeInfo("torch", "TIMEOUT", "probe timed out", {}),
        )
        rc, payload, err = run_envreport(
            ["envreport", "--verify", "--json"],
            repo_root=world.env.tmp_path,
            probe_runner=runner_for(probes),
            resolve_inputs=make_resolvers(world),
        )
        assert rc == 1
        data = json.loads(payload)
        checks, vs = verify_checks(data)
        assert checks["torch_contract"]["status"] == "TIMEOUT"
        assert vs["required_failed"] == 1

    def test_engine_tamper_detected(self, world):
        (world.runtime_root / "versions" / world.runtime_id / "python.exe").write_bytes(b"tampered")
        rc, payload, err = run_envreport(
            ["envreport", "--verify", "--json"],
            repo_root=world.env.tmp_path,
            probe_runner=runner_for(cpu_probe_set()),
            resolve_inputs=make_resolvers(world),
        )
        assert rc == 1
        data = json.loads(payload)
        checks, vs = verify_checks(data)
        assert checks["installed_tree"]["status"] != "OK"
        assert vs["required_failed"] >= 1
        # engine notes may quote tree paths: per-field redaction must keep the payload clean
        assert br.privacy_violations(payload) == []

    def test_engine_builder_error_code_only(self, world, monkeypatch):
        def fake_verify(*args, **kwargs):
            raise br.BuilderError("privacy", "leaked C:\\Users\\Alice\\AppData\\x")

        monkeypatch.setattr(br, "verify_runtime", fake_verify)
        rc, payload, err = run_envreport(
            ["envreport", "--verify", "--json"],
            repo_root=world.env.tmp_path,
            probe_runner=runner_for(cpu_probe_set()),
            resolve_inputs=make_resolvers(world),
        )
        assert rc == 1
        data = json.loads(payload)
        _, vs = verify_checks(data)
        engine = [c for c in vs["checks"] if c["id"] == "engine"]
        assert len(engine) == 1
        assert engine[0]["status"] == "MISMATCH"
        assert engine[0]["note"] == "engine stopped: privacy"
        assert engine[0]["required"] is True
        # the BuilderError message (with a path) must never reach the payload
        assert "Alice" not in payload
        assert "AppData" not in payload

    def test_engine_generic_exception_uses_type_name(self, world, monkeypatch):
        def fake_verify(*args, **kwargs):
            raise RuntimeError("engine secret detail")

        monkeypatch.setattr(br, "verify_runtime", fake_verify)
        rc, payload, err = run_envreport(
            ["envreport", "--verify", "--json"],
            repo_root=world.env.tmp_path,
            probe_runner=runner_for(cpu_probe_set()),
            resolve_inputs=make_resolvers(world),
        )
        assert rc == 1
        data = json.loads(payload)
        _, vs = verify_checks(data)
        engine = [c for c in vs["checks"] if c["id"] == "engine"]
        assert len(engine) == 1
        assert engine[0]["status"] == "INIT_ERROR"
        assert engine[0]["note"] == "engine error: RuntimeError"
        assert engine[0]["required"] is True
        # the exception detail must never reach the payload
        assert "secret" not in payload

    def test_engine_code_mapping(self, world, monkeypatch):
        # Every key of the production mapping table gets a synthetic engine
        # check, so the whole table is pinned; the set guard below makes any
        # future added/renamed/removed production key fail loudly.
        cases = [
            ("c_pass", "PASS", "OK"),
            ("c_mismatch", "MISMATCH", "MISMATCH"),
            ("c_schema_mismatch", "SCHEMA_MISMATCH", "MISMATCH"),
            ("c_source_mismatch", "SOURCE_MISMATCH", "MISMATCH"),
            ("c_hash_mismatch", "HASH_MISMATCH", "MISMATCH"),
            ("c_extra_entry", "EXTRA_ENTRY", "MISMATCH"),
            ("c_present", "PRESENT", "MISMATCH"),
            ("c_missing_manifest", "MISSING_MANIFEST", "MISSING"),
            ("c_missing_interpreter", "MISSING_INTERPRETER", "MISSING"),
            ("c_missing_entry", "MISSING_ENTRY", "MISSING"),
            ("c_unknown_code", "WEIRD_CODE", "MISMATCH"),  # unmapped codes fail closed
        ]
        assert {code for _, code, _ in cases} == set(envreport._ENGINE_CODE_MAP) | {"WEIRD_CODE"}
        fake = [br.Check(name, code) for name, code, _ in cases]
        monkeypatch.setattr(br, "verify_runtime", lambda *a, **k: list(fake))
        rc, payload, err = run_envreport(
            ["envreport", "--verify", "--json"],
            repo_root=world.env.tmp_path,
            probe_runner=runner_for(cpu_probe_set()),
            resolve_inputs=make_resolvers(world),
        )
        assert rc == 1  # several required failures
        data = json.loads(payload)
        checks, vs = verify_checks(data)
        for name, _code, expected in cases:
            assert checks[name]["status"] == expected
        assert vs["required_failed"] >= 10  # all engine checks except c_pass

    def test_inputs_failure_short_circuits_engine(self, world):
        def boom(root, variant):
            raise br.BuilderError("lock", "boom")

        rc, payload, err = run_envreport(
            ["envreport", "--verify", "--json"],
            repo_root=world.env.tmp_path,
            probe_runner=runner_for(cpu_probe_set()),
            resolve_inputs=boom,
        )
        assert rc == 1
        data = json.loads(payload)
        checks, vs = verify_checks(data)
        assert vs["available"] is True
        assert checks["inputs"]["status"] == "UNAVAILABLE"
        assert checks["inputs"]["note"] == "inputs: lock"  # code only, never the message
        assert checks["inputs"]["required"] is True
        assert checks["selector_agreement"]["status"] == "OK"
        assert "manifest_privacy" not in checks  # engine never ran
        assert "installed_tree" not in checks
        assert vs["summary"] == "inputs unresolved; engine not run"
        # probes still collected and rendered
        for name in envreport.PROBES:
            assert f"probe_{name}" in checks

    def test_inputs_exception_without_code_uses_type_name(self, world):
        def boom(root, variant):
            raise RuntimeError("secret path detail")

        rc, payload, err = run_envreport(
            ["envreport", "--verify", "--json"],
            repo_root=world.env.tmp_path,
            probe_runner=runner_for(cpu_probe_set()),
            resolve_inputs=boom,
        )
        assert rc == 1
        data = json.loads(payload)
        checks, vs = verify_checks(data)
        assert checks["inputs"]["note"] == "inputs: RuntimeError"
        assert "secret path detail" not in payload

    def test_selector_missing(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        rc, payload, err = run_envreport(
            ["envreport", "--verify", "--json"],
            repo_root=repo,
            probe_runner=runner_for(cpu_probe_set()),
        )
        assert rc == 1
        data = json.loads(payload)
        checks, vs = verify_checks(data)
        assert vs["available"] is False
        assert checks["target"]["status"] == "MISSING"
        assert checks["target"]["required"] is True
        assert vs["summary"] == "no runnable runtime identified"

    def test_selector_malformed(self, tmp_path):
        repo = tmp_path / "repo"
        (repo / "runtime").mkdir(parents=True)
        (repo / "runtime" / "active-runtime.txt").write_bytes(b"garbage\n")
        # base report: a malformed selector is informational, not fatal
        rc_base, payload_base, _ = run_envreport(
            ["envreport", "--json"], repo_root=repo, probe_runner=runner_for(cpu_probe_set())
        )
        assert rc_base == 0
        data_base = json.loads(payload_base)
        assert data_base["runtime"]["selector"]["valid"] is False
        # verify: no target can be derived -> unavailable
        rc, payload, err = run_envreport(
            ["envreport", "--verify", "--json"],
            repo_root=repo,
            probe_runner=runner_for(cpu_probe_set()),
        )
        assert rc == 1
        data = json.loads(payload)
        checks, vs = verify_checks(data)
        assert checks["target"]["status"] == "MISSING"
        assert vs["available"] is False

    def test_variant_undecidable(self, world):
        # selector points at a runtime id whose tree does not exist
        (world.runtime_root / "active-runtime.txt").write_bytes((("00" * 32) + "\r\n").encode())
        rc, payload, err = run_envreport(
            ["envreport", "--verify", "--json"],
            repo_root=world.env.tmp_path,
            probe_runner=runner_for(cpu_probe_set()),
        )
        assert rc == 1
        data = json.loads(payload)
        checks, vs = verify_checks(data)
        assert vs["available"] is True
        assert vs["target_runtime_id"] == "00" * 32
        assert vs["variant"] is None
        assert checks["variant"]["status"] == "UNAVAILABLE"
        assert checks["selector_agreement"]["status"] == "OK"
        assert vs["summary"] == "variant undecidable; engine not run"

    def test_verify_section_exception_degrades_gracefully(self, world, monkeypatch):
        def boom(repo_root, runtime_info, probe_by_name, resolve_inputs):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(envreport, "_build_verify", boom)
        rc, payload, err = run_envreport(
            ["envreport", "--verify", "--json"],
            repo_root=world.env.tmp_path,
            probe_runner=runner_for(cpu_probe_set()),
        )
        assert rc == 1
        data = json.loads(payload)
        vs = data["verify_section"]
        assert vs["available"] is False
        assert vs["checks"][0]["id"] == "verify"
        assert vs["checks"][0]["status"] == "INIT_ERROR"
        assert "RuntimeError" in vs["checks"][0]["note"]
        # the base report is still rendered in full
        assert data["probes"]


# ---------------------------------------------------------------------------
# Failure safety of the entry point
# ---------------------------------------------------------------------------


class TestEntryPointSafety:
    def test_base_construction_failure_exit_1(self, world, monkeypatch):
        def boom():
            raise ValueError("nope")

        monkeypatch.setattr(envreport, "_os_info", boom)
        rc, out, err = run_envreport(
            ["envreport", "--json"],
            repo_root=world.env.tmp_path,
            probe_runner=runner_for(cpu_probe_set()),
        )
        assert rc == 1
        assert out == ""
        assert err == "envreport: ValueError\n"

    def test_selector_grammar(self, tmp_path):
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        selector = runtime_dir / "active-runtime.txt"
        valid = ("ab" * 32)

        def parse(content: bytes):
            selector.write_bytes(content)
            return envreport._selector_info(tmp_path, br)

        assert parse(valid.encode()).valid is True
        assert parse((valid + "\r\n").encode()).valid is True
        assert parse((valid + "\n").encode()).valid is True
        assert parse(valid.upper().encode()).valid is False
        assert parse(b"").valid is False
        assert parse(b"short\n").valid is False
        assert parse((valid + "extra\n").encode()).valid is False
        assert parse((valid + "\n\n").encode()).valid is False  # two logical lines
        selector.unlink()
        missing = envreport._selector_info(tmp_path, br)
        assert missing.present is False
        assert missing.valid is False


# ---------------------------------------------------------------------------
# Import boundary (subprocess): main.py envreport imports no third-party root
# ---------------------------------------------------------------------------

WRAPPER_SOURCE = """\
import json
import runpy
import sys
from pathlib import Path

BLOCKED = {
    "core", "torch", "cv2", "onnx", "onnxruntime", "numpy", "scipy", "PyQt5",
    "ffmpeg", "colorama", "flatbuffers", "future", "jinja2", "markupsafe",
    "ml_dtypes", "mpmath", "networkx", "numexpr", "packaging", "pip",
    "protobuf", "setuptools", "sympy", "tqdm", "typing_extensions", "filelock",
    "fsspec", "opencv_python", "main",
}


def main():
    main_py, rec_file = sys.argv[1], sys.argv[2]
    app_args = sys.argv[3:]
    recorded = []

    class Recorder:
        def find_spec(self, name, path=None, target=None):
            root = name.split(".", 1)[0]
            if root in BLOCKED:
                recorded.append(name)
                raise ImportError(f"blocked by boundary recorder: {name}")
            return None

    sys.meta_path.insert(0, Recorder())
    sys.path.insert(0, str(Path(main_py).resolve().parent))
    sys.argv = [main_py] + app_args
    try:
        runpy.run_path(main_py, run_name="__main__")
        rc = 0
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
    finally:
        Path(rec_file).write_text(json.dumps(recorded), encoding="utf-8")
    raise SystemExit(rc)


main()
"""


def _boundary_run(tmp_path, app_args, timeout=300):
    wrapper = tmp_path / "boundary_wrapper.py"
    wrapper.write_text(WRAPPER_SOURCE, encoding="utf-8")
    rec_file = tmp_path / "recorded.json"
    cmd = [sys.executable, "-B", "-X", "utf8", str(wrapper), str(MAIN_PY), str(rec_file), *app_args]
    proc = subprocess.run(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout
    )
    recorded = json.loads(rec_file.read_text(encoding="utf-8"))
    return proc.returncode, proc.stdout.decode("utf-8", "replace"), recorded


class TestImportBoundary:
    def test_envreport_dispatch_imports_no_third_party(self, tmp_path):
        rc, stdout, recorded = _boundary_run(tmp_path, ["envreport", "--json"])
        assert rc == 0
        assert recorded == []
        data = json.loads(stdout)
        assert data["schema"] == "dfl-envreport-v1"
        assert data["json"] is True
        assert data["verify"] is False

    def test_non_envreport_path_still_reaches_third_party(self, tmp_path):
        rc, stdout, recorded = _boundary_run(tmp_path, [], timeout=120)
        assert rc != 0
        assert stdout == ""
        assert recorded, "negative control: the heavy dispatch path must import third-party roots"
