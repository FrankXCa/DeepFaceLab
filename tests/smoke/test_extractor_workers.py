"""Phase 9D acceptance: restored extractor worker counts and the
deterministic extractor-worker lifecycle.

Restored USER_LEGACY semantics (pinned in docs/PHASE9_STATE.md §8):

- ``--gpu-worker-count`` / ``--final-worker-count`` CLI flags, added to
  the ``extract`` subparser only (default ``None`` at the argparse
  boundary) and forwarded to ``Extractor.main``;
- ``main()`` resolution: ``None`` + GPU -> prompt "Parallel GPU workers
  per device" (default 1, range 1..32); ``None`` + ``--cpu-only`` ->
  resolves silently to 1 (no prompt); ``None`` final -> prompt with
  default ``min(8, max(1, cpu_count // 2))`` and range
  ``1..cpu_count``;
- ``get_devices_for_config``: each selected GPU contributes
  ``max(1, int(gpu_worker_count))`` workers, every worker staying
  attached to its original ``device_idx`` (names ``#i``-suffixed when
  the count > 1); zero/negative values clamp to 1 (0 NEVER means CPU —
  CPU-only is controlled only by ``--cpu-only`` / an empty device
  list); the CPU-only branch is unchanged (``min(8, cpu//2)`` CPU
  workers, or the single ``CPU`` device for ``landmarks-manual``); the
  ``landmarks-manual`` GPU branch applies the same multiplier on the
  single best device; the final stage uses
  ``max(1, min(final_worker_count, cpu_count))`` CPU workers — an
  explicit count overrides the old hard-coded ``min(8, cpu_count)``;
- the ``7b52151`` Subprocessor lifecycle (deterministic
  close/finalize handshake, kill+join on worker error) is preserved:
  after every stage run, zero owned subprocesses remain.

Coverage:

A. ``get_devices_for_config`` (pure function; fake ``DeviceConfig``
   containers — no hardware, no backend registry):
   1 GPU x 1 / x N; 2 GPUs x N; 0 -> 1; negative -> 1; CPU-only
   branch; landmarks-manual multiplier on the best device; final
   explicit count; final > cpu_count clamp; final <= 0 clamp; the
   DEBUG final branch; device_idx and name preservation.
B. CLI (a real ``python main.py extract`` subprocess — the parser in
   ``main.py`` lives under ``if __name__ == "__main__"`` and is not
   importable by design, so the subprocess IS the plumbing test; a
   source-level AST check additionally pins the argparse boundary):
   both flags parse and propagate (exit 0, NO prompt text);
   omitting ``--final-worker-count`` makes the final prompt fire;
   non-cpu-only without ``--gpu-worker-count`` makes the GPU prompt
   fire (the child is blocked on the open stdin pipe at the prompt —
   it cannot proceed to the pipeline — then terminated: no orphan
   processes are possible).
C. Pipeline integration (real owned worker subprocesses running the
   torch S3FD/FAN extractors, CPU-structured; the CUDA worker path is
   covered by the device-list tests + the Phase 9E CUDA validation):
   in-process ``Extractor.main(cpu_only=True, ...)`` on the
   repo-tracked public fixtures (``doc/mini_tutorial.jpg`` = one
   face, ``doc/meme1.jpg`` = none) -> face counts, DFLJPG output with
   68x2 landmarks, no prompt fired, and zero owned processes left;
   the per-worker device contract (``client_dict`` fields from
   ``process_info_generator``) for CPU and fake-GPU configs.
D. Worker lifecycle (the ``Subprocessor`` machinery, owned ``Process``
   objects — no process-name matching):
   normal completion (final stage, 2 workers, 2 images) -> zero
   owned processes; every owned ``cli.p`` is dead AND joined;
   repeated runs -> no worker accumulation;
   error path (a worker raises during final-stage output write:
   the output directory is deliberately missing, so the write
   failure surfaces as a raised ``AttributeError`` in the child)
   -> the parent ``run()`` returns normally, the failed chunk is
   dropped, the owned worker is reaped, no orphans.

Runs in both validated environments (CPU venv and CUDA venv): every
test here is CPU-structured and deterministic.
"""

import ast
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.leras import nn  # noqa: E402
from core.leras.device import Device, DeviceConfig  # noqa: E402
from facelib import FaceType  # noqa: E402
from mainscripts import Extractor  # noqa: E402
from mainscripts.Extractor import ExtractSubprocessor  # noqa: E402

CPU_COUNT = 12  # fixed pseudo-core count for the clamping tests


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _settle_to(baseline, timeout=30.0):
    """Bounded wait for process-teardown bookkeeping to settle after a
    run() that already closed/killed its owned workers. This is a wait
    for a GUARANTEED event, not a retry of a failure."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if len(multiprocessing.active_children()) <= baseline:
            return True
        time.sleep(0.05)
    return len(multiprocessing.active_children()) <= baseline


def _gpu_config(gpu_count, mem_gb=8):
    """Fake GPU DeviceConfig (no backend registry, no hardware)."""
    return DeviceConfig([
        Device(i, 'GPU', 'FakeGPU%d' % i, mem_gb * 1024**3, mem_gb * 1024**3 // 2)
        for i in range(gpu_count)
    ])


def _cpu_config():
    return DeviceConfig([])


def _restore_process_priority():
    """Extractor.main() demotes the current process to IDLE priority
    (official behavior); restore NORMAL so the rest of the pytest
    session is not slowed down."""
    if sys.platform[0:3] == 'win':
        from ctypes import windll, wintypes
        GetCurrentProcess = windll.kernel32.GetCurrentProcess
        GetCurrentProcess.restype = wintypes.HANDLE
        SetPriorityClass = windll.kernel32.SetPriorityClass
        SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        SetPriorityClass(GetCurrentProcess(), 0x00000020)  # NORMAL


def _final_data(image_path):
    """'final'-stage data with PRECOMPUTED rect+landmarks: the final
    stage never builds a detector, so this is the cheapest real stage
    to drive the Subprocessor machinery through a full lifecycle.
    filepath is a Path, exactly like the real pipeline feeds
    (pathex.get_image_unique_filestem_paths -> Path); final_stage
    calls filepath.stem / filepath.name."""
    d = ExtractSubprocessor.Data(Path(str(image_path)))
    d.rects = [[506, 100, 676, 348]]
    # deterministic pseudo-landmarks (values are irrelevant for
    # MARK_ONLY: no transform is applied). The real pipeline feeds
    # numpy arrays here (FAN's get_pts_from_predict output) and
    # final_stage calls .tolist() on them in the MARK_ONLY branch.
    import numpy as np
    d.landmarks = [np.array([[100 + 2 * i, 150 + 3 * (i % 7)]
                             for i in range(68)], dtype=np.float32)]
    return d


def _run_cli(args, input_dir, output_dir, stdin=subprocess.DEVNULL, timeout=900):
    """Run the REAL `python main.py extract ...` in a subprocess
    (sys.executable = the venv interpreter under test).
    PYTHONUNBUFFERED makes prompts flush immediately to the pipe."""
    cmd = [sys.executable, str(REPO_ROOT / 'main.py'), 'extract',
           '--detector', 's3fd',
           '--input-dir', str(input_dir),
           '--output-dir', str(output_dir),
           '--face-type', 'full_face',
           '--max-faces-from-image', '0',
           '--image-size', '256',
           '--jpeg-quality', '90',
           '--no-output-debug'] + list(args)
    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdin=stdin,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=timeout)


@pytest.fixture
def extract_input(plain_tmp):
    """Two repo-tracked public images: mini_tutorial (1 face), meme1 (0)."""
    src = REPO_ROOT / 'doc'
    inp = Path(plain_tmp) / 'in'
    out = Path(plain_tmp) / 'out'
    inp.mkdir()
    out.mkdir()
    shutil.copy(src / 'mini_tutorial.jpg', inp / 'mini_tutorial.jpg')
    shutil.copy(src / 'meme1.jpg', inp / 'meme1.jpg')
    return inp, out


# ---------------------------------------------------------------------------
# A. get_devices_for_config — worker-count semantics (pure function)
# ---------------------------------------------------------------------------

class TestGetDevicesForConfig(object):

    def test_single_gpu_single_worker(self):
        got = ExtractSubprocessor.get_devices_for_config('all', _gpu_config(1), 1, 1)
        assert got == [(0, 'GPU', 'FakeGPU0', 8.0)]

    def test_single_gpu_n_workers(self):
        got = ExtractSubprocessor.get_devices_for_config('all', _gpu_config(1), 3, 1)
        assert [d[0] for d in got] == [0, 0, 0]          # device_idx preserved
        assert [d[1] for d in got] == ['GPU'] * 3
        assert [d[2] for d in got] == ['FakeGPU0 #0', 'FakeGPU0 #1', 'FakeGPU0 #2']
        assert [d[3] for d in got] == [8.0] * 3

    def test_two_gpus_n_workers(self):
        got = ExtractSubprocessor.get_devices_for_config('all', _gpu_config(2), 3, 1)
        assert len(got) == 6                            # 2 GPUs x 3 workers
        assert [d[0] for d in got] == [0, 0, 0, 1, 1, 1]
        assert [d[2] for d in got][:3] == ['FakeGPU0 #0', 'FakeGPU0 #1', 'FakeGPU0 #2']
        assert [d[2] for d in got][3:] == ['FakeGPU1 #0', 'FakeGPU1 #1', 'FakeGPU1 #2']

    def test_zero_gpu_count_clamps_to_one(self):
        got = ExtractSubprocessor.get_devices_for_config('all', _gpu_config(1), 0, 1)
        assert got == [(0, 'GPU', 'FakeGPU0', 8.0)]

    def test_negative_gpu_count_clamps_to_one(self):
        got = ExtractSubprocessor.get_devices_for_config('all', _gpu_config(1), -7, 1)
        assert got == [(0, 'GPU', 'FakeGPU0', 8.0)]

    def test_cpu_only_workers_unchanged(self, monkeypatch):
        # the CPU-only branch is NOT affected by the GPU worker count
        monkeypatch.setattr(multiprocessing, 'cpu_count', lambda: CPU_COUNT)
        got = ExtractSubprocessor.get_devices_for_config('all', _cpu_config(), 3, 1)
        assert got == [(i, 'CPU', 'CPU%d' % i, 0) for i in range(min(8, CPU_COUNT // 2))]

    def test_cpu_only_landmarks_manual_single(self, monkeypatch):
        monkeypatch.setattr(multiprocessing, 'cpu_count', lambda: CPU_COUNT)
        got = ExtractSubprocessor.get_devices_for_config('landmarks-manual', _cpu_config(), 3, 1)
        assert got == [(0, 'CPU', 'CPU', 0)]

    def test_landmarks_manual_gpu_multiplier_on_best_device(self):
        # get_best_device = highest total_mem -> FakeGPU1 (24 GB)
        cfg = DeviceConfig([
            Device(0, 'GPU', 'FakeGPU0', 8 * 1024**3, 4 * 1024**3),
            Device(1, 'GPU', 'FakeGPU1', 24 * 1024**3, 20 * 1024**3),
        ])
        got = ExtractSubprocessor.get_devices_for_config('landmarks-manual', cfg, 2, 1)
        assert len(got) == 2
        assert [d[0] for d in got] == [1, 1]
        assert [d[2] for d in got] == ['FakeGPU1 #0', 'FakeGPU1 #1']

    def test_final_explicit_count(self, monkeypatch):
        monkeypatch.setattr(multiprocessing, 'cpu_count', lambda: CPU_COUNT)
        got = ExtractSubprocessor.get_devices_for_config('final', _cpu_config(), 1, 4)
        assert got == [(i, 'CPU', 'CPU%d' % i, 0) for i in range(4)]

    def test_final_count_above_cpu_count_clamps(self, monkeypatch):
        monkeypatch.setattr(multiprocessing, 'cpu_count', lambda: 4)
        got = ExtractSubprocessor.get_devices_for_config('final', _cpu_config(), 1, 32)
        assert len(got) == 4
        assert [d[2] for d in got] == ['CPU0', 'CPU1', 'CPU2', 'CPU3']

    def test_final_zero_clamps_to_one(self, monkeypatch):
        monkeypatch.setattr(multiprocessing, 'cpu_count', lambda: CPU_COUNT)
        got = ExtractSubprocessor.get_devices_for_config('final', _cpu_config(), 1, 0)
        assert got == [(0, 'CPU', 'CPU0', 0)]

    def test_final_negative_clamps_to_one(self, monkeypatch):
        monkeypatch.setattr(multiprocessing, 'cpu_count', lambda: CPU_COUNT)
        got = ExtractSubprocessor.get_devices_for_config('final', _cpu_config(), 1, -3)
        assert got == [(0, 'CPU', 'CPU0', 0)]

    def test_final_signature_defaults_single_worker(self, monkeypatch):
        # __init__ defaults are 1/1; production main() always passes the
        # resolved (prompted or explicit) counts, so the default path
        # yields exactly one CPU0 worker.
        monkeypatch.setattr(multiprocessing, 'cpu_count', lambda: CPU_COUNT)
        got = ExtractSubprocessor.get_devices_for_config('final', _cpu_config())
        assert got == [(0, 'CPU', 'CPU0', 0)]

    def test_final_debug_branch_single_cpu0(self, monkeypatch):
        monkeypatch.setattr(multiprocessing, 'cpu_count', lambda: CPU_COUNT)
        monkeypatch.setattr(Extractor, 'DEBUG', True)
        got = ExtractSubprocessor.get_devices_for_config('final', _cpu_config(), 1, 4)
        assert got == [(0, 'CPU', 'CPU0', 0)]


# ---------------------------------------------------------------------------
# B. CLI plumbing (real `main.py` subprocesses + the argparse boundary)
# ---------------------------------------------------------------------------

class TestCliWorkerCountFlags(object):

    def test_argparse_boundary(self):
        """The extract subparser declares both flags (type int, dest,
        default None) and process_extract forwards both to
        Extractor.main. Pinned from source: the parser lives under
        ``if __name__ == "__main__"`` in main.py and is not importable."""
        tree = ast.parse((REPO_ROOT / 'main.py').read_text(encoding='utf-8'))
        # ast.unparse normalizes string literals to single quotes
        main_body = [n for n in tree.body if isinstance(n, ast.If)
                     and ast.unparse(n.test).replace('"', "'") == "__name__ == '__main__'"]
        assert main_body, "main.py CLI section missing"
        func = main_body[0]

        # collect the add_argument(...) calls
        flags = {}
        for node in ast.walk(func):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'add_argument' and node.args
                    and isinstance(node.args[0], ast.Constant)):
                name = node.args[0].value
                kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
                flags[name] = kwargs
        assert '--gpu-worker-count' in flags, "extract parser missing --gpu-worker-count"
        assert '--final-worker-count' in flags, "extract parser missing --final-worker-count"

        def _const(node):
            return node.value if isinstance(node, ast.Constant) else None

        for name in ('--gpu-worker-count', '--final-worker-count'):
            kw = flags[name]
            assert _const(kw['dest']) == name.strip('-').replace('-', '_')
            assert _const(kw['default']) is None
            assert isinstance(kw['type'], ast.Name) and kw['type'].id == 'int'
        # the flags must NOT exist on other subparsers (extract-only):
        # the names are unique in the whole file's add_argument calls
        all_flags = [n.args[0].value for n in ast.walk(func)
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr == 'add_argument' and n.args
                     and isinstance(n.args[0], ast.Constant)]
        assert all_flags.count('--gpu-worker-count') == 1
        assert all_flags.count('--final-worker-count') == 1

        # process_extract forwards both arguments
        procs = [n for n in ast.walk(func)
                 if isinstance(n, ast.FunctionDef) and n.name == 'process_extract']
        assert len(procs) == 1
        call = [n for n in ast.walk(procs[0])
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == 'main']
        assert call, "process_extract must call Extractor.main"
        kw = {k.arg for k in call[0].keywords if k.arg}
        assert 'gpu_worker_count' in kw and 'final_worker_count' in kw

    def test_both_flags_parse_and_propagate(self, extract_input):
        inp, out = extract_input
        proc = _run_cli(['--cpu-only', '--gpu-worker-count', '1',
                         '--final-worker-count', '1'], inp, out)
        assert proc.returncode == 0, (
            'stdout: %s\nstderr: %s' % (proc.stdout[-4000:], proc.stderr[-4000:]))
        # both explicit values reached main(): neither prompt may fire
        assert 'Parallel GPU workers per device' not in proc.stdout
        assert 'Parallel CPU workers for final stage' not in proc.stdout
        # the pipeline completed (mini_tutorial: 1 face; meme1: 0)
        assert 'Images found:        2' in proc.stdout
        assert 'Faces detected:      1' in proc.stdout
        assert len(list(Path(out).glob('mini_tutorial_*.jpg'))) == 1
        assert len(list(Path(out).glob('meme1_*.jpg'))) == 0

    def test_final_prompt_fires_when_flag_absent(self, extract_input):
        inp, out = extract_input
        # cpu-only: the GPU prompt is suppressed (no --cpu-only prompt
        # regression), but the FINAL prompt must fire for the absent flag
        proc = _run_cli(['--cpu-only', '--gpu-worker-count', '1'], inp, out)
        assert proc.returncode == 0, (
            'stdout: %s\nstderr: %s' % (proc.stdout[-4000:], proc.stderr[-4000:]))
        assert 'Parallel CPU workers for final stage' in proc.stdout
        assert 'Parallel GPU workers per device' not in proc.stdout
        # EOF on the prompt -> default value, the run still completes
        assert 'Faces detected:      1' in proc.stdout

    def test_gpu_prompt_fires_when_flag_absent(self, extract_input):
        """Non-cpu-only run without --gpu-worker-count: the GPU-worker
        prompt fires. The child's stdin is an OPEN pipe that is never
        written to, so input() blocks and the child can never reach the
        pipeline -> terminating it can not leak any worker process.

        NOTE: the prompt is printed by builtin input() WITHOUT a trailing
        newline, so line-based reads (readline/iter) block until more
        output arrives; the child stdout must be consumed byte-wise.
        The NN_DEVICE* env vars are stripped so the child performs a
        fresh device enumeration exactly like a real CLI launch."""
        inp, out = extract_input
        cmd = [sys.executable, str(REPO_ROOT / 'main.py'), 'extract',
               '--detector', 's3fd',
               '--input-dir', str(inp), '--output-dir', str(out),
               '--face-type', 'full_face', '--max-faces-from-image', '0',
               '--image-size', '256', '--jpeg-quality', '90',
               '--no-output-debug',
               '--force-gpu-idxs', '0',           # no device-selection prompt
               '--final-worker-count', '1',       # no final prompt
               ]
        env = {k: v for k, v in os.environ.items()
               if not (k.startswith('NN_DEVICE_') or k in ('NN_DEVICES_INITIALIZED',
                                                           'NN_DEVICES_COUNT'))}
        env['PYTHONUNBUFFERED'] = '1'
        baseline = len(multiprocessing.active_children())
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)

        # daemon reader: delivers prompt bytes even without a trailing \n
        out_fd = proc.stdout.fileno()
        collected = []

        def _reader():
            while True:
                try:
                    chunk = os.read(out_fd, 256)
                except OSError:
                    break
                if not chunk:
                    break
                collected.append(chunk)

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        prompt_seen = False
        child_dead_before_prompt = False
        deadline = time.time() + 240
        try:
            while time.time() < deadline:
                buf = b''.join(collected).decode('utf-8', errors='replace')
                if 'Parallel GPU workers per device' in buf:
                    prompt_seen = True
                    break
                if proc.poll() is not None:
                    child_dead_before_prompt = True
                    break
                time.sleep(0.2)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    stream.close()
                except Exception:
                    pass
            reader.join(timeout=5)
        buf = b''.join(collected).decode('utf-8', errors='replace')
        assert not child_dead_before_prompt, \
            'child exited before the GPU prompt. output: %s' % buf[-4000:]
        assert prompt_seen, 'GPU-worker prompt never printed. output: %s' % buf[-4000:]
        # the final-stage prompt comes AFTER the GPU prompt: while the
        # child is blocked on the GPU prompt it can not have started it
        assert 'Parallel CPU workers for final stage' not in buf
        # the blocked child left no owned process behind
        assert _settle_to(baseline)
        assert len(multiprocessing.active_children()) == baseline


# ---------------------------------------------------------------------------
# C. Pipeline integration (torch S3FD/FAN inside real worker processes)
# ---------------------------------------------------------------------------

class TestPipelineIntegration(object):

    def test_worker_device_contract_cpu(self, monkeypatch):
        """process_info_generator must hand every worker its
        device_idx / device_type / device_name (the contract the
        worker's Cli.on_initialize consumes to build the extractors)."""
        monkeypatch.setattr(sys, 'stdin', open(os.devnull))
        sub = ExtractSubprocessor(
            [ExtractSubprocessor.Data(str(REPO_ROOT / 'doc' / 'mini_tutorial.jpg'))],
            'final', 256, 90, FaceType.MARK_ONLY,
            device_config=_cpu_config(),
            gpu_worker_count=1, final_worker_count=2)
        # USER_LEGACY final-stage contract: [(i, 'CPU', 'CPU{i}', 0) ...]
        # — device_idx is the worker ordinal, not 0.
        assert sub.devices == [(0, 'CPU', 'CPU0', 0), (1, 'CPU', 'CPU1', 0)]
        infos = list(sub.process_info_generator())
        assert len(infos) == 2
        for idx, (name, _host, client) in enumerate(infos):
            assert client['device_type'] == 'CPU'
            assert client['device_idx'] == idx
            assert client['type'] == 'final'
            assert client['image_size'] == 256
            assert client['jpeg_quality'] == 90
            assert client['face_type'] == FaceType.MARK_ONLY
            assert name == client['device_name']

    def test_worker_device_contract_fake_gpu(self, monkeypatch):
        """Each worker keeps the device_idx of the GPU it was spawned
        for (2 fake GPUs x 3 workers -> 6 workers, idx 0,0,0,1,1,1)."""
        monkeypatch.setattr(sys, 'stdin', open(os.devnull))
        sub = ExtractSubprocessor(
            [ExtractSubprocessor.Data(str(REPO_ROOT / 'doc' / 'mini_tutorial.jpg'))],
            'all', 256, 90, FaceType.FULL,
            device_config=_gpu_config(2),
            gpu_worker_count=3, final_worker_count=1)
        assert [d[0] for d in sub.devices] == [0, 0, 0, 1, 1, 1]
        infos = list(sub.process_info_generator())
        assert len(infos) == 6
        assert [c['device_idx'] for _n, _h, c in infos] == [0, 0, 0, 1, 1, 1]
        assert [c['device_type'] for _n, _h, c in infos] == ['GPU'] * 6
        assert [c['device_name'] for _n, _h, c in infos] == [
            'FakeGPU0 #0', 'FakeGPU0 #1', 'FakeGPU0 #2',
            'FakeGPU1 #0', 'FakeGPU1 #1', 'FakeGPU1 #2']

    def test_no_tensorflow_in_pipeline_boundary(self):
        """No TensorFlow runtime may remain in the extraction pipeline
        module boundary (mainscripts.Extractor pulls in facelib, which
        pulls in the torch S3FD/FAN extractors)."""
        for name in list(sys.modules):
            assert not name == 'tensorflow' and not name.startswith('tensorflow.'), \
                'tensorflow is loaded in the pipeline import boundary: %s' % name
        for rel in ('mainscripts/Extractor.py',
                    'facelib/S3FDExtractor.py',
                    'facelib/FANExtractor.py'):
            # 'utf-8-sig': mainscripts/Extractor.py carries the
            # repo's original UTF-8 BOM; ast.parse rejects U+FEFF.
            tree = ast.parse((REPO_ROOT / rel).read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        assert not a.name.startswith('tensorflow'), (rel, a.name)
                elif isinstance(node, ast.ImportFrom):
                    assert not (node.module or '').startswith('tensorflow'), rel

    def test_pipeline_e2e_all_stage_cpu_inprocess(self, plain_tmp, monkeypatch, capsys):
        """Real in-process Extractor.main() on CPU: the 'all' stage
        (torch S3FD + torch FAN in every worker subprocess) and the
        final stage (explicit final_worker_count=2) run to completion;
        no prompt fires; zero owned processes remain.

        This is also the normal-completion lifecycle proof for the
        heavy model-loading stage (the 'final'-stage lifecycle tests
        below cover the dedicated repeated/error paths)."""
        src = REPO_ROOT / 'doc'
        inp = Path(plain_tmp) / 'in'
        out = Path(plain_tmp) / 'out'
        inp.mkdir()
        out.mkdir()
        shutil.copy(src / 'mini_tutorial.jpg', inp / 'mini_tutorial.jpg')
        shutil.copy(src / 'meme1.jpg', inp / 'meme1.jpg')

        # a fired prompt would read EOF (no hang) and its text would be
        # captured below, failing the no-prompt assertions
        monkeypatch.setattr(sys, 'stdin', open(os.devnull))

        baseline = len(multiprocessing.active_children())
        try:
            Extractor.main(detector='s3fd',
                           input_path=inp,
                           output_path=out,
                           output_debug=False,
                           face_type='full_face',
                           max_faces_from_image=0,
                           image_size=256,
                           jpeg_quality=90,
                           cpu_only=True,
                           gpu_worker_count=1,
                           final_worker_count=2)
        finally:
            _restore_process_priority()

        captured = capsys.readouterr()
        assert 'Parallel GPU workers per device' not in captured.out
        assert 'Parallel CPU workers for final stage' not in captured.out
        assert 'Images found:        2' in captured.out
        assert 'Faces detected:      1' in captured.out

        # the detected face produced a DFLJPG with 68x2 landmarks
        files = list(Path(out).glob('mini_tutorial_*.jpg'))
        assert len(files) == 1
        from DFLIMG import DFLJPG
        dfl = DFLJPG.load(str(files[0]))
        assert dfl.has_data()
        assert dfl.get_face_type() == 'full_face'
        lm = dfl.get_landmarks()
        assert lm.shape == (68, 2)
        assert dfl.get_source_filename() == 'mini_tutorial.jpg'
        # meme1 (no face) produced no output
        assert len(list(Path(out).glob('meme1_*.jpg'))) == 0

        # zero owned subprocesses remain (all workers finalized+joined)
        assert _settle_to(baseline)
        assert len(multiprocessing.active_children()) == baseline


# ---------------------------------------------------------------------------
# D. Worker lifecycle (Subprocessor machinery, owned-Process tracking)
# ---------------------------------------------------------------------------

class TestWorkerLifecycle(object):

    def test_normal_completion_zero_owned(self, plain_tmp, monkeypatch):
        """A stage run exits with zero owned subprocesses remaining:
        every owned worker was gracefully closed (close -> finalized
        handshake) and its Process joined."""
        monkeypatch.setattr(sys, 'stdin', open(os.devnull))
        out = Path(plain_tmp) / 'final_out'
        out.mkdir()
        data = [_final_data(REPO_ROOT / 'doc' / 'mini_tutorial.jpg') for _ in range(2)]
        baseline = len(multiprocessing.active_children())
        sub = ExtractSubprocessor(data, 'final', 256, 90, FaceType.MARK_ONLY,
                                  final_output_path=out,
                                  device_config=_cpu_config(),
                                  gpu_worker_count=1, final_worker_count=2)
        result = sub.run()
        assert len(result) == 2
        assert all(d.faces_detected == 1 for d in result)
        # the deterministic close contract: every cli finalized (state 2)
        # and its owned Process is dead AND reaped (exitcode set)
        assert all(cli.state == 2 for cli in sub.clis)
        assert all(cli.p is not None for cli in sub.clis)
        assert all(not cli.p.is_alive() for cli in sub.clis)
        assert all(cli.p.exitcode is not None for cli in sub.clis)
        assert _settle_to(baseline)
        assert len(multiprocessing.active_children()) == baseline

    def test_repeated_runs_do_not_accumulate(self, plain_tmp, monkeypatch):
        """Repeated construct/run/finalize cycles must not accumulate
        owned workers (no generation leaks)."""
        monkeypatch.setattr(sys, 'stdin', open(os.devnull))
        out = Path(plain_tmp) / 'final_out'
        out.mkdir()
        baseline = len(multiprocessing.active_children())
        for i in range(2):
            data = [_final_data(REPO_ROOT / 'doc' / 'mini_tutorial.jpg') for _ in range(2)]
            sub = ExtractSubprocessor(data, 'final', 256, 90, FaceType.MARK_ONLY,
                                      final_output_path=out,
                                      device_config=_cpu_config(),
                                      gpu_worker_count=1, final_worker_count=2)
            sub.run()
            assert _settle_to(baseline)
            assert len(multiprocessing.active_children()) == baseline, \
                'run %d leaked owned workers' % (i + 1)

    def test_error_path_worker_reaped(self, plain_tmp, monkeypatch, capsys):
        """A worker that raises during stage execution is killed+joined
        by the host, the failed data is dropped, and run() returns
        normally with zero owned processes remaining. Verified failure
        chain in the child (cv2 5.x does NOT raise on
        jpeg_quality=None): the deliberately missing final_output_path
        dir -> cv2_imwrite's open(..., 'wb') swallowed by its bare
        except -> DFLJPG.load returns None -> AttributeError at
        dflimg.set_face_type -> the child sends the 'error' op -> the
        host logs 'Error while processing data' and kill+joins the
        worker."""
        monkeypatch.setattr(sys, 'stdin', open(os.devnull))
        out = Path(plain_tmp) / 'final_out_err'
        # deliberately NOT created: the child fails before any file write
        data = [_final_data(REPO_ROOT / 'doc' / 'mini_tutorial.jpg')]
        baseline = len(multiprocessing.active_children())
        sub = ExtractSubprocessor(data, 'final', 256, None, FaceType.MARK_ONLY,
                                  final_output_path=out,
                                  device_config=_cpu_config(),
                                  gpu_worker_count=1, final_worker_count=1)
        result = sub.run()
        captured = capsys.readouterr()
        assert result == []            # the failed data was dropped
        assert 'Error while processing data' in captured.out
        # the single owned worker was killed+joined, no orphans remain
        assert _settle_to(baseline)
        assert len(multiprocessing.active_children()) == baseline
        assert not out.exists() or not list(out.glob('*.jpg'))
