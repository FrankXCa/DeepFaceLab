#!/usr/bin/env python3
"""Failure-safe env probe worker for the P13 ``dfl envreport`` command.

This file is the CHILD (worker) process of ``scripts/envreport.py``.
The parent starts it with ``python -I -B`` and a very large wall-clock
deadline (120 s) as a hard backstop, but the normal path never gets
close to that: every risky operation below is either a pure
introspection of the already-running interpreter or a bounded child
command with its own small timeout, so the parent's deadline exists
only to catch an environment so broken that even ``import torch`` or
the child commands hang.

Protocol (one JSON line on stdout, one worker per probe name)
-------------------------------------------------------------
The parent invokes this worker once per probe, passing the probe name
as the single command-line argument, and consumes the emitted line for
that probe only:

    {"name": "<probe>", "status": "<status>", "note": "<short
     non-sensitive note>", "fields": {"<field>": ...}}

The line is the requested probe's own result record (``_result``), so
the status, note and fields on the line are exactly what the parent
reports for that probe.  A name outside the handler table is an
internal contract violation: it is reported as a stable UNAVAILABLE
with the fixed, non-sensitive note "unknown probe" (the token is
never echoed) and the worker still exits 0, so the parent's parse
path stays well-formed.  Worker-internal errors likewise downgrade
gracefully: every handler converts its failure modes into a stable
status, and the last-resort crash guard below emits a well-formed
UNAVAILABLE line instead of a bare traceback; the parent treats any
other deviation (no line, malformed JSON) as an UNAVAILABLE probe
result.

Nested external-tool capture
----------------------------
``probe_nvidia_smi`` and ``probe_system_tool`` invoke system
executables (``nvidia-smi``, ``ffmpeg``, ``ffprobe``) through
``_run_capped``.  That runner is memory-bounded with the same design
the parent uses for the worker itself: a daemon reader thread STORES
at most ``_CHILD_OUT_LIMIT`` bytes of tool stdout (kept equal to the
parent's ``_PROBE_OUT_LIMIT``) and keeps draining -- discarding the
excess -- until EOF.  No shell is involved (``shell=False`` equivalent
via CPython's C ``CreateProcess``); on Windows the tool's stderr goes to the NUL
privacy sink; a bounded timeout always applies.  A hostile or verbose
tool therefore can neither pin unbounded worker memory nor deadlock
the pipe.

Containment is per-process-tree and is established BEFORE the tool
executes its first instruction.  On Windows the runner drives the raw
Win32 API through ``ctypes`` (stdlib only -- no new dependencies): a
per-invocation job object is created, ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``
is configured and re-verified with a Set/Query round trip (a single
typed ABI spelled out below -- the public SDK field spellings in the
running kernel's 144-byte extended layout; a job whose round trip does
not verify is never trusted and the runner fails closed), the
tool process is created SUSPENDED via CPython's C ``CreateProcess``
(``subprocess._winapi``) with its stdout wired to a pipe through an
EXTENDED ``STARTUPINFO`` object (marshaled by the stdlib C code as
``EXTENDED_STARTUPINFO_PRESENT`` plus a
``PROC_THREAD_ATTR_HANDLE_LIST`` naming exactly the two inheritable
handles -- the pipe write end and the NUL sink; this is the same form
stdlib ``Popen`` itself uses, so the std-handle wiring behaves
identically to a Popen-launched child.  The child's startup flags
carry the real ``STARTF_USESTDHANDLES`` value 0x100 -- 0x1 is
``STARTF_USESHOWWINDOW``, and a second-round draft that set the wrong
flag is the recorded root cause of an earlier unwired pipe, see
``docs/PHASE13_STATE.md``), the parent's copy of the pipe write end
is closed before the child runs so EOF is reachable, and stderr goes
to the NUL sink, the child is assigned to the
job with the assignment result checked (plus a documented
``JobObjectProcessIdList`` membership cross-check: the child's PID
must appear in the job's process list), and only then is the thread
resumed.  A
suspended child cannot spawn or write before assignment, so there is
no pre-assignment race: the tool and every descendant it later spawns
are members of the job for their entire life.  Any failure of job
creation, configuration/verification, child creation, or assignment
is FAIL-CLOSED: the suspended child is terminated without ever being
resumed, every handle is closed, and the probe reports a stable,
non-sensitive ``UNAVAILABLE`` note -- the tool is simply never run
(no direct-execution fallback, no third state).  On a timeout -- or
when the direct child exited but a descendant that inherited the
stdout pipe outlives it and keeps the pipe open -- the WHOLE job tree
is terminated with ``TerminateJobObject``; closing the job handle
afterwards applies the verified kill-on-close to any remaining member
on every path, so no member of a job this runner created can outlive
the worker.  Every cleanup step is budgeted (at most one reap window
and two reader-settle windows: ``_REAP_BUDGET`` plus two
``_SETTLE_BUDGET`` windows, pinned together as ``_CLEANUP_BUDGET_MAX``
= 9 s), so the runner returns within ``timeout + 9 s + epsilon`` and
never waits indefinitely for an inherited pipe to close; the parent's
read end is closed only after the reader thread has finished.  On POSIX the legacy direct-child
termination is kept (job objects are a Windows mechanism).  The probe
reports ``TIMEOUT`` for any stalled capture; a tool whose output
exceeds the capture limit reports ``UNAVAILABLE`` with a fixed note.
Raw output is discarded and never surfaced; only structured,
whitelisted facts survive into the protocol line.

The third-party imports themselves (``torch``, ``onnx``,
``onnxruntime``) happen INSIDE the per-probe handlers, so a broken or
missing package can report its failure without taking down the
worker.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import subprocess
import struct
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Stable status vocabulary (must stay in sync with scripts/envreport.py)
# ---------------------------------------------------------------------------

STATUS_OK = "OK"
STATUS_MISSING = "MISSING"
STATUS_IMPORT_ERROR = "IMPORT_ERROR"
STATUS_INIT_ERROR = "INIT_ERROR"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_MISMATCH = "MISMATCH"

# Bounded timeout for the short-lived child subprocesses (nvidia-smi and
# ffmpeg/ffprobe). The parent enforces its own, larger cap on the whole
# worker; this internal bound lets the worker report TIMEOUT on its own
# before the parent kills it, keeping the captured result deterministic.
_CHILD_CMD_TIMEOUT = 15.0
# Hard cap on bytes of nested-tool stdout we are willing to STORE.  It is
# enforced while draining (the reader stops storing at this limit and
# discards the rest), so worker memory never scales with tool output.
# Keep equal to scripts/envreport._PROBE_OUT_LIMIT: one 65536-byte cap
# across the whole envreport feature.
_CHILD_OUT_LIMIT = 65536

# Bounded cleanup budgets for the nested-tool runner.  After a timeout (or
# a stalled capture) every cleanup step is capped: at most one reap
# window (``_REAP_BUDGET``) plus one settle window per drained pipe
# (``_SETTLE_BUDGET``; two pipes on Windows: stdout and stderr, one on
# POSIX), so ``_run_capped`` always returns within ``timeout +
# _CLEANUP_BUDGET_MAX + epsilon`` (timeout + 9 s + epsilon on the worst
# Windows path) and never waits indefinitely for an inherited pipe to
# close.  The derived budget is pinned by tests; do not edit these
# constants without updating that record and the bound in the
# ``_run_capped`` docstring.
_REAP_BUDGET = 5.0
_SETTLE_BUDGET = 2.0
# Worst-case cleanup budget derived from the constants above (pinned by
# tests and referenced by the ``_run_capped`` return-bound contract).
_CLEANUP_BUDGET_MAX = _REAP_BUDGET + 2 * _SETTLE_BUDGET  # 9.0 s

# --- Windows process-tree containment (ctypes, stdlib only) ---------------
# Containment is established BEFORE the tool executes its first
# instruction, and it is fail-closed: the tool process is created
# SUSPENDED, attached to a job object whose KILL_ON_CLOSE flag was
# configured AND re-verified, the assignment result is checked, and only
# then is the thread resumed.  A suspended child cannot spawn or write
# before assignment, so there is no pre-assignment race and no
# orphanable window.  Any setup failure terminates the suspended child
# without resuming it and reports a stable UNAVAILABLE status; the tool
# is simply never run (no direct-execution fallback, no third state).
# All Win32 symbols are resolved lazily so importing this module stays
# failure-safe.
_JOB_EXTENDED_LIMIT = 9  # JobObjectExtendedLimitInformation
# JobObjectProcessIdList: the documented class for the post-assignment
# membership cross-check.  Its response is a JOBOBJECT_BASIC_PROCESS_ID_LIST
# -- DWORD NumberOfAssignedProcesses, DWORD NumberOfProcessIdsInList, then
# one ULONG_PTR PID per listed process -- and the check requires the
# child's PID to be among the listed PIDs (a count check would be wrong:
# modern kernels list bookkeeping entries alongside the real PIDs).
# (Class 7 is JobObjectAssociateCompletionPortInformation: completion-
# port association data, NOT a process list, and is not used.)
_JOB_PROCESS_ID_LIST = 3  # JobObjectProcessIdList
_JOB_KILL_ON_CLOSE = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
_JOB_EXIT_CODE = 0xFFFE  # exit status TerminateJobObject records
# One bounded class-3 (JobObjectProcessIdList) query buffer: 64 KiB holds
# up to 8191 pointer-width PIDs -- far more than a single probe child
# plus its descendants can occupy.  There is deliberately no resize/retry
# model: a response that does not fit the fixed buffer is a failed
# verification and the runner fails closed.
_PID_LIST_QUERY_SIZE = 65536
# winbase.h: value GetExitCodeProcess reports while a process still runs
# (NOT 0xFFFFFFFF -- that constant does not exist in the Windows API).
_STILL_ACTIVE = 259
_CREATE_SUSPENDED = 0x00000004
# winbase.h: STARTF_USESTDHANDLES -- the real value is 0x00000100 (the
# value 0x00000001 is STARTF_USESHOWWINDOW).  A second-round draft set
# 0x1, so the kernel correctly left the child's std handles untouched
# and the child's stdout silently fell back to the console (unwired
# pipe); that flag mistake was the true root cause of that failure,
# not a kernel quirk.  Matches subprocess._winapi.STARTF_USESTDHANDLES.
_STARTF_USESTDHANDLES = 0x00000100
_HANDLE_FLAG_INHERIT = 0x0001
_WAIT_OBJECT_0 = 0x00000000
_WAIT_TIMEOUT = 0x00000102
# --- The typed job-object ABI ----------------------------------------------
# The x64 JOBOBJECT_*_LIMIT_INFORMATION layout this runner drives,
# declared as the documented public SDK structures (Windows SDK
# 10.0.26100.0 ``winnt.h``) at native C alignment:
#
#   IO_COUNTERS: the six documented ULONGLONG fields = 48 bytes;
#   JOBOBJECT_BASIC_LIMIT_INFORMATION: the nine documented fields --
#     LARGE_INTEGER PerProcess/PerJobUserTimeLimit, DWORD LimitFlags
#     (offset 16), SIZE_T Minimum/MaximumWorkingSetSize, DWORD
#     ActiveProcessLimit, ULONG_PTR Affinity, DWORD PriorityClass and
#     SchedulingClass = 64 bytes;
#   JOBOBJECT_EXTENDED_LIMIT_INFORMATION: basic (64) + IO_COUNTERS
#     (48) + four SIZE_T limit/peak fields (32) = 144 bytes.
#
# On x64 the native alignment of the nine documented basic fields is
# exactly 64 bytes: the 4-byte gaps at offsets 20..23 and 44..47 are
# ordinary C alignment padding between documented fields, and the
# values the kernel reports at offsets 56/60 on Query are the
# documented PriorityClass (0x20 on this kernel) and SchedulingClass
# (0x5 on this kernel) fields -- not an opaque tail.  The running
# kernel (Windows build 26200, measured live) accepts exactly the
# 144-byte extended spelling for JobObjectExtendedLimitInformation
# (class 9); shorter or longer buffers (128, 136, 152, 184 and 352
# bytes were all measured) are rejected with ERROR_MORE_DATA
# (winerror 24).  The runner writes only ``LimitFlags`` and trusts a
# job only after the Set -> Query round trip in
# ``_win_job_configure_kill_on_close`` proves the KILL_ON_CLOSE flag
# on the running kernel.  Full measurement history:
# docs/PHASE13_STATE.md.


class _IO_COUNTERS(ctypes.Structure):
    """x64 JOBOBJECT_IO_COUNTERS: the six documented ``ULONGLONG``
    fields of the public SDK spelling (Windows SDK 10.0.26100.0
    ``winnt.h``) at native alignment, 48 bytes.  The struct is trusted
    on a job only after the Set -> Query round trip proves the limit
    flag on the running kernel (see
    ``_win_job_configure_kill_on_close``)."""

    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    """x64 JOBOBJECT_BASIC_LIMIT_INFORMATION: the nine documented
    fields of the public SDK spelling (Windows SDK 10.0.26100.0
    ``winnt.h``) at native C alignment, 64 bytes:

    ===========================  ===============  ======
    field                        type             offset
    ===========================  ===============  ======
    PerProcessUserTimeLimit      LARGE_INTEGER    0
    PerJobUserTimeLimit          LARGE_INTEGER    8
    LimitFlags                   DWORD            16
    MinimumWorkingSetSize        SIZE_T           24
    MaximumWorkingSetSize        SIZE_T           32
    ActiveProcessLimit           DWORD            40
    Affinity                     ULONG_PTR        48
    PriorityClass                DWORD            56
    SchedulingClass              DWORD            60
    ===========================  ===============  ======

    The 4-byte gaps at offsets 20..23 and 44..47 are ordinary C
    alignment padding between documented fields; there is no opaque
    tail and no kernel-internal region.  This runner writes and reads
    only ``LimitFlags`` and trusts the job only after the Set -> Query
    round trip in ``_win_job_configure_kill_on_close`` proves the
    KILL_ON_CLOSE bit back from the running kernel.
    """

    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),  # LARGE_INTEGER
        ("PerJobUserTimeLimit", ctypes.c_longlong),      # LARGE_INTEGER
        ("LimitFlags", ctypes.c_ulong),                  # DWORD
        ("MinimumWorkingSetSize", ctypes.c_size_t),      # SIZE_T
        ("MaximumWorkingSetSize", ctypes.c_size_t),      # SIZE_T
        ("ActiveProcessLimit", ctypes.c_ulong),          # DWORD
        ("Affinity", ctypes.c_size_t),                   # ULONG_PTR
        ("PriorityClass", ctypes.c_ulong),               # DWORD
        ("SchedulingClass", ctypes.c_ulong),             # DWORD
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    """x64 JOBOBJECT_EXTENDED_LIMIT_INFORMATION: the documented
    composition -- JOBOBJECT_BASIC_LIMIT_INFORMATION (64 bytes),
    IO_COUNTERS (48) and the four documented ``SIZE_T`` memory
    limit/peak fields (32) = 144 bytes, exactly the length the running
    kernel accepts for JobObjectExtendedLimitInformation (class 9); the
    Set -> Query round trip on every job re-proves the configuration
    before use."""

    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


# The suspended child is created through CPython's C
# ``subprocess._winapi.CreateProcess`` with a ``subprocess.STARTUPINFO``
# object: the stdlib C code marshals it as an EXTENDED startup info
# (``EXTENDED_STARTUPINFO_PRESENT`` plus a
# ``PROC_THREAD_ATTR_HANDLE_LIST`` naming exactly the handles the
# child may inherit), the same form stdlib ``Popen`` itself uses, so
# the std-handle wiring behaves identically to a Popen-launched child.
# The child's ``dwFlags`` must carry the real ``STARTF_USESTDHANDLES``
# value (0x00000100, matching ``subprocess._winapi.STARTF_USESTDHANDLES``);
# 0x00000001 is ``STARTF_USESHOWWINDOW``.  A second-round draft set 0x1
# instead, so the kernel legitimately left the child's std handles
# untouched and the child's stdout fell back to the console (unwired
# pipe) -- the true root cause of that failure, NOT a kernel quirk.
# Both handles named in the list must carry the inherit flag (the
# kernel rejects a launch whose handle list contains a non-inheritable
# handle), and both are marked inheritable before the call.  The C
# call returns the tuple ``(hProcess, hThread, pid, tid)`` of Python
# ints.


class _WinChild:
    """A suspended tool process plus the parent's read ends of its pipes.

    ``h_thread`` is zeroed once resumed; ``file`` is the (binary) Python
    stream wrapping the read end of the stdout pipe and ``err_file``
    wraps the read end of the stderr pipe (whose content the runner
    drains and discards; the tool probe may consult its first line ONLY
    as the tool's own version banner).  All fields are owned by the
    runner and released in its ``finally`` block.
    """

    __slots__ = ("h_process", "h_thread", "pid", "file", "err_file")

    def __init__(self) -> None:
        self.h_process = 0
        self.h_thread = 0
        self.pid = 0
        self.file = None
        self.err_file = None


class _ContainmentSetupError(RuntimeError):
    """Internal: process-tree containment could not be established.

    Raised by the Windows runner when job creation, kill-on-close
    configuration/verification, child creation, or job assignment
    fails.  At that point the tool has NOT run (it was never resumed),
    so the probe handlers map this to a stable, non-sensitive
    UNAVAILABLE note instead of ever executing the tool outside
    containment.  The message is internal only and must never reach
    product output.
    """


_WIN_API = None


def _win_api():
    """Lazily bind the Win32 prototypes (Windows only, stdlib ctypes)."""
    global _WIN_API
    if _WIN_API is not None:
        return _WIN_API
    import ctypes.wintypes as wt  # noqa: PLC0415 - Windows-only import

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wt.HANDLE
    k32.CreateJobObjectW.argtypes = [wt.LPVOID, wt.LPCWSTR]
    k32.SetInformationJobObject.restype = wt.BOOL
    k32.SetInformationJobObject.argtypes = [
        wt.HANDLE,
        wt.DWORD,
        ctypes.POINTER(ctypes.c_uint8),
        wt.DWORD,
    ]
    k32.QueryInformationJobObject.restype = wt.BOOL
    k32.QueryInformationJobObject.argtypes = [
        wt.HANDLE,
        wt.DWORD,
        ctypes.POINTER(ctypes.c_uint8),
        wt.DWORD,
        ctypes.POINTER(wt.DWORD),
    ]
    k32.AssignProcessToJobObject.restype = wt.BOOL
    k32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
    k32.TerminateJobObject.restype = wt.BOOL
    k32.TerminateJobObject.argtypes = [wt.HANDLE, wt.DWORD]
    k32.ResumeThread.restype = wt.DWORD
    k32.ResumeThread.argtypes = [wt.HANDLE]
    k32.TerminateProcess.restype = wt.BOOL
    k32.TerminateProcess.argtypes = [wt.HANDLE, wt.DWORD]
    k32.WaitForSingleObject.restype = wt.DWORD
    k32.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
    k32.GetExitCodeProcess.restype = wt.BOOL
    k32.GetExitCodeProcess.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD)]
    k32.CreatePipe.restype = wt.BOOL
    k32.CreatePipe.argtypes = [
        ctypes.POINTER(wt.HANDLE),
        ctypes.POINTER(wt.HANDLE),
        wt.LPVOID,
        wt.DWORD,
    ]
    k32.SetHandleInformation.restype = wt.BOOL
    k32.SetHandleInformation.argtypes = [wt.HANDLE, wt.DWORD, wt.DWORD]
    k32.CloseHandle.restype = wt.BOOL
    k32.CloseHandle.argtypes = [wt.HANDLE]
    _WIN_API = k32
    return k32


def _win_job_create(api):
    """Create the per-invocation job object (0 on failure, never raises)."""
    job = api.CreateJobObjectW(None, None)
    return job or 0


def _win_job_configure_kill_on_close(api, job) -> bool:
    """Set and VERIFY ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` on ``job``.

    A single typed ABI (``_JOBOBJECT_EXTENDED_LIMIT_INFORMATION``:
    144 bytes on x64, ``LimitFlags`` at offset 16 -- see the ABI note
    above) is written with ``SetInformationJobObject`` and read back
    with ``QueryInformationJobObject``; the KILL_ON_CLOSE bit must
    survive the round trip on the running kernel.  An unverified job is
    never used: if the set or the query fails, or the bit does not
    come back, this returns False and the runner fails closed.  No
    raw buffer, offset guess, or alternative layout is ever
    consulted.
    """
    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = _JOB_KILL_ON_CLOSE
    info_ptr = ctypes.cast(
        ctypes.byref(info), ctypes.POINTER(ctypes.c_uint8)
    )
    if not api.SetInformationJobObject(
        job, _JOB_EXTENDED_LIMIT, info_ptr, ctypes.sizeof(info)
    ):
        return False
    back = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    back_ptr = ctypes.cast(
        ctypes.byref(back), ctypes.POINTER(ctypes.c_uint8)
    )
    written = ctypes.c_ulong(0)
    if not api.QueryInformationJobObject(
        job, _JOB_EXTENDED_LIMIT, back_ptr, ctypes.sizeof(back),
        ctypes.byref(written)
    ):
        return False
    return bool(back.BasicLimitInformation.LimitFlags & _JOB_KILL_ON_CLOSE)


def _win_spawn_suspended(api, cmd):
    """Create the tool SUSPENDED: stdout and stderr piped.

    Returns a ``_WinChild``, or ``None`` when any Win32 step fails (the
    tool cannot run in that case; a created but unresumed child is
    terminated before returning, so nothing is left behind).  The
    parent's copy of both pipe write ends is closed before returning,
    so the readers reach EOF as soon as every job member exits.

    No file handle is created at all: both pipes are the ONLY handles
    the child may inherit (they are named in the extended STARTUPINFO
    handle list), so the NUL sink of the earlier draft -- which needed
    a CreateFileW + SetHandleInformation pair and was the source of the
    round-2 winerror-87 launches -- is gone.  The stderr pipe exists
    because this machine's static gyan.dev ffmpeg/ffprobe build writes
    the ``--version`` banner to stderr (its av_log channel) and exits
    with a build-specific code (0xABAFB008 = 2880417800 for ffmpeg;
    1 for ffprobe) whenever its std handles are explicitly wired; the
    runner drains the pipe and
    discards the content, and the tool probe may consult the first
    line ONLY as the tool's own version banner (no other stderr byte
    ever reaches the protocol line).
    """
    import msvcrt  # noqa: PLC0415 - Windows-only import

    read_h = ctypes.c_void_p()
    write_h = ctypes.c_void_p()
    if not api.CreatePipe(ctypes.byref(read_h), ctypes.byref(write_h), None, 0):
        return None
    if not api.SetHandleInformation(read_h, _HANDLE_FLAG_INHERIT, 0):
        api.CloseHandle(write_h)
        api.CloseHandle(read_h)
        return None
    if not api.SetHandleInformation(write_h, _HANDLE_FLAG_INHERIT, _HANDLE_FLAG_INHERIT):
        api.CloseHandle(write_h)
        api.CloseHandle(read_h)
        return None
    err_read_h = ctypes.c_void_p()
    err_write_h = ctypes.c_void_p()
    if not api.CreatePipe(
        ctypes.byref(err_read_h), ctypes.byref(err_write_h), None, 0
    ):
        api.CloseHandle(write_h)
        api.CloseHandle(read_h)
        return None
    if not api.SetHandleInformation(err_read_h, _HANDLE_FLAG_INHERIT, 0):
        api.CloseHandle(err_write_h)
        api.CloseHandle(err_read_h)
        api.CloseHandle(write_h)
        api.CloseHandle(read_h)
        return None
    if not api.SetHandleInformation(
        err_write_h, _HANDLE_FLAG_INHERIT, _HANDLE_FLAG_INHERIT
    ):
        api.CloseHandle(err_write_h)
        api.CloseHandle(err_read_h)
        api.CloseHandle(write_h)
        api.CloseHandle(read_h)
        return None
    # Extended STARTUPINFO via stdlib C marshaling (see the module
    # note above): the attribute list names exactly the two handles
    # the child may inherit -- the two pipe write ends -- so no other
    # parent handle can leak into the job tree.  Both are marked
    # inheritable above: the kernel rejects a launch whose handle list
    # contains a non-inheritable handle, so both write ends must carry
    # the flag (the pipe ends are inheritable by default, except the
    # read ends which were explicitly cleared).
    si = subprocess.STARTUPINFO()
    si.dwFlags = _STARTF_USESTDHANDLES
    si.hStdInput = None
    si.hStdOutput = write_h.value
    si.hStdError = err_write_h.value
    si.lpAttributeList = {"handle_list": [write_h.value, err_write_h.value]}
    try:
        h_process, h_thread, pid, _tid = subprocess._winapi.CreateProcess(
            None,
            subprocess.list2cmdline(cmd),
            None,
            None,
            True,  # bInheritHandles: ONLY the handle_list handles inherit
            _CREATE_SUSPENDED,
            None,
            None,
            si,
        )
    except OSError:
        api.CloseHandle(err_write_h)
        api.CloseHandle(err_read_h)
        api.CloseHandle(write_h)
        api.CloseHandle(read_h)
        return None
    # The child now owns its write ends; drop the parent's copies so
    # the pipes can ever report EOF.
    api.CloseHandle(write_h)
    api.CloseHandle(err_write_h)
    child = _WinChild()
    child.h_process = h_process
    child.h_thread = h_thread
    child.pid = pid
    try:
        # c_void_p exposes its address as a Python int via ``.value``;
        # the O_* access flags come from the ``os`` module.
        fd = msvcrt.open_osfhandle(
            read_h.value, os.O_RDONLY | os.O_NOINHERIT | os.O_BINARY
        )
    except OSError:
        # Pathological: the stdout read end cannot be wrapped; fail
        # closed WITHOUT ever resuming the child.
        api.TerminateProcess(h_process, _JOB_EXIT_CODE)
        api.WaitForSingleObject(h_process, 1000)
        api.CloseHandle(h_thread)
        api.CloseHandle(h_process)
        api.CloseHandle(read_h)
        api.CloseHandle(err_read_h)
        return None
    child.file = os.fdopen(fd, "rb")
    try:
        err_fd = msvcrt.open_osfhandle(
            err_read_h.value, os.O_RDONLY | os.O_NOINHERIT | os.O_BINARY
        )
    except OSError:
        # Pathological: the stderr read end cannot be wrapped; the same
        # fail-closed rule applies (and the already-wrapped stdout fd
        # is released, which closes its handle).
        try:
            child.file.close()
        except Exception:  # noqa: BLE001
            pass
        child.file = None
        api.TerminateProcess(h_process, _JOB_EXIT_CODE)
        api.WaitForSingleObject(h_process, 1000)
        api.CloseHandle(h_thread)
        api.CloseHandle(h_process)
        api.CloseHandle(err_read_h)
        return None
    child.err_file = os.fdopen(err_fd, "rb")
    return child


class _JobPidListError(ValueError):
    """A ``JobObjectProcessIdList`` (class 3) response is malformed; the
    membership verification must fail closed."""


def _parse_job_process_id_list(data: bytes) -> list[int]:
    """Strictly parse a ``JobObjectProcessIdList`` (class 3) response.

    The documented response is a ``JOBOBJECT_BASIC_PROCESS_ID_LIST``:

        DWORD     NumberOfAssignedProcesses   (offset 0)
        DWORD     NumberOfProcessIdsInList    (offset 4)
        ULONG_PTR ProcessIdList[]             (offset 8)

    i.e. an 8-byte little-endian header followed by one native
    pointer-width element per listed process.  Process IDs are
    pointer-width values (8 bytes on x64), never 4-byte DWORDs.
    Documented count semantics: ``NumberOfAssignedProcesses`` is the
    total number of processes currently assigned to the job, while
    ``NumberOfProcessIdsInList`` is the number of PID entries actually
    returned in the current buffer; a valid response therefore always
    satisfies ``NumberOfProcessIdsInList <= NumberOfAssignedProcesses``
    (the returned list may be truncated by available buffer space, so
    ``listed < assigned`` is legitimate), and a response with
    ``NumberOfProcessIdsInList > NumberOfAssignedProcesses`` is
    rejected.  The buffer length must equal exactly ``8 +
    NumberOfProcessIdsInList * sizeof(ULONG_PTR)``; a shorter buffer
    (truncation -- including a header that claims PIDs the buffer does
    not contain), a longer one (a non-integral trailing element or
    other trailing bytes), or an inverted count relationship is
    rejected and the caller fails closed.  The parser reads ``data``
    only and never allocates or extends it.
    """
    if len(data) < 8:
        raise _JobPidListError(
            f"response shorter than the 8-byte header ({len(data)} bytes)"
        )
    n_assigned, n_listed = struct.unpack_from("<II", data, 0)
    if n_listed > n_assigned:
        raise _JobPidListError(
            f"implausible counts: {n_listed} listed > {n_assigned} assigned"
        )
    pid_size = ctypes.sizeof(ctypes.c_size_t)  # native ULONG_PTR width
    required = 8 + n_listed * pid_size
    if len(data) != required:
        raise _JobPidListError(
            "buffer length %d != 8 + %d * %d (%d)"
            % (len(data), n_listed, pid_size, required)
        )
    return [
        int.from_bytes(data[8 + i * pid_size : 8 + (i + 1) * pid_size], "little")
        for i in range(n_listed)
    ]


def _win_job_attach(api, job, child) -> bool:
    """Assign the suspended child to the job and verify the assignment.

    The ``AssignProcessToJobObject`` return value is the mandatory
    check and is never discarded.  A second, correctly documented
    cross-check then queries the job's process list (class 3,
    ``JobObjectProcessIdList``) with a single fixed-size
    (``_PID_LIST_QUERY_SIZE``) buffer and no retry model, and
    ``_parse_job_process_id_list`` decodes the documented
    ``JOBOBJECT_BASIC_PROCESS_ID_LIST`` response strictly: a
    ``DWORD`` header of two counts plus one ``ULONG_PTR`` PID per
    listed process, at exactly the implied buffer length.  The
    child's PID must be among the listed PIDs.  Modern kernels may
    list extra bookkeeping entries next to the child's PID
    (process-group records), so this is a membership check, never a
    count check.  If the query fails, the response is malformed, or
    the PID is not listed, the assignment is treated as unverified
    and the runner fails closed (the suspended child is terminated
    without being resumed).
    """
    if not api.AssignProcessToJobObject(job, child.h_process):
        return False
    qbuf = (ctypes.c_uint8 * _PID_LIST_QUERY_SIZE)()
    written = ctypes.c_ulong(0)
    if not api.QueryInformationJobObject(
        job, _JOB_PROCESS_ID_LIST, qbuf, _PID_LIST_QUERY_SIZE,
        ctypes.byref(written),
    ):
        return False
    try:
        pids = _parse_job_process_id_list(bytes(qbuf[: written.value]))
    except _JobPidListError:
        return False
    return child.pid in pids


def _win_job_resume(api, child) -> None:
    """Resume the suspended child; called ONLY after verified assignment."""
    api.ResumeThread(child.h_thread)
    api.CloseHandle(child.h_thread)
    child.h_thread = 0


def _win_job_kill(api, job) -> None:
    """Terminate the whole job tree; never raises."""
    try:
        api.TerminateJobObject(job, _JOB_EXIT_CODE)
    except Exception:  # noqa: BLE001 - containment must not break a probe
        pass


def _win_job_close(api, job) -> None:
    """Close the job handle; never raises.

    With KILL_ON_CLOSE verified, closing the handle additionally
    terminates any remaining job member on every path, so no member of
    a job this runner created can outlive the worker.
    """
    try:
        api.CloseHandle(job)
    except Exception:  # noqa: BLE001
        pass


def _win_process_code(api, handle) -> int:
    """``GetExitCodeProcess`` for ``handle``.

    Returns ``_STILL_ACTIVE`` (259, winbase.h) while the process runs,
    its real exit code once terminated, or -1 when the query itself
    fails (which is never treated as "alive").
    """
    code = ctypes.c_ulong(0)
    if not api.GetExitCodeProcess(handle, ctypes.byref(code)):
        return -1
    return code.value


def _win_wait(api, handle, seconds: float) -> int:
    """``WaitForSingleObject`` with a millisecond budget; never overruns it."""
    ms = max(0, int(seconds * 1000))
    return int(api.WaitForSingleObject(handle, ms))


def _emit(obj: dict) -> None:
    """Write the single protocol line to stdout and flush it."""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _result(name: str, status: str, note: str = "", fields: dict | None = None) -> dict:
    return {"name": name, "status": status, "note": note, "fields": fields or {}}


def _drain_capped(stream, buf: bytearray, over_cap: list) -> None:
    """Drain the tool's stdout: store up to the cap, discard the rest.

    Runs on a daemon thread and stops at EOF (every job member exited
    / the pipe write ends closed).  It KEEPS READING after the store
    cap is reached so a verbose tool can never block forever on a full
    pipe; excess bytes are counted, not stored.
    """
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            if len(buf) < _CHILD_OUT_LIMIT:
                room = _CHILD_OUT_LIMIT - len(buf)
                buf.extend(chunk[:room])
                if len(chunk) > room:
                    over_cap[0] = True
            else:
                over_cap[0] = True
    except Exception:  # noqa: BLE001 - draining is best-effort containment
        return


def _first_nonempty_line(text: str) -> str:
    """Return the first stripped non-empty line of ``text`` (else ``""``).

    Used ONLY to read a tool's own first-line version banner (stdout, or
    stderr for builds that emit the banner on their av_log channel --
    see the module note).  The returned line is a bounded prefix of the
    tool's own output, never of any other process's.
    """
    for ln in text.splitlines():
        if ln.strip():
            return ln.strip()
    return ""


def _run_capped(
    cmd: list[str], timeout: float | None = None, with_stderr: bool = False
):
    """Run a bounded child command with race-free process-tree containment.

    stdout is drained by a daemon reader thread into a buffer that stops
    storing at ``_CHILD_OUT_LIMIT`` bytes and keeps discarding the
    excess until EOF, so a verbose or hostile tool cannot pin unbounded
    worker memory or deadlock the pipe.  No shell is involved; on
    Windows the tool's stderr goes to a second pipe that the runner
    drains and discards (POSIX: DEVNULL).

    On Windows the containment sequence is strict and fail-closed (see
    ``_run_capped_win``): create job -> configure + verify
    KILL_ON_CLOSE -> create the tool SUSPENDED -> assign to the job
    (result checked) -> resume.  A suspended child cannot spawn or
    write before assignment, so there is no pre-assignment race; any
    setup failure terminates the suspended child without ever resuming
    it and raises ``_ContainmentSetupError`` so the probe reports a
    stable, non-sensitive UNAVAILABLE note -- the tool is never run in
    a third state.  On a timeout, or when the direct child exited but
    a descendant keeps the inherited pipe open, the WHOLE job tree is
    terminated with ``TerminateJobObject`` and closing the job handle
    applies the verified kill-on-close to any remaining member.

    Every cleanup step is budgeted: at most one reap window
    (``_REAP_BUDGET``) plus one reader-settle window per drained pipe
    (``_SETTLE_BUDGET``; two pipes on Windows, one on POSIX), so this
    function returns within ``timeout + _CLEANUP_BUDGET_MAX + epsilon``
    (timeout + 9 s + epsilon on the worst Windows path) and never waits
    indefinitely for an inherited pipe to close.  The parent's read end
    is closed ONLY once the reader thread has finished -- closing it
    while the reader is still draining would race the buffered-stream
    lock.

    Returns ``(timed_out, returncode, stdout_text_capped, over_cap)``.
    ``stdout_text_capped`` holds at most ``_CHILD_OUT_LIMIT`` bytes
    (decoded lossily); ``over_cap`` is True when the tool emitted more
    than the cap, in which case the captured prefix is unusable by
    design and callers must report a stable status WITHOUT surfacing
    it.  With ``with_stderr=True`` the runner additionally returns the
    drained-and-capped stderr prefix as a fourth element, i.e. the
    five-tuple ``(timed_out, returncode, stdout_text_capped,
    stderr_text_capped, over_cap)``; the stderr buffer is stored and
    drained exactly like stdout.  Callers may use it ONLY to read the
    tool's own first-line version banner (see ``probe_system_tool``);
    no other stderr content is ever surfaced.  The only exception to
    "never raise" is ``_ContainmentSetupError`` (Windows), which means
    the tool never ran.  On POSIX the legacy direct-child termination
    is kept and the stderr prefix is always ``""``.
    """
    if timeout is None:
        timeout = _CHILD_CMD_TIMEOUT
    if os.name == "nt":
        result = _run_capped_win(cmd, timeout)
    else:
        result = _run_capped_posix(cmd, timeout)
    if with_stderr:
        return result
    return result[0], result[1], result[2], result[4]


def _run_capped_posix(cmd: list[str], timeout: float):
    """POSIX branch: legacy Popen + direct-child termination.

    (Job objects are a Windows mechanism; POSIX descendants are a known
    narrower contract for this probe, which only ever shells out to
    short-lived, well-behaved system tools.)
    """
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=False,
        )
    except Exception:  # noqa: BLE001 - stable probe contract
        return False, None, "", "", False
    buf = bytearray()
    over_cap = [False]
    reader = threading.Thread(
        target=_drain_capped, args=(proc.stdout, buf, over_cap), daemon=True
    )
    reader.start()
    timed_out = False
    returncode = None
    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        reader.join(_SETTLE_BUDGET)
        if reader.is_alive():
            # The stdout pipe is still held open after the direct
            # child exited (or timed out): a descendant inherited
            # it.  Kill the direct child so the write ends close
            # and the reader can reach EOF.  A stalled capture is
            # a timeout from the probe's point of view, so map it
            # conservatively to TIMEOUT.
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
            timed_out = True
        try:
            proc.wait(timeout=_REAP_BUDGET)
        except subprocess.TimeoutExpired:
            pass
        reader.join(_SETTLE_BUDGET)
        if not reader.is_alive():
            try:
                proc.stdout.close()
            except Exception:  # noqa: BLE001
                pass
    if timed_out:
        return True, None, "", "", False
    text = bytes(buf).decode("utf-8", "replace")
    # POSIX: stderr is DEVNULL, so there is no stderr prefix to report.
    return False, returncode, text, "", over_cap[0]


def _run_capped_win(cmd: list[str], timeout: float):
    """Windows branch: race-free, fail-closed process-tree containment.

    Strict sequence (every step checked; any failure of the setup steps
    is FAIL-CLOSED, in which case the tool never ran; a failure after
    resume takes the whole job tree down):

      1. ``CreateJobObjectW`` -- a fresh per-invocation job;
      2. configure + verify KILL_ON_CLOSE (``_win_job_configure_kill_on_close``);
      3. spawn the tool with CREATE_SUSPENDED via CPython's C
         ``CreateProcess`` (``subprocess._winapi``), passing an
         extended ``STARTUPINFO`` object (the same stdlib form ``Popen``
         uses) whose flags carry the real ``STARTF_USESTDHANDLES``
         value 0x100 and whose handle list names exactly the two pipe
         write ends, wiring the child's stdout to the first pipe and
         stderr to the second; the stderr pipe is drained and
         discarded by a second reader thread (the tool probe may
         consult its first line ONLY as the tool's own version
         banner);
      4. ``AssignProcessToJobObject`` -- result checked (plus the
         ``JobObjectProcessIdList`` membership cross-check: the
         response is strictly parsed as a documented
         ``JOBOBJECT_BASIC_PROCESS_ID_LIST`` -- a two-DWORD header
         plus one ``ULONG_PTR`` PID per listed process -- and the
         child's PID must be among the listed PIDs; a malformed
         response is a failed verification);
      5. ``ResumeThread`` -- only now may the tool execute its first
         instruction; it is already a verified member of the job, so it
         and every descendant it spawns stay contained for life.

    On timeout -- or when the direct child exited but a descendant keeps
    the inherited pipe open -- the whole job tree is terminated with
    ``TerminateJobObject`` and the child is reaped within the budget;
    closing the job handle in ``finally`` applies the verified
    kill-on-close to any remaining member on every path.
    """
    api = _win_api()
    job = 0
    child = None
    reader = None
    err_reader = None
    buf = bytearray()
    err_buf = bytearray()
    over_cap = [False]
    err_over = [False]
    timed_out = False
    returncode = None
    resumed = False
    try:
        job = _win_job_create(api)
        if not job:
            raise _ContainmentSetupError("job object creation failed")
        if not _win_job_configure_kill_on_close(api, job):
            raise _ContainmentSetupError("kill-on-close configuration unverified")
        child = _win_spawn_suspended(api, cmd)
        if child is None:
            raise _ContainmentSetupError("suspended child creation failed")
        if not _win_job_attach(api, job, child):
            raise _ContainmentSetupError("job assignment failed")
        # Containment is established BEFORE execution: from here on the
        # child is resumed only as a verified member of the job.
        _win_job_resume(api, child)
        resumed = True
        reader = threading.Thread(
            target=_drain_capped, args=(child.file, buf, over_cap), daemon=True
        )
        reader.start()
        # The stderr pipe is drained the same way; its stored prefix is
        # only ever consulted for the tool's own first-line version
        # banner (probe_system_tool), never surfaced otherwise.
        err_reader = threading.Thread(
            target=_drain_capped, args=(child.err_file, err_buf, err_over), daemon=True
        )
        err_reader.start()
        if _win_wait(api, child.h_process, timeout) == _WAIT_TIMEOUT:
            timed_out = True
        else:
            returncode = _win_process_code(api, child.h_process)
            # The direct child exited; a descendant that inherited
            # either pipe may still keep it open.
            reader.join(_SETTLE_BUDGET)
            err_reader.join(_SETTLE_BUDGET)
            if reader.is_alive() or err_reader.is_alive():
                timed_out = True
        if timed_out:
            # Take down the whole job tree so every inherited write end
            # closes and both readers can reach EOF; then reap the
            # child.
            _win_job_kill(api, job)
            if _win_process_code(api, child.h_process) == _STILL_ACTIVE:
                _win_wait(api, child.h_process, _REAP_BUDGET)
            returncode = None
            reader.join(_SETTLE_BUDGET)
            err_reader.join(_SETTLE_BUDGET)
    except Exception as exc:  # noqa: BLE001 - fail closed on ANY failure
        # Every failure of containment setup (or of the contained run
        # itself) is converted to the stable setup error: the tool is
        # never left half-run.  If it was never resumed it is terminated
        # while still suspended (it cannot have spawned anything); if it
        # was already running, the whole job tree is taken down; and the
        # job close in ``finally`` applies the kill-on-close backstop.
        if child is not None and child.h_process:
            if not resumed:
                api.TerminateProcess(child.h_process, _JOB_EXIT_CODE)
            else:
                _win_job_kill(api, job)
            if _win_process_code(api, child.h_process) == _STILL_ACTIVE:
                _win_wait(api, child.h_process, _REAP_BUDGET)
        if not isinstance(exc, _ContainmentSetupError):
            raise _ContainmentSetupError("containment setup failed") from exc
        raise
    finally:
        if child is not None:
            if child.file is not None and (reader is None or not reader.is_alive()):
                try:
                    child.file.close()
                except Exception:  # noqa: BLE001
                    pass
                child.file = None
            if child.err_file is not None and (
                err_reader is None or not err_reader.is_alive()
            ):
                try:
                    child.err_file.close()
                except Exception:  # noqa: BLE001
                    pass
                child.err_file = None
            if child.h_process:
                api.CloseHandle(child.h_process)
                child.h_process = 0
            if child.h_thread:
                api.CloseHandle(child.h_thread)
                child.h_thread = 0
        if job:
            _win_job_close(api, job)
    if timed_out:
        return True, None, "", "", False
    text = bytes(buf).decode("utf-8", "replace")
    err_text = bytes(err_buf).decode("utf-8", "replace")
    return False, returncode, text, err_text, over_cap[0]


# ---------------------------------------------------------------------------
# Probe handlers
# ---------------------------------------------------------------------------


def _local_tag(version: str) -> str:
    """Classify a Torch version string into the locked local tag.

    This MUST stay character-for-character in sync with the parent's
    ``_local_tag`` in ``scripts/envreport.py``: the parent's required
    ``torch_contract`` check compares the worker's ``local_tag`` field
    against its own ``_local_tag(locked_version)``, so both functions
    must produce the same tag for the same version string (raw suffix
    forms such as ``cu130`` / ``cpu``, ``stable`` for an untagged
    version, ``custom`` for an unrecognised suffix).
    """
    if not isinstance(version, str) or "+" not in version:
        return "stable"
    tail = version.rsplit("+", 1)[1].strip().lower()
    if tail.startswith("cpu"):
        return "cpu"
    m = re.match(r"^cu(\d+)$", tail)
    if m:
        return f"cu{m.group(1)}"
    return tail or "custom"


def _torch_field(torch_mod) -> dict:
    """Build the torch fields dict from an imported module (no I/O)."""
    # ``torch.version`` is a small module whose OPTIONAL attributes
    # differ between wheels: this machine's torch 2.14.0+cu130 wheel
    # exposes ``cuda`` and ``hip`` but has NO ``cudnn`` attribute, so a
    # direct ``.cudnn`` read raises AttributeError and turned an
    # otherwise healthy torch into an IMPORT_ERROR (and failed the
    # parent's required ``torch_contract`` check).  Every
    # version-module attribute is therefore read through ``getattr``
    # with a None default: a missing attribute degrades to None instead
    # of a crash, and the status stays OK.
    version_mod = getattr(torch_mod, "version", None)
    fields = {
        "version": torch_mod.__version__,
        "local_tag": _local_tag(torch_mod.__version__),
        "build_name": getattr(torch_mod, "__build_name__", None),
        # The parent contract (PROBE_FIELD_WHITELIST["torch"] and the
        # required ``torch_contract`` check) reads the CUDA line from the
        # ``cuda`` key; the value is ``torch.version.cuda`` (e.g. "13.0"
        # on a +cu130 wheel, None on a cpu build).  Extra diagnostic keys
        # (build_name, hip, cudnn, is_built_with_cuda, nccl) are dropped
        # by the parent's field whitelist before any display.
        "cuda": getattr(version_mod, "cuda", None),
        "hip": bool(getattr(version_mod, "hip", None)),
        "cudnn": getattr(version_mod, "cudnn", None),
        "is_built_with_cuda": bool(torch_mod.cuda.is_available()),
        "nccl": None,
    }
    try:
        fields["nccl"] = torch_mod.cuda.nccl_version()
    except Exception:  # noqa: BLE001 - NCCL is optional metadata
        pass
    return fields


def probe_torch(name: str) -> dict:
    """Report the exact Torch the selected interpreter would use.

    A bare ``import torch`` on the already-running interpreter: no
    module load, no DLL load, no subprocess, no GPU work, no
    initialization.  The report deliberately does NOT call
    ``torch.cuda.init()`` -- that would be heavy, could touch driver
    state, and would violate the "report only, never initialize or
    load" contract.
    """
    try:
        import torch  # noqa: PLC0415 - the import IS the probe

        fields = _torch_field(torch)
        # CUDA/HIP presence as reported by the installed build, without
        # initializing the CUDA context (torch.cuda.is_available() on
        # a build without CUDA is a cheap attribute check; with CUDA it
        # may run a small device query, which is exactly the fact we
        # need and never a heavy init).
        return _result(name, STATUS_OK, fields=fields)
    except ImportError:
        return _result(name, STATUS_IMPORT_ERROR, "torch is not installed")
    except Exception as exc:  # noqa: BLE001 - degrade, never raise
        # A broken torch install (e.g. a failed DLL load) must not take
        # the worker down: report it as an import-level failure.
        return _result(name, STATUS_IMPORT_ERROR, f"torch import failed: {type(exc).__name__}")


def probe_cuda(name: str) -> dict:
    """Report the CUDA runtime the selected interpreter would use.

    The runtime field is a pure attribute read (no device work).  The
    device fields use torch's cheap query API (``is_available`` /
    ``device_count`` / ``get_device_name`` / ``get_device_properties``)
    and are reported only when a device is actually available: on a
    driverless machine the probe reports OK with ``available=False``
    and no device fields, and any query that raises degrades the probe
    to INIT_ERROR instead of crashing the worker.
    """
    try:
        import torch  # noqa: PLC0415

        # torch.version.cuda reports the CUDA runtime the installed
        # wheel was built against (e.g. "13.0") or None for a CPU
        # build.  Read through getattr so a wheel without the
        # attribute degrades to a CPU-build report, not a crash.
        runtime = getattr(getattr(torch, "version", None), "cuda", None)
        fields = {"runtime": runtime}
        if not runtime:
            return _result(
                name, STATUS_MISSING,
                "installed torch was built without CUDA support",
                fields=fields,
            )
        available = bool(torch.cuda.is_available())
        fields["available"] = available
        if available:
            fields["device_count"] = int(torch.cuda.device_count())
            fields["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            fields["vram_gb"] = round(props.total_memory / 2**30, 1)
            fields["compute_capability"] = f"{props.major}.{props.minor}"
        return _result(name, STATUS_OK, fields=fields)
    except ImportError:
        return _result(name, STATUS_IMPORT_ERROR, "torch is not installed")
    except Exception as exc:  # noqa: BLE001 - degrade, never raise
        return _result(name, STATUS_INIT_ERROR, f"cuda probe failed: {type(exc).__name__}")


def probe_nvidia_smi(name: str) -> dict:
    """Run ``nvidia-smi --query-gpu=...`` bounded; return structured rows.

    The executable is located with ``shutil.which``; if it is absent the
    probe reports MISSING.  The query requests ``compute_cap`` -- the
    field name installed nvidia-smi versions actually accept (the
    longer spelling ``compute_capability`` is rejected with exit code
    2); the parsed output keeps the field name ``compute_capability``
    for model stability.  A non-zero exit degrades to INIT_ERROR, a
    timeout to TIMEOUT, and empty output, an over-cap capture, or a
    containment setup failure to UNAVAILABLE -- each with a short,
    non-sensitive note.  Raw output is never surfaced.  A tool that emits more than ``_CHILD_OUT_LIMIT`` bytes
    is a contract violation (the query above is a bounded CSV): the
    captured prefix is discarded and UNAVAILABLE is reported instead of
    surfacing it.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return _result(name, STATUS_MISSING, "nvidia-smi executable not found")
    cmd = [
        exe,
        "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        timed_out, rc, out, over_cap = _run_capped(cmd)
    except _ContainmentSetupError:
        # Fail-closed: job create/configure/verify/assign/launch failed,
        # so the tool never ran; report a stable, non-sensitive note.
        # Display name, fixed, non-sensitive: consistent with every
        # other note in this handler ("nvidia-smi ...").
        return _result(
            name, STATUS_UNAVAILABLE,
            "nvidia-smi could not be executed in a contained process tree",
        )
    if timed_out:
        return _result(name, STATUS_TIMEOUT, "nvidia-smi timed out")
    if rc is None or rc != 0:
        code = rc if rc is not None else -1
        # Present tool, failed execution: INIT_ERROR, the same mapping
        # the parent applies to a failing probe worker.
        return _result(name, STATUS_INIT_ERROR, f"nvidia-smi exited with code {code}")
    if over_cap:
        # Contract violation: never surface the discarded prefix.
        return _result(
            name,
            STATUS_UNAVAILABLE,
            f"nvidia-smi output exceeded the {_CHILD_OUT_LIMIT}-byte capture limit",
        )
    rows = [ln for ln in out.strip().splitlines() if ln.strip()]
    if not rows:
        return _result(name, STATUS_UNAVAILABLE, "nvidia-smi returned no GPU rows",
                       fields={"gpu_count": 0})
    # Bounded parse: the query returns one 4-column CSV row per GPU;
    # cap the rows we turn into fields so a hostile or verbose output
    # cannot inflate the protocol line (the capture itself is already
    # capped at ``_CHILD_OUT_LIMIT`` bytes).
    driver_version = None
    gpu_rows = []
    for ln in rows[:16]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 4:
            continue
        if driver_version is None:
            driver_version = parts[1]
        try:
            vram_mb = int(parts[2])
        except ValueError:
            vram_mb = None
        gpu_rows.append(
            {
                "name": parts[0],
                "vram_mb": vram_mb,
                "compute_capability": parts[3],
            }
        )
    fields = {"driver_version": driver_version, "gpu_count": len(gpu_rows), "gpus": gpu_rows}
    return _result(name, STATUS_OK, fields=fields)


def probe_system_tool(name: str, tool: str) -> dict:
    """Run ``<tool> --version`` bounded; capture the first line.

    ``tool`` is ``ffmpeg`` or ``ffprobe``.  If the executable is not on
    PATH the probe reports MISSING; a timeout degrades to TIMEOUT, and a
    containment setup failure to UNAVAILABLE with a stable note.

    A non-zero exit is normally INIT_ERROR, with one build-specific
    exception: this machine's static gyan.dev ffmpeg/ffprobe build
    (binary mtime 2024-11-07, unchanged since) writes the
    ``--version`` banner to its stderr (av_log) channel and exits
    with a build-specific code (0xABAFB008 = 2880417800 for ffmpeg;
    1 for ffprobe) whenever its standard handles are explicitly
    wired.  So when the tool exits non-zero but
    its (drained, capped) stderr starts with ``<tool> version `` and its
    stdout did not overflow the capture limit, that first stderr line
    is accepted as the tool's own version banner and the probe
    reports OK.  No other stderr content is ever surfaced, and the
    acceptance is limited to the banner prefix above.

    The version field holds the version identifier that the tool's own
    banner reports: the first whitespace-separated token after the
    ``<tool> version `` prefix (e.g. ``4.4.4`` or this machine's gyan
    build id ``2024-11-06-git-4047b887fc-full_build-www.gyan.dev``),
    bounded to 256 characters.  No other part of the tool's output is
    ever surfaced.
    """
    exe = shutil.which(tool)
    if exe is None:
        return _result(name, STATUS_MISSING, f"{tool} executable not found")
    try:
        timed_out, rc, out, err, over_cap = _run_capped(
            [exe, "--version"], with_stderr=True
        )
    except _ContainmentSetupError:
        # Fail-closed: the tool never ran; stable, non-sensitive note.
        return _result(
            name, STATUS_UNAVAILABLE,
            f"{tool} could not be executed in a contained process tree",
        )
    if timed_out:
        return _result(name, STATUS_TIMEOUT, f"{tool} timed out")
    if rc is not None and rc != 0 and not over_cap:
        code = rc
        banner = _first_nonempty_line(err)
        if banner.startswith(f"{tool} version "):
            # The tool's own version banner, emitted on stderr by this
            # build: accept its version identifier -- the first
            # whitespace-separated token after the ``<tool> version ``
            # prefix (e.g. this machine's gyan build id
            # ``2024-11-06-git-4047b887fc-full_build-www.gyan.dev``) --
            # and nothing else (see the docstring).
            tail = banner[len(f"{tool} version "):]
            token = tail.split()
            version = token[0] if token else banner
            return _result(name, STATUS_OK, fields={"version": version[:256]})
        # Present tool, failed execution: INIT_ERROR, the same mapping
        # the parent applies to a failing probe worker.
        return _result(name, STATUS_INIT_ERROR, f"{tool} exited with code {code}")
    if rc is None or rc != 0:
        # Non-zero exit with an over-cap capture: the prefix is
        # unusable and the banner check above was not applicable.
        code = rc if rc is not None else -1
        return _result(name, STATUS_INIT_ERROR, f"{tool} exited with code {code}")
    if over_cap:
        # ``--version`` never emits megabytes; treat an over-cap capture
        # as a contract violation and discard the prefix.
        return _result(
            name,
            STATUS_UNAVAILABLE,
            f"{tool} output exceeded the {_CHILD_OUT_LIMIT}-byte capture limit",
        )
    version = _first_nonempty_line(out)[:256] or "unknown"
    return _result(name, STATUS_OK, fields={"version": version})


def probe_onnx(name: str) -> dict:
    """Report the onnx package without importing it (metadata only)."""
    try:
        import importlib.metadata as md  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return _result(name, STATUS_INIT_ERROR, "importlib.metadata unavailable")
    try:
        version = md.version("onnx")
    except md.PackageNotFoundError:
        return _result(name, STATUS_MISSING, "onnx is not installed")
    except Exception as exc:  # noqa: BLE001 - degrade, never raise
        return _result(name, STATUS_INIT_ERROR, f"onnx metadata failed: {type(exc).__name__}")
    return _result(name, STATUS_OK, fields={"version": version})


def probe_onnxruntime(name: str) -> dict:
    """Report the onnxruntime package without importing it (metadata only).

    Importing ``onnxruntime`` loads its native runtime; the env report
    must never trigger that, so only installed metadata is reported.
    """
    try:
        import importlib.metadata as md  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return _result(name, STATUS_INIT_ERROR, "importlib.metadata unavailable")
    try:
        version = md.version("onnxruntime")
    except md.PackageNotFoundError:
        return _result(name, STATUS_MISSING, "onnxruntime is not installed")
    except Exception as exc:  # noqa: BLE001 - degrade, never raise
        return _result(name, STATUS_INIT_ERROR, f"onnxruntime metadata failed: {type(exc).__name__}")
    return _result(name, STATUS_OK, fields={"version": version})


def probe_ffmpeg(name: str) -> dict:
    """Run ``ffmpeg --version`` bounded; see ``probe_system_tool``."""
    return probe_system_tool(name, "ffmpeg")


def probe_ffprobe(name: str) -> dict:
    """Run ``ffprobe --version`` bounded; see ``probe_system_tool``."""
    return probe_system_tool(name, "ffprobe")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

HANDLERS = {
    "torch": probe_torch,
    "cuda": probe_cuda,
    "nvidia_smi": probe_nvidia_smi,
    "ffmpeg": probe_ffmpeg,
    "ffprobe": probe_ffprobe,
    "onnx": probe_onnx,
    "onnxruntime": probe_onnxruntime,
}


def main(argv: list[str]) -> int:
    """Run the probe named in ``argv[1]`` and emit its protocol line.

    The parent (``scripts/envreport.py``) starts this worker once per
    probe and consumes the emitted line for THAT probe only: its
    ``status``, ``note`` and ``fields`` become the parent's ProbeInfo.
    Emitting a batch line (all probes in one record) would make the
    parent report a uniform OK status with every field silently
    dropped, so the per-name record is the only correct protocol.

    A name outside the handler table is an internal contract
    violation: it is reported as a stable UNAVAILABLE with the fixed,
    non-sensitive note "unknown probe" (the token is never echoed) and
    the worker still exits 0, so the parent's parse path stays
    well-formed.

    Each handler is exception-safe by construction (it converts every
    failure mode into a stable status), so the dispatch below cannot
    abort the worker; the surrounding ``except`` is a last-resort
    guard that still emits a well-formed line.
    """
    name = argv[1] if len(argv) > 1 else ""
    handler = HANDLERS.get(name)
    if handler is None:
        # Fixed note: the unknown token is never echoed (privacy gate).
        _emit({"name": name, "status": STATUS_UNAVAILABLE, "note": "unknown probe", "fields": {}})
        return 0
    try:
        rec = handler(name)
    except Exception as exc:  # noqa: BLE001 - absolute last resort
        rec = _result(name, STATUS_UNAVAILABLE, f"probe error: {type(exc).__name__}")
    _emit(rec)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception as exc:  # noqa: BLE001 - keep the protocol line well-formed
        _emit(
            {
                "ok": False,
                "name": "<worker>",
                "status": "UNAVAILABLE",
                "note": f"probe worker error: {type(exc).__name__}",
            }
        )
        sys.exit(0)
