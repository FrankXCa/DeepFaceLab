@echo off
rem ============================================================================
rem dfl.bat - DeepFaceLab portable-runtime application launcher (Phase 13).
rem
rem Contract (docs/PHASE13_STATE.md sections 15, 17, 18):
rem   * The repository root is derived from this file's own location
rem     (launchers\..); the caller's current directory is never used.
rem   * Reads runtime\active-runtime.txt, validates the FILE AS DATA
rem     (size, line count, character-set - never by substituting raw
rem     selector bytes into a parsed command) against the restricted
rem     runtime-id grammar (exactly 64 lowercase hex characters on the
rem     single logical line), and only then reads the proven-safe value;
rem     it launches that immutable version's bundled interpreter in
rem     isolated mode:
rem         <runtime>\python.exe -I -B <repo>\scripts\runtime_entry.py [args...]
rem   * NO install, NO update, NO network access, NO runtime-tree mutation,
rem     NO temporary-file writes (the selector read uses no helper file).
rem   * Clears inherited host Python environment state (PYTHONHOME,
rem     PYTHONPATH, PYTHONUSERBASE, PYTHONSTARTUP) and publishes
rem     PYTHONNOUSERSITE=1 / PYTHONDONTWRITEBYTECODE=1 as defense in depth;
rem     runtime_entry.py re-sanitizes at the Python level. Stale DFL device
rem     state (NN_DEVICES_INITIALIZED, NN_DEVICES_COUNT and every variable
rem     named NN_DEVICE_* in ANY letter case) is cleared via a tested,
rem     injection-safe enumeration; the variable NN_DEVICE (no trailing
rem     underscore) and NN_DEVICES_* are not part of that family and are
rem     left untouched. CUDA toolkit sanitation (CUDA_PATH, CUDA_HOME,
rem     CUDA_VISIBLE_DEVICES and toolkit PATH entries) is owned by
rem     runtime_entry.py, not by this launcher.
rem   * The selected runtime's bundled python.exe is the ONLY interpreter
rem     used: no host/system/venv Python fallback exists or is attempted.
rem   * User arguments are forwarded unchanged (raw %%*), and the
rem     application exit code is returned unchanged.
rem
rem Exit codes: the application exit code is propagated; 1 = the selected
rem runtime (or its bundled interpreter / the repository bootstrap) is
rem missing; 2 = the active-runtime selector is missing or malformed.
rem ============================================================================
setlocal

rem ---- repository root from this launcher's location (never from CWD) ----
set "ROOT=%~dp0.."
cd /d "%ROOT%"
rem From here on all paths are relative to the repository root: for /f
rem file operands must be relative paths in this cmd build (quoted
rem absolute paths are misread as string literals), and the application
rem child process inherits the repository root as its working directory.

rem ---- host Python environment isolation (defense in depth; child-local)
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONUSERBASE="
set "PYTHONSTARTUP="
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"

rem ---- stale DFL device state: exact names plus the NN_DEVICE_ family.
rem Windows environment names are case-insensitive, so the family match
rem uses findstr /B /I (prefix match in any letter case: nn_device_*,
rem Nn_DeViCe_* ...). Each matched name is cleared with a QUOTED set, so
rem cmd metacharacters inside the NAME cannot be interpreted as commands.
rem Only the name (for /f tokens=1) ever reaches this process's command
rem line; the variable VALUE is never expanded or executed, so hostile
rem values (e.g. "& <command>") are inert. No CALL and no delayed
rem expansion are used over attacker-controlled state.
set "NN_DEVICES_INITIALIZED="
set "NN_DEVICES_COUNT="
for /f "tokens=1 delims==" %%V in ('set ^| findstr /B /I "NN_DEVICE_"') do set "%%V="

rem ---- active-runtime selector: required, exactly 64 lowercase hex ----
rem CORE RULE: unvalidated selector bytes must NEVER be substituted into
rem a parsed batch command - not even inside `set "VAR=..."`. On this
rem cmd build a double quote inside substituted text can end the quoted
rem SET token, and the remainder of the value is then parsed as command
rem syntax (a first line of `"& <command> & rem` executes). Quoting the
rem value "safely" is therefore not an option: the file is validated AS
rem DATA first, using tools that treat file contents as data; only after
rem the complete selector has been proven safe is its content loaded
rem into a normal variable.
rem (1) the file exists;
rem (2) its size is 64, 65 or 66 bytes (`for %%F in (...) do %%~zF`
rem     reads the size without touching the content): 64 = a bare
rem     64-hex line, 65 = 64-hex + LF (the Commit-2 selector writer
rem     format), 66 = 64-hex + CRLF; anything else is malformed before
rem     a single content byte is examined;
rem (3) it holds exactly ONE logical line (`find /c /v ""`): a second
rem     line - useful or empty - makes the selector malformed, so a
rem     valid second line can never rescue a malformed first line;
rem (4) no line of the file contains a character outside 0-9a-f
rem     (`findstr /R /C:"[^0-9abcdef]"`): the class must be an explicit
rem     CHARACTER LIST, not a range - on this cmd build a range class such
rem     as [a-f] matches case-insensitively (so [^0-9a-f] would let
rem     uppercase A-F through), while a list [abcdef] is case-sensitive,
rem     so uppercase A-F as well as every quote, letter, space, tab,
rem     colon, percent, metacharacter and CR is a match. This proves the
rem     single line is PURE lowercase hex before any of its bytes is ever
rem     read into this process.
rem ONLY THEN is the line captured (`for /f "delims="` over the file + a
rem quoted set; a plain file input, because the `usebackq` `(<"file")`
rem form is rejected by this cmd build): the captured text is provably
rem pure hex - no quote, no metacharacter, no % - so substituting it
rem below is inert BY CONSTRUCTION. The final gate is an exact length probe on that pure
rem value: a character exists at position 64 (0-based 63) AND no
rem character exists at position 65. It rejects a size-65 or size-66
rem file whose extra byte is a 65th/66th hex character rather than a
rem line terminator (65-hex / 66-hex lines without a terminator) and a
rem size-64 file whose last byte is an LF after only 63 hex chars.
rem No helper file, no temp file, no CALL, no delayed expansion: the
rem normal launch is filesystem-read-only.
if not exist "runtime\active-runtime.txt" goto :dfl-no-selector
for %%F in (runtime\active-runtime.txt) do set "DFL_SELSZ=%%~zF"
if not "%DFL_SELSZ%"=="64" if not "%DFL_SELSZ%"=="65" if not "%DFL_SELSZ%"=="66" goto :dfl-bad-selector
set "DFL_SELLINES="
rem find /c prints "---------- FILE: N" for a single file argument: token 2
rem after the colon carries the count (possibly with a leading space on this
rem cmd build); strip spaces before the exact comparison.
for /f "tokens=2 delims=:" %%N in ('find /c /v "" "runtime\active-runtime.txt" 2^>nul') do set "DFL_SELLINES=%%N"
if not defined DFL_SELLINES goto :dfl-bad-selector
set "DFL_SELLINES=%DFL_SELLINES: =%"
if not "%DFL_SELLINES%"=="1" goto :dfl-bad-selector
findstr /R /C:"[^0-9abcdef]" "runtime\active-runtime.txt" >nul 2>nul
if not errorlevel 1 goto :dfl-bad-selector
rem -- the file is now proven to be one line of pure lowercase hex --
set "DFL_ID="
for /f "delims=" %%L in (runtime\active-runtime.txt) do set "DFL_ID=%%L"
if not defined DFL_ID goto :dfl-bad-selector
if "%DFL_ID:~63,1%"=="" goto :dfl-bad-selector
if not "%DFL_ID:~64,1%"=="" goto :dfl-bad-selector

rem ---- the selected version must live under runtime\versions (the 64-hex
rem grammar structurally contains no separator, so no path traversal is
rem possible) and its bundled interpreter must be present. The interpreter
rem check uses a file pattern (PATHEXT-aware, matching the execution
rem semantics of the launch line below); the version directory itself is
rem verified byte-exactly by the Commit-2 activation primitive, so no
rem unverified interpreter can be reached this way.
if not exist "runtime\versions\%DFL_ID%\" goto :dfl-no-runtime
set "PYFOUND="
for %%F in (runtime\versions\%DFL_ID%\python.exe*) do set "PYFOUND=1"
if not defined PYFOUND goto :dfl-no-python
if not exist "scripts\runtime_entry.py" goto :dfl-no-entry

"runtime\versions\%DFL_ID%\python.exe" -I -B "scripts\runtime_entry.py" %*
exit /b %ERRORLEVEL%

:dfl-no-selector
echo dfl: no active runtime selected: runtime\active-runtime.txt is missing. Run launchers\dfl-setup-runtime.bat ^<variant^> to build and activate a runtime first. >&2
exit /b 2

:dfl-bad-selector
echo dfl: active-runtime.txt holds an invalid runtime id (expected exactly 64 lowercase hex characters on its single line). Do not edit it by hand; run launchers\dfl-setup-runtime.bat to activate a verified runtime. >&2
exit /b 2

:dfl-no-runtime
echo dfl: selected runtime is missing: runtime\versions\%DFL_ID% does not exist. Re-run launchers\dfl-setup-runtime.bat or restore the version directory. >&2
exit /b 1

:dfl-no-python
echo dfl: selected runtime has no bundled interpreter: runtime\versions\%DFL_ID%\python.exe is missing. The runtime is incomplete or corrupted; rebuild it with launchers\dfl-setup-runtime.bat. >&2
exit /b 1

:dfl-no-entry
echo dfl: repository bootstrap is missing: scripts\runtime_entry.py must sit next to launchers\ in a DeepFaceLab source tree. >&2
exit /b 1
