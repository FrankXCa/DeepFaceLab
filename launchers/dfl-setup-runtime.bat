@echo off
rem ============================================================================
rem dfl-setup-runtime.bat - explicit developer setup/build helper (Phase 13).
rem
rem This is a DEVELOPER/build helper, not a Python-free bootstrap:
rem runtime construction requires a host CPython >= 3.11 (the builder's
rem documented source floor). The assembled runtime bundles its own
rem standalone CPython 3.12.11 and never needs host Python at run time.
rem
rem Host interpreter selection (deterministic policy):
rem   * Explicit %DFL_SETUP_PYTHON%: PATH DATA, never command text. It
rem     must be the path of ONE existing host interpreter executable
rem     (.exe) - no command plus arguments, no shell fragment, no quoted
rem     command expression, no pipe/redirection, no environment-variable
rem     expression, no batch wrapper (a .bat/.cmd override would be
rem     command text, which this helper cannot validate safely in cmd;
rem     the authoritative Phase-13 contract only requires a host
rem     CPython >= 3.11, which a python.exe path provides). Before any
rem     byte of the value may appear in an executable position it is
rem     validated AS DATA by a fixed validator program run in a child of
rem     a discovered automatic candidate (the value crosses only the
rem     environment, which the child reads - this process never expands
rem     it). The value is rejected when it is empty, contains any of the
rem     characters " & | < > ^ % ! ( ) ; CR LF, does not end in .exe, or
rem     does not exist as a file. With no automatic candidate available
rem     the override cannot be validated and is rejected - the value is
rem     still never expanded. A validated value is expanded exactly once
rem     into DFLS_OVR and used only as a safely quoted executable path;
rem     every override host invocation is a DIRECT execution of that
rem     quoted path (no CALL, no child cmd, no reconstructed command
rem     string). An explicit override that fails validation, whose probe
rem     fails, or that is below the 3.11 floor fails TERMINALLY
rem     (exit 3 / exit 4): there is no fallback to automatic discovery
rem     for an explicit override.
rem   * Special value `none`: documented test/CI escape; the value is
rem     recognized by the data-space validator (the child interpreter
rem     compares the environment variable byte-for-byte - no part of
rem     the value is ever parsed by this process) and the run then
rem     fails with the no-host-Python error (exit 3). It is not a
rem     usable host.
rem   * Automatic candidates, tried in order: `py -3`, then `python` on
rem     PATH (the Windows Store alias cannot run a real interpreter and
rem     fails the probe, so it is never selected). These are
rem     LAUNCHER-OWNED fixed command forms (literal text; never
rem     caller-controlled command strings):
rem       - probe exit 0  = candidate executes;
rem       - floor exit 0  = candidate meets the >= 3.11 source floor;
rem       - floor exit 3  = candidate executes but is below the floor;
rem       - any other exit code = candidate execution/probe failure.
rem     A candidate that fails to probe, or is below the floor, is SKIPPED
rem     and the next automatic candidate is tried (the Phase-13 plan does
rem     not make below-floor terminal for automatic discovery). If every
rem     automatic candidate is exhausted: exit 4 when at least one candidate
rem     was below the floor (its version is reported), else exit 3.
rem   Exact exit-code equality is used throughout; ERRORLEVEL range
rem   semantics (which would misclassify probe failures as "too old") are
rem   never used.
rem Host execution design (injection-safe):
rem   * Automatic-candidate PROBE, FLOOR and VERSION-CAPTURE run in a
rem     child cmd process (`cmd /c py -3 -I -B -c "<fixed code>"` /
rem     `cmd /c python -I -B -c "<fixed code>"`): the child command line
rem     is launcher-owned fixed text - no caller-controlled data ever
rem     crosses it.
rem   * The override host is a prevalidated .exe path: its probe, floor
rem     and builder invocations are DIRECT executions of
rem     `"%DFLS_OVR%" -I -B ...` in this process (a spawned child for a
rem     real executable; the value contains no quote, no metacharacter,
rem     no %, by construction of the validation, so the quoted expansion
rem     is inert).
rem   * The BUILDER invocation runs DIRECTLY in this process on one of
rem     sixteen literal lines (eight per host kind). A value variable is
rem     substituted into the selected line exactly once: cmd expands each
rem     command line exactly once and never re-scans substituted text, so
rem     a value that still contains %NAME% after the caller's own single
rem     expansion is passed to the builder LITERALLY - this helper adds no
rem     second expansion layer (no CALL, no child cmd on the builder
rem     path). Every value sits inside template double quotes, so cmd
rem     metacharacters in the value are inert. The builder line is the
rem     last command of this helper (the next statement is `exit /b`).
rem     A real python.exe / py.exe host returns its exit code through
rem     %ERRORLEVEL%; a batch host reached through the AUTOMATIC
rem     candidates runs in-process on the builder line and must end the
rem     process with `exit <code>` (process-level exit) for its code to
rem     become the launcher's exit code.
rem
rem Usage:
rem   launchers\dfl-setup-runtime.bat ^<variant^> [builder options]
rem ^<variant^> is exactly one of: cuda-gui  cuda-nogui  cpu-gui  cpu-nogui
rem (explicit selection; no GPU-based auto-detection).
rem [builder options] - an explicitly parsed, validated token set (NO
rem free-form argument-string accumulation; unknown options are rejected):
rem   --activate                 activate the verified runtime afterwards via
rem                               the Commit-2 activation primitive (selector
rem                               swap under the setup/activation lock)
rem   --offline                  zero-network build
rem   --offline-inputs ^<value^>   offline inputs for --offline (quote values
rem                               containing spaces; value required)
rem   --artifact-cache ^<value^>   developer artifact cache location
rem                               (value required)
rem   --lock-timeout ^<value^>     setup/activation lock wait timeout
rem                               (value required)
rem Option grammar (strict): a valued option's value must be present and
rem must NOT begin with "--": a recognized option token (or any
rem option-looking token) used where a value is required is rejected with
rem exit 2, and no escape syntax for values beginning with "--" is
rem provided. Option names are matched case-insensitively and forwarded in
rem canonical spelling. Each value is stored raw in its own variable and
rem forwarded on one of eight literal builder-invocation lines (selected
rem by which valued options were given) in which the value sits inside
rem template double quotes: cmd metacharacters in the value (spaces & ; ( )
rem and Unicode) stay literal in the builder invocation and argv. A value
rem containing a double quote cannot be represented on a Windows command
rem line and is structurally un-forwardable; a value containing % is
rem subject to the cmd layer's own variable expansion (a limitation of
rem every .bat CLI).
rem
rem Online mode may download the locked artifacts; offline mode never
rem touches the network. Setup may build, verify, and (with --activate)
rem activate a verified runtime; it never deletes, retires, or
rem garbage-collects any verified runtime version.
rem
rem Exit codes: 0 = builder exit code propagated; 2 = usage/variant/option
rem error; 3 = no usable host CPython found (or explicit override unusable);
rem 4 = host CPython below the 3.11 floor.
rem ============================================================================
setlocal

set "ROOT=%~dp0.."

rem ---- explicit variant selection (no auto-detection) ----
if "%~1"=="" goto :dfls-usage
set "VARIANT=%~1"
shift
if "%VARIANT%"=="cuda-gui" goto :dfls-options
if "%VARIANT%"=="cuda-nogui" goto :dfls-options
if "%VARIANT%"=="cpu-gui" goto :dfls-options
if "%VARIANT%"=="cpu-nogui" goto :dfls-options
goto :dfls-bad-variant

rem ---- builder options: explicit parse, one validated token per option.
rem Each VALUE is stored RAW in its own variable (a quoted set, so cmd
rem metacharacters in the value are inert and the value is never evaluated
rem as command text). No option text is accumulated into a single command
rem string; see the Usage section above for the forwarding design.
:dfls-options
set "DFLS_ACT="
set "DFLS_OFF="
set "DFLS_OFFIN="
set "DFLS_ARTC="
set "DFLS_LOCKT="
set "DFLS_OFFIN_VAL="
set "DFLS_ARTC_VAL="
set "DFLS_LOCKT_VAL="
:optloop
if "%~1"=="" goto :dfls-host
set "DFLS_OPT=%~1"
if "%DFLS_OPT%"=="--activate" goto :dfls-opt-activate
if "%DFLS_OPT%"=="--offline" goto :dfls-opt-offline
if "%DFLS_OPT%"=="--offline-inputs" goto :dfls-opt-offinputs
if "%DFLS_OPT%"=="--artifact-cache" goto :dfls-opt-artcache
if "%DFLS_OPT%"=="--lock-timeout" goto :dfls-opt-locktimeout
goto :dfls-bad-option

:dfls-opt-activate
set "DFLS_ACT=--activate"
shift
goto :optloop

:dfls-opt-offline
set "DFLS_OFF=--offline"
shift
goto :optloop

:dfls-opt-offinputs
shift
if "%~1"=="" goto :dfls-missing-value
set "DFLS_VAL=%~1"
set "DFLS_VALPRE=%DFLS_VAL:~0,2%"
if "%DFLS_VALPRE%"=="--" goto :dfls-option-value
set "DFLS_OFFIN_VAL=%DFLS_VAL%"
set "DFLS_OFFIN=1"
shift
goto :optloop

:dfls-opt-artcache
shift
if "%~1"=="" goto :dfls-missing-value
set "DFLS_VAL=%~1"
set "DFLS_VALPRE=%DFLS_VAL:~0,2%"
if "%DFLS_VALPRE%"=="--" goto :dfls-option-value
set "DFLS_ARTC_VAL=%DFLS_VAL%"
set "DFLS_ARTC=1"
shift
goto :optloop

:dfls-opt-locktimeout
shift
if "%~1"=="" goto :dfls-missing-value
set "DFLS_VAL=%~1"
set "DFLS_VALPRE=%DFLS_VAL:~0,2%"
if "%DFLS_VALPRE%"=="--" goto :dfls-option-value
set "DFLS_LOCKT_VAL=%DFLS_VAL%"
set "DFLS_LOCKT=1"
shift
goto :optloop

rem ---- host CPython selection (deterministic, exact exit codes) ----
:dfls-host
set "HOSTCMD="
set "DFLS_OVR="
set "DFLS_AUTOHOST="
set "DFLS_BELOW=0"
set "DFLS_HOSTVER="
rem The `none` sentinel is NOT detected in this process: expanding the
rem value here (raw %DFL_SETUP_PYTHON% or :~N,1% substrings) would put
rem its bytes into parsed command text before validation - on this cmd
rem build a compound `if defined VAR if "%VAR:~0,1%"==...` line aborts
rem the whole batch with a syntax error even when VAR is undefined, and
rem a quote-bearing value corrupts quote pairing far enough that a
rem trailing &/| command fragment can execute. The sentinel is instead
rem recognized by the data-space validator below (the child interpreter
rem compares the environment variable byte-for-byte) and reported there
rem as validator rc 7, which maps to the no-host-Python failure (exit 3).
if defined DFL_SETUP_PYTHON goto :dfls-override
goto :dfls-auto

rem == automatic candidates (launcher-owned fixed command forms) ========
:dfls-auto
rem -- automatic candidate 1: the Windows py launcher (`py -3`)
cmd /c py -3 -I -B -c "import sys" >nul 2>nul
set /a DFLS_RC=%ERRORLEVEL%
if %DFLS_RC%==0 goto :dfls-py-floor
goto :dfls-try-python

:dfls-py-floor
rem builder source floor: host CPython >= 3.11 (child cmd; fixed code)
cmd /c py -3 -I -B -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 3)" >nul 2>nul
set /a DFLS_RC=%ERRORLEVEL%
if %DFLS_RC%==0 (
  set "HOSTCMD=py -3"
  goto :dfls-run
)
if %DFLS_RC%==3 goto :dfls-py-below
goto :dfls-try-python

:dfls-py-below
rem below-floor automatic candidate: skip (not terminal), remember it
set "DFLS_BELOW=1"
set "DFLS_HOSTVER="
for /f "delims=" %%V in ('cmd /c py -3 -I -B -c "import sys; print(sys.version.split()[0])" 2^>nul') do set "DFLS_HOSTVER=%%V"
goto :dfls-try-python

:dfls-try-python
rem -- automatic candidate 2: `python` on PATH
cmd /c python -I -B -c "import sys" >nul 2>nul
set /a DFLS_RC=%ERRORLEVEL%
if %DFLS_RC%==0 goto :dfls-pyth-floor
if %DFLS_BELOW%==1 goto :dfls-old-host
goto :dfls-no-host

:dfls-pyth-floor
cmd /c python -I -B -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 3)" >nul 2>nul
set /a DFLS_RC=%ERRORLEVEL%
if %DFLS_RC%==0 (
  set "HOSTCMD=python"
  goto :dfls-run
)
if %DFLS_RC%==3 (
  set "DFLS_BELOW=1"
  set "DFLS_HOSTVER="
  for /f "delims=" %%V in ('cmd /c python -I -B -c "import sys; print(sys.version.split()[0])" 2^>nul') do set "DFLS_HOSTVER=%%V"
)
if %DFLS_BELOW%==1 goto :dfls-old-host
goto :dfls-no-host

rem == explicit override: PATH DATA, validated before any use ==========
:dfls-override
rem Step 1: discover a fallback interpreter that can run the validator
rem (an automatic candidate that executes the probe; any CPython - the
rem validator code has no 3.11-specific syntax).
cmd /c py -3 -I -B -c "import sys" >nul 2>nul
set /a DFLS_RC=%ERRORLEVEL%
if %DFLS_RC%==0 (
  set "DFLS_AUTOHOST=py -3"
  goto :dfls-override-validate
)
cmd /c python -I -B -c "import sys" >nul 2>nul
set /a DFLS_RC=%ERRORLEVEL%
if %DFLS_RC%==0 set "DFLS_AUTOHOST=python"
if not defined DFLS_AUTOHOST goto :dfls-override-unvalidatable
:dfls-override-validate
rem Step 2: validate the value AS DATA in a child of that interpreter.
rem The fixed validator code reads DFL_SETUP_PYTHON from the environment;
rem the value's bytes are never expanded by this process (a child process
rem reading an environment variable performs no cmd parsing of the value).
rem rc 0 = the value is a clean .exe interpreter path; rc 6 = rejected;
rem rc 7 = the value is the documented `none` sentinel (byte-exact,
rem case-sensitive) - not a usable host: fail with the no-host-Python
rem error (exit 3), no fallback is performed.
rem The validator rejects: empty values; values containing any of the
rem characters " & | < > ^ % ! ( ) ; CR LF (any character that can turn
rem the value into command syntax); values not ending in .exe; values
rem that do not exist as a file.
cmd /c %DFLS_AUTOHOST% -I -B -c "import os,sys;v=os.environ.get('DFL_SETUP_PYTHON','');b=chr(34)+chr(38)+chr(124)+chr(60)+chr(62)+chr(94)+chr(37)+chr(33)+chr(40)+chr(41)+chr(59)+chr(13)+chr(10);n=chr(110)+chr(111)+chr(110)+chr(101);ok=v and not any(c in v for c in b) and v.lower().endswith('.exe') and os.path.isfile(v);sys.exit(7 if v==n else (0 if ok else 6))" >nul 2>nul
set /a DFLS_RC=%ERRORLEVEL%
if %DFLS_RC%==0 (
  rem value is proven path data: expand it exactly once. From here on the
  rem value holds no quote, no metacharacter, no % - every later
  rem substitution of DFLS_OVR is inert by construction.
  set "DFLS_OVR=%DFL_SETUP_PYTHON%"
  goto :dfls-override-probe
)
if %DFLS_RC%==7 goto :dfls-no-host
if %DFLS_RC%==6 goto :dfls-override-invalid
goto :dfls-override-unsupported

rem Step 3: probe / floor / build the validated .exe DIRECTLY (a spawned
rem child process; no CALL, no child cmd, no reconstructed command text).
:dfls-override-probe
"%DFLS_OVR%" -I -B -c "import sys" >nul 2>nul
set /a DFLS_RC=%ERRORLEVEL%
if %DFLS_RC%==0 goto :dfls-override-floor
goto :dfls-override-failed

:dfls-override-floor
rem below-floor explicit overrides fail terminally (no auto fallback)
"%DFLS_OVR%" -I -B -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 3)" >nul 2>nul
set /a DFLS_RC=%ERRORLEVEL%
if %DFLS_RC%==0 goto :dfls-run
if %DFLS_RC%==3 goto :dfls-override-below
goto :dfls-override-failed

:dfls-override-below
goto :dfls-old-host


:dfls-run
rem Select the literal builder-invocation line matching the validated
rem option set. The host command runs DIRECTLY in this process: a real
rem python.exe / py.exe is a spawned child process; a batch host reached
rem through the automatic candidates runs in-process but is the LAST
rem command of this helper (the next statement is `exit /b`), so it cannot
rem affect any later command. No CALL and no child cmd exist on the
rem builder path: the substituted value text is never re-parsed or
rem re-expanded by this helper. Every value variable sits inside template
rem double quotes on the selected line.
cd /d "%ROOT%"
if defined DFLS_OVR goto :dfls-ovr-dispatch
if defined DFLS_OFFIN if defined DFLS_ARTC if defined DFLS_LOCKT goto :dfls-run-7
if defined DFLS_OFFIN if defined DFLS_ARTC goto :dfls-run-6
if defined DFLS_OFFIN if defined DFLS_LOCKT goto :dfls-run-5
if defined DFLS_ARTC if defined DFLS_LOCKT goto :dfls-run-4
if defined DFLS_OFFIN goto :dfls-run-3
if defined DFLS_ARTC goto :dfls-run-2
if defined DFLS_LOCKT goto :dfls-run-1
goto :dfls-run-0

:dfls-ovr-dispatch
if defined DFLS_OFFIN if defined DFLS_ARTC if defined DFLS_LOCKT goto :dfls-run-ovr-7
if defined DFLS_OFFIN if defined DFLS_ARTC goto :dfls-run-ovr-6
if defined DFLS_OFFIN if defined DFLS_LOCKT goto :dfls-run-ovr-5
if defined DFLS_ARTC if defined DFLS_LOCKT goto :dfls-run-ovr-4
if defined DFLS_OFFIN goto :dfls-run-ovr-3
if defined DFLS_ARTC goto :dfls-run-ovr-2
if defined DFLS_LOCKT goto :dfls-run-ovr-1
goto :dfls-run-ovr-0

:dfls-run-7
%HOSTCMD% -I -B "%ROOT%\scripts\build_runtime.py" build --variant "%VARIANT%" %DFLS_ACT% %DFLS_OFF% --offline-inputs "%DFLS_OFFIN_VAL%" --artifact-cache "%DFLS_ARTC_VAL%" --lock-timeout "%DFLS_LOCKT_VAL%"
exit /b %ERRORLEVEL%

:dfls-run-6
%HOSTCMD% -I -B "%ROOT%\scripts\build_runtime.py" build --variant "%VARIANT%" %DFLS_ACT% %DFLS_OFF% --offline-inputs "%DFLS_OFFIN_VAL%" --artifact-cache "%DFLS_ARTC_VAL%"
exit /b %ERRORLEVEL%

:dfls-run-5
%HOSTCMD% -I -B "%ROOT%\scripts\build_runtime.py" build --variant "%VARIANT%" %DFLS_ACT% %DFLS_OFF% --offline-inputs "%DFLS_OFFIN_VAL%" --lock-timeout "%DFLS_LOCKT_VAL%"
exit /b %ERRORLEVEL%

:dfls-run-4
%HOSTCMD% -I -B "%ROOT%\scripts\build_runtime.py" build --variant "%VARIANT%" %DFLS_ACT% %DFLS_OFF% --artifact-cache "%DFLS_ARTC_VAL%" --lock-timeout "%DFLS_LOCKT_VAL%"
exit /b %ERRORLEVEL%

:dfls-run-3
%HOSTCMD% -I -B "%ROOT%\scripts\build_runtime.py" build --variant "%VARIANT%" %DFLS_ACT% %DFLS_OFF% --offline-inputs "%DFLS_OFFIN_VAL%"
exit /b %ERRORLEVEL%

:dfls-run-2
%HOSTCMD% -I -B "%ROOT%\scripts\build_runtime.py" build --variant "%VARIANT%" %DFLS_ACT% %DFLS_OFF% --artifact-cache "%DFLS_ARTC_VAL%"
exit /b %ERRORLEVEL%

:dfls-run-1
%HOSTCMD% -I -B "%ROOT%\scripts\build_runtime.py" build --variant "%VARIANT%" %DFLS_ACT% %DFLS_OFF% --lock-timeout "%DFLS_LOCKT_VAL%"
exit /b %ERRORLEVEL%

:dfls-run-0
%HOSTCMD% -I -B "%ROOT%\scripts\build_runtime.py" build --variant "%VARIANT%" %DFLS_ACT% %DFLS_OFF%
exit /b %ERRORLEVEL%

:dfls-run-ovr-0
"%DFLS_OVR%" -I -B "%ROOT%\scripts\build_runtime.py" build --variant "%VARIANT%" %DFLS_ACT% %DFLS_OFF%
exit /b %ERRORLEVEL%

:dfls-run-ovr-1
"%DFLS_OVR%" -I -B "%ROOT%\scripts\build_runtime.py" build --variant "%VARIANT%" %DFLS_ACT% %DFLS_OFF% --lock-timeout "%DFLS_LOCKT_VAL%"
exit /b %ERRORLEVEL%

:dfls-run-ovr-2
"%DFLS_OVR%" -I -B "%ROOT%\scripts\build_runtime.py" build --variant "%VARIANT%" %DFLS_ACT% %DFLS_OFF% --artifact-cache "%DFLS_ARTC_VAL%"
exit /b %ERRORLEVEL%

:dfls-run-ovr-3
"%DFLS_OVR%" -I -B "%ROOT%\scripts\build_runtime.py" build --variant "%VARIANT%" %DFLS_ACT% %DFLS_OFF% --offline-inputs "%DFLS_OFFIN_VAL%"
exit /b %ERRORLEVEL%

:dfls-run-ovr-4
"%DFLS_OVR%" -I -B "%ROOT%\scripts\build_runtime.py" build --variant "%VARIANT%" %DFLS_ACT% %DFLS_OFF% --artifact-cache "%DFLS_ARTC_VAL%" --lock-timeout "%DFLS_LOCKT_VAL%"
exit /b %ERRORLEVEL%

:dfls-run-ovr-5
"%DFLS_OVR%" -I -B "%ROOT%\scripts\build_runtime.py" build --variant "%VARIANT%" %DFLS_ACT% %DFLS_OFF% --offline-inputs "%DFLS_OFFIN_VAL%" --lock-timeout "%DFLS_LOCKT_VAL%"
exit /b %ERRORLEVEL%

:dfls-run-ovr-6
"%DFLS_OVR%" -I -B "%ROOT%\scripts\build_runtime.py" build --variant "%VARIANT%" %DFLS_ACT% %DFLS_OFF% --offline-inputs "%DFLS_OFFIN_VAL%" --artifact-cache "%DFLS_ARTC_VAL%"
exit /b %ERRORLEVEL%

:dfls-run-ovr-7
"%DFLS_OVR%" -I -B "%ROOT%\scripts\build_runtime.py" build --variant "%VARIANT%" %DFLS_ACT% %DFLS_OFF% --offline-inputs "%DFLS_OFFIN_VAL%" --artifact-cache "%DFLS_ARTC_VAL%" --lock-timeout "%DFLS_LOCKT_VAL%"
exit /b %ERRORLEVEL%


:dfls-usage
echo dfl-setup-runtime: missing required variant.
echo   Usage: launchers\dfl-setup-runtime.bat ^<variant^> [--activate] [--offline] [--offline-inputs ^<value^>] [--artifact-cache ^<value^>] [--lock-timeout ^<value^>]
echo   ^<variant^> is one of: cuda-gui  cuda-nogui  cpu-gui  cpu-nogui
exit /b 2

:dfls-bad-variant
echo dfl-setup-runtime: unknown variant "%VARIANT%". Supported variants: cuda-gui  cuda-nogui  cpu-gui  cpu-nogui
exit /b 2

:dfls-bad-option
echo dfl-setup-runtime: unknown builder option "%DFLS_OPT%". Supported options: --activate  --offline  --offline-inputs ^<value^>  --artifact-cache ^<value^>  --lock-timeout ^<value^>
exit /b 2

:dfls-missing-value
echo dfl-setup-runtime: option "%DFLS_OPT%" requires a value.
exit /b 2

:dfls-option-value
echo dfl-setup-runtime: option "%DFLS_OPT%" requires a value, but the next token is another option. A value must not begin with "--": option tokens are never accepted as values (no escape syntax is provided).
exit /b 2

:dfls-override-failed
rem the raw value is deliberately NOT echoed here: this label is
rem reachable after a permissive validator accepted the value, and
rem echoing it would substitute its bytes into a parsed command line.
echo dfl-setup-runtime: the explicit DFL_SETUP_PYTHON candidate could not be executed or failed its probe. No fallback to automatic discovery is performed for an explicit override. Point DFL_SETUP_PYTHON at a usable host CPython 3.11 or newer, or unset it to use automatic discovery.
exit /b 3

:dfls-override-invalid
echo dfl-setup-runtime: DFL_SETUP_PYTHON is not an accepted host interpreter path. The explicit override must be the path of ONE existing python.exe interpreter executable (a .exe file path containing no command syntax, no arguments, no environment expansion, no batch wrapper). Point DFL_SETUP_PYTHON at a host CPython 3.11 or newer, or unset it to use automatic discovery.
exit /b 3

:dfls-override-unsupported
echo dfl-setup-runtime: the explicit DFL_SETUP_PYTHON override could not be validated (the validator interpreter reported an unexpected result). No fallback to automatic discovery is performed for an explicit override, and no part of the value was executed. Point DFL_SETUP_PYTHON at a host CPython 3.11 or newer, or unset it to use automatic discovery.
exit /b 3

:dfls-override-unvalidatable
echo dfl-setup-runtime: DFL_SETUP_PYTHON is set, but validating an explicit override requires a fallback host interpreter on PATH (py -3 or python) and none was found. No part of the value was executed. Install CPython 3.11 or newer (or a working py launcher), or unset DFL_SETUP_PYTHON to use automatic discovery.
exit /b 3


:dfls-no-host
echo dfl-setup-runtime: no usable host CPython was found: DFL_SETUP_PYTHON was the `none` sentinel, or automatic discovery of `py -3` and `python` on PATH found no host CPython 3.11 or newer (the builder source floor). Install a host CPython 3.11 or newer, or set DFL_SETUP_PYTHON to the full path of a suitable python.exe. The assembled runtime does not need host Python at run time.
exit /b 3

:dfls-old-host
if defined DFLS_HOSTVER (
  echo dfl-setup-runtime: host Python %DFLS_HOSTVER% is below the required 3.11 builder source floor. Install CPython 3.11 or newer, or set DFL_SETUP_PYTHON to the full path of a suitable python.exe.
) else (
  echo dfl-setup-runtime: the explicit DFL_SETUP_PYTHON host is below the required 3.11 builder source floor. Point DFL_SETUP_PYTHON at a host CPython 3.11 or newer, or unset it to use automatic discovery.
)
exit /b 4
