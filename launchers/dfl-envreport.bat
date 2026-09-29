@echo off
rem ============================================================================
rem dfl-envreport.bat - diagnostic alias for the runtime environment report.
rem
rem Thin batch-to-batch alias (docs/PHASE13_STATE.md section 17): it hands
rem the `envreport` command to the normal launcher chain, so the report runs
rem under the SAME isolated chain as the application - the selected
rem runtime's bundled interpreter, -I -B isolation, sanitized environment,
rem argument forwarding and exit-code propagation are all inherited from
rem dfl.bat. `call` is used because batch-to-batch return propagation is
rem required.
rem
rem The report command itself is provided by the runtime entry point
rem (environment-report commit); until then the runtime entry point rejects
rem the unknown command and the alias propagates that failure.
rem ============================================================================
call "%~dp0dfl.bat" envreport %*
exit /b %ERRORLEVEL%
