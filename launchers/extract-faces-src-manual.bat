@echo off
rem ============================================================================
rem extract-faces-src-manual - extract aligned data_src faces manually with the interactive window
rem
rem Curated workflow launcher (Phase-13 PRE_BASELINE;
rem docs/IMPLEMENTATION_PLAN_v3.md section 18). Thin wrapper
rem around the shared dfl.bat launcher: this file never runs an
rem interpreter itself; selector resolution, runtime isolation,
rem environment sanitation, argument forwarding and exit-code
rem propagation are all inherited from dfl.bat.
rem
rem Mapped command:
rem   extract --input-dir "workspace\data_src" --output-dir "workspace\data_src\aligned" --detector manual
rem
rem Manual mode opens the interactive face-window UI; no detector weights are
rem required.
rem ============================================================================
call "%~dp0dfl.bat" extract --input-dir "workspace\data_src" --output-dir "workspace\data_src\aligned" --detector manual
exit /b %ERRORLEVEL%
