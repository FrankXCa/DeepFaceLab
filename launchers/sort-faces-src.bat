@echo off
rem ============================================================================
rem sort-faces-src - sort the aligned data_src faces by histogram similarity
rem
rem Curated workflow launcher (Phase-13 PRE_BASELINE;
rem docs/IMPLEMENTATION_PLAN_v3.md section 18). Thin wrapper
rem around the shared dfl.bat launcher: this file never runs an
rem interpreter itself; selector resolution, runtime isolation,
rem environment sanitation, argument forwarding and exit-code
rem propagation are all inherited from dfl.bat.
rem
rem Mapped command:
rem   sort --input-dir "workspace\data_src\aligned" --by hist
rem
rem --by hist pins the sort method. The reference launcher omitted --by and
rem showed an interactive method menu; hist is the method the sort CLI itself
rem selects as its unattended default (method id 5).
rem ============================================================================
call "%~dp0dfl.bat" sort --input-dir "workspace\data_src\aligned" --by hist
exit /b %ERRORLEVEL%
