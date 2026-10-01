@echo off
rem ============================================================================
rem extract-frames-dst-full-fps - decode the data_dst target video into raw frame images at full frame rate
rem
rem Curated workflow launcher (Phase-13 PRE_BASELINE;
rem docs/IMPLEMENTATION_PLAN_v3.md section 18). Thin wrapper
rem around the shared dfl.bat launcher: this file never runs an
rem interpreter itself; selector resolution, runtime isolation,
rem environment sanitation, argument forwarding and exit-code
rem propagation are all inherited from dfl.bat.
rem
rem Mapped command:
rem   videoed extract-video --input-file "workspace\data_dst.*" --output-dir "workspace\data_dst" --fps 0
rem
rem The wildcard --input-file value selects the first data_dst video in the
rem workspace root; output frames go to workspace\data_dst. --fps 0 pins full
rem frame rate explicitly (the launcher's distinguishing behavior).
rem ============================================================================
call "%~dp0dfl.bat" videoed extract-video --input-file "workspace\data_dst.*" --output-dir "workspace\data_dst" --fps 0
exit /b %ERRORLEVEL%
