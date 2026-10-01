@echo off
rem ============================================================================
rem extract-frames-src - decode the data_src source video into raw frame images
rem
rem Curated workflow launcher (Phase-13 PRE_BASELINE;
rem docs/IMPLEMENTATION_PLAN_v3.md section 18). Thin wrapper
rem around the shared dfl.bat launcher: this file never runs an
rem interpreter itself; selector resolution, runtime isolation,
rem environment sanitation, argument forwarding and exit-code
rem propagation are all inherited from dfl.bat.
rem
rem Mapped command:
rem   videoed extract-video --input-file "workspace\data_src.*" --output-dir "workspace\data_src"
rem
rem The wildcard --input-file value selects the first data_src video in the
rem workspace root; output frames go to workspace\data_src at the video's native
rem frame rate (no --fps pin).
rem ============================================================================
call "%~dp0dfl.bat" videoed extract-video --input-file "workspace\data_src.*" --output-dir "workspace\data_src"
exit /b %ERRORLEVEL%
