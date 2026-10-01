@echo off
rem ============================================================================
rem denoise-frames-dst - denoise the raw data_dst frame sequence
rem
rem Curated workflow launcher (Phase-13 PRE_BASELINE;
rem docs/IMPLEMENTATION_PLAN_v3.md section 18). Thin wrapper
rem around the shared dfl.bat launcher: this file never runs an
rem interpreter itself; selector resolution, runtime isolation,
rem environment sanitation, argument forwarding and exit-code
rem propagation are all inherited from dfl.bat.
rem
rem Mapped command:
rem   videoed denoise-image-sequence --input-dir "workspace\data_dst"
rem
rem Operates on the raw frames in workspace\data_dst (not on aligned faces); the
rem denoise factor is left to the CLI default.
rem ============================================================================
call "%~dp0dfl.bat" videoed denoise-image-sequence --input-dir "workspace\data_dst"
exit /b %ERRORLEVEL%
