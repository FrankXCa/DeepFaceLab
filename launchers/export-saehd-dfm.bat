@echo off
rem ============================================================================
rem export-saehd-dfm - export the trained SAEHD model as a .dfm for external tools
rem
rem Curated workflow launcher (Phase-13 PRE_BASELINE;
rem docs/IMPLEMENTATION_PLAN_v3.md section 18). Thin wrapper
rem around the shared dfl.bat launcher: this file never runs an
rem interpreter itself; selector resolution, runtime isolation,
rem environment sanitation, argument forwarding and exit-code
rem propagation are all inherited from dfl.bat.
rem
rem Mapped command:
rem   exportdfm --model-dir "workspace\model" --model SAEHD
rem
rem Exports from the user's trained model directory (user-supplied resource).
rem ============================================================================
call "%~dp0dfl.bat" exportdfm --model-dir "workspace\model" --model SAEHD
exit /b %ERRORLEVEL%
