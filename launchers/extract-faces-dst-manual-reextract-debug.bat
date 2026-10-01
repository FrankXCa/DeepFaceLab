@echo off
rem ============================================================================
rem extract-faces-dst-manual-reextract-debug - manually re-extract deleted aligned_debug frames for data_dst
rem
rem Curated workflow launcher (Phase-13 PRE_BASELINE;
rem docs/IMPLEMENTATION_PLAN_v3.md section 18). Thin wrapper
rem around the shared dfl.bat launcher: this file never runs an
rem interpreter itself; selector resolution, runtime isolation,
rem environment sanitation, argument forwarding and exit-code
rem propagation are all inherited from dfl.bat.
rem
rem Mapped command:
rem   extract --input-dir "workspace\data_dst" --output-dir "workspace\data_dst\aligned" --detector manual --max-faces-from-image 0 --output-debug --manual-output-debug-fix
rem
rem Re-extracts frames that were deleted from aligned_debug via the manual
rem output-debug-fix window.
rem ============================================================================
call "%~dp0dfl.bat" extract --input-dir "workspace\data_dst" --output-dir "workspace\data_dst\aligned" --detector manual --max-faces-from-image 0 --output-debug --manual-output-debug-fix
exit /b %ERRORLEVEL%
