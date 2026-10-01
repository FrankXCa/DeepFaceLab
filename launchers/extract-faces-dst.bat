@echo off
rem ============================================================================
rem extract-faces-dst - extract aligned data_dst faces with the S3FD detector (debug output, unlimited faces per image)
rem
rem Curated workflow launcher (Phase-13 PRE_BASELINE;
rem docs/IMPLEMENTATION_PLAN_v3.md section 18). Thin wrapper
rem around the shared dfl.bat launcher: this file never runs an
rem interpreter itself; selector resolution, runtime isolation,
rem environment sanitation, argument forwarding and exit-code
rem propagation are all inherited from dfl.bat.
rem
rem Mapped command:
rem   extract --input-dir "workspace\data_dst" --output-dir "workspace\data_dst\aligned" --detector s3fd --max-faces-from-image 0 --output-debug
rem
rem --max-faces-from-image 0 means no per-image face cap; --output-debug writes
rem aligned_debug output; uses the tracked S3FD detector weights.
rem ============================================================================
call "%~dp0dfl.bat" extract --input-dir "workspace\data_dst" --output-dir "workspace\data_dst\aligned" --detector s3fd --max-faces-from-image 0 --output-debug
exit /b %ERRORLEVEL%
