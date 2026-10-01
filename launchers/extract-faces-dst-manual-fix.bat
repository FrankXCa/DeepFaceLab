@echo off
rem ============================================================================
rem extract-faces-dst-manual-fix - extract aligned data_dst faces with S3FD, then open the manual-fix window for missed frames
rem
rem Curated workflow launcher (Phase-13 PRE_BASELINE;
rem docs/IMPLEMENTATION_PLAN_v3.md section 18). Thin wrapper
rem around the shared dfl.bat launcher: this file never runs an
rem interpreter itself; selector resolution, runtime isolation,
rem environment sanitation, argument forwarding and exit-code
rem propagation are all inherited from dfl.bat.
rem
rem Mapped command:
rem   extract --input-dir "workspace\data_dst" --output-dir "workspace\data_dst\aligned" --detector s3fd --max-faces-from-image 0 --output-debug --manual-fix
rem
rem After the S3FD pass the manual-fix window lets the user add faces the
rem detector missed.
rem ============================================================================
call "%~dp0dfl.bat" extract --input-dir "workspace\data_dst" --output-dir "workspace\data_dst\aligned" --detector s3fd --max-faces-from-image 0 --output-debug --manual-fix
exit /b %ERRORLEVEL%
