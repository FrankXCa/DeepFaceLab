@echo off
rem ============================================================================
rem extract-faces-src - extract aligned data_src faces with the S3FD detector
rem
rem Curated workflow launcher (Phase-13 PRE_BASELINE;
rem docs/IMPLEMENTATION_PLAN_v3.md section 18). Thin wrapper
rem around the shared dfl.bat launcher: this file never runs an
rem interpreter itself; selector resolution, runtime isolation,
rem environment sanitation, argument forwarding and exit-code
rem propagation are all inherited from dfl.bat.
rem
rem Mapped command:
rem   extract --input-dir "workspace\data_src" --output-dir "workspace\data_src\aligned" --detector s3fd
rem
rem Uses the tracked S3FD detector weights from the facelib package.
rem ============================================================================
call "%~dp0dfl.bat" extract --input-dir "workspace\data_src" --output-dir "workspace\data_src\aligned" --detector s3fd
exit /b %ERRORLEVEL%
