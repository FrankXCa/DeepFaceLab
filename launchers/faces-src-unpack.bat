@echo off
rem ============================================================================
rem faces-src-unpack - unpack a packed data_src faceset back into aligned face images
rem
rem Curated workflow launcher (Phase-13 PRE_BASELINE;
rem docs/IMPLEMENTATION_PLAN_v3.md section 18). Thin wrapper
rem around the shared dfl.bat launcher: this file never runs an
rem interpreter itself; selector resolution, runtime isolation,
rem environment sanitation, argument forwarding and exit-code
rem propagation are all inherited from dfl.bat.
rem
rem Mapped command:
rem   util --input-dir "workspace\data_src\aligned" --unpack-faceset
rem
rem Contract: fixed arguments (no user forwarding). Paths are relative to the
rem repository root (dfl.bat sets the working directory), so this launcher works
rem from any current directory.
rem ============================================================================
call "%~dp0dfl.bat" util --input-dir "workspace\data_src\aligned" --unpack-faceset
exit /b %ERRORLEVEL%
