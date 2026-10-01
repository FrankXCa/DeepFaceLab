@echo off
rem ============================================================================
rem cut-video - cut a dropped video file (optionally trimmed) for later frame extraction
rem
rem Curated workflow launcher (Phase-13 PRE_BASELINE;
rem docs/IMPLEMENTATION_PLAN_v3.md section 18). Thin wrapper
rem around the shared dfl.bat launcher: this file never runs an
rem interpreter itself; selector resolution, runtime isolation,
rem environment sanitation, argument forwarding and exit-code
rem propagation are all inherited from dfl.bat.
rem
rem Mapped command:
rem   videoed cut-video --input-file <dropped-file> [extra cut options]
rem
rem Contract: argument forwarding. Drop a video file on this launcher; its path
rem becomes the --input-file value. Optional cut options may follow and are
rem forwarded unmodified: --from-time, --to-time, --audio-track-id, --bitrate.
rem The launcher forwards all remaining arguments unchanged (the only curated
rem launcher that forwards user arguments).
rem ============================================================================
call "%~dp0dfl.bat" videoed cut-video --input-file %*
exit /b %ERRORLEVEL%
