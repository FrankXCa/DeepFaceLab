@echo off
rem ============================================================================
rem result-video-mov-lossless - encode the merged data_dst faces into a lossless mov result video plus a lossless mask video
rem
rem Curated workflow launcher (Phase-13 PRE_BASELINE;
rem docs/IMPLEMENTATION_PLAN_v3.md section 18). Thin wrapper
rem around the shared dfl.bat launcher: this file never runs an
rem interpreter itself; selector resolution, runtime isolation,
rem environment sanitation, argument forwarding and exit-code
rem propagation are all inherited from dfl.bat.
rem
rem Mapped commands:
rem   videoed video-from-sequence --input-dir "workspace\data_dst\merged" --output-file "workspace\result.mov" --reference-file "workspace\data_dst.*" --include-audio --lossless
rem   videoed video-from-sequence --input-dir "workspace\data_dst\merged_mask" --output-file "workspace\result_mask.mov" --reference-file "workspace\data_dst.*" --lossless
rem
rem Two commands run in order; if the first fails, the second is skipped and
rem the first exit code is propagated. Both videos are encoded lossless; the
rem wildcard --reference-file supplies the frame timing and the original audio
rem track.
rem ============================================================================
call "%~dp0dfl.bat" videoed video-from-sequence --input-dir "workspace\data_dst\merged" --output-file "workspace\result.mov" --reference-file "workspace\data_dst.*" --include-audio --lossless
if errorlevel 1 exit /b %ERRORLEVEL%
call "%~dp0dfl.bat" videoed video-from-sequence --input-dir "workspace\data_dst\merged_mask" --output-file "workspace\result_mask.mov" --reference-file "workspace\data_dst.*" --lossless
exit /b %ERRORLEVEL%
