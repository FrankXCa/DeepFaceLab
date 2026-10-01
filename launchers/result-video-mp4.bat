@echo off
rem ============================================================================
rem result-video-mp4 - encode the merged data_dst faces into the final mp4 result video plus a lossless mask video
rem
rem Curated workflow launcher (Phase-13 PRE_BASELINE;
rem docs/IMPLEMENTATION_PLAN_v3.md section 18). Thin wrapper
rem around the shared dfl.bat launcher: this file never runs an
rem interpreter itself; selector resolution, runtime isolation,
rem environment sanitation, argument forwarding and exit-code
rem propagation are all inherited from dfl.bat.
rem
rem Mapped commands:
rem   videoed video-from-sequence --input-dir "workspace\data_dst\merged" --output-file "workspace\result.mp4" --reference-file "workspace\data_dst.*" --include-audio
rem   videoed video-from-sequence --input-dir "workspace\data_dst\merged_mask" --output-file "workspace\result_mask.mp4" --reference-file "workspace\data_dst.*" --lossless
rem
rem Two commands run in order; if the first fails, the second is skipped and
rem the first exit code is propagated. The mask video is always encoded
rem lossless; the wildcard --reference-file supplies the frame timing and the
rem original audio track.
rem ============================================================================
call "%~dp0dfl.bat" videoed video-from-sequence --input-dir "workspace\data_dst\merged" --output-file "workspace\result.mp4" --reference-file "workspace\data_dst.*" --include-audio
if errorlevel 1 exit /b %ERRORLEVEL%
call "%~dp0dfl.bat" videoed video-from-sequence --input-dir "workspace\data_dst\merged_mask" --output-file "workspace\result_mask.mp4" --reference-file "workspace\data_dst.*" --lossless
exit /b %ERRORLEVEL%
