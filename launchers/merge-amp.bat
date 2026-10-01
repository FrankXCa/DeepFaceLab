@echo off
rem ============================================================================
rem merge-amp - merge aligned data_dst faces with the AMP model into full-size faces (with masks)
rem
rem Curated workflow launcher (Phase-13 PRE_BASELINE;
rem docs/IMPLEMENTATION_PLAN_v3.md section 18). Thin wrapper
rem around the shared dfl.bat launcher: this file never runs an
rem interpreter itself; selector resolution, runtime isolation,
rem environment sanitation, argument forwarding and exit-code
rem propagation are all inherited from dfl.bat.
rem
rem Mapped command:
rem   merge --input-dir "workspace\data_dst" --output-dir "workspace\data_dst\merged" --output-mask-dir "workspace\data_dst\merged_mask" --aligned-dir "workspace\data_dst\aligned" --model-dir "workspace\model" --model AMP
rem
rem Uses the user's trained model directory (user-supplied resource).
rem ============================================================================
call "%~dp0dfl.bat" merge --input-dir "workspace\data_dst" --output-dir "workspace\data_dst\merged" --output-mask-dir "workspace\data_dst\merged_mask" --aligned-dir "workspace\data_dst\aligned" --model-dir "workspace\model" --model AMP
exit /b %ERRORLEVEL%
