@echo off
rem ============================================================================
rem merge-quick96 - merge aligned data_dst faces with the Quick96 model
rem
rem Curated workflow launcher. Thin wrapper around the shared dfl.bat
rem launcher; packaged-runtime selection, environment isolation and exit-code
rem propagation are inherited from that launcher.
rem
rem Mapped command:
rem   merge --input-dir "workspace\data_dst" --output-dir "workspace\data_dst\merged" --output-mask-dir "workspace\data_dst\merged_mask" --aligned-dir "workspace\data_dst\aligned" --model-dir "workspace\model" --model Quick96
rem
rem Uses the user's trained model directory. Explicit device selection remains
rem available through the generic dfl.bat merge command.
rem ============================================================================
call "%~dp0dfl.bat" merge --input-dir "workspace\data_dst" --output-dir "workspace\data_dst\merged" --output-mask-dir "workspace\data_dst\merged_mask" --aligned-dir "workspace\data_dst\aligned" --model-dir "workspace\model" --model Quick96
exit /b %ERRORLEVEL%
