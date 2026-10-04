@echo off
rem ============================================================================
rem train-quick96 - train the Quick96 model from aligned src/dst facesets
rem
rem Curated workflow launcher. Thin wrapper around the shared dfl.bat
rem launcher; packaged-runtime selection, environment isolation and exit-code
rem propagation are inherited from that launcher.
rem
rem Mapped command:
rem   train --training-data-src-dir "workspace\data_src\aligned" --training-data-dst-dir "workspace\data_dst\aligned" --model-dir "workspace\model" --model Quick96 --no-preview --silent-start
rem
rem No pretrained model is assumed. Users with optional pretrained Quick96
rem components or explicit device selection can use the generic dfl.bat CLI.
rem ============================================================================
call "%~dp0dfl.bat" train --training-data-src-dir "workspace\data_src\aligned" --training-data-dst-dir "workspace\data_dst\aligned" --model-dir "workspace\model" --model Quick96 --no-preview --silent-start
exit /b %ERRORLEVEL%
