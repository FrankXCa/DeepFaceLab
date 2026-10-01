@echo off
rem ============================================================================
rem train-amp-src-src - train the AMP model using the aligned data_src faceset as both source and target
rem
rem Curated workflow launcher (Phase-13 PRE_BASELINE;
rem docs/IMPLEMENTATION_PLAN_v3.md section 18). Thin wrapper
rem around the shared dfl.bat launcher: this file never runs an
rem interpreter itself; selector resolution, runtime isolation,
rem environment sanitation, argument forwarding and exit-code
rem propagation are all inherited from dfl.bat.
rem
rem Mapped command:
rem   train --training-data-src-dir "workspace\data_src\aligned" --training-data-dst-dir "workspace\data_src\aligned" --model-dir "workspace\model" --model AMP --no-preview --silent-start
rem
rem SRC-SRC variant: both training directories point at data_src\aligned.
rem No --pretraining-data-dir: the launcher does not assume any bundled
rem pretrain pack; the CLI accepts training without a pretraining set.
rem --no-preview --silent-start keep the run headless for launcher use: no
rem interactive preview window, best device auto-selected, latest saved
rem model resumed. A first run still prompts for the model name and the
rem model options in the console.
rem ============================================================================
call "%~dp0dfl.bat" train --training-data-src-dir "workspace\data_src\aligned" --training-data-dst-dir "workspace\data_src\aligned" --model-dir "workspace\model" --model AMP --no-preview --silent-start
exit /b %ERRORLEVEL%
