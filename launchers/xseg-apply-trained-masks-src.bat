@echo off
rem ============================================================================
rem xseg-apply-trained-masks-src - apply masks predicted by the user-trained XSeg model to data_src faces
rem
rem Curated workflow launcher (Phase-13 PRE_BASELINE;
rem docs/IMPLEMENTATION_PLAN_v3.md section 18). Thin wrapper
rem around the shared dfl.bat launcher: this file never runs an
rem interpreter itself; selector resolution, runtime isolation,
rem environment sanitation, argument forwarding and exit-code
rem propagation are all inherited from dfl.bat.
rem
rem Mapped command:
rem   xseg apply --input-dir "workspace\data_src\aligned" --model-dir "workspace\model"
rem
rem The model is the user's own trained model in workspace\model (user-supplied
rem resource); no bundled model pack is assumed.
rem ============================================================================
call "%~dp0dfl.bat" xseg apply --input-dir "workspace\data_src\aligned" --model-dir "workspace\model"
exit /b %ERRORLEVEL%
