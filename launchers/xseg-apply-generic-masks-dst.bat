@echo off
rem ============================================================================
rem xseg-apply-generic-masks-dst - apply masks predicted by a generic XSeg model to data_dst faces
rem
rem Curated workflow launcher (Phase-13 PRE_BASELINE;
rem docs/IMPLEMENTATION_PLAN_v3.md section 18). Thin wrapper
rem around the shared dfl.bat launcher: this file never runs an
rem interpreter itself; selector resolution, runtime isolation,
rem environment sanitation, argument forwarding and exit-code
rem propagation are all inherited from dfl.bat.
rem
rem Mapped command:
rem   xseg apply --input-dir "workspace\data_dst\aligned" --model-dir "resources\xseg_generic_model"
rem
rem Resource policy (P13-GENERIC-XSEG-RESOURCE-POLICY):
rem The generic XSeg model is a USER-PROVIDED resource at the documented
rem host-local, untracked location resources\xseg_generic_model (relative to
rem the repository root). This distribution bundles no XSeg model bytes,
rem redistributes no historical DeepFaceLab model pack, and performs no
rem downloads: place a compatible model directory you have obtained and
rem verified yourself there (it must contain XSeg_256.npy, optionally
rem XSeg_data.dat), or run the CLI with your own compatible --model-dir.
rem Without the resource the command fails with an actionable diagnostic
rem (exit code 1); it never silently substitutes another model location.
rem ============================================================================
call "%~dp0dfl.bat" xseg apply --input-dir "workspace\data_dst\aligned" --model-dir "resources\xseg_generic_model"
exit /b %ERRORLEVEL%
