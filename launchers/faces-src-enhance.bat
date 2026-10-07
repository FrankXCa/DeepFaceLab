@echo off
rem ============================================================================
rem faces-src-enhance - enhance the aligned data_src faces with FaceEnhancer
rem
rem Thin wrapper around the shared dfl.bat packaged-runtime launcher. This file
rem never selects or runs an interpreter itself and intentionally accepts no
rem forwarded arguments. Use dfl.bat directly for advanced device selection.
rem
rem Mapped command:
rem   facesettool enhance --input-dir "workspace\data_src\aligned"
rem
rem Paths are relative to the repository root (dfl.bat sets the working
rem directory), so this launcher is safe to invoke from any current directory.
rem ============================================================================
call "%~dp0dfl.bat" facesettool enhance --input-dir "workspace\data_src\aligned"
exit /b %ERRORLEVEL%
