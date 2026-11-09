@echo off
rem Builds hybrid.exe into bin\ (needs Visual Studio 2022 Build Tools with clang).
setlocal
set ROOT=%~dp0
set LL=%ROOT%third_party\llama.cpp
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
rem Import libraries for the llama.cpp DLLs in bin\, made from the .def files in lib\.
for %%n in (llama ggml ggml-base) do (
  if not exist "%ROOT%lib\%%n.lib" lib /nologo /machine:x64 /def:"%ROOT%lib\%%n.def" /out:"%ROOT%lib\%%n.lib" || exit /b 1
)
"%VCINSTALLDIR%Tools\Llvm\x64\bin\clang-cl.exe" /nologo /O2 /EHsc /std:c++17 /W3 ^
  /I"%LL%\include" /I"%LL%\ggml\include" ^
  "%ROOT%src\hybrid.cpp" /Fo"%ROOT%lib\hybrid.obj" /Fe"%ROOT%bin\hybrid.exe" ^
  /link /LIBPATH:"%ROOT%lib" llama.lib ggml.lib ggml-base.lib || exit /b 1
echo built %ROOT%bin\hybrid.exe
