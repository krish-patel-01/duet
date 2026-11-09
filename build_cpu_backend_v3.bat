@echo off
rem Builds all 14 x64 ggml CPU backend variants from third_party\llama.cpp-b11969\ggml (llama.cpp
rem b11969 = v0.8.0 + our decode attention kernel, patches\fast-decode-attention-b11969.patch) into
rem build-cpu-v3\, using the GNU-style clang driver (clang-cl skips 5 variants).
rem Output DLLs are copied to build-cpu-v3\variants\.
setlocal
set ROOT=%~dp0
set GG=%ROOT%cpu-backend-v3
set BLD=%ROOT%build-cpu-v3
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set CMAKE=%VSINSTALLDIR%Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set NINJA=%VSINSTALLDIR%Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe
set LLVM=%VCINSTALLDIR%Tools\Llvm\x64
set LLVM=%LLVM:\=/%
if not exist "%BLD%\build.ninja" (
  "%CMAKE%" -S "%GG%" -B "%BLD%" -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="%LLVM%/bin/clang.exe" -DCMAKE_CXX_COMPILER="%LLVM%/bin/clang++.exe" ^
    -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=ON -DGGML_LLAMAFILE=ON ^
    -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF ^
    "-DOpenMP_C_FLAGS=-fopenmp" "-DOpenMP_CXX_FLAGS=-fopenmp" ^
    -DOpenMP_C_LIB_NAMES=libomp -DOpenMP_CXX_LIB_NAMES=libomp -DOpenMP_libomp_LIBRARY="%LLVM%/lib/libomp.lib" ^
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,/MANIFEST:NO -DCMAKE_SHARED_LINKER_FLAGS=-Wl,/MANIFEST:NO -DCMAKE_MODULE_LINKER_FLAGS=-Wl,/MANIFEST:NO || exit /b 1
)
set TARGETS=ggml-cpu-x64 ggml-cpu-sse42 ggml-cpu-sandybridge ggml-cpu-ivybridge ggml-cpu-piledriver ggml-cpu-haswell ggml-cpu-skylakex ggml-cpu-cannonlake ggml-cpu-cascadelake ggml-cpu-icelake ggml-cpu-cooperlake ggml-cpu-zen4 ggml-cpu-alderlake ggml-cpu-sapphirerapids
if not "%1"=="" set TARGETS=%*
"%CMAKE%" --build "%BLD%" --target %TARGETS% || exit /b 1
if not exist "%BLD%\variants" mkdir "%BLD%\variants"
copy /y "%BLD%\bin\ggml-cpu-*.dll" "%BLD%\variants\" >nul
echo built into build-cpu-v3\variants
