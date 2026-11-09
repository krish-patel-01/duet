@echo off
rem Builds the ggml CPU backend variants from third_party\llama.cpp\ggml (with our decode
rem attention kernel) into build-cpu\.
rem   build_cpu_backend.bat            build ggml-cpu-alderlake.dll and install it into bin\
rem   build_cpu_backend.bat all        build all 14 x64 variants (for the portable drive) with the
rem                                    GNU-style clang driver; copies them to build-cpu-all\variants\
rem                                    without touching bin\
rem The prebuilt alderlake DLL is kept as dll-variants\prebuilt.dll.
setlocal
set ROOT=%~dp0
set GG=%ROOT%cpu-backend
set BLD=%ROOT%build-cpu
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set CMAKE=%VSINSTALLDIR%Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set NINJA=%VSINSTALLDIR%Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe
set LLVM=%VCINSTALLDIR%Tools\Llvm\x64
rem cmake wants forward slashes in compiler paths
set LLVM=%LLVM:\=/%
rem The OpenMP_* settings link LLVM's libomp.dll like the prebuilt DLLs (not MSVC's libomp140).
if not exist "%BLD%\build.ninja" (
  "%CMAKE%" -S "%GG%" -B "%BLD%" -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="%LLVM%/bin/clang-cl.exe" -DCMAKE_CXX_COMPILER="%LLVM%/bin/clang-cl.exe" ^
    -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=ON -DGGML_LLAMAFILE=ON ^
    -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF ^
    "-DOpenMP_C_FLAGS=-Xclang -fopenmp" "-DOpenMP_CXX_FLAGS=-Xclang -fopenmp" ^
    -DOpenMP_C_LIB_NAMES=libomp -DOpenMP_CXX_LIB_NAMES=libomp -DOpenMP_libomp_LIBRARY="%LLVM%/lib/libomp.lib" ^
    -DCMAKE_EXE_LINKER_FLAGS=/MANIFEST:NO -DCMAKE_SHARED_LINKER_FLAGS=/MANIFEST:NO -DCMAKE_MODULE_LINKER_FLAGS=/MANIFEST:NO || exit /b 1
)

if /i "%1"=="all" (
  rem All variants need the GNU-style clang driver: with clang-cl, ggml skips ivybridge,
  rem piledriver, cooperlake, zen4 and sapphirerapids (MSVC-only limitations).
  set BLD=%ROOT%build-cpu-all
  if not exist "%ROOT%build-cpu-all\build.ninja" (
    "%CMAKE%" -S "%GG%" -B "%ROOT%build-cpu-all" -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="%LLVM%/bin/clang.exe" -DCMAKE_CXX_COMPILER="%LLVM%/bin/clang++.exe" ^
      -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=ON -DGGML_LLAMAFILE=ON ^
      -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF ^
      "-DOpenMP_C_FLAGS=-fopenmp" "-DOpenMP_CXX_FLAGS=-fopenmp" ^
      -DOpenMP_C_LIB_NAMES=libomp -DOpenMP_CXX_LIB_NAMES=libomp -DOpenMP_libomp_LIBRARY="%LLVM%/lib/libomp.lib" ^
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,/MANIFEST:NO -DCMAKE_SHARED_LINKER_FLAGS=-Wl,/MANIFEST:NO -DCMAKE_MODULE_LINKER_FLAGS=-Wl,/MANIFEST:NO || exit /b 1
  )
  "%CMAKE%" --build "%ROOT%build-cpu-all" --target ggml-cpu-x64 ggml-cpu-sse42 ggml-cpu-sandybridge ggml-cpu-ivybridge ggml-cpu-piledriver ggml-cpu-haswell ggml-cpu-skylakex ggml-cpu-cannonlake ggml-cpu-cascadelake ggml-cpu-icelake ggml-cpu-cooperlake ggml-cpu-zen4 ggml-cpu-alderlake ggml-cpu-sapphirerapids || exit /b 1
  if not exist "%ROOT%build-cpu-all\variants" mkdir "%ROOT%build-cpu-all\variants"
  copy /y "%ROOT%build-cpu-all\bin\ggml-cpu-*.dll" "%ROOT%build-cpu-all\variants\" >nul
  echo built all variants into build-cpu-all\variants
  exit /b 0
)

"%CMAKE%" --build "%BLD%" --target ggml-cpu-alderlake || exit /b 1
rem backups live outside bin\: the backend loader tries every ggml-cpu-*.dll it finds there
if not exist "%ROOT%dll-variants" mkdir "%ROOT%dll-variants"
if not exist "%ROOT%dll-variants\prebuilt.dll" copy /y "%ROOT%bin\ggml-cpu-alderlake.dll" "%ROOT%dll-variants\prebuilt.dll" >nul
for /r "%BLD%\ggml" %%f in (ggml-cpu-alderlake.dll) do if exist "%%f" copy /y "%%f" "%ROOT%bin\ggml-cpu-alderlake.dll" >nul
echo installed bin\ggml-cpu-alderlake.dll
