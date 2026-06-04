# vkperf

Small Vulkan compute benchmark CLI 
These test suites are from different places: some are extracted from [GPUPerfTests](https://github.com/ChipsandCheese/gpuperftests), some are written by myself.

The program writes benchmark results to stdout as CSV by default. Diagnostic logs go to stderr, so scripts can redirect stdout directly to a CSV file.

## Build on Windows

```powershell
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
cmake -S . -B build-msvc-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-msvc-release
```

If CMake cannot find Vulkan, install the Vulkan SDK and make sure `VULKAN_SDK` is visible in the shell.

## Build for Android

```powershell
cmake -S . -B build-android `
  -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE=$env:ANDROID_NDK_HOME\build\cmake\android.toolchain.cmake `
  -DANDROID_ABI=arm64-v8a `
  -DANDROID_PLATFORM=android-28
cmake --build build-android
```

Push and run:

```powershell
adb push build-android\vkperf /data/local/tmp/vkperf
adb shell chmod +x /data/local/tmp/vkperf
adb shell /data/local/tmp/vkperf --test vk_list
```

## Usage

```powershell
vkperf --list-tests
vkperf --test vk_list
vkperf --test vk_info --device 0
vkperf --test vk_cuprobe --device 0 > cuprobe.csv
vkperf --test vk_resident_wave_slots --device 0 > resident_wave_slots.csv
vkperf --test vk_rate_fp32_add --device 0 > result.csv
```

Options:

- `--test`, `-t`: test id to run.
- `--device`, `-d`: Vulkan physical device index. Default `-1` runs matching benchmark tests on all devices.
- `--format`, `-f`: `csv`, `raw`, or `readable`. Default `csv`.
- `--list-tests`: print available test IDs to stderr.
