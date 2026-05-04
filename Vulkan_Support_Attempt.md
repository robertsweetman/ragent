# Vulkan GPU Acceleration Attempt — Snapdragon X Elite (Adreno X1-85)

**Date:** May 2026  
**Hardware:** Microsoft Surface, Snapdragon X Elite X1E80100 (12-core, 3.4 GHz), Adreno X1-85 GPU, 32 GB unified memory  
**OS:** Windows 11 ARM64  
**Goal:** Replace CPU-only llama.cpp inference (via LM Studio) with GPU-accelerated inference via Vulkan, expecting faster token generation for use with ragent.

---

## Motivation

LM Studio on this hardware reports "0 GPUs detected" and runs `llama.cpp-win-arm64` (CPU only).
`lms runtime get -l` confirmed no DirectML or QNN runtime is available in LM Studio 0.4.12.
The Adreno X1-85 supports Vulkan 1.3 natively, so building llama.cpp from source with
`-DGGML_VULKAN=ON` appeared to be the best available path to GPU acceleration.

---

## Prerequisites Discovered

### Vulkan SDK — already installed
```
vulkaninfo --summary
GPU0: Qualcomm(R) Adreno(TM) X1-85 GPU
  apiVersion         = 1.3.295
  driverName         = Qualcomm Technologies Inc. Adreno Vulkan Driver
  conformanceVersion = 1.3.9.1        ← passed conformance tests
  deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
```
The native Qualcomm Vulkan driver was present and conformant. SDK installed at
`C:\VulkanSDK\1.4.341.1\`, `glslc.exe` at `C:\VulkanSDK\1.4.341.1\Bin\glslc.exe`.

### Build tools — partially present
- CMake 3.30.4 ✅
- Ninja 1.12.0 (via Strawberry Perl at `C:\Strawberry\c\bin\ninja.exe`) ✅
- Clang 18.1.8 (standalone LLVM at `C:\Program Files\LLVM\`) ✅
- VS Build Tools 18 (VS 2026 Preview) installed but **ARM64 components missing** initially ✅ after fix

---

## Build Process

### Step 1 — Add ARM64 build tools to VS
The standalone LLVM clang could compile ARM64 code but couldn't link: `msvcrtd.lib` and
`oldnames.lib` were not found. Root cause: VS Build Tools 18 was installed with x64 libraries
only. Fix: opened Visual Studio Installer → Modify → added
`MSVC v14x ARM64 build tools` + `Windows 11 SDK`.

### Step 2 — Use the correct compiler
Several cmake attempts failed before finding the right combination:

| Attempt | Compiler | Error |
|---|---|---|
| `clang.exe` (standalone LLVM) | `C:/Program Files/LLVM/bin/clang.exe` | `oldnames.lib` not found |
| `cl.exe` (MSVC via Developer PowerShell) | auto-detected | llama.cpp CMakeLists rejects MSVC for ARM: "use clang" |
| `clang-cl.exe` (standalone LLVM) | `C:/Program Files/LLVM/bin/clang-cl.exe` | `mainCRTStartup` undefined — wrong LIB path |
| `clang-cl.exe` (VS ARM64 bundled) | `...BuildTools/VC/Tools/Llvm/ARM64/bin/clang-cl.exe` | Same — Developer PowerShell used mixed env |
| `clang-cl.exe` (VS ARM64) via `vcvarsarm64.bat` in `cmd.exe` | ✅ **Success** | — |

The key insight: `.bat` environment variables only persist within the same `cmd.exe` session.
The VS Developer PowerShell does **not** provide a clean ARM64-native environment. You must
use a plain `cmd.exe` and call `vcvarsarm64.bat` explicitly.

### Step 3 — Working cmake command

Run from a plain `cmd.exe` after calling `vcvarsarm64.bat`:

```cmd
"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsarm64.bat"

cd C:\llama.cpp
mkdir build && cd build

cmake .. -G Ninja ^
  -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_METAL=OFF ^
  -DVULKAN_SDK="C:\VulkanSDK\1.4.341.1" ^
  -DCMAKE_C_COMPILER="C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\Llvm\ARM64\bin\clang-cl.exe" ^
  -DCMAKE_CXX_COMPILER="C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\Llvm\ARM64\bin\clang-cl.exe"
```

### Step 4 — Fix missing OpenMP DLL
After build, running any binary failed with:
> `libomp140.aarch64.dll was not found`

Fix: copy from VS redistributables:
```cmd
copy "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Redist\MSVC\14.50.35710\debug_nonredist\arm64\Microsoft.VC145.OpenMP.LLVM\libomp140.aarch64.dll" "C:\llama.cpp\build\bin\"
```

### Step 5 — Fix zero-byte llama-server.exe
The main build produced a valid `llama-cli.exe` but a 0-byte `llama-server.exe`.
Ninja treated the link failure as a warning and left a stale empty file.

Fix:
```cmd
del C:\llama.cpp\build\bin\llama-server.exe
ninja llama-server
```

### Step 6 — Fix --ngl flag
`--ngl` (double dash) is not recognised. Use `-ngl` (single dash) or `--n-gpu-layers`.

---

## Verification — Vulkan Device Detected

```
C:\llama.cpp\build\bin\llama-cli.exe --list-devices

Available devices:
  Vulkan0: Qualcomm(R) Adreno(TM) X1-85 GPU (20260 MiB, 20260 MiB free)
```

20 GB of unified memory visible to the GPU — the full shared memory pool.

---

## Server Startup — All Layers Offloaded

```cmd
C:\llama.cpp\build\bin\llama-server.exe ^
  -m "C:\Users\rober\.lmstudio\models\unsloth\Qwen3.5-4B-GGUF\Qwen3.5-4B-Q4_K_S.gguf" ^
  --device Vulkan0 ^
  --n-gpu-layers 99 ^
  --ctx-size 16384 ^
  --alias qwen3.5-4b ^
  --port 8080
```

Key startup lines confirmed full GPU offload:
```
load_tensors: offloaded 33/33 layers to GPU
Vulkan0 model buffer size =  1962.65 MiB
Vulkan0 KV buffer size    =   512.00 MiB
sched_reserve: Flash Attention was auto, set to enabled
main: server is listening on http://127.0.0.1:8080
```

---

## Result — Slower Than CPU

A test request from ragent (4822 token system prompt) connected at `20:40:05` and timed out
at `20:50:04` — exactly 10 minutes — while the server had only processed 42% of the prefill
tokens. The server log confirmed it was stuck in the prompt processing phase.

This is significantly slower than LM Studio's CPU-only path (~30–60 seconds for equivalent prefill).

### Root Cause Analysis

| Phase | Characteristic | CPU result | Vulkan result |
|---|---|---|---|
| Prefill (input tokens) | Compute-bound GEMM | ~30–60s for 4822 tokens | >10 min (timeout) |
| Generation (output tokens) | Memory-bandwidth bound | ~15 tok/s (estimated) | Never reached |

**Why CPU won:**

1. **Shader compilation overhead** — Vulkan compiles GLSL compute shaders on first use.
   The first forward pass triggers compilation for every unique kernel, which can take
   several minutes. Subsequent runs cache these, but we timed out before seeing the benefit.

2. **Optimised CPU path** — llama.cpp's ARM64 CPU backend uses `DOTPROD`, `MATMUL_INT8`,
   and `ARM_FMA` instructions (confirmed in system_info output). The Oryon CPU's
   high-performance cores with these extensions are extremely competitive for Q4 GEMM
   at 4B parameter scale.

3. **Adreno compute vs graphics** — The Adreno X1-85 is designed and optimised for graphics
   workloads. Its Vulkan compute path (used for GPGPU/ML workloads) is less mature and
   less optimised in the driver than its graphics pipeline.

4. **Small model, bandwidth-bound generation** — At 4B Q4_K_S, the model weights are ~2.4 GB.
   Generation is memory-bandwidth bound. The unified memory architecture means CPU and GPU
   share the same physical bandwidth (~135 GB/s), removing the GPU's usual bandwidth advantage.

---

## Workloads Where Vulkan/Adreno Might Still Win

The bottleneck is specifically **long-input prefill** (compute-bound GEMM). Scenarios that
avoid this could still benefit:

| Scenario | Why it might help |
|---|---|
| **Short system prompt** — disable ragent's `auto_detect` project context | Reduces 4822 → ~200 tokens; prefill becomes trivial |
| **Generation-heavy workloads** — short prompt, long response | Skips the GEMM-heavy phase; generation is bandwidth-bound where GPU/CPU are similar |
| **Larger models (9B+)** | More layers = more parallelism = better GPU utilisation |
| **Image generation** (Stable Diffusion) | Designed for GPU execution; CNN/attention workloads suit Adreno better |
| **After shader cache is warm** | Second+ runs avoid recompilation; need to test actual generation speed |

---

## What to Watch For

The real acceleration path for Snapdragon X is **QNN (Qualcomm Neural Networks SDK)** via
the Hexagon NPU (45 TOPS), not Vulkan via the Adreno GPU. This requires:

- LM Studio to ship a QNN runtime (watch `lms runtime get -l`)
- Or Ollama to add QNN support
- Or a separate tool using Qualcomm's AI Engine Direct SDK

**Monitoring:**
- `lms runtime get -l` — check periodically for new runtimes
- LM Studio Discord `#announcements` — new runtime support announced here first
- `github.com/lmstudio-ai/lmstudio-releases` — changelogs
- `r/LocalLLaMA` — Snapdragon X community benchmarks

---

## Current Recommended Setup

Until QNN support arrives, LM Studio with `llama.cpp-win-arm64` (CPU) remains the
best available option for this hardware. ragent is configured to use it:

```toml
backend = "openai"

[openai]
url = "http://localhost:1234/v1"
model = "qwen3.5-4b"
```

With `qwen3.5-4b` loaded at 16384 context in LM Studio and ragent's `max_iterations = 30`
with budget warning injection, the setup is stable and functional.
