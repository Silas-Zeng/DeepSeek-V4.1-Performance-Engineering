# Experiment environment

The repository contains the measurement helpers and experiment contract. The
full DeepSeek-V4.1 run must be performed in a Linux NVIDIA environment; the
local RTX 3050 Ti is useful for editing and validating the helpers but cannot
serve the full model or its SM90/SM100 kernels.

## Local helper environment

On Windows or Linux, create a lightweight virtual environment for the scripts:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
```

On Linux, use the equivalent `.venv/bin/python` path.

## GPU experiment prerequisites

Use a Linux x86_64 GPU host or the framework's CUDA container and record the
exact versions before every run:

- NVIDIA driver, CUDA toolkit and GPU model;
- Python and PyTorch;
- the pinned vLLM and SGLang commits;
- FlashInfer, DeepGEMM and any model-specific kernel package;
- the model revision and quantization configuration.

The framework-specific source setup is documented in
[`frameworks.md`](frameworks.md). vLLM and SGLang are installed into separate
environments because their CUDA wheels and kernel dependencies are
architecture- and release-specific. After both are installed, run:

```bash
python scripts/collect_environment.py --output reports/environment.json
```

## Profiling tools

Install Nsight Systems and Nsight Compute on the GPU host. Run normal timing
and profiler timing separately. The profiler wrapper records a bounded trace:

```bash
bash scripts/profile_nsys.sh --output profiles/v41_decode -- \
  python path/to/repro.py --config configs/workload-matrix.yaml
```

The `--` separates wrapper options from the command being profiled. Keep raw
`.nsys-rep` and `.ncu-rep` files out of git; commit reduced tables and notes.
