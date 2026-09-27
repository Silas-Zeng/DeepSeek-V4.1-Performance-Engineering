# vLLM and SGLang experiment environments

The performance comparison uses two isolated source installations:

```text
.framework-env/
  src/vllm     + venvs/vllm
  src/sglang   + venvs/sglang
```

Do not install both frameworks into one Python environment. Their PyTorch,
CUDA extension and kernel dependencies can differ, and a shared environment
would make a benchmark hard to reproduce.

## Linux GPU host setup

The current local machine is Windows with an RTX 3050 Ti. It cannot run the
DeepSeek-V4.1 production path, so run these commands on the target Linux GPU
host. vLLM's official installation guide supports source installation with
`uv pip install -e . --torch-backend=auto`; SGLang should be installed from
its source checkout for a pinned benchmark commit.

First record the host and choose commits:

```bash
python scripts/collect_environment.py --output reports/environment-host.json
export VLLM_REF=<vllm-commit>
export SGLANG_REF=<sglang-commit>
```

Then install separately:

```bash
bash environment/setup_framework.sh vllm "$VLLM_REF"
bash environment/setup_framework.sh sglang "$SGLANG_REF"
```

The script uses Python 3.12, checks out the exact commit, installs from source,
and records package/GPU metadata. If a framework requires a vendor CUDA image,
run the same script inside that image and keep the image digest in the report.

## First smoke checks

```bash
.framework-env/venvs/vllm/bin/python -c 'import torch, vllm; print(torch.__version__, vllm.__version__)'
.framework-env/venvs/sglang/bin/python -c 'import torch, sglang; print(torch.__version__, sglang.__version__)'
nvidia-smi
nvcc --version
```

Only after both smoke checks pass should you launch DeepSeek-V4.1. Start with
the same model revision, precision, TP size and workload from
`configs/workload-matrix.yaml`; record backend selection from each framework's
startup log instead of forcing different backends by hand.
