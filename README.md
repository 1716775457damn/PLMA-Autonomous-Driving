<div align="center">

# PLMA

### Predictive Local Multiple Attention for Real-Time Risk Assessment in Autonomous Driving

<p>
  <a href="https://ieeexplore.ieee.org/document/11513734"><img src="https://img.shields.io/badge/Paper-IEEE%20Xplore-00629B?style=flat-square&logo=ieee" alt="Paper"></a>
  <a href="https://github.com/1716775457damn/PLMA-Autonomous-Driving/releases"><img src="https://img.shields.io/github/v/release/1716775457damn/PLMA-Autonomous-Driving?style=flat-square&color=brightgreen" alt="Release"></a>
  <a href="https://github.com/1716775457damn/PLMA-Autonomous-Driving/blob/main/LICENSE"><img src="https://img.shields.io/github/license/1716775457damn/PLMA-Autonomous-Driving?style=flat-square" alt="License"></a>
  <a href="https://github.com/1716775457damn/PLMA-Autonomous-Driving/issues"><img src="https://img.shields.io/github/issues/1716775457damn/PLMA-Autonomous-Driving?style=flat-square" alt="Issues"></a>
</p>

<p>
  <strong>Research code for attention-based risk assessment in autonomous driving.</strong><br>
  The repository preserves the reference implementation used during the research process and provides a foundation for reproducible follow-up experiments.
</p>

</div>

---

## Overview

Accurate and efficient risk assessment is an important component of safe autonomous-driving systems. **PLMA** is an attention-based framework designed to capture local and multidimensional driving-risk features while keeping the model suitable for real-time-oriented analysis.

The implementation includes:

- **Local Refinement Attention Module (LRAM)** for local feature refinement;
- **Multidimensional Focused Attention Fusion Module (MFAFM)** for feature fusion;
- the complete **Predictive Local Multiple Attention (PLMA)** model;
- baseline models for comparative experiments;
- training, validation, evaluation, cross-subject analysis, and ablation utilities.

> **Release status:** `v0.1.0` — initial public research-code release.

## Paper

**Predictive Local Multiple Attention For Real-time Risk Assessment in Autonomous Driving**

- [IEEE Xplore: Document 11513734](https://ieeexplore.ieee.org/document/11513734)
- [Citation metadata](CITATION.cff)
- [Initial release notes](https://github.com/1716775457damn/PLMA-Autonomous-Driving/releases/tag/v0.1.0)

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/1716775457damn/PLMA-Autonomous-Driving.git
cd PLMA-Autonomous-Driving
```

### 2. Create an environment

```bash
python -m venv .venv
```

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA-enabled training, install the PyTorch build that matches your CUDA version before installing the remaining dependencies.

### 4. Run the release smoke checks

```bash
python -m compileall -q src scripts
python scripts/check_release.py
```

### 5. Run the historical reference workflow

```bash
python scripts/run_reference.py
```

> The reference workflow may launch a long-running experiment. Review the dataset path and configuration before starting it.

## Data

The datasets used in the paper are **not included** in this repository. This is intentional: the data may be subject to separate access, privacy, and redistribution requirements.

Before running a full experiment, provide a license-compliant local dataset matching the serialized data structure expected by the reference implementation. The initial release preserves historical path assumptions, so local path adjustment may be required.

Please do **not** commit any of the following:

- raw driving data or personal information;
- pretrained weights or checkpoints;
- experiment logs and generated results;
- private paper drafts, presentations, or review materials.

## Repository structure

```text
PLMA-Autonomous-Driving/
├── configs/
│   └── plma.yaml                  # Configuration template
├── docs/
│   ├── CANONICAL_IMPLEMENTATION.md
│   ├── REPRODUCIBILITY.md
│   └── THIRD_PARTY.md
├── scripts/
│   └── run_reference.py           # Reference launcher
├── src/plma/
│   ├── plma_legacy_reference.py   # Primary reference implementation
│   ├── plma_scientific_variant.py # Historical scientific variant
│   └── plma_ablation_reference.py # Ablation/extended experiments
├── tests/
│   └── test_release_layout.py
├── CITATION.cff
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

## Reproducibility

The reported results depend on the exact dataset split, preprocessing pipeline, prediction horizon, random seed, and hardware environment. For a faithful reproduction:

1. obtain the appropriate dataset through its official source;
2. verify the data format and local paths;
3. use the configuration and preprocessing assumptions described in the paper;
4. record the Python, PyTorch, CUDA, and GPU versions;
5. save the random seed and all experiment settings;
6. compare both accuracy and computational-efficiency metrics.

More details are available in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Third-party code and licenses

The original workspace contained external or reference projects, including RiskBench and PODAR-related code. They are **not redistributed** in this repository. Obtain third-party components from their official sources and follow their licenses and citation requirements.

See [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md) before adding external code.

## Citation

If you use this repository, please cite the associated paper:

```bibtex
@article{tao_plma,
  title   = {Predictive Local Multiple Attention For Real-time Risk Assessment in Autonomous Driving},
  author  = {Tao, Xinwang and Zhang, Jinlai and Wang, Lecu and Gao, Kai and Liu, Zhizhen and Hu, Lin},
  journal = {IEEE Xplore},
  note    = {Document 11513734},
  url     = {https://ieeexplore.ieee.org/document/11513734}
}
```

The machine-readable citation is available in [`CITATION.cff`](CITATION.cff).

## Contributing

Issues and pull requests are welcome. Before submitting a contribution:

- keep the original research behavior reproducible;
- avoid committing datasets, credentials, checkpoints, or generated artifacts;
- document changes to preprocessing, model architecture, or evaluation;
- include a focused test or reproducibility note where appropriate.

## License

Original code in this repository is released under the [MIT License](LICENSE). Datasets, external implementations, and other third-party materials remain subject to their respective licenses.

<div align="center">

If this work is useful to you, please consider citing the paper and starring the repository.

</div>
