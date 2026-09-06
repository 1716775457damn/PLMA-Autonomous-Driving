# PLMA: Predictive Local Multiple Attention

Official code release for **Predictive Local Multiple Attention For Real-time Risk Assessment in Autonomous Driving**.

- Paper: [IEEE Xplore, document 11513734](https://ieeexplore.ieee.org/document/11513734)
- Main reference: `src/plma/plma_legacy_reference.py`

## Installation

```bash
git clone https://github.com/1716775457damn/PLMA-Autonomous-Driving.git
cd PLMA-Autonomous-Driving
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The main reference contains PLMA, LRAM, MFAFM, baselines, training, evaluation, cross-subject analysis, and ablation utilities. The other source files are historical research variants retained for transparency.

## Data

The dataset is not included. Provide a license-compliant local dataset matching the PODAR-style serialized data reader before running the historical reference script. The legacy script still contains path assumptions and may require local path edits.

```bash
python scripts/run_reference.py
```

Do not upload restricted driving data, personal information, checkpoints, or generated results.

## Layout

```text
configs/   configuration template
docs/      reproducibility and third-party notes
scripts/   launch and release checks
src/plma/  reference implementations
tests/     release smoke tests
```

## Citation and license

Please cite the paper in `CITATION.cff`. Original code is MIT licensed; datasets and third-party implementations retain their own licenses.
