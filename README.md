# iMSMD-tool: Interactive Time–Frequency Masking for Non-Stationary Signal Reconstruction

iMSMD-tool is a Python/PyQt5 application for interactive time–frequency masking and reconstruction of non-stationary signals.  
**Entry script:** `zhonggoucg1_msst.py`

## Features
- Interactive time–frequency masking workflow
- Signal reconstruction utilities
- GUI based on PyQt5
- Core algorithms provided in `algorithms/`

## Repository structure
- `zhonggoucg1_msst.py`: main entry script
- `algorithms/`: core time–frequency and reconstruction algorithms
- `data/`: (optional) example data
- `assets/`: (optional) GUI resources (images/fonts)

## Installation
### Option A: Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
