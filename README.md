# CellFate

**Predicting cell fate with single-cell data**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Tutorial](#tutorial)
- [Using NumPy 2.0](#using-numpy-20)
- [Citation](#citation)
- [License](#license)

---

## Prerequisites

| Package | Version | Note |
| --- | --- | --- |
| [minepy](https://github.com/minepy/minepy) | 1.2.6 (modified) | Included; no separate installation needed |
| [NumPy](https://numpy.org/) | >=1.24.1 | Supports NumPy 1.x and 2.x ([details](#using-numpy-20)) |

Building from source requires Linux and GCC with OpenMP support (libgomp). On Python 3.12, NumPy >=1.26 is required.

---

## Installation

```bash
git clone https://github.com/Irrationall/CellFate.git
cd CellFate

conda create -n cellfate "python>=3.10,<3.13" pip
conda activate cellfate

pip install .
```

The included minepy and RD extensions are compiled automatically during installation.

---

## Tutorial

See the [`tutorial/`](tutorial/) directory.

---

## Using NumPy 2.0

CellFate supports NumPy 1.x and 2.x. The included modified minepy is compiled automatically during installation, so no separate minepy installation or manual recompilation is needed. pip selects a NumPy version compatible with Scanpy, SciPy and Numba.

<details>
<summary><b>Show step-by-step guide</b></summary>

From the CellFate source directory:

```bash
conda create -n cellfate-np2 "python>=3.10,<3.13" pip
conda activate cellfate-np2

pip install "numpy>=2" .
python -c "import numpy; import cellfate as cf; print(numpy.__version__); print(cf.backend_info())"
```

</details>

---

## Citation

If you use CellFate in your research, please cite:

**CellFate**

```bibtex
TBD
```

**Relative Dimension** ([DynGDim](https://github.com/barahona-research-group/DynGDim))

```bibtex
@article{peach2022relative,
  title={Relative, local and global dimension in complex networks},
  author={Peach, Robert and Arnaudon, Alexis and Barahona, Mauricio},
  journal={Nature Communications},
  volume={13},
  number={1},
  pages={3088},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

