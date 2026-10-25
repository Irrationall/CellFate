# CellFate

**Predicting cell fate with single-cell data**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.6%E2%80%933.12-blue)

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
| [minepy](https://github.com/minepy/minepy) | 1.2.6 | [Documentation](https://minepy.readthedocs.io/en/latest/) |
| numpy | < 2.0 | minepy does not support NumPy 2.0 out of the box ([workaround](#using-numpy-20)) |

---

## Installation

```bash
git clone https://github.com/Irrationall/CellFate.git
cd CellFate

# Install in editable mode
pip install -e .
```

---

## Tutorial

Example notebooks are available in the [`tutorial/`](tutorial/) directory.

---

## Using NumPy 2.0

**Scanpy 1.10.3** works with **NumPy 2.0** by default, but **minepy** does not.
To use them together, recompile minepy in an environment where NumPy 2.0 is installed.

<details>
<summary><b>Show step-by-step guide</b></summary>

&nbsp;

**Step 1.** Create a virtual environment (conda is used here).

```bash
conda create -n newminepy python=3.10
conda activate newminepy
```

**Step 2.** Install Scanpy and clone the minepy repository.
Recent versions of Scanpy install NumPy ≥ 2.0 automatically.

```bash
pip install scanpy
git clone https://github.com/minepy/minepy.git
```

**Step 3.** Generate the C source and compile it.

```bash
cd minepy
pip install cython
bash compile_pyx.sh   # creates minepy/mine.c
```

```bash
gcc -shared -pthread -fPIC -fwrapv -O2 \
    -I"$CONDA_PREFIX/include/python3.10" \
    -I"$(python -c 'import numpy; print(numpy.get_include())')" \
    -o minepy/mine.new.so minepy/mine.c

pip install .
```

> [!NOTE]
> The include paths above are resolved automatically from the active conda environment.
> If you are not using conda, replace them with your own Python and NumPy include directories.

**Step 4.** Check that minepy works (run this outside the `minepy` folder).

```python
from minepy import MINE
import numpy as np

print(f"NumPy version: {np.__version__}")

x = [1, 2, 3, 4, 5]
y = [5, 4, 3, 2, 1]

mine = MINE()
mine.compute_score(x, y)
print([mine.mic(), mine.mas(), mine.mev(), mine.mcn(), mine.tic()])
```

Expected output:

```text
NumPy version: 2.0.2
[0.9709505944546685, 0.0, 0.9709505944546685, 2.0, 0.9709505944546685]
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
