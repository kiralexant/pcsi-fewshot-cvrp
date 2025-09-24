# Parameter Control Strategy Inference for Randomized-Heuristic-Search Applied to CVRP


## Installation

Clone the repository and install in editable mode (development), so changes in Python code take effect immediately without reinstalling:
```bash
git clone --recurse-submodules https://github.com/kiralexant/FewShotCVRP.git
cd FewShotCVRP
python -m pip install -e .
```

---

## Recompiling Only the C++ Extension

If you change only the C++ code and want to rebuild without reinstalling the package:
```bash
python setup.py build_ext --inplace
```

---

## Fetching External Dependencies

Clone the external repository as a dependancy:
```bash
git submodule update --init --recursive
```


---

## License
This project is licensed under the MIT License.
