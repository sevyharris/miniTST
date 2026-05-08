# minitst

A lightweight Python package for generating and manipulating 3-D molecular geometries and conformers.

Extracted from [AutoTST](https://github.com/ReactionMechanismGenerator/AutoTST) — contains only `species.py` and `geometry.py`, with no dependency on the rest of the AutoTST codebase.

## What's included

| Module | Contents |
|---|---|
| `minitst.geometry` | `Bond`, `Angle`, `Torsion`, `CisTrans`, `ChiralCenter` data classes |
| `minitst.species` | `Conformer` (single 3-D structure) and `Species` (collection of resonance structures + conformers) |

## Dependencies

| Package | Purpose |
|---|---|
| `rdkit` | 3-D embedding, bond/torsion perception |
| `ase` | Atoms object, distance/angle/dihedral manipulation |
| `rmgpy` | Molecule representation, SMILES/adjacency-list parsing, symmetry |
| `numpy` | Coordinate arithmetic |
| `py3Dmol` *(optional)* | Interactive 3-D viewer in Jupyter notebooks |

## Installation

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate minitst
pip install -e .
```

### pip only

```bash
pip install -e .
```

> **Note:** `rdkit` and `rmgpy` are best installed via `conda-forge`. Pure-pip installs may require extra steps.

### Build the conda package locally

```bash
conda install conda-build
conda build conda.recipe/
```

## Quick start

```python
from minitst import Conformer, Species

# Single conformer from a SMILES string
conf = Conformer(smiles="CCC")

print(conf.bonds)       # list of Bond objects
print(conf.torsions)    # list of Torsion objects

# Rotate a torsion
conf.set_torsion(torsion_index=0, dihedral=60.0)

# Get XYZ coordinates
print(conf.get_xyz_block())

# All resonance structures of a species
spc = Species(smiles=["CC=O"])
print(spc.conformers)   # {smiles: [Conformer, ...]}
```

## Running tests

```bash
pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).
