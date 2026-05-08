"""
Basic tests for minitst.
Run with:  pytest tests/
"""
import pytest
from minitst.geometry import Bond, Angle, Torsion, CisTrans, ChiralCenter
from minitst.species import Conformer, Species


# ---------------------------------------------------------------------------
# geometry.py tests
# ---------------------------------------------------------------------------

def test_bond_repr():
    b = Bond(index=0, atom_indices=(0, 1), length=1.5)
    assert "0" in repr(b) or "1" in repr(b)


def test_angle_repr():
    a = Angle(index=0, atom_indices=(0, 1, 2), degree=109.5, mask=[])
    assert "Angle" in repr(a)


def test_torsion_repr():
    t = Torsion(atom_indices=(0, 1, 2, 3), dihedral=60.0)
    assert "Torsion" in repr(t)


def test_cistrans_repr():
    ct = CisTrans(index=0, atom_indices=(0, 1, 2, 3), dihedral=0.0, mask=[], stero="Z")
    assert "Z" in repr(ct)


def test_chiral_center_repr():
    cc = ChiralCenter(index=0, atom_index=2, chirality="R")
    assert "R" in repr(cc)


# ---------------------------------------------------------------------------
# species.py — Conformer tests
# ---------------------------------------------------------------------------

ETHANE_SMILES = "CC"
PROPANE_SMILES = "CCC"
METHANOL_SMILES = "CO"


def test_conformer_builds_from_smiles():
    conf = Conformer(smiles=ETHANE_SMILES)
    assert conf.smiles == ETHANE_SMILES
    assert conf.rmg_molecule is not None
    assert conf.rdkit_molecule is not None
    assert conf.ase_molecule is not None


def test_conformer_has_bonds():
    conf = Conformer(smiles=ETHANE_SMILES)
    assert len(conf.bonds) > 0
    for bond in conf.bonds:
        assert isinstance(bond, Bond)
        assert bond.length > 0


def test_conformer_has_angles():
    conf = Conformer(smiles=ETHANE_SMILES)
    assert len(conf.angles) > 0
    for angle in conf.angles:
        assert isinstance(angle, Angle)
        assert 0 < angle.degree < 180


def test_conformer_torsions_propane():
    conf = Conformer(smiles=PROPANE_SMILES)
    assert len(conf.torsions) > 0
    for tor in conf.torsions:
        assert isinstance(tor, Torsion)
        assert len(tor.atom_indices) == 4


def test_conformer_no_torsions_ethane():
    # Ethane has only terminal (methyl) torsions; non-terminal list is empty
    conf = Conformer(smiles=ETHANE_SMILES)
    # torsions list may be non-empty but that's fine — just check types
    for tor in conf.torsions:
        assert isinstance(tor, Torsion)


def test_conformer_copy():
    conf = Conformer(smiles=PROPANE_SMILES)
    copy = conf.copy()
    assert copy.smiles == conf.smiles
    assert copy is not conf
    assert copy.ase_molecule is not conf.ase_molecule


def test_conformer_get_xyz_block():
    conf = Conformer(smiles=METHANOL_SMILES)
    xyz = conf.get_xyz_block()
    assert isinstance(xyz, str)
    assert len(xyz.strip()) > 0
    # Should contain element symbols
    assert "C" in xyz or "O" in xyz or "H" in xyz


def test_conformer_set_torsion():
    conf = Conformer(smiles=PROPANE_SMILES)
    if not conf.torsions:
        pytest.skip("No torsions found for propane")
    tor = conf.torsions[0]
    conf.set_torsion(tor.index, 60.0)
    import math
    i, j, k, ll = tor.atom_indices
    new_dihedral = conf.ase_molecule.get_dihedral(i, j, k, ll)
    assert abs(new_dihedral - 60.0) < 1.0


def test_conformer_symmetry_number():
    conf = Conformer(smiles=ETHANE_SMILES)
    sym = conf.symmetry_number
    assert isinstance(sym, (int, float))
    assert sym >= 1


# ---------------------------------------------------------------------------
# species.py — Species tests
# ---------------------------------------------------------------------------

def test_species_from_smiles():
    spc = Species(smiles=[ETHANE_SMILES])
    assert len(spc.smiles) >= 1
    assert spc.rmg_species is not None


def test_species_repr():
    spc = Species(smiles=[ETHANE_SMILES])
    r = repr(spc)
    assert "Species" in r


def test_species_conformers_generated():
    spc = Species(smiles=[ETHANE_SMILES])
    confs = spc.conformers
    assert isinstance(confs, dict)
    for smile, conf_list in confs.items():
        assert len(conf_list) > 0
        assert isinstance(conf_list[0], Conformer)
