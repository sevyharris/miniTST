#!/usr/bin/python
# -*- coding: utf-8 -*-

##########################################################################
#
#   AutoTST - Automated Transition State Theory
#
#   Copyright (c) 2015-2020 Richard H. West (r.west@northeastern.edu)
#   and the AutoTST Team
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the 'Software'),
#   to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense,
#   and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.
#
##########################################################################

import itertools
import logging
import os
import time
import pandas as pd
import numpy as np
import functools
import multiprocessing
from copy import deepcopy

import ase
import ase.io
import ase.io.extxyz
import ase.units
import ase.calculators.calculator
import ase.optimize
import ase.constraints

import rdkit.Chem

import rmgpy.exceptions
import rmgpy.molecule

import autotst
from ..species import Conformer
from ..reaction import TS
from .utilities import get_energy, find_terminal_torsions


def find_all_combos(
        conformer,
        delta=float(120),
        cistrans=True,
        chiral_centers=True):
    """
    A function to find all possible conformer combinations for a given conformer

    Params:
    - conformer (`Conformer`) an AutoTST `Conformer` object of interest
    - delta (int or float): a number between 0 and 180 or how many conformers to generate per dihedral
    - cistrans (bool): indication of if one wants to consider cistrans bonds
    - chiral_centers (bool): indication of if one wants to consider chiral centers bonds

    Returns:
    - all_combos (list): a list corresponding to the number of unique conformers created.
    """

    conformer.get_geometries()

    _, torsions = find_terminal_torsions(conformer)

    torsion_angles = np.arange(0, 360, delta)
    torsion_combos = list(itertools.product(
        torsion_angles, repeat=len(torsions)))

    if cistrans:
        cistranss = []
        cistrans_options = ["E", "Z"]
        try:
            ring_info = conformer._pseudo_geometry.GetRingInfo()
        except AttributeError:
            ring_info = conformer.rdkit_molecule.GetRingInfo()

        for cistrans in conformer.cistrans:
            i, j, k, _ = cistrans.atom_indices
            if (ring_info.NumAtomRings(i) != 0) or (ring_info.NumAtomRings(k) != 0):
                continue
            cistranss.append(cistrans)

        cistrans_combos = list(itertools.product(
            cistrans_options, repeat=len(cistranss)))

    else:
        cistrans_combos = [()]

    if chiral_centers:
        chiral_centerss = []
        chiral_options = ["R", "S"]
        try:
            ring_info = conformer._pseudo_geometry.GetRingInfo()
        except AttributeError:
            ring_info = conformer.rdkit_molecule.GetRingInfo()

        for center in conformer.chiral_centers:
            if ring_info.NumAtomRings(center.atom_index) != 0:
                continue
            chiral_centerss.append(center)

        chiral_combos = list(itertools.product(
            chiral_options, repeat=len(chiral_centerss)))

    else:
        chiral_combos = [()]

    all_combos = list(
        itertools.product(
            torsion_combos,
            cistrans_combos,
            chiral_combos))
    return all_combos


def calculate_garbage_score(atoms, verbose=False):
    # copied from https://github.com/sevyharris/reaction_calculator/blob/main/garbage.py
    garbage_score = 0
    min_H_dist = 0.75
    min_heavy_dist = 1.1
    nn_dist = 1.5
    nn_threshold = 4

    # do a nearest neighbor check to make sure no atom is too close to another
    num_combos = len(atoms) * (len(atoms) - 1) / 2
    for i in range(0, len(atoms)):
        # count the atoms within 1.5A of this atom
        n_close = 0

        for j in range(i + 1, len(atoms)):
            dist = atoms.get_distance(i, j)
            if atoms[i].symbol == 'H' or atoms[j].symbol == 'H':
                threshold = min_H_dist
            else:
                threshold = min_heavy_dist
            if dist < threshold:
                atomic_weight = atoms.get_atomic_numbers()[i] + atoms.get_atomic_numbers()[j]
                garbage_score += atomic_weight / num_combos + threshold - dist
                if verbose:
                    print(f'Atom {i} and atom {j} are too close ({dist:.3f} < {threshold:.3f})')
            if dist < nn_dist:
                n_close += 1
        for j in range(0, i):
            dist = atoms.get_distance(i, j)
            if dist < nn_dist:
                n_close += 1

        if n_close > nn_threshold:
            garbage_score += n_close / 10.0
            if verbose:
                print(f'Atom {i} has {n_close} neighbors')
    return garbage_score


def opt_conf(i):
    """
    A helper function to optimize the geometry of a conformer.
    param i: index of the conformer or a conformer object
    """
    try:
        conformer = conformers[i]  # use the global object
    except (TypeError, KeyError):
        # When running tests for this single function, it's hard to create a global
        # conformers dict. This step allows users to pass in conformer objects rather
        # than specify a global dict
        conformer = i

    if conformer.save_results:
        # check for previous results
        result_num = i + conformer.save_offset
        result_file = os.path.join(conformer.results_dir, f"results_{result_num:05}.xyz")

        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                atoms_gen = ase.io.extxyz.read_extxyz(f)
                try:
                    atoms = next(atoms_gen)
                except StopIteration:
                    logging.info(f"No atoms object in {result_file}")
            try:
                energy = atoms.info['energy']
                logging.info(f"Skipping calculation due to previous results in {result_file}")
                return energy
            except KeyError:
                logging.info(f"No energy results in {result_file}")
        if not os.path.exists(result_file) and conformer.save_results:
            logging.debug(f'No result file for {result_num:05}. Calculating...')

    if not isinstance(conformer, TS):
        reference_mol = conformer.rmg_molecule.copy(deep=True)
        reference_mol = reference_mol.to_single_bonds()
    calculator = conformer.ase_molecule.get_calculator()

    if calculator == 'SKIP':
        logging.info("Skipping calculation per ase calculator instruction")
        return 1e5  # the one case we don't need to save xyz files

    # do sanity check on geometry to check for garbage
    garbage_score = calculate_garbage_score(conformer.ase_molecule)
    if garbage_score > 1.0:
        logging.info(f"Skipping calculation due to garbage score of {garbage_score}")
        # save result anyways so we can skip later
        energy = 1e5
        if conformer.save_results:
            with open(result_file, "w") as f:
                ase.io.extxyz.write_xyz(f, conformer.ase_molecule, comment=f"energy={energy}")
        return energy

    calculator.__init__()
    calculator = deepcopy(calculator)
    labels = []
    for bond in conformer.get_bonds():
        labels.append(bond.atom_indices)

    if isinstance(conformer, TS):
        label = conformer.reaction_label
        if conformer.reaction_family == 'Disproportionation':
            ind1 = conformer.rmg_molecule.get_labeled_atoms("*1")[0].sorting_label
            ind2 = conformer.rmg_molecule.get_labeled_atoms("*2")[0].sorting_label
            ind3 = conformer.rmg_molecule.get_labeled_atoms("*4")[0].sorting_label
        else:
            ind1 = conformer.rmg_molecule.get_labeled_atoms("*1")[0].sorting_label
            ind2 = conformer.rmg_molecule.get_labeled_atoms("*3")[0].sorting_label
            ind3 = conformer.rmg_molecule.get_labeled_atoms("*2")[0].sorting_label

        if [ind1, ind2] not in labels and [ind2, ind1] not in labels:
            labels.append([ind1, ind2])
        if [ind2, ind3] not in labels and [ind3, ind2] not in labels:
            labels.append([ind2, ind3])
        if [ind1, ind3] not in labels and [ind3, ind1] not in labels:
            labels.append([ind1, ind3])

        # TODO - not sure if this is helping things
        # if conformer.reaction_family == 'Disproportionation':
        #     ind1 = conformer.rmg_molecule.get_labeled_atoms("*1")[0].sorting_label
        #     ind2 = conformer.rmg_molecule.get_labeled_atoms("*2")[0].sorting_label
        # else:
        #     ind1 = conformer.rmg_molecule.get_labeled_atoms("*1")[0].sorting_label
        #     ind2 = conformer.rmg_molecule.get_labeled_atoms("*3")[0].sorting_label
        # labels.append([ind1, ind2])
        type = 'ts'

    else:
        label = conformer.smiles
        type = 'species'

    if isinstance(calculator, ase.calculators.calculator.FileIOCalculator):
        if calculator.directory:
            directory = calculator.directory
        else:
            directory = 'conformer_logs'
        calculator.label = f"{conformer.smiles}_{conformer.index}"
        calculator.directory = os.path.join(directory, label, f'{conformer.smiles}_{conformer.index}')
        if not os.path.exists(calculator.directory):
            try:
                os.makedirs(calculator.directory)
            except OSError:
                logging.info(f"An error occured when creating {calculator.directory}")

        calculator.atoms = conformer.ase_molecule
    conformer.ase_molecule.set_calculator(calculator)
    opt = ase.optimize.BFGS(conformer.ase_molecule, logfile=None)
    if type == 'species':
        if isinstance(conformer.index, int):
            c = ase.constraints.FixBondLengths(labels)
            conformer.ase_molecule.set_constraint(c)
        try:
            opt.run(steps=1e6)
        except RuntimeError:
            logging.info("Optimization failed...we will use the unconverged geometry")
            pass
        except AssertionError:
            logging.info("Bad eigenvalues probably ... use the unconverged geometry?")
            pass
        if str(conformer.index) == 'ref':
            conformer.update_coords_from("ase")
            try:
                rmg_mol = rmgpy.molecule.Molecule()
                rmg_mol.from_xyz(
                    conformer.ase_molecule.arrays["numbers"],
                    conformer.ase_molecule.arrays["positions"]
                )
                if not rmg_mol.is_isomorphic(reference_mol):
                    logging.info(f"{conformer}_{str(conformer.index)} is not isomorphic with reference mol")
                    return False
            except rmgpy.exceptions.AtomTypeError:
                logging.info("Could not create a RMG Molecule from optimized conformer coordinates...assuming not isomorphic")
                return False
    converged = False
    if type == 'ts':
        c = ase.constraints.FixBondLengths(labels)
        conformer.ase_molecule.set_constraint(c)
        try:
            opt.run(fmax=0.20, steps=1e6)
            # runs the optimization, but where does it save the result? or is this supposed to happen in place??

        except RuntimeError:
            logging.info("Optimization failed...we will use the unconverged geometry")
            converged = True
            pass
        except AssertionError:
            logging.info("Bad eigenvalues probably ... use the unconverged geometry")
            converged = True
            pass
    conformer.update_coords_from("ase")
    try:
        energy = conformer.ase_molecule.get_potential_energy()
    except (RuntimeError, AssertionError, NotImplementedError):
        if not converged:
            logging.error("Unable to parse energy from unconverged geometry")
        else:
            logging.error("Unable to parse energy from geometry")
        energy = 1e5
    try:
        conformers[i] = conformer  # update the conformer from old object
    except TypeError:
        logging.error('Could not add updated conformer to conformers dict')

    if conformer.save_results:
        if not os.path.exists(conformer.results_dir):
            try:
                os.makedirs(conformer.results_dir)
            except OSError:
                logging.info(f"An error occured when creating {conformer.results_dir}")
        with open(result_file, "w") as f:
            ase.io.extxyz.write_xyz(f, conformer.ase_molecule, comment=f"energy={energy}")

    logging.debug(f'Returning optimization energy {energy}')
    return energy  # return energy


def sorted_hash(atoms):
    my_string = ''
    O_atoms = [atom for atom in atoms if atom.symbol == 'O']
    C_atoms = [atom for atom in atoms if atom.symbol == 'C']
    H_atoms = [atom for atom in atoms if atom.symbol == 'H']

    O_atoms = sorted(O_atoms, key=functools.cmp_to_key(lambda item1, item2: np.linalg.norm(item1.position) - np.linalg.norm(item2.position)))
    C_atoms = sorted(C_atoms, key=functools.cmp_to_key(lambda item1, item2: np.linalg.norm(item1.position) - np.linalg.norm(item2.position)))
    H_atoms = sorted(H_atoms, key=functools.cmp_to_key(lambda item1, item2: np.linalg.norm(item1.position) - np.linalg.norm(item2.position)))

    round_digits = 1
    for atom in O_atoms:
        my_string += f'O{np.round(atom.position[0], round_digits):}{np.round(atom.position[1], round_digits)}{np.round(atom.position[2], round_digits)}\n'
    for atom in C_atoms:
        my_string += f'C{np.round(atom.position[0], round_digits):}{np.round(atom.position[1], round_digits)}{np.round(atom.position[2], round_digits)}\n'
    for atom in H_atoms:
        my_string += f'H{np.round(atom.position[0], round_digits):}{np.round(atom.position[1], round_digits)}{np.round(atom.position[2], round_digits)}\n'
    return my_string


def overlap_molecule(atomsA, atomsB, translate_indices):  # assumes order is same
    # returns an atoms object of atoms B transformed to overlap with atomsA
    new_atoms = deepcopy(atomsB)

    # Translate
    translation_vector = atomsA.positions[translate_indices[0], :] - atomsB.positions[translate_indices[0], :]
    for i in range(new_atoms.positions.shape[0]):
        new_atoms.positions[i, :] += translation_vector

    # Rotate
    vector12A = atomsA.positions[translate_indices[1], :] - atomsA.positions[translate_indices[0], :]
    vector12B = new_atoms.positions[translate_indices[1], :] - new_atoms.positions[translate_indices[0], :]
    new_atoms.rotate(vector12B, vector12A, center=new_atoms.positions[translate_indices[0], :])
    return new_atoms


def check_redundant(atoms1, atoms2):
    # try rotating one molecule into the other's orientation and see if they produce the same hash string
    # returns True if redundant
    A = sorted_hash(atoms1)
    B = sorted_hash(atoms2)
    if A == B:
        return True
    combos = list(itertools.combinations(range(len(atoms1)), 2))
    for i, c in enumerate(combos[:10]):
        new_atoms = overlap_molecule(atoms1, atoms2, c)
        B = sorted_hash(new_atoms)
        if A == B:
            return True
    return False


def systematic_search(
    conformer,
    delta=float(120),
    energy_cutoff=10.0,  # kcal/mol
    rmsd_cutoff=0.5,  # angstroms
    cistrans=True,
    chiral_centers=True,
    multiplicity=False,
    max_combos=-1,  # default is no maximum
    max_conformers=-1,  # default is no maximum
    count_combos=False,  # return the number of combinations
    return_combos=False  # return the conformer geometries without optimization
):
    """
    Perfoms a systematic conformer analysis of a `Conformer` or a `TS` object

    Variables:
    - conformer (`Conformer` or `TS`): a `Conformer` or `TS` object of interest
    - delta (int or float): a number between 0 and 180 or how many conformers to generate per dihedral
    - energy_cutoff (str or float): energy in kcal/mol
    - rmsd_cutoff (str or float): root mean square deviation of inter atomic positions
    - cistrans (bool): indication of if one wants to consider cistrans bonds
    - chiral_centers (bool): indication of if one wants to consider chiral centers bonds
    - max_combos is the maximum number of conformers to investigate with the cheap ase calculator
    - max_conformers is the maximum number of conformers to return for detailed investigation

    Returns:
    - confs (list): a list of unique `Conformer` objects within 10 kcal/mol of the lowest energy conformer determined
    """

    rmsd_cutoff_options = {
        'loose': 1.0,
        'default': 0.5,
        'tight': 0.1
    }

    energy_cutoff_options = {
        'high': 50.0,
        'default': 10.0,
        'low': 5.0
    }

    if isinstance(rmsd_cutoff, str):
        rmsd_cutoff = rmsd_cutoff.lower()
        assert rmsd_cutoff in rmsd_cutoff_options.keys(), 'rmsd_cutoff options are loose, default, and tight'
        rmsd_cutoff = rmsd_cutoff_options[rmsd_cutoff]

    if isinstance(energy_cutoff, str):
        energy_cutoff = energy_cutoff.lower()
        assert energy_cutoff in energy_cutoff_options.keys(), 'energy_cutoff options are low, default, and high'
        energy_cutoff = energy_cutoff_options[energy_cutoff]

    if not isinstance(conformer, TS):
        reference_mol = conformer.rmg_molecule.copy(deep=True)
        reference_mol = reference_mol.to_single_bonds()

    # if not isinstance(conformer,TS):
    #    calc = conformer.ase_molecule.get_calculator()
    #    reference_conformer = conformer.copy()
    #    if opt_conf(reference_conformer, calc, 'ref', rmsd_cutoff):
    #        conformer = reference_conformer

    combos = find_all_combos(
        conformer,
        delta=delta,
        cistrans=cistrans,
        chiral_centers=chiral_centers)

    if count_combos:
        return len(combos)
    if max_combos > 0 and len(combos) > max_combos:
        combos = combos[0:max_combos]

    if len(combos) == 0:
        logging.info(
            "This species has no torsions, cistrans bonds, or chiral centers")
        logging.info("Returning origional conformer")
        return [conformer]

    _, non_terminal_torsions = find_terminal_torsions(conformer)

    calc = conformer.ase_molecule.get_calculator()
    if isinstance(calc, ase.calculators.calculator.FileIOCalculator):
        logging.info("The calculator generates input and output files.")

    results = []
    global conformers
    global df
    df = pd.DataFrame()
    conformers = {}
    combinations = {}
    logging.info(f"There are {len(combos)} possible conformers to investigate...")

    for index, combo in enumerate(combos):

        combinations[index] = combo

        torsion_angles, cistrans, chiral_centers = combo
        assert len(non_terminal_torsions) == len(torsion_angles), "The number of torsions and torsion angles do not match"
        copy_conf = conformer.copy()

        for i, torsion_angle in enumerate(torsion_angles):

            torsion_object = non_terminal_torsions[i]  # originally this information came from copy_conf, but we need the rotor index...
            atom1, atom2, atom3, atom4 = torsion_object.atom_indices
            mask = torsion_object.mask

            torsion_angle += copy_conf.ase_molecule.get_dihedral(atom1, atom2, atom3, atom4)

            copy_conf.ase_molecule.set_dihedral(
                a1=atom1,
                a2=atom2,
                a3=atom3,
                a4=atom4,
                angle=torsion_angle,
                mask=mask
            )
            copy_conf.update_coords_from("ase")

        # for i, e_z in enumerate(cistrans):
        #     ct = copy_conf.cistrans[i]
        #     copy_conf.set_cistrans(ct.index, e_z)

        # for i, s_r in enumerate(chiral_centers):
        #     center = copy_conf.chiral_centers[i]
        #     copy_conf.set_chirality(center.index, s_r)

        copy_conf.update_coords_from("ase")
        copy_conf.ase_molecule.set_calculator(calc)

        conformers[index] = copy_conf

    logging.info(f"Conformers to investigate: {len(conformers)}")

    if return_combos:
        return conformers

    num_threads = multiprocessing.cpu_count() - 1 or 1
    logging.debug(f'Running optimizations with {num_threads} threads')
    pool = multiprocessing.Pool(processes=num_threads)
    results = pool.map(opt_conf, range(len(conformers)))
    pool.close()
    pool.join()

    logging.debug('optimization of all conformers is complete')

    energies = []
    for i, energy in enumerate(results):
        energies.append((conformers[i], energy))

    df = pd.DataFrame(energies, columns=["conformer", "energy"])
    df = df[df.energy < df.energy.min() + (energy_cutoff * ase.units.kcal / ase.units.mol
            / ase.units.eV)].sort_values("energy").reset_index(drop=True)
    # cut to 1.5x length of maximum we'll keep to save time on redundancy check
    first_cut = int(3 * max_conformers)
    if max_conformers > 0 and first_cut < len(df):
        df = df[:first_cut]

    redundant = []
    for i, j in itertools.combinations(range(len(df.conformer)), 2):
        if check_redundant(df.conformer[i].ase_molecule, df.conformer[j].ase_molecule):
            redundant.append(j)
    redundant = list(set(redundant))

    # never drop the first one
    if 0 in redundant:
        redundant.remove(0)

    df.drop(df.index[redundant], inplace=True)

    if len(df) == 0:
        print('SOMETHING WENT HORRIBLY WRONG WITH REDUNDANCY CHECK')
        # this is a bandaid because I haven't figured out why df is dropping all conformers
        # grab at least one conformer geometry...
        df = pd.DataFrame(energies, columns=["conformer", "energy"])
        df = df[df.energy < df.energy.min() + (energy_cutoff * ase.units.kcal / ase.units.mol
                / ase.units.eV)].sort_values("energy").reset_index(drop=True)
        N = np.min(len(df), 3)
        df = df.iloc[0:N]

    if multiplicity and conformer.rmg_molecule.multiplicity > 2:
        rads = conformer.rmg_molecule.get_radical_count()
        if rads % 2 == 0:
            multiplicities = range(1, rads + 2, 2)
        else:
            multiplicities = range(2, rads + 2, 2)
    else:
        multiplicities = [conformer.rmg_molecule.multiplicity]

    confs = []
    if max_conformers > 0 and len(df) < max_conformers:
        df = df.sort_values('energy')[0:max_conformers]
    i = 0
    for conf in df.conformer:
        if multiplicity:
            for mult in multiplicities:
                conf_copy = conf.copy()
                conf_copy.index = i
                conf_copy.rmg_molecule.multiplicity = mult
                confs.append(conf_copy)
                i += 1
        else:
            conf.index = i
            confs.append(conf)
            i += 1

    logging.info(f"We have identified {len(confs)} unique, low-energy conformers for {conformer}")

    return confs
