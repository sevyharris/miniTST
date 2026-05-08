import os
import pickle
import rmgpy.chemkin
import sys
import hashlib
import minitst.species
import xtb.ase.calculator
import glob


my_library = '/home/moon/library_mini'


# read in the input file
species_dictionary_file = sys.argv[1]

my_species = rmgpy.chemkin.load_species_dictionary(species_dictionary_file)

# RMG rules, assume working directory is the same as the input file
library_directory = os.path.dirname(species_dictionary_file)

existing_species_names = glob.glob(os.path.join(library_directory, '*'))
existing_species_names = [os.path.basename(x) for x in existing_species_names]


# update the existing species names with the names of the species in the input file
for sp_name, species in my_species.items():
    # come up with a nice name for the species:

    adj_list_str = species.to_adjacency_list()
    h = hashlib.md5(string=adj_list_str.encode())
    
    sp_name = species.molecule[0].get_formula()

    # see if the name already exists, if so, add a number to the index until we get a unique name
    if sp_name in existing_species_names:
        i = 1
        while sp_name + '_' + str(i) in existing_species_names:
            i += 1
        sp_name = sp_name + '_' + str(i)
    
    sp_dir = os.path.join(library_directory, sp_name)
    if not os.path.exists(sp_dir):
        os.makedirs(sp_dir, exist_ok=True)

    # TODO, save the species object as a pickle file in the directory or just the species dictionary? we'll see

    # TODO, save some sort of persistent database matching file

    # Make an autotst species
    my_species = minitst.species.Species(species)


    # make conformers directory
    conformers_dir = os.path.join(sp_dir, "conformers")
    if not os.path.exists(conformers_dir):
        os.makedirs(conformers_dir, exist_ok=True)

    calc = xtb.ase.calculator.XTB()
    my_species.generate_conformers(
        ase_calculator=calc,
        max_combos=1000,
        max_conformers=10,
        results_dir=sp_dir,
        save_results=True,
    )

    break