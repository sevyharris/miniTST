import os
import pickle
import rmgpy.chemkin
import sys
import hashlib
import minitst.species
import xtb.ase.calculator

my_library = '/home/moon/library_mini'


# read in the input file
species_dictionary_file = sys.argv[1]

my_species = rmgpy.chemkin.load_species_dictionary(species_dictionary_file)


for sp_name, species in my_species.items():
    # come up with a nice name for the species:

    adj_list_str = species.to_adjacency_list()
    h = hashlib.md5(string=adj_list_str.encode())
    
    sp_name = species.molecule[0].get_formula() + '_' + h.hexdigest()[:8]


    # TODO, save some sort of persistent database matching file

    # make a working directory for that folder
    sp_dir = os.path.join(my_library, sp_name)
    if not os.path.exists(sp_dir):
        os.makedirs(sp_dir)

    # Make an autotst species
    my_species = minitst.species.Species(species)

    # generate the conformers
    calc = xtb.ase.calculator.XTB()

    my_species.generate_conformers(
        ase_calculator=calc,
        max_combos=1000,
        max_conformers=10,
        results_dir=sp_dir,
        save_results=True,
    )

    break