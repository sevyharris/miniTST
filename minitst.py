import os
import pickle

import logging
import rmgpy.chemkin
import sys
import hashlib
import minitst.species
import xtb.ase.calculator
import yaml
import glob


my_library = '/home/moon/library_mini'


# read in the input file
species_dictionary_file = sys.argv[1]

my_species = rmgpy.chemkin.load_species_dictionary(species_dictionary_file)

# RMG rules, assume working directory is the same as the input file
library_directory = os.path.dirname(species_dictionary_file)

existing_species_names = glob.glob(os.path.join(library_directory, '*'))
existing_species_names = [os.path.basename(x) for x in existing_species_names]


species_database_file = os.path.join(library_directory, 'database.yaml')
if not os.path.exists(species_database_file):
    with open(species_database_file, 'w') as f:
        yaml.safe_dump([], f)

with open(species_database_file, 'r') as f:
    species_database = yaml.safe_load(f)


# update the existing species names with the names of the species in the input file
for sp_given_name, species in my_species.items():

    # look up the species name in the database
    for entry in species_database:
        adj_list = entry['adjacency_list']
        if species.is_isomorphic(rmgpy.molecule.Molecule().from_adjacency_list(adj_list)):
            logging.info(f'Found {sp_given_name} in the database with name {entry["name"]}. Skipping...')
            break
    else:
        logging.info(f'{sp_given_name} not found in the database. Adding to the library...')
        if sp_given_name in ['AUTO']:
            names_in_database = [entry['name'] for entry in species_database]
            # make up a name for the species if it doesn't have one
            new_name = species.molecule[0].get_formula()
            if new_name in names_in_database:
                i = 1
                while new_name + '_' + str(i) in names_in_database:
                    i += 1
                new_name = new_name + '_' + str(i)
            sp_given_name = new_name

        # add the species to the database
        new_entry = {
            'name': sp_given_name,
            'adjacency_list': species.to_adjacency_list()
        }

        species_database.append(new_entry)

with open(species_database_file, 'w') as f:
    yaml.safe_dump(species_database, f)


for entry in species_database:

    adj_list_str = entry['adjacency_list']
    sp_name = entry['name']    
    sp_dir = os.path.join(library_directory, sp_name)
    if not os.path.exists(sp_dir):
        os.makedirs(sp_dir, exist_ok=True)

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

    # break