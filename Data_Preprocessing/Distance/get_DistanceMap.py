import argparse
import numpy as np
from Bio.PDB import PDBParser

parser = argparse.ArgumentParser()
parser.add_argument("-in","--path_input", type=str, help="the path of input PDB file")
parser.add_argument("-out","--path_output", type=str, help="the path of output distance map file")


def read_pdb_file(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)
    return structure


def get_ca_coordinates(structure):
    ca_coordinates = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id("CA"):
                    ca_coordinates.append(residue["CA"].get_coord())

    return np.array(ca_coordinates)


def calculate_distance_matrix(ca_coordinates):
    distance_matrix = np.zeros((len(ca_coordinates), len(ca_coordinates)))

    for i, coord1 in enumerate(ca_coordinates):
        for j, coord2 in enumerate(ca_coordinates):
            distance_matrix[i, j] = np.linalg.norm(coord1 - coord2)

    return distance_matrix


def save_distance_map(distance_matrix, output_file):
    np.savetxt(output_file, distance_matrix, fmt="%.3f")


def main(pdb_file, output_file):
    structure = read_pdb_file(pdb_file)
    ca_coordinates = get_ca_coordinates(structure)
    distance_matrix = calculate_distance_matrix(ca_coordinates)
    save_distance_map(distance_matrix, output_file)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.path_input, args.path_output)