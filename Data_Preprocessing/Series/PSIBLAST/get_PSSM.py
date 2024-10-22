import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-in","--path_input", type=str, help="the path of input PSSM file")
parser.add_argument("-out","--path_output", type=str, help="the path of output Binary matrix file")

def read_pssm(pssm_file):
    with open(pssm_file, 'r') as file:
        lines = file.readlines()

    # Trim the header and footer
    lines = lines[3:-6]

    # Create a list of lists, where each sublist contains the values for one row
    matrix = [list(map(int, line.split()[22:42])) for line in lines]
    
    # Convert to numpy array
    matrix = np.array(matrix)

    return matrix

def save_binary_matrix(matrix, output_path):
    np.savetxt(output_path, matrix)

def main(pssm_file, output_file):
    matrix = read_pssm(pssm_file)
    save_binary_matrix(matrix, output_file)

    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args.path_input, args.path_output)