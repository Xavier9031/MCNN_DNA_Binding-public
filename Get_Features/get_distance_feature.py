import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-in","--path_input", type=str, help="the path of input file")
parser.add_argument("-out","--path_output", type=str, help="the path of output file")
parser.add_argument("-d","--distance_map", type=str, help="the path of distance map file")
parser.add_argument("-n","--top_n", type=int, help="the top n smallest distance")

def loadData(path):
    Data = np.loadtxt(path)
    return Data

def saveData(path, data):
    np.save(path, data)

def get_top_n_smallest_indices(distance_matrix, row, top_n):
    smallest_indices = np.argpartition(distance_matrix[row], top_n)[:top_n]
    sorted_indices = smallest_indices[np.argsort(distance_matrix[row][smallest_indices])]
    return sorted_indices.tolist()

def get_distance_feature(data, distance_map, top_n=17):
    n_1, n_2 = data.shape
    features = np.zeros((n_1, top_n, n_2))
    for i in range(n_1):
        top_n_indices = get_top_n_smallest_indices(distance_map, i, top_n)
        features[i] = data[top_n_indices]
    return features

def main(path_input, path_output, distance_map, top_n):
    data = loadData(path_input)
    distance_map = loadData(distance_map)
    result = get_distance_feature(data, distance_map, top_n)
    saveData(path_output, result)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.path_input, args.path_output, args.distance_map, args.top_n)

