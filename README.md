# MCNN_DNA_Binding

# I. Summery Series Preprocessing 
| get.py              | Input | Output       | Old_ext | New_ext    | Args               |
|---------------------|-------|--------------|---------|------------|--------------------|
| get_BinaryMatrix.py | FASTA | BinaryMatrix | .fasta  | .binary    | -in -out           |
| get_MMseqs2.py      | FASTA | MMseqs2      | .fasta  | .mmseqs2   | -in -out  -db_path |
| get_ProtTrans.py    | FASTA | ProtTrans    | .fasta  | .porttrans | -in -out           |

---
# Usage Guide for get_BinaryMatrix.py


## Description

`get_BinaryMatrix.py` is a Python script that converts a FASTA file into a binary matrix file. The binary matrix is a one-hot encoding of the amino acid sequence in the FASTA file.

## Requirements

- Python 3.5 or later
- tqdm library (can be installed using pip)
## Usage

To use the script, follow these steps:

1. Open a terminal window and navigate to the directory containing the Python script.

2. Run the script using the following command:

    ```
    python get_BinaryMatrix.py -in input_file -out output_file
    ```

    Replace `input_file` with the path to your FASTA file and `output_file` with the desired path for your binary matrix file.

3. Wait for the script to finish processing the FASTA file.

4. Check the output file for the binary matrix.

## Arguments

The script takes the following arguments:

- `-in` or `--path_input`: The path of the input FASTA file.
- `-out` or `--path_output`: The path of the output binary matrix file.

Note: The argument names in the command must match the ones listed above.

---
# Usage Guide for get_MMseqs2.py


## Description

`get_MMseqs2.py` is a Python script that converts a folder of FASTA files into MMseqs2 files. MMseqs2 is a fast and sensitive protein search and clustering algorithm. The script uses MMseqs2 to generate a profile matrix for each FASTA file, which is then converted into an MMseqs2 file.

## Requirements

- Python 3.5 or later
- MMseqs2 software (must be installed and accessible via the command line)
- NumPy library (can be installed using pip)

## Usage

To use the script, follow these steps:

1. Open a terminal window and navigate to the directory containing the Python script.

2. Run the script using the following command:

    ```
    python get_MMseqs2.py -in input_folder -out output_folder -db_path database_path
    ```

    Replace `input_folder` with the path to your folder containing the FASTA files, `output_folder` with the desired path for your MMseqs2 files, and `database_path` with the path to your MMseqs2 database.

3. Wait for the script to finish processing the FASTA files.

4. Check the output folder for the MMseqs2 files.

## Arguments

The script takes the following arguments:

- `-in` or `--path_input`: The path of the input folder containing the FASTA files.
- `-out` or `--path_output`: The path of the output folder for the MMseqs2 files.
- `-db_path` or `--database_path`: The path of the MMseqs2 database.

Note: The argument names in the command must match the ones listed above.

---
# Usage Guide for get_ProtTrans.py


## Description

`get_ProtTrans.py` is a Python script that generates protein embeddings using the ProtTrans model. The script reads in a single FASTA file and outputs a file containing the embeddings.

## Requirements

- Python 3.5 or later
- PyTorch library
- Hugging Face Transformers library
- NumPy library (can be installed using pip)

## Usage

To use the script, follow these steps:

1. Open a terminal window and navigate to the directory containing the Python script.

2. Run the script using the following command:

    ```
    python get_ProtTrans.py -in input_file -out output_file
    ```

    Replace `input_file` with the path to your input FASTA file, and `output_file` with the desired path for your output file.

3. Wait for the script to finish generating the embeddings.

4. Check the output file for the embeddings.

## Arguments

The script takes the following arguments:

- `-in` or `--path_input`: The path of the input FASTA file.
- `-out` or `--path_output`: The path of the output file for the embeddings.

Note: The argument names in the command must match the ones listed above.



# II. Summery Distance Preprocessing 

| get.py             | Input | Output      | Old_ext | New_ext | Args     |
|--------------------|-------|-------------|---------|---------|----------|
| get_AF2_PDB.py     | FASTA | PDB         | .fasta  | .pdb    | -in -out |
| get_DistanceMap.py | PDB   | DistanceMap | .pdb    | .dmap   | -in -out |

```bash
# Example CMD for single processing
python get.py -in path/input/file.old_ext -out path/output/file.new_ext

# Example CMD for multi processing
python batch_proc.py -in path/input/folder -out path/output/folder -script path/get.py -num num_threads -old_ext .old_ext -new_ext .new_ext
```
---
# Usage Guide for get_DistanceMap.py

## Description

`get_DistanceMap.py` is a Python script that reads a PDB file and generates a distance map file. The distance map file contains the distances between all pairs of C-alpha atoms in the protein structure.

## Requirements

- Python 3.5 or later
- NumPy library (can be installed using pip)
- Biopython library (can be installed using pip)

## Usage

To use the script, follow these steps:

1. Open a terminal window and navigate to the directory containing the Python script.

2. Run the script using the following command:

    ```
    python get_DistanceMap.py -in input_pdb_file -out output_distance_map_file
    ```

    Replace `input_pdb_file` with the path to your PDB file, and `output_distance_map_file` with the desired path for your distance map file.

3. Wait for the script to finish processing the PDB file.

4. Check the output folder for the distance map file.

## Arguments

The script takes the following arguments:

- `-in` or `--path_input`: The path of the input PDB file.
- `-out` or `--path_output`: The path of the output distance map file.

Note: The argument names in the command must match the ones listed above.

## Example

To generate a distance map file for a PDB file called `1abc.pdb` located in the folder `/home/user/pdb_files/`, and save the distance map file as `1abc_distance_map.txt` in the folder `/home/user/output/`, run the following command:

    python get_DistanceMap.py -in /home/user/pdb_files/1abc.pdb -out /home/user/output/1abc_distance_map.txt



---
<br>


# II. Get DNA Binding Features

# Usage Guide for get_distance_feature.py


## Description

`get_distance_feature.py` is a Python script that calculates distance-based features for input data based on a distance map file. The distance map file should be a symmetric matrix of pairwise distances between residues in a protein. The script calculates the top `n` smallest distances for each residue in the protein and returns the corresponding `n` rows from the input data as features. The output is a 3D tensor of shape `(num_residues, n, num_features)`.

## Requirements

- Python 3.5 or later
- NumPy library (can be installed using pip)

## Usage

To use the script, follow these steps:

1. Open a terminal window and navigate to the directory containing the Python script.

2. Run the script using the following command:

    ```
    python get_distance_feature.py -in input_file -out output_file -d distance_map -n top_n
    ```

    Replace `input_file` with the path to your input data file, `output_file` with the desired path for your output file, `distance_map` with the path to your distance map file, and `top_n` with the desired number of top distances to use as features.

3. Wait for the script to finish processing the input file.

4. Check the output file for the calculated features.

## Arguments

The script takes the following arguments:

- `-in` or `--path_input`: The path of the input file containing the data to calculate features for.
- `-out` or `--path_output`: The path of the output file for the calculated features.
- `-d` or `--distance_map`: The path of the distance map file.
- `-n` or `--top_n`: The number of top distances to use as features.

Note: The argument names in the command must match the ones listed above.

---
# Usage Guide for batch_get_distance_feature.py

## Description

`batch_get_distance_feature.py` is a Python script that processes a folder of input files and generates a corresponding folder of output files. The script calls `get_distance_feature.py` for each input file to generate its corresponding output file. The `get_distance_feature.py` script generates a feature matrix for each input file, where the feature matrix is based on the distances between amino acids in the protein structure. The `batch_get_distance_feature.py` script takes care of parallel processing of the input files to speed up the processing.

## Requirements

- Python 3.5 or later
- `get_distance_feature.py` script (must be in the same directory or in your system's path)
- NumPy library (can be installed using pip)
- tqdm library (can be installed using pip)

## Usage

To use the script, follow these steps:

1. Open a terminal window and navigate to the directory containing the Python script.

2. Run the script using the following command:

    ```
    python batch_get_distance_feature.py -in input_folder -out output_folder -script path/to/get_distance_feature.py -num num_threads -old_ext old_extension -new_ext new_extension -d path/to/distance_map_file -n top_n
    ```

    Replace `input_folder` with the path to your input folder, `output_folder` with the desired path for your output folder, `path/to/get_distance_feature.py` with the path to your `get_distance_feature.py` script, `num_threads` with the number of threads you want to use for parallel processing, `old_extension` with the file extension of the input files, `new_extension` with the new file extension for the output files, `path/to/distance_map_file` with the path to your distance map file, and `top_n` with the number of top smallest distances to consider for generating the feature matrix.

3. Wait for the script to finish processing the input files.

4. Check the output folder for the output files.

## Arguments

The script takes the following arguments:

- `-in` or `--path_input`: The path of the input folder containing the input files.
- `-out` or `--path_output`: The path of the output folder for the output files.
- `-script` or `--path_script`: The path of the `get_distance_feature.py` script.
- `-num` or `--num_threads`: The number of threads to use for parallel processing.
- `-old_ext` or `--old_filename_extension`: The extension of the input files to be processed.
- `-new_ext` or `--new_filename_extension`: The new extension of the output files.
- `-d` or `--distance_map`: The path of the distance map file.
- `-n` or `--top_n`: The number of top smallest distances to consider for generating the feature matrix.

Note: The argument names in the command must match the ones listed above.

---
# Usage Guide for get_series_feature.py

## Description

`get_series_feature.py` is a Python script that takes a 2D array of numerical data and generates a series feature file. The series feature file contains a sliding window representation of the input data, where each window has a specified window size and the window shifts by one data point at a time.

## Requirements

- Python 3.5 or later
- NumPy library (can be installed using pip)

## Usage

To use the script, follow these steps:

1. Open a terminal window and navigate to the directory containing the Python script.

2. Run the script using the following command:

    ```
    python get_series_feature.py -in input_file -out output_file -w window_size
    ```

    Replace `input_file` with the path to your input data file, `output_file` with the desired path for your series feature file, and `window_size` with the desired window size.

3. Wait for the script to finish processing the input data.

4. Check the output folder for the series feature file.

## Arguments

The script takes the following arguments:

- `-in` or `--path_input`: The path of the input file containing the 2D array of numerical data.
- `-out` or `--path_output`: The path of the output file for the series feature.
- `-w` or `--window_size`: The window size of the sliding window representation.

Note: The argument names in the command must match the ones listed above.

## Example

To generate a series feature file for an input data file called `data.txt` located in the folder `/home/user/data/`, and save the series feature file as `series_feature.npy` in the folder `/home/user/output/`, with a window size of 5, run the following command:

    python get_series_feature.py -in /home/user/data/data.txt -out /home/user/output/series_feature.npy -w 5

---
# Usage Guide for batch_get_series_feature.py


## Description

`batch_get_series_feature.py` is a Python script that processes a batch of input files in a specified folder to generate a series feature file for each input file. The series feature file contains a sliding window representation of the input data, where each window has a specified window size and the window shifts by one data point at a time.

## Requirements

- Python 3.5 or later
- NumPy library (can be installed using pip)
- tqdm library (can be installed using pip)

## Usage

To use the script, follow these steps:

1. Open a terminal window and navigate to the directory containing the Python script.

2. Run the script using the following command:

    ```
    python batch_get_series_feature.py -in input_folder -out output_folder -script script_path -num num_threads -old_ext old_filename_extension -new_ext new_filename_extension -w window_size
    ```

    Replace `input_folder` with the path to your input data folder, `output_folder` with the desired path for your output folder, `script_path` with the path to your series feature script, `num_threads` with the desired number of threads for parallel processing, `old_filename_extension` with the extension of the files to be processed, `new_filename_extension` with the new extension of the output files, and `window_size` with the desired window size.

3. Wait for the script to finish processing the input data.

4. Check the output folder for the series feature files.

## Arguments

The script takes the following arguments:

- `-in` or `--path_input`: The path of the input folder containing the files to be processed.
- `-out` or `--path_output`: The path of the output folder for the series feature files.
- `-script` or `--path_script`: The path of the Python script that generates the series feature files.
- `-num` or `--num_threads`: The number of threads to use for parallel processing.
- `-old_ext` or `--old_filename_extension`: The extension of the files to be processed.
- `-new_ext` or `--new_filename_extension`: The new extension of the output files.
- `-w` or `--window_size`: The window size of the sliding window representation.

Note: The argument names in the command must match the ones listed above.

## Example

To process a batch of input files with extension `.txt` located in the folder `/home/user/input/`, generate a series feature file with a window size of 5 for each input file, and save the output files with extension `.npy` in the folder `/home/user/output/`, run the following command:

    python batch_get_series_feature.py -in /home/user/input/ -out /home/user/output/ -script get_series_feature.py -num 4 -old_ext .txt -new_ext .npy -w 5

Note: The number of threads (in this example, 4) should be set based on the number of available CPU cores.
