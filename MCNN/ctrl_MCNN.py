import subprocess
import sys
import os

def run_mcnn(params):
    cmd = ['python', 'MCNN.py'] + params_to_args(params)
    cmd = " ".join(cmd)
    print(cmd)
    with open(params['path_result'] + '.log', 'w') as log_file:
        mcnn_process = subprocess.Popen(['python', 'MCNN.py'] + params_to_args(params),
                                        stdout=log_file,
                                        stderr=subprocess.STDOUT,
                                        text=True)
        mcnn_process.wait()


def params_to_args(params):
    args = []
    for key, value in params.items():
        if isinstance(value, list):
            for v in value:
                args.append('--' + key)
                args.append(str(v))
        else:
            args.append('--' + key)
            args.append(str(value))
    return args


DATA_TYPE = "MMseqs2"

path_x_train = os.path.join("D:", os.sep, "htchang", "DPCR", "DataSet", "DBpred","Series", "ProtTrans", "Train")
path_y_train = os.path.join("D:", os.sep, "htchang", "DPCR", "Data", "DBpred", "FASTA", "Train", "label")
path_x_test = os.path.join("D:", os.sep, "htchang", "DPCR", "DataSet", "DBpred","Series", "ProtTrans", "Test")
path_y_test = os.path.join("D:", os.sep, "htchang", "DPCR", "Data", "DBpred", "FASTA", "Test", "label")

if __name__ == "__main__":
    params = {
        'path_x_train': path_x_train,
        'path_y_train': path_y_train,
        'path_x_test': path_x_test,
        'path_y_test': path_y_test,
        'path_result': 'D:\htchang\DPCR\ProtTrans.result',
        'num_dependent': 8,
        'num_filter': 256,
        'num_hidden': 1024,
        'batch_size': 1000,
        'window_sizes': [2, 4, 8],
        'num_classes': 2,
        'num_feature': 20,
        'epochs': 60,
        'validation_mod': 'independent',
        'num_k_fold': 10
    }

    run_mcnn(params)

