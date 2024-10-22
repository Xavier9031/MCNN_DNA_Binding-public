# Dependency
import h5py
import os
from tqdm import tqdm
from time import gmtime, strftime
import numpy as np
import math
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import roc_curve
import tensorflow as tf
from tensorflow.keras import Model, layers
from sklearn.model_selection import KFold
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-x_train","--path_x_train", type=str, help="the path of input path_x_train")
parser.add_argument("-y_train","--path_y_train", type=str, help="the path of input path_y_train")
parser.add_argument("-x_test","--path_x_test", type=str, help="the path of input path_x_test")
parser.add_argument("-y_test","--path_y_test", type=str, help="the path of input path_y_test")
parser.add_argument("-r","--path_result", type=str, help="the path of output results")
parser.add_argument("-n_dep","--num_dependent", type=int, default=8, help="the number of dependent variables")
parser.add_argument("-n_fil","--num_filter", type=int, default=256, help="the number of filters in the convolutional layer")
parser.add_argument("-n_hid","--num_hidden", type=int, default=1024, help="the number of hidden units in the dense layer")
parser.add_argument("-bs","--batch_size", type=int, default=1000, help="the batch size")
parser.add_argument("-ws","--window_sizes", nargs="+", type=int, default=[2,4,8], help="the window sizes for convolutional filters")
parser.add_argument("-n_cls","--num_classes", type=int, default=2, help="the number of classes")
parser.add_argument("-n_feat","--num_feature", type=int, default=20, help="the number of features")
parser.add_argument("-e","--epochs", type=int, default=60, help="the number of epochs for training")
parser.add_argument("-val","--validation_mod", type=str, default="independent", help="the mod for validation 'cross' or 'independent'")
parser.add_argument("-k_fold","--num_k_fold", type=int, default=10, help="the number of k for k_fold cross validation")

NUM_CLASSES = 2
# Time_log
def time_log(message):
    print(message," : ",strftime("%Y-%m-%d %H:%M:%S", gmtime()))

def MCNN_data_load(x_folder, y_folder):
    x_train = []
    y_train = []

    x_files = [file for file in os.listdir(x_folder) if file.endswith('.set.npy')]
    
    # Iterate through x_folder with tqdm
    for file in tqdm(x_files, desc="Loading data", unit="file"):
        x_path = os.path.join(x_folder, file)
        x_data = np.load(x_path)
        x_train.append(x_data)

        # Get the corresponding y file
        y_file = file[:-8] + '.label'
        y_path = os.path.join(y_folder, y_file)

        try:
            with open(y_path, 'r') as y_f:
                lines = y_f.readlines()
                y_data = np.array([int(x) for x in lines[1].strip()])
                y_train.append(y_data)
        except:
             print(f"No such file or directory : {y_path}")

    # Concatenate all the data
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Add new dimensions to x_train and y_train
    x_train = np.expand_dims(x_train, axis=1)
    y_train = np.expand_dims(y_train, axis=1)
    y_train = tf.keras.utils.to_categorical(y_train,NUM_CLASSES)
    
    return x_train, y_train

class DeepScan(Model):
	def __init__(self,
	            maxseq=17,
                num_feature=20,
	            window_sizes=[2,4,6,8],
	            num_filters=256,
	            num_hidden=1024):
		super(DeepScan, self).__init__()
		# Add input layer
		self.input_layer = tf.keras.Input((1, maxseq, num_feature))
		self.window_sizes = window_sizes
		self.conv2d = []
		self.maxpool = []
		self.flatten = []
		for window_size in self.window_sizes:
			self.conv2d.append(
			 layers.Conv2D(filters=num_filters,
			               kernel_size=(1, window_size),
			               activation=tf.nn.relu,
			               padding='valid',
			               bias_initializer=tf.constant_initializer(0.1),
			               kernel_initializer=tf.keras.initializers.GlorotUniform()))
			self.maxpool.append(
			 layers.MaxPooling2D(pool_size=(1, maxseq - window_size + 1),
			                     strides=(1, maxseq),
			                     padding='valid'))
			self.flatten.append(layers.Flatten())
		self.dropout = layers.Dropout(rate=0.7)
		self.fc1 = layers.Dense(
		 num_hidden,
		 activation=tf.nn.relu,
		 bias_initializer=tf.constant_initializer(0.1),
		 kernel_initializer=tf.keras.initializers.GlorotUniform())
		self.fc2 = layers.Dense(NUM_CLASSES,
		                        activation='softmax',
		                        kernel_regularizer=tf.keras.regularizers.l2(1e-3))

		# Get output layer with `call` method
		self.out = self.call(self.input_layer)

	def call(self, x, training=False):
		_x = []
		for i in range(len(self.window_sizes)):
			x_conv = self.conv2d[i](x)
			x_maxp = self.maxpool[i](x_conv)
			x_flat = self.flatten[i](x_maxp)
			_x.append(x_flat)

		x = tf.concat(_x, 1)
		x = self.dropout(x, training=training)
		x = self.fc1(x)
		x = self.fc2(x)  #Best Threshold
		return x


# Model Train
def model_train(x_train,y_train,BATCH_SIZE,EPOCHS,MAXSEQ,NUM_FEATURE,WINDOW_SIZES,NUM_FILTER,NUM_HIDDEN):
    model = DeepScan(
        maxseq=MAXSEQ,
        num_feature=NUM_FEATURE,
        num_filters=NUM_FILTER,
        num_hidden=NUM_HIDDEN,
        window_sizes=WINDOW_SIZES)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.build(input_shape=x_train.shape)
    model.summary()

    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
    )
    return model


def model_test(model, x_test, y_test):

    pred_test = model.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test[:,1], pred_test[:, 1])
    AUC = metrics.auc(fpr, tpr)

    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print(f'Best Threshold={thresholds[ix]}, G-Mean={gmeans[ix]}')
    threshold = thresholds[ix]

    y_pred = (pred_test[:, 1] >= threshold).astype(int)

    TN, FP, FN, TP =  metrics.confusion_matrix(y_test[:,1], y_pred).ravel()

    Sens = TP/(TP+FN) if TP+FN > 0 else 0.0
    Spec = TN/(FP+TN) if FP+TN > 0 else 0.0
    Acc = (TP+TN)/(TP+FP+TN+FN)
    MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) if TP+FP > 0 and FP+TN > 0 and TP+FN and TN+FN else 0.0
    F1 = 2*TP/(2*TP+FP+FN)
    print(f'TP={TP}, FP={FP}, TN={TN}, FN={FN}, Sens={Sens:.4f}, Spec={Spec:.4f}, Acc={Acc:.4f}, MCC={MCC:.4f}\n')
    return TP,FP,TN,FN,Sens,Spec,Acc,MCC,AUC


def main(path_x_train, path_y_train, path_x_test, path_y_test, path_result, NUM_DEPENDENT, WINDOW_SIZES, NUM_FILTER, NUM_HIDDEN, BATCH_SIZE, EPOCHS, NUM_CLASSES, MAXSEQ, NUM_FEATURE, VAL_MOD, K_FOLD):
    print(f"VAL_MOD = {VAL_MOD}")
    print(f"WINDOW_SIZES = {WINDOW_SIZES}")
    print(f"NUM_FILTER = {NUM_FILTER}")
    print(f"NUM_HIDDEN = {NUM_HIDDEN}")
    print(f"BATCH_SIZE = {BATCH_SIZE}")
    print(f"EPOCHS = {EPOCHS}")
    print(f"NUM_DEPENDENT = {NUM_DEPENDENT}")
    print(f"MAXSEQ = {MAXSEQ}")
    print(f"NUM_FEATURE = {NUM_FEATURE}")
    prams_log=f"{NUM_DEPENDENT},{WINDOW_SIZES},{NUM_FILTER},{NUM_HIDDEN},{BATCH_SIZE},{EPOCHS},{NUM_CLASSES},{MAXSEQ},{NUM_FEATURE},{VAL_MOD},{K_FOLD}"
    # Data Load
    time_log("Start Load Train Data")
    x_train, y_train = MCNN_data_load(path_x_train, path_y_train)
    print(f"x_train.shape = {x_train.shape}")
    print(f"y_train.shape = {y_train.shape}")
    time_log("End Load Train Data")
    
    
    if(VAL_MOD=="cross"):
        print(f"K_FOLD = {K_FOLD}")
        time_log("Start Cross Validation")
        kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=42)
        results = []
        for i, (train_index, test_index) in tqdm(enumerate(kf.split(x_train)), total=K_FOLD, desc="Cross Validation"):
            x_train_cv, x_test_cv = x_train[train_index], x_train[test_index]
            y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
            model = model_train(x_train_cv, y_train_cv, BATCH_SIZE, EPOCHS, MAXSEQ, NUM_FEATURE, WINDOW_SIZES, NUM_FILTER, NUM_HIDDEN)
            TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC = model_test(model, x_test_cv, y_test_cv)
            results.append([TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC])
        #取K_FOLD個的平均
        mean_results = np.mean(results, axis=0)
        time_log("End Cross Validation")

        with open(path_result, 'w') as f:
            f.write(f"{prams_log},{mean_results[0]},{mean_results[1]},{mean_results[2]},{mean_results[3]},{mean_results[4]},{mean_results[5]},{mean_results[6]},{mean_results[7]},{mean_results[8]}")
    
    elif(VAL_MOD=="independent"):
        time_log("Start Load Train Data")
        x_test, y_test = MCNN_data_load(path_x_test, path_y_test)
        print(f"x_test.shape = {x_test.shape}")
        print(f"y_test.shape = {y_test.shape}")
        time_log("End Load Train Data")

        time_log("Start Model Train")
        model = model_train(x_train,y_train,BATCH_SIZE,EPOCHS,MAXSEQ,NUM_FEATURE,WINDOW_SIZES,NUM_FILTER,NUM_HIDDEN)
        time_log("End Model Train")

        time_log("Start Model Test")
        TP,FP,TN,FN,Sens,Spec,Acc,MCC,AUC = model_test(model, x_test, y_test)
        time_log("End Model Test")

        with open(path_result, 'w') as f:
             f.write(f"{prams_log},{TP},{FP},{TN},{FN},{Sens},{Spec},{Acc},{MCC},{AUC}")
    else:
         print("Invalid VAL_MOD")
         return
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.path_x_train,
         args.path_y_train,
         args.path_x_test,
         args.path_y_test,
         args.path_result,
         args.num_dependent,
         args.window_sizes,
         args.num_filter,
         args.num_hidden,
         args.batch_size,
         args.epochs,
         args.num_classes,
         args.num_dependent*2+1,
         args.num_feature,
         args.validation_mod,
         args.num_k_fold
         )