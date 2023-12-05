import os
import _pickle as pickle

from metrics import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from sklearn.metrics import f1_score
import tensorflow_addons as tfa
#from keras.backend import set_session

import numpy as np

"""
from loss import medical_codes_loss
from metrics import EvaluateCodesCallBack
from utils import DataGenerator
"""

mimic3_path = os.path.join('data', 'mimic3')
encoded_path = os.path.join(mimic3_path, 'encoded')
standard_path = os.path.join(mimic3_path, 'standard')


def load_data():
    labs_dataset = pickle.load(open(os.path.join(standard_path, 'labs_dataset.pkl'), 'rb'))
    return labs_dataset


if __name__ == '__main__':
    labs_dataset = load_data()
    train_labs_data, valid_labs_data, test_labs_data = labs_dataset['train_labs_data'], labs_dataset['valid_labs_data'], labs_dataset['test_labs_data']
    

    (train_single_lab_x, train_single_lab_y) = train_labs_data
    (valid_single_lab_x, valid_single_lab_y) = valid_labs_data
    (test_single_lab_x, test_single_lab_y) = test_labs_data
    
    train_single_lab_x = tf.convert_to_tensor(train_single_lab_x, dtype=tf.float32)
    train_single_lab_y = tf.convert_to_tensor(train_single_lab_y, dtype=tf.float32)
    valid_single_lab_x = tf.convert_to_tensor(valid_single_lab_x, dtype=tf.float32)
    valid_single_lab_y = tf.convert_to_tensor(valid_single_lab_y, dtype=tf.float32)
    test_single_lab_x = tf.convert_to_tensor(test_single_lab_x, dtype=tf.float32)
    test_single_lab_y = tf.convert_to_tensor(test_single_lab_y, dtype=tf.float32)
    
    """
    config = tf.compat.v1.ConfigProto
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)
    """
    print(len(train_single_lab_x), len(valid_single_lab_x), len(test_single_lab_x))     ### 30859, 1816, 3630
    
    
    ### Lab
    item_num = len(train_single_lab_x[0])       #682
    code_num = len(train_single_lab_y[0])       #6398
    
    model = keras.Sequential()
    model.add(keras.Input(shape = (item_num, )))     ### Input layer: n*max_admission*out_dim
    model.add(layers.Dense(200, activation = "relu"))
    model.add(layers.Dense(code_num, activation = "sigmoid"))
    
    #tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer='adam',
              loss= tf.keras.losses.BinaryCrossentropy(from_logits=False), 
              metrics=[tf.keras.metrics.Recall(top_k = 20, thresholds = 0.1),
                       tf.keras.metrics.Recall(top_k = 40, thresholds = 0.1),
                       tfa.metrics.F1Score(num_classes = code_num, average="weighted", threshold = 0.1, name = "weighted_f1")])
    
    #tf.keras.metrics.F1Score(average="weighted", threshold = 0.1, name = "weighted_f1")
    #tfa.metrics.F1Score(num_classes = code_num, average="weighted", threshold = 0.1, name = "weighted_f1")
    model.fit(train_single_lab_x, train_single_lab_y, validation_data = (valid_single_lab_x, valid_single_lab_y), batch_size=32, epochs = 100, verbose = 2)
    model.evaluate(test_single_lab_x, test_single_lab_y)
    
    ### Save Pre-trained Model
    pre_trained = os.path.join(mimic3_path, 'Pre-trained')
    if not os.path.exists(pre_trained):
        os.makedirs(pre_trained)
    model.save(pre_trained)
    
    
    """
    sess.close()
    tf.reset_default_graph()
    """
    
    