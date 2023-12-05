import os
import _pickle as pickle

from metrics import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback, LearningRateScheduler

import numpy as np
from utils import DataGenerator
from metrics import EvaluateCodesCallBackRNNOnly
from loss import medical_codes_loss
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

mimic3_path = os.path.join('data', 'mimic3')
standard_path = os.path.join(mimic3_path, 'standard')

def load_data():
    codes_dataset = pickle.load(open(os.path.join(standard_path, 'codes_dataset.pkl'), 'rb'))
    return codes_dataset


def convert_hot(x):
    result = np.zeros_like(x ,dtype=int)
    for i, examples in enumerate(x):
        for j, step in enumerate(examples):
            if np.all(x[i][j]==0): break
            hot_index = [t-1 for t in x[i][j] if t!= 0]
            result[i][j][hot_index] = 1
    return result


class RNN(keras.layers.Layer):
    def __init__(self, unit, steps, input_dim):
        super().__init__()
        self.block0 = tf.keras.layers.Masking(mask_value=0, input_shape = (steps, input_dim))
        self.block1 = tf.keras.layers.GRU(unit)            ###kernel_regularizer=tf.keras.regularizers.l2(0.01)

    def call(self, codes):
        x = self.block0(codes)
        x = self.block1(x)
        return x
        

class Classifier(keras.layers.Layer):
    def __init__(self, output_dim, name='classifier'):
        super().__init__(name=name)
        self.dense = keras.layers.Dense(output_dim, activation=None)
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x):
        x = self.dropout(x)
        output = self.dense(x)
        return output


class Model(keras.Model):
    def __init__(self, output_dim, steps, input_dim):
        super().__init__()
        self.rnn = RNN(100, steps, input_dim)
        self.classifier = Classifier(output_dim)
    
    def call(self, inputs):
        x = self.rnn(inputs['codes_x'])
        output = self.classifier(x)
        return output



def lr_schedule_fn(epoch, lr):
        if epoch < 20:
            lr = 0.01
        elif epoch < 100:
            lr = 0.001
        elif epoch < 200:
            lr = 0.0001
        else:
            lr = 0.00001
        return lr
    

if __name__ == '__main__':
    more_valid, seed = True, 6666
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    codes_dataset = load_data()
    train_codes_data, valid_codes_data, test_codes_data = codes_dataset['train_codes_data'], codes_dataset['valid_codes_data'], codes_dataset['test_codes_data']
    

    (train_codes_x, train_codes_y, _, _, _) = train_codes_data
    (valid_codes_x, valid_codes_y, _, _, _) = valid_codes_data
    (test_codes_x, test_codes_y, _, _, _) = test_codes_data
    train_codes_x, valid_codes_x, test_codes_x = convert_hot(train_codes_x), convert_hot(valid_codes_x), convert_hot(test_codes_x)
    
    ###
    if more_valid:
        num_valid = int(0.4*(len(valid_codes_x)+len(test_codes_x)))
        tv_codes_x = np.append(test_codes_x, valid_codes_x, axis = 0)
        tv_codes_y = np.append(test_codes_y, valid_codes_y, axis = 0)
        tv_codes_x, tv_codes_y = shuffle(tv_codes_x, tv_codes_y, random_state = seed)
        valid_codes_x, valid_codes_y = tv_codes_x[:num_valid], tv_codes_y[:num_valid]
        test_codes_x, test_codes_y = tv_codes_x[num_valid:], tv_codes_y[num_valid:]
    ###
    
    print(train_codes_x.shape, valid_codes_x.shape, test_codes_x.shape)     #(5331, 14, 6398) (322, 14, 6398) (483, 14, 6398)
    
    
    train_codes_x = tf.convert_to_tensor(train_codes_x, dtype=tf.float32)
    train_codes_y = tf.convert_to_tensor(train_codes_y, dtype=tf.float32)
    valid_codes_x = tf.convert_to_tensor(valid_codes_x, dtype=tf.float32)
    valid_codes_y = tf.convert_to_tensor(valid_codes_y, dtype=tf.float32)
    test_codes_x = tf.convert_to_tensor(test_codes_x, dtype=tf.float32)
    test_codes_y = tf.convert_to_tensor(test_codes_y, dtype=tf.float32)
    
    maxf1 = (0, 0)
    valid_codes_gen = DataGenerator([valid_codes_x], shuffle=False)
    valid_callback = EvaluateCodesCallBackRNNOnly(valid_codes_gen, valid_codes_y, maxf1)
    test_codes_gen = DataGenerator([test_codes_x], shuffle=False)
    test_callback = EvaluateCodesCallBackRNNOnly(test_codes_gen, test_codes_y, None)
    
    
    ### RNN: diagnoses-> diagnoses
    max_admission = len(train_codes_x[0])
    in_dim = len(train_codes_x[0][0])
    out_dim = len(train_codes_y[0])
    
    lr_scheduler = LearningRateScheduler(lr_schedule_fn)
    
    model = Model(out_dim, max_admission, in_dim)
    model.compile(optimizer='adam', loss = medical_codes_loss)
    
    #validation_data = ({'codes_x': valid_codes_x}, valid_codes_y), 
    model.fit(x = {'codes_x': train_codes_x}, y = train_codes_y, validation_data = ({'codes_x': valid_codes_x}, valid_codes_y),
                    batch_size=32, callbacks=[valid_callback, test_callback], epochs = 200, verbose = 2)
    #plt.plot(history.history['val_loss'])
    #plt.savefig('./Result-Plain/kkk.png')
    
    model.summary()
    print(maxf1)
    
    