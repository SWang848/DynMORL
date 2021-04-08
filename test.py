import numpy as np
import random
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import *
from scipy.spatial import distance


def cal_similarity(old, current):
    state = {'old':old, 'current':current}
    for key, value in state.items():

        input_t = tf.convert_to_tensor(value, dtype=np.float32)
        x = Lambda(lambda x: x / 255., name="input_normalizer")(input_t)

        x = TimeDistributed(Conv2D(filters=32, kernel_size=6, strides=2, 
                                    activation='relu', kernel_initializer='glorot_uniform',
                                    input_shape=x.shape))(x)
        x = TimeDistributed(MaxPool2D())(x)

        x = TimeDistributed(Conv2D(filters=48, kernel_size=5, strides=2, 
                                    activation='relu', kernel_initializer='glorot_uniform'))(x)
        x = TimeDistributed(MaxPool2D())(x)

        x = Dense(256, activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Dense(128, activation='relu', kernel_initializer='glorot_uniform')(x)
        x = Dense(64, activation='relu', kernel_initializer='glorot_uniform')(x)

        x = Flatten()(x)
        state[key] = x.numpy()

    dist = np.linalg.norm(state['current']-state['old'])
    
    # print(dist)
    return dist


# old = np.random.randint(256, size=(1, 2, 48, 48, 3))
# current = np.random.randint(256, size=(1, 2, 48, 48, 3))
# old = np.ones((1, 2, 48, 48, 3))
# current = np.zeros((1, 2, 48, 48, 3))
# if (old == current).all():
#     print(np.linalg.norm(old-current))

if np.array_equal(old, current):
    print(np.linalg.norm(old-current))

# a = np.ones((1, 2))
# b = np.ones((1, 2))

# print(np.sqrt(sum_sq))
# print(np.linalg.norm(a - b))

cal_similarity(old, current)