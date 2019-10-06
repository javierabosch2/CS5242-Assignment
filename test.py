import numpy as np
import warnings
warnings.filterwarnings('ignore')

from nn.layers import Pool2D
from utils.tools import rel_error

from keras import Sequential
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D

input = np.random.uniform(size=(10, 3, 30, 30))
params = { 
    'pool_type': 'avg',
    'pool_height': 4,
    'pool_width': 4,
    'pad': 2,
    'stride': 2,
}
pool = Pool2D(params)
out = pool.forward(input)

keras_pool = Sequential([
    AveragePooling2D(pool_size=(params['pool_height'], params['pool_width']),
                 strides=params['stride'],
                 padding='same',
                 data_format='channels_first',
                 input_shape=input.shape[1:])
])
keras_out = keras_pool.predict(input, batch_size=input.shape[0])

print('Relative error (<1e-6 will be fine): ', rel_error(out, keras_out))