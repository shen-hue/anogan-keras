import innvestigate
import numpy as np
import tensorflow as tf
from keras.models import  Model
from keras.layers import Input,  Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.layers import  ReLU

def discriminator_model():
    ## CNN model
    inputs = Input((28,))
    # fc1 = Flatten(input_shape=X_train.shape[1])(inputs)
    fc1 = Dense(512, input_dim=28)(inputs)
    fc1 = ReLU(0.2)(fc1)
    fc2 = Dense(256)(fc1)
    fc2 = ReLU(0.2)(fc2)
    outputs = Dense(1)(fc2)
    # fc3 = Dense(1)(fc2)         # not for simple NN model(WGAN-GP)
    # outputs = Activation('sigmoid')(fc3)    # not for simple NN model(WGAN-GP)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

inputs = np.load('result_credit_NN/X_test.npy')

model = discriminator_model()
# model.load_weights('weights/discriminator.h5')
analyzer = innvestigate.create_analyzer("gradient", model)
analysis = analyzer.analyze(inputs)
print(analysis)