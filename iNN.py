import innvestigate
import numpy as np
import tensorflow as tf
from keras.models import  Model
from keras.layers import Input,  Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.layers import  ReLU

def discriminator_model():
    ## CNN model
    inputs = Input((28, 28, 1))
    conv1 = Conv2D(64, (5, 5), padding='same')(inputs)
    conv1 = ReLU(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, (5, 5), padding='same')(pool1)
    conv2 = ReLU(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    fc1 = Flatten()(pool2)
    fc1 = Dense(1)(fc1)
    outputs = Activation('sigmoid')(fc1)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

inputs = np.load('result_credit_CNN/X_test.npy')

model = discriminator_model()
# model.load_weights('weights/discriminator.h5')
analyzer = innvestigate.create_analyzer("gradient", model)
analysis = analyzer.analyze(inputs)
print(analysis)