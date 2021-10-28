from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Input, Reshape, Dense, Dropout, MaxPooling2D, Conv2D, Flatten, Bidirectional
from keras.layers import Conv2DTranspose, LeakyReLU,LSTM, TimeDistributed, RepeatVector
from keras.layers.core import Activation
from keras.layers import BatchNormalization, Lambda
from tensorflow.keras.optimizers import Adam,RMSprop
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras import initializers
import tensorflow as tf
import numpy as np
# from tqdm import tqdm
import cv2
import math
from keras import backend as K
from functools import partial
import pandas as pd

from keras.utils. generic_utils import Progbar

### combine images for visualization
# def combine_images(generated_images):
#     num = generated_images.shape[0]
#     width = int(math.sqrt(num))
#     height = int(math.ceil(float(num)/width))
#     shape = generated_images.shape[1:4]
#     ### CNN model
#     # image = np.zeros((height*shape[0], width*shape[1], shape[2]),
#     #                  dtype=generated_images.dtype)
#     # for index, img in enumerate(generated_images):
#     #     i = int(index/width)
#     #     j = index % width
#     #     image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],:] = img[:, :, :]
#
#     ### simple NN model
#     image = np.zeros((height*shape[0], width*shape[1], shape[2]),
#                      dtype=generated_images.dtype)
#     for index, img in enumerate(generated_images):
#         i = int(index/width)
#         j = index % width
#         image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],:] = img[:, :, :]
#
#     return image

### simple NNmodel(WGAN-GP)
def wassersterin_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)

### simple NNmodel(WGAN-GP)
def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,axis=np.arange(1,len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)



### generator model define
def generator_model():
    ### simple NN model
    # inputs = Input((10,))
    # fc1 = Dense(256, input_dim=10)(inputs)
    # fc1 = Activation('relu')(fc1)
    # # fc1 = BatchNormalization(momentum=0.8)(fc1)
    # fc2 = Dense(512)(fc1)
    # fc2 = Activation('relu')(fc2)
    # # fc2 = BatchNormalization(momentum=0.8)(fc2)
    # fc3 = Dense(1024)(fc2)
    # fc3 = Activation('relu')(fc3)
    # # fc3 = BatchNormalization(momentum=0.8)(fc3)
    # outputs = Dense(6)(fc3)
    # # outputs = Activation('relu')(fc4)
    # # outputs = Reshape(X_train.shape[1])(fc4)

    ### LSTM model
    inputs = Input((6,1,))
    # fc1 = Dense(input_dim=10, units=6 * 10)(inputs)
    # fc1 = BatchNormalization()(fc1)
    # fc1 = LeakyReLU(0.2)(fc1)
    # fc1 = Reshape((6, 10))(fc1)
    # fc1 = RepeatVector(6)(inputs)
    lstm1 = LSTM(8, activation='tanh', return_sequences=True)(inputs)
    lstm2 = LSTM(4, activation='tanh', return_sequences=True)(lstm1)
    lstm3 = LSTM(1, activation='tanh', return_sequences=True)(lstm2)
    # def norm2(result):
    #     ill = TimeDistributed(Dense(1)(result))
    #     return ill
    # timedis = Lambda(norm2)(lstm3)
    d = Dense(units=1)(lstm3)
    outputs = Activation('tanh')(d)

    model = Model(inputs=[inputs], outputs=[outputs])
    # model.summary()
    return model

### discriminator model define
def discriminator_model():
    ### simple NN model
    # inputs = Input((6,))
    # # fc1 = Flatten(input_shape=X_train.shape[1])(inputs)
    # fc1 = Dense(512, input_dim=6)(inputs)
    # fc1 = Activation('relu')(fc1)
    # fc2 = Dense(256)(fc1)
    # fc2 = LeakyReLU(0.2)(fc2)
    # outputs = Dense(1)(fc2)
    # # fc3 = Dense(1)(fc2)         # not for simple NN model(WGAN-GP)
    # # outputs = Activation('sigmoid')(fc3)    # not for simple NN model(WGAN-GP)

    ### LSTM model
    inputs = Input((6,1,))
    lstm1 = LSTM(8, return_sequences=True)(inputs)
    lstm2 = LSTM(4, return_sequences=True)(lstm1)
    fc1 = Dense(1)(lstm2)
    fc2 = Flatten()(fc1)
    fc3 = Dense(1)(fc2)
    outputs = Activation('sigmoid')(fc3)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

### d_on_g model for training generator
def generator_containing_discriminator(g, d):
    d.trainable = False
    ganInput = Input(shape=(6,1,))
    x = g(ganInput)
    ganOutput = d(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    # gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

def load_model():
    d = discriminator_model()
    g = generator_model()
    d_optim = Adam()
    g_optim = Adam()
    g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    d.load_weights('./weights/discriminator.h5')
    g.load_weights('./weights/generator.h5')
    return g, d

### train generator and discriminator
def train(BATCH_SIZE, X_train):
    
    ### model define
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = RMSprop(lr=0.001)
    g_optim = RMSprop(lr=0.001)
    g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    

    for epoch in range(250):
        print ("Epoch is", epoch)
        n_iter = int(X_train.shape[0]/BATCH_SIZE)
        progress_bar = Progbar(target=n_iter)
        
        for index in range(n_iter):
            # create random noise -> U(0,1) 10 latent vectors
            noise = np.random.uniform(0, 1, size=(BATCH_SIZE, 6, 1))


            # load real data & generate fake data
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            
            # visualize training results
            # if index % 20 == 0:
            #     image = combine_images(generated_images)
            #     # image = image*127.5+127.5
            #     cv2.imwrite('./result/'+str(epoch)+"_"+str(index)+".png", image)

            # attach label for training discriminator
            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)     # 0 for simple NN model, -1 for simple NN model(WGAN-GP)
            
            # training discriminator
            d_loss = d.train_on_batch(X, y)

            # training generator
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, np.array([1] * BATCH_SIZE))
            d.trainable = True

            progress_bar.update(index, values=[('g',g_loss), ('d',d_loss)])
        print ('')

        # save weights for each epoch
        g.save_weights('weights/generator.h5', True)
        d.save_weights('weights/discriminator.h5', True)
    return d, g

### generate images
def generate(BATCH_SIZE):
    g = generator_model()
    g.load_weights('weights/generator.h5')
    plot_model(g, to_file='model_g.png', show_shapes=True)
    noise = np.random.uniform(0, 1, (BATCH_SIZE, 6, 1))
    generated_images = g.predict(noise)
    return generated_images

### anomaly loss function 
def sum_of_residual(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred))

### discriminator intermediate layer feautre extraction
def feature_extractor(d=None):
    if d is None:
        d = discriminator_model()
        d.load_weights('weights/discriminator.h5')
    plot_model(d, to_file='model_d.png', show_shapes=True)
    intermidiate_model = Model(inputs=d.layers[0].input, outputs=d.layers[-4].output)     ####-5 for simple model,-7 for CNN model
    intermidiate_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return intermidiate_model

### anomaly detection model define
def anomaly_detector(g=None, d=None):
    if g is None:
        g = generator_model()
        g.load_weights('weights/generator.h5')
    intermidiate_model = feature_extractor(d)
    intermidiate_model.trainable = False
    # g = Model(inputs=g.layers[1].input, outputs=g.layers[-1].output)
    g.trainable = False
    # Input layer cann't be trained. Add new layer as same size & same distribution
    aInput = Input(shape=(6,1,))
    # fc1 = Dense(512)(aInput)
    # fc2 = Activation('relu')(fc1)
    # fc3 = Dense(256)(fc2)
    # fc4 = Activation('relu')(fc3)
    gInput = Dense(1)(aInput)
    # gInput = Activation('relu')(gInput)
    # aInput = Input(shape=(10,))
    # gInput = Dense((10), trainable=True)(aInput)
    # gInput = Activation('sigmoid')(gInput)
    
    # G & D feature
    G_out = g(gInput)
    # D_out= intermidiate_model(G_out)
    model = Model(inputs=[aInput], outputs=[G_out])
    # model.compile(optimizer='rmsprop',loss='mse')
    model.compile(loss=sum_of_residual, optimizer='adam')
    # batchnorm learning phase fixed (test) : make non trainable
    K.set_learning_phase(0)
    # model.summary()
    
    return model

### anomaly detection
def compute_anomaly_score(model, x, iterations=50, d=None):
    z = np.random.uniform(0, 1, size=(1, 6,1))
    
    intermidiate_model = feature_extractor(d)
    d_x = intermidiate_model.predict(x)

    # learning for changing latent
    loss = model.fit(z, x, batch_size=1, epochs=iterations, verbose=1)
    similar_data = model.predict(z)
    loss = loss.history['loss'][-1]
    
    return loss, similar_data
