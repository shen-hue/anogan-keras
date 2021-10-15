from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Input, Reshape, Dense, Dropout, MaxPooling2D, Conv2D, Flatten
from keras.layers import Conv2DTranspose, LeakyReLU
from keras.layers.core import Activation
from keras.layers import BatchNormalization
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

# ### simple NNmodel(WGAN-GP)
# class RandomWeightedAverage(_Merge):
#     def _merge_function(self, inputs,BATCH_SIZE):
#         weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
#         return (weights * inputs[0]) + ((1 - weights) * inputs[1])


### generator model define
def generator_model():
    ### simple NN model
    inputs = Input((10,))
    fc1 = Dense(256, input_dim=10)(inputs)
    fc1 = LeakyReLU(0.2)(fc1)
    fc1 = BatchNormalization(momentum=0.8)(fc1)
    fc2 = Dense(512)(fc1)
    fc2 = LeakyReLU(0.2)(fc2)
    fc2 = BatchNormalization(momentum=0.8)(fc2)
    fc3 = Dense(1024)(fc2)
    fc3 = LeakyReLU(0.2)(fc3)
    fc3 = BatchNormalization(momentum=0.8)(fc3)
    fc4 = Dense(6)(fc3)
    outputs = Activation('tanh')(fc4)

    # outputs = Reshape(X_train.shape[1])(fc4)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

### discriminator model define
def discriminator_model():
    ### simple NN model
    inputs = Input((6,))
    # fc1 = Flatten(input_shape=X_train.shape[1])(inputs)
    fc1 = Dense(512, input_dim=6)(inputs)
    fc1 = LeakyReLU(0.2)(fc1)
    fc2 = Dense(256)(fc1)
    fc2 = LeakyReLU(0.2)(fc2)
    outputs = Dense(1)(fc2)
    # fc3 = Dense(1)(fc2)         # not for simple NN model(WGAN-GP)
    # outputs = Activation('sigmoid')(fc3)    # not for simple NN model(WGAN-GP)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

### d_on_g model for training generator
def generator_containing_discriminator(g, d):
    d.trainable = False
    ganInput = Input(shape=(10,))
    x = g(ganInput)
    ganOutput = d(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    # gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

def load_model():
    d = discriminator_model()
    g = generator_model()
    d_optim = RMSprop()
    g_optim = RMSprop(lr=0.0002)
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
    d_optim = RMSprop(lr=0.0004)
    g_optim = RMSprop(lr=0.0002)
    g.compile(loss='mse', optimizer=g_optim)
    d_on_g.compile(loss='mse', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='mse', optimizer=d_optim)
    

    for epoch in range(50):
        print ("Epoch is", epoch)
        n_iter = int(X_train.shape[0]/BATCH_SIZE)
        progress_bar = Progbar(target=n_iter)
        
        for index in range(n_iter):
            # create random noise -> U(0,1) 10 latent vectors
            noise = np.random.uniform(0, 1, size=(BATCH_SIZE, 10))

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
            y = np.array([1] * BATCH_SIZE + [-1] * BATCH_SIZE)     # 0 for simple NN model, -1 for simple NN model(WGAN-GP)
            
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
    noise = np.random.uniform(0, 1, (BATCH_SIZE, 10))
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
    g = Model(inputs=g.layers[1].input, outputs=g.layers[-1].output)
    g.trainable = False
    # Input layer cann't be trained. Add new layer as same size & same distribution
    aInput = Input(shape=(6,))
    fc1 = Dense(512, input_dim=6)(aInput)
    fc2 = LeakyReLU(0.2)(fc1)
    fc3 = Dense(256)(fc2)
    fc4 = LeakyReLU(0.2)(fc3)
    gInput = Dense(10)(fc4)
    
    # G & D feature
    G_out = g(gInput)
    # D_out= intermidiate_model(G_out)
    model = Model(inputs=aInput, outputs=G_out)
    model.compile(optimizer='rmsprop',loss='mse')
    
    # batchnorm learning phase fixed (test) : make non trainable
    K.set_learning_phase(0)
    
    return model

### anomaly detection
def compute_anomaly_score(model, x, iterations=500, d=None):
    z = np.random.uniform(0, 1, size=(1, 10))
    
    intermidiate_model = feature_extractor(d)
    d_x = intermidiate_model.predict(x)

    # learning for changing latent
    loss = model.fit(z, [x, d_x], batch_size=1, epochs=iterations, verbose=0)
    similar_data, _ = model.predict(z)
    
    loss = loss.history['loss'][-1]
    
    return loss, similar_data
