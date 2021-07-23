from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Input, Reshape, Dense, Dropout, MaxPooling2D, Conv2D, Flatten
from keras.layers import Conv2DTranspose, LeakyReLU
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras import initializers
import tensorflow as tf
import numpy as np
# from tqdm import tqdm
import cv2
import math
from functools import partial
from keras.layers.merge import _Merge


from keras.utils.generic_utils import Progbar


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
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


### simple NNmodel(WGAN-GP)
def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


### simple NNmodel(WGAN-GP)
class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs, BATCH_SIZE=64):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


### generator model define
def generator_model():
    ### CNN model
    # inputs = Input((10,))
    # fc1 = Dense(input_dim=10, units=128*7*7)(inputs)
    # fc1 = BatchNormalization()(fc1)
    # fc1 = LeakyReLU(0.2)(fc1)
    # fc2 = Reshape((7, 7, 128), input_shape=(128*7*7,))(fc1)
    # up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(fc2)
    # conv1 = Conv2D(64, (3, 3), padding='same')(up1)
    # conv1 = BatchNormalization()(conv1)
    # conv1 = Activation('relu')(conv1)
    # up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv1)
    # conv2 = Conv2D(1, (5, 5), padding='same')(up2)
    # outputs = Activation('tanh')(conv2)
    ### simple NN model
    model = Sequential()
    model.add(Dense(256, input_dim=10))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(28))
    model.add(Activation('tanh'))

    return model


### discriminator model define
def discriminator_model():
    ### CNN model
    # inputs = Input((28, 28, 1))
    # conv1 = Conv2D(64, (5, 5), padding='same')(inputs)
    # conv1 = LeakyReLU(0.2)(conv1)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # conv2 = Conv2D(128, (5, 5), padding='same')(pool1)
    # conv2 = LeakyReLU(0.2)(conv2)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # fc1 = Flatten()(pool2)
    # fc1 = Dense(1)(fc1)
    # outputs = Activation('sigmoid')(fc1)
    ### simple NN model
    model = Sequential()
    model.add(Dense(512, input_dim=28))
    model.add(LeakyReLU(0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1))

    return model


### d_on_g model for training generator
# def generator_containing_discriminator(g, d):
#     d.trainable = False
#     ganInput = Input(shape=(10,))
#     x = g(ganInput)
#     ganOutput = d(x)
#     gan = Model(inputs=ganInput, outputs=ganOutput)
#     # gan.compile(loss='binary_crossentropy', optimizer='adam')
#     return gan


def load_model():
    # load discriminator_model
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
    TRAINING_RATIO = 5
    ### model define
    # define discriminator and generator model
    d_model = discriminator_model()
    g_model = generator_model()

    ## trainable of generator model
    for layer in d_model.layers:
        layer.trainable = False
    d_model.trainable = False
    ##inputs of generator model
    generator_input = Input(shape=(10,))
    generator_layers = g_model(generator_input)
    discriminator_layers_for_generator = d_model(generator_layers)
    ## generator model
    g = Model(inputs=[generator_input],
                            outputs=[discriminator_layers_for_generator])
    g.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=wasserstein_loss)

    ## trainable of discriminator
    for layer in d_model.layers:
        layer.trainable = True
    for layer in g_model.layers:
        layer.trainable = False
    d_model.trainable = True
    g_model.trainable = False
    ## inputs of discriminator model
    real_samples = Input(shape=(28,))
    generator_input_for_discriminator = Input(shape=(10,))
    ## outputs of discriminator model
    generated_samples_for_discriminator = g_model(generator_input_for_discriminator)
    discriminator_output_from_generator = d_model(generated_samples_for_discriminator)
    discriminator_output_from_real_samples = d_model(real_samples)
    averaged_samples = RandomWeightedAverage()([real_samples,generated_samples_for_discriminator])
    averaged_samples_out = d_model(averaged_samples)
    ## loss of discriminator model
    GRADIENT_PENALTY_WEIGHT = 10
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
    partial_gp_loss.__name__ = 'gradient_penalty'
    ## discriminator model
    d = Model(inputs=[real_samples,generator_input_for_discriminator],
              outputs=[discriminator_output_from_real_samples,
                       discriminator_output_from_generator,
                       averaged_samples_out])
    d_optim = Adam(0.0001, beta_1=0.5, beta_2=0.9)
    d.trainable = True
    d.compile(optimizer=d_optim, loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])


    minibatches_size = BATCH_SIZE * TRAINING_RATIO
    for epoch in range(10):
        print("Epoch is", epoch)
        n_iter = int(X_train.shape[0] // (minibatches_size))
        progress_bar = Progbar(target=n_iter)

        for index in range(n_iter):
            discriminator_minibatches = X_train[index * minibatches_size:
                                                (index + 1) * minibatches_size]
            discriminator_minibatches = discriminator_minibatches.astype(np.float32)
            for j in range(TRAINING_RATIO):
                image_batch = discriminator_minibatches[j * BATCH_SIZE:
                                                        (j + 1) * BATCH_SIZE]
                noise = np.random.rand(BATCH_SIZE, 10).astype(np.float32)
                positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
                negative_y = -positive_y
                dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
                d_loss = d.train_on_batch([image_batch,noise],[positive_y,negative_y,dummy_y])
            g_loss = g.train_on_batch(np.random.rand(BATCH_SIZE,10),positive_y)

        progress_bar.update(index, values=[('g', g_loss), ('d', d_loss[0])])
        print('')

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
    intermidiate_model = Model(inputs=d.layers[0].input,
                               outputs=d.layers[-4].output)  ####-5 for simple model,-7 for CNN model
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
    aInput = Input(shape=(10,))
    gInput = Dense((10), trainable=True)(aInput)
    gInput = Activation('sigmoid')(gInput)

    # G & D feature
    G_out = g(gInput)
    D_out = intermidiate_model(G_out)
    model = Model(inputs=aInput, outputs=[G_out, D_out])
    model.compile(loss=sum_of_residual, loss_weights=[0.90, 0.10], optimizer='rmsprop')

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
