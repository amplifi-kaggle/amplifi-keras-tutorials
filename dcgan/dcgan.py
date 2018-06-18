'''

    DCGAN.py

'''
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Reshape, Flatten, Dropout, Input

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist

from tqdm import tqdm
import numpy as np

#%matplotlib inline
import matplotlib.pyplot as plt

def Generator():

    main_input = Input(shape=(100, ), dtype='float32', name='main_input')
    main_input = Dense(1024)(main_input)

    aux_input = Input(shape=(10, ), dtype='int32', name='aux_input')
    aux_input = Dense(256)(aux_input)

    x = keras.layers.concatenate([main_input, aux_input])
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(x)

    x = Dense(128 * 7 * 7)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(x)

    x = Reshape((7, 7, 128))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, kernel_size=(5, 5), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, kernel_size=(5, 5), padding='same')(x)
    generated_image = Activation('tanh')(x)

    model = Model(inputs=[main_input, aux_input], outputs=[generated_image])


    return model


def Discriminator():


    main_input = Input(shape=(28, 28, 1), dtype='float32', name='main_input')
    main_input = Reshape((784, ))(main_input)

    aux_input = Input(shape=(10, ), dtype='int32', name='aux_input')
    aux_input = Dense(240)(aux_input)

    # 784 + 240 = 1024
    x = keras.layers.concatenate([main_input, aux_input])
    x = Reshape((32, 32))(x)
    x = Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(32, 32, 1))(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, kernel_size=(5, 5), strides=(2, 2))(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)
    probability = Activation('sigmoid')(x)


    model = Model(inputs=[main_input, aux_input], outputs=[probability])

    optimizer = Adam(lr=1e-5, beta_1=0.1)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model



def DCGAN(generator, discriminator):
    discriminator.trainable = False

    model = Sequential()
    model.add(generator)
    model.add(discriminator)

    optimizer = Adam(lr=2e-4, beta_1=0.5)
    # compile: configure the learning process
    # loss: The objective that the model will try to minimize
    # optimizer: could be the string identifier (rmsprop, or adagrad)
    #            or an instance of the Optimizer class
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model


if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train.astype('float32')
    X_test.astype('float32')
    X_train = X_train / 127.5 - 1.0
    X_test = X_test / 127.5 - 1.0
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    print('X_train_shape', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    generator = Generator()
    discriminator = Discriminator()
    gan = DCGAN(generator, discriminator)

    epochs = 30
    batch_size = 32
    input_size = 100

    num_batches = int(X_train.shape[0] / batch_size)

    # To show the progress bar
    # https://pypi.python.org/pypi/tqdm
    pbar = tqdm(total=epochs * num_batches)

    generator_loss = []
    discriminator_loss = []

    for epoch in range(epochs):
        for index in range(num_batches):
            pbar.update(1)


            real_images = X_train[index * batch_size : (index + 1) * batch_size]
            real_labels = y_train[index * batch_size : (index + 1) * batch_size]

            noise = np.random.uniform(-1, 1, size=[batch_size, input_size])			
            condition = np.eye(10)[real_labels]
            noise_condition = np.concatenate((noise, condition), axis=1)

            real_images_condition = np.concatenate((real_images, condition), axis=1)

            # inference
            generated_images = generator.fit([noise, condition], [], epochs=0, batch_size=batch_size)


            # Discriminator tring to:
            # D(x) -> 0 (real)
            # D(G(z)) -> 1 (fake)
            x = np.vstack((generated_images, real_images_condition))
            y = np.zeros(2 * batch_size)
            y[:batch_size] = 1

            # Train discriminator
            d_loss = discriminator.train_on_batch(x=x, y=y)

            # Train generator
            noise = np.random.uniform(-1, 1, size=[batch_size, input_size])
            noise_condition = np.concatenate((noise, condition), axis=1)
            y = np.zeros(batch_size)
            # discriminator.trainable = False (discriminator parameters are fixed)
            # Generator trying to make D(G(z)) -> 0 (real)
            g_loss = gan.train_on_batch(x=noise_condition, y=y)

            discriminator_loss.append(d_loss)
            generator_loss.append(g_loss)


        # Plot losses
        # figsize = (width, height) in inches
        fig = plt.figure(figsize=(10, 5))
        fig.suptitle('epoch: ' + str(epoch + 1))
        plt.plot(discriminator_loss, label="discriminiator's loss", color='b')
        plt.plot(generator_loss, label="generator's loss", color='r')
        plt.xlim([0, epochs * num_batches])
        plt.legend()
        plt.savefig('./dcgan-loss/' + str(epoch + 1) + '.png')
        plt.close()

        # Visualize generated data
        noise = np.random.uniform(-1, 1, size=[10, input_size])
        digits = list(range(10))
        condition = np.eye(10)[digits]
        noise_condition = np.concatenate((noise, condition), axis=1)
        generated_images = generator.predict(noise_condition)

        # figsize = (width, height) in inches
        fig = plt.figure(figsize=(9, 9))
        for i in range(9):
            # (nrows, ncols, index) = return an Axes, at position index of a grid of nrows by ncols axes
            plt.subplot(3, 3, i+1)
            img = generated_images[i, :] * 0.5 + 0.5
            img = img.reshape((28, 28))
            plt.tight_layout()
            plt.imshow(img, cmap='gray')
            plt.axis('off') # do now show any axes
        plt.savefig('./dcgan-images/' + str(epoch + 1) + '.png')
        plt.close()

    pbar.close()



