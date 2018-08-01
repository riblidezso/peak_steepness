import keras
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU, GlobalAveragePooling2D

def get_original_model():
    
    model = Sequential()

    #L1
    model.add(Conv2D(4, (3, 3), input_shape=(256, 256, 1), name='L01_convolution'))
    model.add(LeakyReLU(alpha=0.03, name='L01_activation'))

    #L2
    model.add(AveragePooling2D(pool_size=(2, 2), name='L02_avg_pool'))
    
    #L3
    model.add(Conv2D(12, (3, 3), name='L03_convolution'))
    model.add(LeakyReLU(alpha=0.03, name='L03_activation'))

    #L4
    model.add(Conv2D(12, (3, 3), padding='same', name='L04_convolution'))
    model.add(LeakyReLU(alpha=0.03, name='L04_activation'))

    #L5
    model.add(AveragePooling2D(pool_size=(5, 5), strides=(2, 2), name='L05_avg_pool'))

    #L6
    model.add(Conv2D(32, (3, 3), name='L06_convolution'))

    #L7
    model.add(AveragePooling2D(pool_size=(2, 2), name='L07_avg_pool'))

    #L8
    model.add(Conv2D(64, (3, 3), name='L08_convolution'))

    #L9
    model.add(AveragePooling2D(pool_size=(2, 2), name='L09_avg_pool'))

    #L10
    model.add(AveragePooling2D(pool_size=(2, 2), name='L10_avg_pool'))

    #L11
    model.add(Flatten(name='L11_flatten'))

    #L12
    model.add(Dense(1024, name='L12_dense'))
    model.add(LeakyReLU(alpha=0.03, name='L12_activation'))

    #L13
    model.add(Dropout(0.5, name='L13_dropout'))

    #L14
    model.add(Dense(256, name='L14_dense'))
    model.add(LeakyReLU(alpha=0.03, name='L14_activation'))

    #L15
    model.add(Dropout(0.5, name='L15_dropout'))

    #L16
    model.add(Dense(10, name='L16_dense'))
    model.add(LeakyReLU(alpha=0.03, name='L16_activation'))

    #L17
    model.add(Dropout(0.5, name='L17_dropout'))

    #L18
    model.add(Dense(2, name='L18_dense'))
    model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999), loss='mae')

    return model

def get_new_model():  
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 1), name='L01', activation='relu'))
    model.add(Conv2D(32, (3, 3), name='L02', activation='relu'))

    model.add(AveragePooling2D(pool_size=(2, 2), name='L03'))
    model.add(Conv2D(64, (3, 3), name='L04', activation='relu'))
    model.add(Conv2D(64, (3, 3), name='L05', activation='relu'))
    
    model.add(AveragePooling2D(pool_size=(2, 2), name='L06'))
    
    model.add(Conv2D(128, (3, 3), name='L07', activation='relu'))
    model.add(Conv2D(128, (3, 3), name='L08', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), name='L09'))

    model.add(Conv2D(128, (3, 3), name='L10', activation='relu'))
    model.add(Conv2D(128, (3, 3), name='L11', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), name='L12'))
    
    model.add(Conv2D(128, (3, 3), name='L13', activation='relu'))
    model.add(Conv2D(128, (3, 3), name='L14', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), name='L15'))
    
    model.add(Flatten())
    model.add(Dense(256, name='L16', activation='relu'))
    model.add(Dense(256, name='L17', activation='relu'))

    model.add(Dense(2, name='L18'))
    model.compile(optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999), loss='mse')

    return model


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, baseDIR, batch_size=32, dim=(256, 256), n_channels=1, shuffle=True, rotate=True):
        'Initialization'
        self.dim = dim
        self.rotate = rotate
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.baseDIR = baseDIR
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim, self.n_channels))
        y = np.empty((self.batch_size, 2))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store image
            img = np.load(self.baseDIR + ID + '.npy').reshape(256, 256, 1)
            if(self.rotate):
                if(np.random.rand() > 0.5): #rotate image
                    img = np.rot90(img)
            X[i,] = img

            # Store omega and sigma
            y[i,0] = np.float(ID.split('_')[0][2:]) # Omega
            y[i,1] = np.float(ID.split('_')[1][2:]) # Sigma

        return X, y
