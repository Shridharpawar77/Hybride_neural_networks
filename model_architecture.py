import keras
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM

from keras.layers import Input
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Dense, Dropout, Activation


## define the model
def define_model(X_train,loss='mse',dropout=0.1):
    ## model 1
    inputs_1 = keras.Input(batch_shape=(20, X_train.shape[1], 1))
    cnn1 = Conv1D(filters=5, kernel_size=2, activation='relu',padding='same')(inputs_1)
    cnn1 = MaxPooling1D(pool_size=2)(cnn1)
    cnn1 = Dropout(dropout)(cnn1)

    cnn1 = Conv1D(filters=5, kernel_size=2, activation='relu',padding='same')(cnn1)
    cnn1 = MaxPooling1D(pool_size=2)(cnn1)
    cnn1 = Dropout(dropout)(cnn1)

    cnn1 = LSTM(384, return_sequences=True, stateful=True)(cnn1)
    cnn1 = Dense(100, activation='relu')(cnn1)
    cnn1 = Flatten()(cnn1)
    
    ## model 2
    inputs_2 = keras.Input(batch_shape=(20, X_train.shape[1], 1))
    cnn2 = Conv1D(filters=5, kernel_size=2, activation='relu',padding='same')(inputs_2)
    cnn2 = MaxPooling1D(pool_size=2)(cnn2)
    cnn2 = Dropout(dropout)(cnn2)

    cnn2 = LSTM(128, return_sequences=True, stateful=True)(cnn2)
    cnn2 = Dense(50, activation='relu')(cnn2)
    cnn2 = Flatten()(cnn2)
    
    ## model 3
    inputs_3 = keras.Input(batch_shape=(20, X_train.shape[1], 1))
    cnn3 = LSTM(64, return_sequences=True, stateful=True)(inputs_3)
    cnn3 = Dense(25, activation='relu')(cnn3)
    cnn3 = Flatten()(cnn3)
    
    ## merge all models
    merge = concatenate([cnn1, cnn2, cnn3], axis=1)
    dense = Dense(500, activation='relu')(merge)
    output = Dense(1)(dense)
    model = Model(inputs=[inputs_1,inputs_2,inputs_3], outputs=output)
    model.compile(optimizer='adam', loss=loss)
    
    return model