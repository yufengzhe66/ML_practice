#from scipy.misc import imread, imresize
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Dropout,MaxPool2D,Flatten, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
import sys




def build_model(out_dims, img_size):
    inputs_dim = Input((img_size, img_size, 3))

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation = 'relu')(inputs_dim)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same',activation = 'relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same',activation = 'relu')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same',activation = 'relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',activation = 'relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',activation = 'relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x_flat = Flatten()(x)
    dp_1 = Dropout(0.5)(x_flat)
    fc2 = Dense(out_dims)(dp_1)
    fc2 = Activation('softmax')(fc2)

    model = Model(inputs=inputs_dim, outputs=fc2)
    return model


if __name__ == '__main__':	
    #X_train,X_val,y_train,y_val,X_test,y_test=data_process()
    #simple_model = build_simple_model(10,32)
    #simple_model.summary()
    model = build_model(out_dims=10, img_size=32)
    model.summary()