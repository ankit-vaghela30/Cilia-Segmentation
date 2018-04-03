from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

'''
We use the same unet model which was created by our team member Vyom Shrivastava during project 3 with Team Canady.
This model was based on the paper: https://arxiv.org/pdf/1707.06314.pdf
'''
def unet():
    inputs = Input((256, 256, 1))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    b1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(b1)
    b1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(b1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    b2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(b2)
    b2 = BatchNormalization()(conv2)
    drop2 = Dropout(0.5)(b2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    b3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(b3)
    b3 = BatchNormalization()(conv3)
    drop3 = Dropout(0.25)(b3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    b4= BatchNormalization()(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(b4)
    b4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(b4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    b5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(b5)
    b5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(b5)

    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(drop5), drop4], axis=3)
    b6 = BatchNormalization(momentum=0.5)(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(b6)
    b6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(b6)
    b6 = BatchNormalization()(conv6)
    drop6 = Dropout(0.5)(b6)

    up7 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(drop6), drop3], axis=3)
    b7 = BatchNormalization(momentum=0.5)(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(b7)
    b7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(b7)
    b7 = BatchNormalization()(conv7)
    drop7 = Dropout(0.5)(b7)

    up8 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(drop7), drop2], axis=3)
    b8 = BatchNormalization(momentum=0.5)(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(b8)
    b8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(b8)
    b8 = BatchNormalization()(conv8)
    drop8 = Dropout(0.5)(b8)

    up9 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(drop8), conv1], axis=3)
    b9 = BatchNormalization(momentum=0.5)(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(b9)
    b9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(b9)
    b9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(b9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model
