from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def residual_block(x, f_in, f_out):
    shortcut = x
    x = BatchNormalization()(x); x = ReLU()(x)
    x = Conv2D(f_in, 1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x); x = ReLU()(x)
    x = Conv2D(f_in, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x); x = ReLU()(x)
    x = Conv2D(f_out, 1, padding='same', kernel_initializer='he_normal')(x)
    if shortcut.shape[-1] != f_out:
        shortcut = Conv2D(f_out, 1, padding='same', kernel_initializer='he_normal')(shortcut)
    return ReLU()(Add()([x, shortcut]))

def build_model(input_shape=(40, 40, 1)):
    inputs = Input(shape=input_shape)
    x = Conv2D(16, 3, padding='same')(inputs)
    x = BatchNormalization()(x); x = ReLU()(x)
    x = MaxPool2D()(x)

    x = residual_block(x, 16, 32); x = MaxPool2D()(x)
    x = residual_block(x, 32, 32)
    x = residual_block(x, 32, 64); x = MaxPool2D()(x)
    x = residual_block(x, 64, 64)

    x = GlobalAveragePooling2D()(x)
    x = Dense(32)(x); x = BatchNormalization()(x); x = ReLU()(x); x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
