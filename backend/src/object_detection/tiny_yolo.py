import tensorflow as tf
from constants import *

def get_tiny_yolo():
    inp = tf.keras.layers.Input((IMAGE_H, IMAGE_W, NUM_C), dtype = 'float32')
    
    x = tf.keras.layers.Conv2D(
        16, 
        (3, 3),
        use_bias=False, 
        padding='same', 
        strides=(1, 1),
        name = 'conv_1'
    )(inp)
    x = tf.keras.layers.BatchNormalization(name='norm_1')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        padding = 'same',
        strides = (2, 2)
    )(x)

    x = tf.keras.layers.Conv2D(
        32, 
        (3, 3),
        use_bias=False, 
        padding='same', 
        strides=(1, 1), 
        name = 'conv_2'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='norm_2')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), 
        padding = 'same',
        strides = (1, 1)
    )(x) 

    x = tf.keras.layers.Conv2D(
        64, 
        (3, 3),
        use_bias=False, 
        padding='same', 
        strides=(1, 1), 
        name = 'conv_3'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='norm_3')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), 
        padding = 'same',
        strides = (2, 2)
    )(x)

    x = tf.keras.layers.Conv2D(
        128, 
        (3, 3),
        use_bias=False, 
        padding='same', 
        strides=(1, 1), 
        name = 'conv_4'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='norm_4')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(1, 1), 
        padding = 'same',
        strides = (1, 1)
    )(x)

    x = tf.keras.layers.Conv2D(
        256, 
        (3, 3),
        use_bias=False, 
        padding='same', 
        strides=(1, 1), 
        name = 'conv_5'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='norm_5')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), 
        padding = 'same',
        strides = (2, 2)
    )(x)

    x = tf.keras.layers.Conv2D(
        512, 
        (3, 3),
        use_bias=False, 
        padding='same', 
        strides=(1, 1), 
        name = 'conv_6'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='norm_6')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), 
        padding = 'same',
        strides = (1, 1)
    )(x)

    x = tf.keras.layers.Conv2D(
        1024, 
        (3, 3),
        use_bias=False, 
        padding='same', 
        strides=(1, 1), 
        name = 'conv_7'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='norm_7')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), 
        padding = 'same',
        strides = (2, 2)
    )(x)

    x = tf.keras.layers.Conv2D(
        1024, 
        (3, 3),
        use_bias=False, 
        padding='same', 
        strides=(1, 1), 
        name = 'conv_8'
    )(x)
    x = tf.keras.layers.BatchNormalization(name='norm_8')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), 
        padding = 'same',
        strides = (2, 2)
    )(x)

    x = tf.keras.layers.Conv2D(
        BOX * (4 + 1 + CLASS), 
        (3, 3),
        use_bias=False, 
        padding='same', 
        strides=(1, 1), 
        name = 'conv_9'
    )(x)
    output = tf.keras.layers.Reshape((GRID_W, GRID_H, BOX, 4 + 1 + CLASS))(x)
    model = tf.keras.Model(inputs = inp, outputs = output) 
    return model

#model = get_tiny_yolo()
#print(model.summary())
