from src.object_detection.constants import *
import tensorflow as tf

def get_depthwise_yolov2_model(plot_model=False):
# Custom Keras layer
    class SpaceToDepth(tf.keras.layers.Layer):

        def __init__(self, block_size, **kwargs):
            self.block_size = block_size
            super(SpaceToDepth, self).__init__(**kwargs)

        def call(self, inputs):
            x = inputs
            batch, height, width, depth = tf.keras.backend.int_shape(x)
            batch = -1
            reduced_height = height // self.block_size
            reduced_width = width // self.block_size
            y = tf.keras.backend.reshape(x, (batch, reduced_height, self.block_size,
                                 reduced_width, self.block_size, depth))
            z = tf.keras.backend.permute_dimensions(y, (0, 1, 3, 2, 4, 5))
            t = tf.keras.backend.reshape(z, (batch, reduced_height, reduced_width, depth * self.block_size **2))
            return t

        def compute_output_shape(self, input_shape):
            shape =  (input_shape[0], input_shape[1] // self.block_size, input_shape[2] // self.block_size,
                      input_shape[3] * self.block_size **2)
            return tf.TensorShape(shape)

    input_image = tf.keras.layers.Input((IMAGE_H, IMAGE_W, NUM_C), dtype='float32')

    # Layer 1
    x = tf.keras.layers.SeparableConv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
    x = tf.keras.layers.BatchNormalization(name='norm_1')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = tf.keras.layers.SeparableConv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_2')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = tf.keras.layers.SeparableConv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_3')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Layer 4
    x = tf.keras.layers.SeparableConv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_4')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Layer 5
    x = tf.keras.layers.SeparableConv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_5')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = tf.keras.layers.SeparableConv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_6')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Layer 7
    x = tf.keras.layers.SeparableConv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_7')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Layer 8
    x = tf.keras.layers.SeparableConv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_8')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = tf.keras.layers.SeparableConv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_9')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Layer 10
    x = tf.keras.layers.SeparableConv2D(128, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_10')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Layer 11
    x = tf.keras.layers.SeparableConv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_11')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Layer 12
    x = tf.keras.layers.SeparableConv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_12')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Layer 13
    x = tf.keras.layers.SeparableConv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_13')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    skip_connection = x

    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 14
    x = tf.keras.layers.SeparableConv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_14')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = tf.keras.layers.SeparableConv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_15')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = tf.keras.layers.SeparableConv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_16')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Layer 17
    x = tf.keras.layers.SeparableConv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_17')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Layer 18
    x = tf.keras.layers.SeparableConv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_18')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Layer 19
    x = tf.keras.layers.SeparableConv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_19')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = tf.keras.layers.SeparableConv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_20')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Layer 21
    skip_connection = tf.keras.layers.SeparableConv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
    skip_connection = tf.keras.layers.BatchNormalization(name='norm_21')(skip_connection)
    skip_connection = tf.keras.layers.LeakyReLU(alpha=0.1)(skip_connection)

    skip_connection = SpaceToDepth(block_size=2)(skip_connection)

    x = tf.keras.layers.concatenate([skip_connection, x])

    # Layer 22
    x = tf.keras.layers.SeparableConv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(name='norm_22')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    features = tf.keras.layers.Dropout(0.3)(x) # add dropout

    # Layer 23
    x = tf.keras.layers.SeparableConv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(features)
    output = tf.keras.layers.Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

    model = tf.keras.models.Model(input_image, output)

    """# 2. Load YOLO pretrained weigts"""

    class WeightReader:
        def __init__(self, weight_file):
            self.offset = 4
            self.all_weights = np.fromfile(weight_file, dtype='float32')
            
        def read_bytes(self, size):
            self.offset = self.offset + size
            return self.all_weights[self.offset-size:self.offset]
        
        def reset(self):
            self.offset = 4
    """
    weight_reader = WeightReader('yolo.weights')

    weight_reader.reset()
    nb_conv = 23

    for i in range(1, nb_conv+1):
        conv_layer = model.get_layer('conv_' + str(i))
        conv_layer.trainable = True
        
        if i < nb_conv:
            norm_layer = model.get_layer('norm_' + str(i))
            norm_layer.trainable = True
            
            size = np.prod(norm_layer.get_weights()[0].shape)

            beta  = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean  = weight_reader.read_bytes(size)
            var   = weight_reader.read_bytes(size)

            weights = norm_layer.set_weights([gamma, beta, mean, var])
            
        if len(conv_layer.get_weights()) > 1:
            bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel])

    layer   = model.layers[-2] # last convolutional layer
    layer.trainable = True


    weights = layer.get_weights()

    new_kernel = np.random.normal(size=weights[0].shape)/(GRID_H*GRID_W)
    new_bias   = np.random.normal(size=weights[1].shape)/(GRID_H*GRID_W)

    layer.set_weights([new_kernel, new_bias])
    """
    if plot_model:
        tf.keras.utils.plot_model(model, to_file='yolov2_model.png', show_shapes=True)
    return model
