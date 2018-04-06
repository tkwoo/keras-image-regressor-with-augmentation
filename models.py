from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D
from keras.layers import SeparableConv2D, LeakyReLU
# from keras.layers import Flatten
from keras import layers
from keras import backend as K
from keras.regularizers import l2

def vgg_like_regressor(flag):
    img_size = flag.image_size
    # num_classes = flag.num_classes

    inputs = Input((img_size, img_size, 3))
    # Block 1
    x = Conv2D(64, (3, 3), activation=None, padding='same', name='block1_conv1')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(64, (3, 3), activation=None, padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation=None, padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (3, 3), activation=None, padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation=None, padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (3, 3), activation=None, padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (3, 3), activation=None, padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation=None, padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(512, (3, 3), activation=None, padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(512, (3, 3), activation=None, padding='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    # x = Conv2D(512, (3, 3), activation=None, padding='same', name='block5_conv1')(x)
    # # x = BatchNormalization()(x)
    # x = LeakyReLU(0.2)(x)
    # x = Conv2D(512, (3, 3), activation=None, padding='same', name='block5_conv2')(x)
    # # x = BatchNormalization()(x)
    # x = LeakyReLU(0.2)(x)
    # x = Conv2D(512, (3, 3), activation=None, padding='same', name='block5_conv3')(x)
    # # x = BatchNormalization()(x)
    # x = LeakyReLU(0.2)(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Conv2D(1024, (3,3), activation=None, padding='same', name='conv6')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2, name='last_logit')(x)
    # x = Dropout(0.2)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='sigmoid', name='predictions')(x)
    # x = Dense(2, name='predictions')(x)

    # Create model.
    model = Model(inputs, x, name='vgg')
    return model

def vgg_like_regressor_1out(flag):
    # img_size = flag.image_size
    # num_classes = flag.num_classes

    inputs = Input((flag.image_height, flag.image_width, 3))
    # Block 1
    x = Conv2D(32, (3, 3), activation=None, padding='same', name='block1_conv1')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(32, (3, 3), activation=None, padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(64, (3, 3), activation=None, padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(64, (3, 3), activation=None, padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(128, (3, 3), activation=None, padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (3, 3), activation=None, padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (3, 3), activation=None, padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(256, (3, 3), activation=None, padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (3, 3), activation=None, padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (3, 3), activation=None, padding='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    # x = Conv2D(512, (3, 3), activation=None, padding='same', name='block5_conv1')(x)
    # # x = BatchNormalization()(x)
    # x = LeakyReLU(0.2)(x)
    # x = Conv2D(512, (3, 3), activation=None, padding='same', name='block5_conv2')(x)
    # # x = BatchNormalization()(x)
    # x = LeakyReLU(0.2)(x)
    # x = Conv2D(512, (3, 3), activation=None, padding='same', name='block5_conv3')(x)
    # # x = BatchNormalization()(x)
    # x = LeakyReLU(0.2)(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Conv2D(1024, (3,3), activation=None, padding='same', name='conv6')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2, name='last_logit')(x)
    # x = Dropout(0.2)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)
    # x = Dense(2, name='predictions')(x)

    # Create model.
    model = Model(inputs, x, name='vgg')
    return model