import tensorflow as tf
import numpy as np

from Define import *

# init_fn = tf.contrib.layers.xavier_initializer()
D_init_fn = tf.truncated_normal_initializer(stddev=0.02)
G_init_fn = tf.truncated_normal_initializer(stddev=0.02)

def residule_block(x, features, name='res', init_fn = G_init_fn):
    last_x = x

    x = tf.layers.conv2d(inputs = x, filters = features, kernel_size = [3, 3], strides = 1, padding = 'SAME', kernel_initializer = init_fn, name = name + '_conv_1')
    x = tf.contrib.layers.instance_norm(x, scope = name + '_instance_norm_1')
    x = tf.nn.relu(x, name = name + '_relu_1')
    
    x = tf.layers.conv2d(inputs = x, filters = features, kernel_size = [3, 3], strides = 1, padding = 'SAME', kernel_initializer = init_fn, name = name + '_conv_2')
    x = tf.contrib.layers.instance_norm(x, scope = name + '_instance_norm_2')
    #x = tf.nn.relu(x, name = name + '_relu_2')
    
    x = tf.nn.relu(last_x + x, name = name + '_relu_2')
    return x

def Generator(inputs, reuse = False, name = 'Generator'):
    with tf.variable_scope(name, reuse = reuse):
        #x = inputs / 127.5 - 1
        x = inputs

        x = tf.layers.conv2d(inputs = x, filters = DEFAULT_FEATURES, kernel_size = [7, 7], strides = 1, padding = 'SAME', kernel_initializer = G_init_fn, name = 'conv_1')
        x = tf.contrib.layers.instance_norm(x, scope = 'instance_norm_1')
        x = tf.nn.relu(x, name = 'relu_1')

        x = tf.layers.conv2d(inputs = x, filters = DEFAULT_FEATURES * 2, kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = G_init_fn, name = 'conv_2')
        x = tf.contrib.layers.instance_norm(x, scope = 'instance_norm_2')
        x = tf.nn.relu(x, name = 'relu_2')

        x = tf.layers.conv2d(inputs = x, filters = DEFAULT_FEATURES * 4, kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = G_init_fn, name = 'conv_3')
        x = tf.contrib.layers.instance_norm(x, scope = 'instance_norm_3')
        x = tf.nn.relu(x, name = 'relu_3')

        for resnet_index in range(1, RESNET_LAYERS + 1):
            x = residule_block(x, DEFAULT_FEATURES * 4, name = 'residual_block_' + str(resnet_index))
    
        # Deconv
        x = tf.layers.conv2d_transpose(inputs = x, filters = DEFAULT_FEATURES * 2, kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = G_init_fn, name = 'deconv_1')
        x = tf.contrib.layers.instance_norm(x, scope = 'instance_norm_4')
        x = tf.nn.relu(x, name = 'relu_4')

        x = tf.layers.conv2d_transpose(inputs = x, filters = DEFAULT_FEATURES, kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = G_init_fn, name = 'deconv_2')
        x = tf.contrib.layers.instance_norm(x, scope = 'instance_norm_5')
        x = tf.nn.relu(x, name = 'relu_5')

        # Output
        x = tf.layers.conv2d(inputs = x, filters = OUTPUT_IMAGE_CHANNEL, kernel_size = [7, 7], strides = 1, padding = 'SAME', kernel_initializer = G_init_fn, name = 'last_conv')
        x = tf.nn.tanh(x, name = name + '_tanh')

    return x

def Discriminator(inputs, reuse = False, name = 'Discriminator'):
    with tf.variable_scope(name, reuse = reuse):
        #x = inputs / 127.5 - 1
        x = inputs

        x = tf.layers.conv2d(inputs = x, filters = DEFAULT_FEATURES, kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = D_init_fn, name = 'conv_1') # (128, 128, ...)
        x = tf.contrib.layers.instance_norm(x, scope = 'instance_norm_1')
        x = tf.nn.leaky_relu(x, name = 'leaky_relu_1')

        x = tf.layers.conv2d(inputs = x, filters = DEFAULT_FEATURES * 2, kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = D_init_fn, name = 'conv_2') # (64, 64, ...)
        x = tf.contrib.layers.instance_norm(x, scope = 'instance_norm_2')
        x = tf.nn.leaky_relu(x, name = 'leaky_relu_2')
        
        x = tf.layers.conv2d(inputs = x, filters = DEFAULT_FEATURES * 4, kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = D_init_fn, name = 'conv_3') # (32, 32, ...)
        x = tf.contrib.layers.instance_norm(x, scope = 'instance_norm_3')
        x = tf.nn.leaky_relu(x, name = 'leaky_relu_3')

        x = tf.layers.conv2d(inputs = x, filters = DEFAULT_FEATURES * 8, kernel_size = [3, 3], strides = 1, padding = 'SAME', kernel_initializer = D_init_fn, name = 'conv_4') # (32, 32, ...)
        x = tf.contrib.layers.instance_norm(x, scope = 'instance_norm_4')
        x = tf.nn.leaky_relu(x, name = 'leaky_relu_4')

        x = tf.layers.conv2d(inputs = x, filters = 1, kernel_size = [3, 3], strides = 1, padding = 'SAME', kernel_initializer = D_init_fn, name = 'conv_5') # (32, 32, 1)
        #x = tf.nn.sigmoid(x)

    return x

def CycleGAN(real_A_input_var, real_B_input_var):

    real_to_fake_A_input_op = Generator(real_B_input_var, False, 'G_BtoA')
    real_to_fake_B_input_op = Generator(real_A_input_var, False, 'G_AtoB')

    fake_to_fake_B_input_op = Generator(real_to_fake_A_input_op, True, 'G_AtoB')
    fake_to_fake_A_input_op = Generator(real_to_fake_B_input_op, True, 'G_BtoA')

    D_fake_A_1_op = Discriminator(real_to_fake_A_input_op, False, 'D_A')
    D_fake_B_1_op = Discriminator(real_to_fake_B_input_op, False, 'D_B')

    D_fake_A_2_op = Discriminator(fake_to_fake_A_input_op, True, 'D_A')
    D_fake_B_2_op = Discriminator(fake_to_fake_B_input_op, True, 'D_B')

    D_real_A_op = Discriminator(real_A_input_var, True, 'D_A')
    D_real_B_op = Discriminator(real_B_input_var, True, 'D_B')
    
    return [D_real_A_op, D_real_B_op, D_fake_A_1_op, D_fake_B_1_op, D_fake_A_2_op, D_fake_B_2_op], \
                [real_to_fake_A_input_op, real_to_fake_B_input_op, fake_to_fake_A_input_op, fake_to_fake_B_input_op]

if __name__ == '__main__':
    real_A_input_var = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_HEIGHT, INPUT_IMAGE_CHANNEL])
    real_B_input_var = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_HEIGHT, INPUT_IMAGE_CHANNEL])

    CycleGAN(real_A_input_var, real_B_input_var)
