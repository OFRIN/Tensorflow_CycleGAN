
import os
import cv2
import sys
import time
import glob

import numpy as np
import tensorflow as tf

from Define import *
from Utils import *
from CycleGAN import *
from CycleGAN_Loss import *

os.environ["CUDA_VISIBLE_DEVICES"]="2" # 2

# 1. load dataset
root_dir = 'D:/_ImageDataset/'

train_A_image_paths = glob.glob(root_dir + '/horse2zebra/trainA/*')
train_B_image_paths = glob.glob(root_dir + '/horse2zebra/trainB/*')

test_A_image_paths = glob.glob(root_dir + '/horse2zebra/testA/*')
test_B_image_paths = glob.glob(root_dir + '/horse2zebra/testB/*')

print('[i] train A : {}'.format(len(train_A_image_paths)))
print('[i] train B : {}'.format(len(train_B_image_paths)))
print('[i] test A : {}'.format(len(test_A_image_paths)))
print('[i] test B : {}'.format(len(test_B_image_paths)))

# 2. build model
real_A_input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_IMAGE_CHANNEL])
real_B_input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_IMAGE_CHANNEL])

discriminator_ops, generator_ops = CycleGAN(real_A_input_var, real_B_input_var)
real_to_fake_A_input_op, real_to_fake_B_input_op, fake_to_fake_A_input_op, fake_to_fake_B_input_op = generator_ops

# 3. loss
total_G_loss_op, G_loss_op, D_loss_op, G_style_loss_op = CycleGAN_Loss(real_A_input_var, real_B_input_var, discriminator_ops, generator_ops, True)

# 4. select variables
vars = tf.trainable_variables()
D_vars = [var for var in vars if 'D_' in var.name]
G_vars = [var for var in vars if 'G_' in var.name]

print('[i] Discriminator')
for var in D_vars:
    print(var)

print('[i] Generator')
for var in G_vars:
    print(var)

# 5. optimizer
learning_rate_var = tf.placeholder(tf.float32, name = 'learning_rate')
D_train_op = tf.train.AdamOptimizer(learning_rate_var, beta1 = 0.5).minimize(D_loss_op, var_list = D_vars)
G_train_op = tf.train.AdamOptimizer(learning_rate_var, beta1 = 0.5).minimize(total_G_loss_op, var_list = G_vars)

# 6. train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
#saver.restore(sess, './model/CycleGAN_176.ckpt')

train_iteration = min(len(train_A_image_paths), len(train_B_image_paths)) // BATCH_SIZE

DECAY_EPOCHS = MAX_EPOCHS // 2
for epoch in range(176 + 1, MAX_EPOCHS + 1):

    if epoch > DECAY_EPOCHS:
        learning_rate = INIT_LEARNING_RATE * ((MAX_EPOCHS + 1) - epoch) / 100
    else:
        learning_rate = INIT_LEARNING_RATE

    st_time = time.time()
    D_loss_list = []
    G_loss_list = []
    G_style_loss_list = []

    np.random.shuffle(train_A_image_paths)
    np.random.shuffle(train_B_image_paths)

    for iter in range(train_iteration):
        #Generator
        batch_A_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_IMAGE_CHANNEL), dtype = np.float32)
        batch_B_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_IMAGE_CHANNEL), dtype = np.float32)

        for index, image_path in enumerate(train_A_image_paths[iter * BATCH_SIZE : (iter + 1) * BATCH_SIZE]):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

            #cv2.imshow('A', image)
            #cv2.waitKey(1)

            batch_A_image_data[index] = image.astype(np.float32) / 127.5 - 1

        for index, image_path in enumerate(train_B_image_paths[iter * BATCH_SIZE : (iter + 1) * BATCH_SIZE]):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

            #cv2.imshow('B', image)
            #cv2.waitKey(1)

            batch_B_image_data[index] = image.astype(np.float32) / 127.5 - 1
        
        _, G_loss, G_style_loss = sess.run([G_train_op, G_loss_op, G_style_loss_op], feed_dict={real_A_input_var : batch_A_image_data, real_B_input_var : batch_B_image_data, learning_rate_var : learning_rate })
        G_loss_list.append(G_loss)
        G_style_loss_list.append(G_style_loss)

        #Discriminator
        _, D_loss = sess.run([D_train_op, D_loss_op], feed_dict={real_A_input_var : batch_A_image_data, real_B_input_var : batch_B_image_data, learning_rate_var : learning_rate })
        D_loss_list.append(D_loss)

        sys.stdout.write('\r[{}/{}] D_loss : {:.5f}, G_loss : {:.5f}, G_style_loss : {:.5f}'.format(iter, train_iteration, D_loss, G_loss, G_style_loss))
        sys.stdout.flush()

    end_time = time.time() - st_time
    D_loss = np.mean(D_loss_list)
    G_loss = np.mean(G_loss_list)
    G_style_loss = np.mean(G_style_loss_list)

    print()
    print("[i] epoch : {}, D_loss : {:.5f}, G_loss : {:.5f}, G_style_loss : {:.5f}, time : {}sec".format(epoch, D_loss, G_loss, G_style_loss, int(end_time)))

    fake_images = np.zeros((SAVE_WIDTH * SAVE_HEIGHT, IMAGE_HEIGHT, IMAGE_WIDTH, OUTPUT_IMAGE_CHANNEL), dtype = np.float32)

    for i in range(SAVE_WIDTH):
        image = cv2.imread(test_A_image_paths[i])
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        fake_images[SAVE_WIDTH * 0 + i] = image.astype(np.float32) / 127.5 - 1
        fake_images[SAVE_WIDTH * 1 + i] = sess.run(real_to_fake_B_input_op, feed_dict = {real_A_input_var : [image]})[0]

    for i in range(SAVE_WIDTH):
        image = cv2.imread(test_B_image_paths[i])
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        fake_images[SAVE_WIDTH * 2 + i] = image.astype(np.float32) / 127.5 - 1
        fake_images[SAVE_WIDTH * 3 + i] = sess.run(real_to_fake_A_input_op, feed_dict = {real_B_input_var : [image]})[0]

    Save(fake_images, './results/{}.jpg'.format(epoch))
    saver.save(sess, './model/CycleGAN_{}.ckpt'.format(epoch))
