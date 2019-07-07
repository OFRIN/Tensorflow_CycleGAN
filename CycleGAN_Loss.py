import tensorflow as tf

from Define import *

def L1_Loss(outputs, targets):
    return tf.reduce_mean(tf.abs(outputs - targets))

def L2_Loss(outputs, targets):
    return tf.reduce_mean((outputs - targets) ** 2)

def CycleGAN_Loss(real_A_input_var, real_B_input_var, discriminator_ops, generator_ops, cross_entropy = True):
    #real_A_input_var = real_A_input_var / 127.5 - 1
    #real_B_input_var = real_B_input_var / 127.5 - 1
    
    D_real_A_op, D_real_B_op, D_fake_A_1_op, D_fake_B_1_op, D_fake_A_2_op, D_fake_B_2_op = discriminator_ops
    real_to_fake_A_input_op, real_to_fake_B_input_op, fake_to_fake_A_input_op, fake_to_fake_B_input_op = generator_ops

    # Generator Loss
    G_A_style_loss = LOSS_LAMBDA * L1_Loss(fake_to_fake_A_input_op, real_A_input_var)
    G_B_style_loss = LOSS_LAMBDA * L1_Loss(fake_to_fake_B_input_op, real_B_input_var)

    if cross_entropy:
        G_AtoB_loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_B_1_op, labels = tf.ones_like(D_fake_B_1_op)))
        G_BtoA_loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_A_1_op, labels = tf.ones_like(D_fake_A_1_op)))
    else:
        G_AtoB_loss_op = L2_Loss(D_fake_B_1_op, tf.ones_like(D_fake_B_1_op))
        G_BtoA_loss_op = L2_Loss(D_fake_A_1_op, tf.ones_like(D_fake_A_1_op))

    G_style_loss_op = G_A_style_loss + G_B_style_loss
    G_loss_op = G_AtoB_loss_op + G_BtoA_loss_op
    total_G_loss_op = G_loss_op + G_style_loss_op
    
    # Discriminator Loss
    if cross_entropy:
        DA_real_loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real_A_op, labels = tf.ones_like(D_real_A_op)))
        DA_fake_loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_A_1_op, labels = tf.zeros_like(D_fake_A_1_op)))

        DB_real_loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real_B_op, labels = tf.ones_like(D_real_B_op)))
        DB_fake_loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_B_1_op, labels = tf.zeros_like(D_fake_B_1_op)))
    else:
        DA_real_loss_op = L2_Loss(D_real_A_op, tf.ones_like(D_real_A_op))
        DA_fake_loss_op = L2_Loss(D_fake_A_1_op, tf.zeros_like(D_fake_A_1_op))

        DB_real_loss_op = L2_Loss(D_real_B_op, tf.ones_like(D_real_B_op))
        DB_fake_loss_op = L2_Loss(D_fake_B_1_op, tf.zeros_like(D_fake_B_1_op))

    DA_loss_op = (DA_real_loss_op + DA_fake_loss_op) / 2
    DB_loss_op = (DB_real_loss_op + DB_fake_loss_op) / 2
    D_loss_op = DA_loss_op + DB_loss_op

    return total_G_loss_op, G_loss_op, D_loss_op, G_style_loss_op
    
