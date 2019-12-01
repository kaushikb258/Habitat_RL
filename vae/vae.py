import numpy as np
import tensorflow as tf
import sys



winit = tf.contrib.layers.xavier_initializer() 
binit = tf.zeros_initializer()


class VAE():

    def __init__(self, sess, lr=1e-4, batch_size=64, n_z=256, img_size=192, nchannels=3, recon_loss='l2', lat_loss='mmd', scope='vae'):
        self.sess = sess
        self.lr = lr
        self.batch_size = batch_size
        self.n_z = n_z 
        self.big_img = 256
        self.img_size = img_size
        self.nchannels = nchannels
        self.recon_loss = recon_loss 
        self.lat_loss = lat_loss 
        self.scope = scope

        self.placeholders()
        self.network()
        self.loss_and_optim()


    # placeholders
    def placeholders(self):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, self.big_img, self.big_img, self.nchannels])
            self.is_train = tf.placeholder(name='is_train', dtype=tf.bool) 
            self.beta = tf.placeholder(name='beta', dtype=tf.float32, shape=())


    def network(self):
      with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE): 

        def conv_block(inp, f, k, s, p, name, is_train, use_norm=True):
            conv = tf.layers.conv2d(inp, filters=f, kernel_size=(k,k), strides=(s,s), padding=p, activation=None, kernel_initializer=winit, bias_initializer=binit, name=name)
            if (use_norm):
               #conv = tf.layers.batch_normalization(conv, training=is_train)
               conv = tf.contrib.layers.instance_norm(conv)
            #conv = tf.nn.leaky_relu(conv, 0.2)
            conv = tf.nn.relu(conv) 
            return conv

        def conv_resnet_block(inp, f, k, s, p, name, is_train):
            conv = conv_block(inp, f, k, s, p, name, is_train) + inp
            return conv

        def dconv_block(inp, f, k, s, p, name, is_train, use_norm=True):
            conv = tf.layers.conv2d_transpose(inp, filters=f, kernel_size=(k,k), strides=(s,s), padding=p, activation=None, kernel_initializer=winit, bias_initializer=binit, name=name)           
            if (use_norm):
               #conv = tf.layers.batch_normalization(conv, training=is_train)
               conv = tf.contrib.layers.instance_norm(conv) 
            #conv = tf.nn.leaky_relu(conv, 0.2) 
            conv = tf.nn.relu(conv)
            return conv

        def dconv_resnet_block(inp, f, k, s, p, name, is_train):
            conv = dconv_block(inp, f, k, s, p, name, is_train) + inp
            return conv


        # encoder
        self.input_x = tf.image.resize_images(self.x, size=(self.img_size,self.img_size)) 

        # conv blocks 
        conv1 = conv_block(self.input_x, f=64, k=4, s=2, p='valid', name='enc_conv1', is_train=self.is_train)
        conv2 = conv_block(conv1, f=128, k=4, s=2, p='valid', name='enc_conv2', is_train=self.is_train)  
        conv3 = conv_block(conv2, f=256, k=4, s=2, p='valid', name='enc_conv3', is_train=self.is_train)  
        conv4 = conv_block(conv3, f=512, k=4, s=2, p='valid', name='enc_conv4', is_train=self.is_train)  
        conv5 = conv_block(conv4, f=512, k=4, s=1, p='valid', name='enc_conv5', is_train=self.is_train)  
        # [None, 7, 7, 512] 


        # conv resnet blocks
        conv6 = conv_resnet_block(conv5, f=512, k=4, s=1, p='same', name='enc_conv6', is_train=self.is_train)
        conv7 = conv_resnet_block(conv6, f=512, k=4, s=1, p='same', name='enc_conv7', is_train=self.is_train)
        conv8 = conv_resnet_block(conv7, f=512, k=4, s=1, p='same', name='enc_conv8', is_train=self.is_train)

        # last conv
        conv9 = conv_block(conv8, f=512, k=4, s=1, p='same', name='enc_conv9', is_train=self.is_train) 
        conv10 = conv_block(conv9, f=512, k=4, s=1, p='same', name='enc_conv10', is_train=self.is_train) 


        # flat
        flat = tf.layers.flatten(conv10, name='enc_flatten')
        self.z_mu = tf.layers.dense(flat, self.n_z, activation=None, name='enc_mu')
        self.z_log_sigma_sq = tf.layers.dense(flat, self.n_z, activation=None, name='enc_sigma')
        self.z_log_sigma_sq = tf.clip_by_value(self.z_log_sigma_sq, clip_value_min=-10.0, clip_value_max=4.0)        

        eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq), mean=0, stddev=1, dtype=tf.float32)
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps   

 
        # decoder 
        dec1 = tf.layers.dense(self.z, 7*7*512, activation=None, name='dec_fc1') 
        #dec1 = tf.nn.leaky_relu(dec1, 0.2)
        dec1 = tf.nn.relu(dec1)
        dec1 = tf.reshape(dec1, shape=(-1, 7, 7, 512), name='dec_reshape')

        # one dconv
        dconv1 = dconv_block(dec1, f=512, k=4, s=1, p='same', name='dec_dconv1', is_train=self.is_train)
        dconv2 = dconv_block(dconv1, f=512, k=4, s=1, p='same', name='dec_dconv2', is_train=self.is_train)

        # dconv resnet blocks
        dconv3 = dconv_resnet_block(dconv2, f=512, k=4, s=1, p='same', name='dec_dconv3', is_train=self.is_train)
        dconv4 = dconv_resnet_block(dconv3, f=512, k=4, s=1, p='same', name='dec_dconv4', is_train=self.is_train)
        dconv5 = dconv_resnet_block(dconv4, f=512, k=4, s=1, p='same', name='dec_dconv5', is_train=self.is_train)
        
        # dconv blocks
        dconv6 = dconv_block(dconv5, f=512, k=4, s=1, p='valid', name='dec_dconv6', is_train=self.is_train)
        dconv7 = dconv_block(dconv6, f=512, k=4, s=2, p='valid', name='dec_dconv7', is_train=self.is_train)
        dconv8 = dconv_block(dconv7, f=256, k=4, s=2, p='valid', name='dec_dconv8', is_train=self.is_train)
        dconv9 = dconv_block(dconv8, f=128, k=4, s=2, p='valid', name='dec_dconv9', is_train=self.is_train)
        dconv10 = dconv_block(dconv9, f=64, k=4, s=2, p='valid', name='dec_dconv10', is_train=self.is_train)

        self.x_hat = tf.layers.conv2d_transpose(dconv10, filters=self.nchannels, kernel_size=(3,3), strides=(1,1), padding='valid', activation=tf.nn.sigmoid, kernel_initializer=winit, bias_initializer=binit, name='dec_xhat')          



    # mmd
    def calc_mmd(self, sample_pz, sample_qz):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
               n = self.batch_size
               nf = tf.cast(self.batch_size, tf.float32)

               norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
               dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
               distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

               norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
               dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
               distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

               dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
               distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods 

               C_base = 2.0*self.n_z*1.0

               stat = 0.0
               
               for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                  C = C_base * scale
                  res1 = C / (C + distances_qz)
                  res1 += C / (C + distances_pz)
                  res1 = tf.multiply(res1, 1. - tf.eye(n))
                  res1 = tf.reduce_sum(res1) / (nf * nf - nf)
                  res2 = C / (C + distances)
                  res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
                  stat += res1 - res2

               return stat

 

    # loss and optim
    def loss_and_optim(self):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
           # reconstruction loss: L2 loss / L1 loss / mse
           if (self.recon_loss == 'l2'):
               loss = tf.reduce_sum(tf.square(self.x_hat - self.input_x), axis=[1, 2, 3])
               self.rec_loss = tf.reduce_mean(loss)
           elif (self.recon_loss == 'l1'):
               self.rec_loss = tf.reduce_mean(tf.abs(self.x_hat-self.input_x)) 
           elif (self.recon_loss == 'bce'):
               self.rec_loss = -tf.reduce_mean(tf.reduce_sum(self.input_x * tf.log(self.x_hat + 1.0e-8) + (1.0 - self.input_x) * tf.log(1.0 - self.x_hat + 1.0e-8), 1))
           

           # latent loss: KL divergence between p(z) and N(0,1)
           if (self.lat_loss == 'kld'):
               self.latent_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=1)) 
           elif (self.lat_loss == 'mmd'):
               z_gauss = tf.random_normal([self.batch_size, self.n_z])
               self.latent_loss = self.calc_mmd(z_gauss, self.z)
 

           self.total_loss = self.rec_loss + self.beta * self.latent_loss

           update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.scope)
           with tf.control_dependencies(update_ops):
               self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.total_loss)


    # run single step
    def run_single_step(self, x, beta, is_train=True):
        fd = {self.x: x, self.beta: beta, self.is_train: is_train}
        _, loss, rec_loss, latent_loss = self.sess.run([self.train_op, self.total_loss, self.rec_loss, self.latent_loss], feed_dict=fd)
        return loss, rec_loss, latent_loss

    # x -> x_hat
    def reconstruction(self, x, beta, is_train=True):
        fd = {self.x: x, self.beta: beta, self.is_train: is_train}
        x_hat = self.sess.run(self.x_hat, feed_dict=fd)
        return x_hat


    # x -> z
    def encode(self, x, beta, is_train=True):
        fd = {self.x: x, self.beta: beta, self.is_train: is_train} 
        z = self.sess.run(self.z, feed_dict=fd)        
        return z 
        #z_mu = self.sess.run(self.z_mu, feed_dict=fd)
        #return z_mu

    # z -> x
    def generate(self, z, is_train=True):
        fd = {self.z: z, self.is_train: is_train}
        xhat = self.sess.run(self.x_hat, feed_dict=fd)
        return xhat

    def resize_img(self, x):
        input_x = self.sess.run(self.input_x, feed_dict={self.x: x}) 
        return input_x

    # vae trainable vars
    def get_vae_vars(self):
        vae_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        #vae_vars += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.scope)
        return vae_vars

#---------------------------------------------------------------------------------------

