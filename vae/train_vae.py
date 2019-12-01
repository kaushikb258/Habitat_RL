import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from vae import *
from utils_vae import *


learning_rate = 1e-4  # learning rate

train_from_scratch = True


# VAE params
nsteps_vae = 50000  # number of steps to train VAE
batch_size = 64 # batch size
n_z = 128 #256 # latent vector
img_size = 192
big_img = 256

img_train = 'rgb' #'depth'     

#----------------------------------------------------------------------------------------


with tf.Session() as sess:

  if (img_train == 'rgb'):
       nchannels = 3
       recon_loss = 'l2' 
       lat_loss = 'kld' #'mmd'
  elif (img_train == 'depth'):
       nchannels = 1   
       recon_loss = 'l2' #'bce' 
       lat_loss = 'kld' #'mmd' 



  print('-'*25)
  print('img_train: ', img_train)
  print('nchannels: ', nchannels)
  print('recon_loss: ', recon_loss)
  print('lat_loss: ', lat_loss) 
  print('-'*25)


  vae = VAE(sess, lr=learning_rate, batch_size=batch_size, n_z=n_z, img_size=img_size, nchannels=nchannels, recon_loss=recon_loss, lat_loss=lat_loss, scope='vae_' + img_train)

  vae_vars = vae.get_vae_vars() 

  print(vae_vars)

  saver_vae = tf.train.Saver(vae_vars)
  

  if (train_from_scratch == True):
      sess.run(tf.global_variables_initializer())
  else:
      saver_vae.restore(sess, "../ckpt/" + img_train + "/model")  
      print("restored vae_vars")
   
#----------------------------------------------------------------------------------------
# train VAE

  print('-'*25 + '\n')
  print("train VAE ")


  if (img_train == 'depth'):
      d = get_depth()
  elif (img_train == 'rgb'):
      d = get_rgb()       


  for ns in range(nsteps_vae):
   

     # set warm-up parameter beta (ladder VAE)
     #beta = float(ns)/float(nsteps_vae/2) * 1.0
     #beta = np.clip(beta, a_min=0.0, a_max=1.0) 
     beta = 1.0


     # sample mini-batch
     #if (img_train == 'rgb'):
     #   batch = get_batch_rgb(batch_size)
     #elif (img_train == 'depth'):
     #   batch = get_batch_depth(batch_size)
     idx = np.random.randint(0, d.shape[0], batch_size)   
     batch = d[idx,:,:,:]     


     if (img_train == 'rgb'):
         batch = np.array(batch).astype(np.float32)
         batch = batch/255.0


     # train one step
     loss, rec_loss, latent_loss = vae.run_single_step(batch, beta, True)

     print("ns: ", ns, "| loss: ", loss, "| rec_loss: ", rec_loss, "| latent_loss: ", latent_loss)
     f = open("../performance/performance_vae_" + img_train + ".txt", "a+")
     f.write(str(ns) + " " + str(loss) + " " + str(rec_loss) + " " + str(latent_loss) + '\n')  
     f.close()     


     # reconstruct image once every N steps
     if (ns % 100 == 0):
         ii = np.random.randint(low=0, high=batch_size, size=1)
         img_ = batch[ii,:,:,:]
         #img_: shape = (1, 256, 256, 3) for rgb or (1, 256, 256, 1) for depth      

         x_hat = vae.reconstruction(img_, beta, is_train=True)
         x_hat = np.squeeze(x_hat,0)              

         img_ = vae.resize_img(img_)
         img_ = np.squeeze(img_,0)

         if (img_train == 'rgb'):
            img_ *= 255.0
            x_hat *= 255.0
         elif (img_train == 'depth'): 
            img_ = np.squeeze(img_,2)*255.0
            x_hat = np.squeeze(x_hat,2)*255.0

         img_ = np.array(img_,dtype=np.uint8)
         x_hat = np.array(x_hat,dtype=np.uint8)

     
         # combine images
         if (img_train == 'rgb'):
            combined_image = np.zeros((img_size,2*img_size+10,3),dtype=np.uint8) 
            combined_image[:,:img_size,:] = img_
            combined_image[:,-img_size:,:] = x_hat
         elif (img_train == 'depth'): 
            combined_image = np.zeros((img_size,2*img_size+10),dtype=np.uint8) 
            combined_image[:,:img_size] = img_
            combined_image[:,-img_size:] = x_hat


         file_name = "vae_samples/" + img_train + "/reconst_" + img_train + "_" + str(ns) + '.png'
         plt.imshow(combined_image); plt.axis('off'); plt.savefig(file_name); plt.close() 

         # generate images
         zgauss = np.random.normal(loc=0.0, scale=1.0, size=(1,n_z))
         xhat = vae.generate(zgauss)
         xhat = np.squeeze(xhat,0)
         if (img_train == 'depth'):
            xhat = np.squeeze(xhat,2)
         xhat = xhat*255.0; xhat = np.array(xhat,dtype=np.uint8)    
         file_name = "vae_samples/" + img_train + "/generate_" + img_train + "_" + str(ns) + '.png'
         plt.imshow(xhat); plt.axis('off'); plt.savefig(file_name); plt.close() 


         # save model
         saver_vae.save(sess, '../ckpt/vae_' + img_train + '/model') 
         print("saved vae model ")


#----------------------------------------------------------------------------------------

