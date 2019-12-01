import habitat
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

from habitat.utils.visualizations import maps
from habitat.core.simulator import AgentState, SimulatorActions
from utils_kb import *
from ppo import *
from vae.vae import *

#-----------------------------------------------------------------------


print(SimulatorActions)

pwd = "/home/kb/habitat_experiments/code/vae_128/t1/"
config=habitat.get_config(pwd + "pointnav_gibson_kb.yaml")
print(config)

map_res = int(config['TASK']['TOP_DOWN_MAP']['MAP_RESOLUTION'])


EPSILON_CLIP = 0.1
batch_size = 64
n_z = 256
gamma = 0.95
gae_lambda = 1.0
learning_rate = 1.0e-4
img_size = 192
S_DIM = 2*n_z + 3 
A_DIM = 3

# number of test episodes to run
nepisodes_max = 500

save_imgs = False

allowed_actions = [SimulatorActions.MOVE_FORWARD, SimulatorActions.TURN_LEFT, SimulatorActions.TURN_RIGHT]



print('-----------------------HABITAT-------------------')
#------------------------------------------------------------------------------------
'''
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
'''
#------------------------------------------------------------------------------------



def get_reward(dist, new_dist, nsteps, env):

     lam = -0.01

     done = False
     success = False

     reward = dist - new_dist + lam
    
     if (new_dist <= 0.2):
         reward += 10.0 
         done = True
         success = True  
         print('success!!! ')

     if (nsteps >= config.ENVIRONMENT.MAX_EPISODE_STEPS-1):
           done = True 

     return reward, done, success
       
#------------------------------------------------------------------------------------

def perception_embedding(vae_rgb, vae_depth, rgb, depth, pointgoal, heading, S_DIM, n_z, beta_rgb=1.0, beta_depth=1.0):
    
     z_rgb = vae_rgb.encode(np.expand_dims(rgb,0), beta_rgb)
     z_depth = vae_depth.encode(np.expand_dims(depth,0), beta_depth)

     obs = np.zeros((1,S_DIM),dtype=np.float32) 
     obs[0,:n_z] = z_rgb
     obs[0,n_z:2*n_z] = z_depth
     obs[0,2*n_z:2*n_z+2] = pointgoal
     obs[0,2*n_z+2] = heading    


     assert obs.shape == (1,S_DIM)

     return obs, z_rgb, z_depth

#------------------------------------------------------------------------------------


def test_rl():
    print('create env')
    env = habitat.Env(config)

    print('create tf session')
    sess = tf.Session()

    Policy = Policy_net('policy', sess, env, S_DIM, A_DIM)
    Old_Policy = Policy_net('old_policy', sess, env, S_DIM, A_DIM)
    PPO = PPOTrain(Policy, Old_Policy, sess, gamma=gamma, clip_value=EPSILON_CLIP)

    ppo_vars = Policy.get_trainable_variables() + Old_Policy.get_trainable_variables()
    saver_ppo = tf.train.Saver(ppo_vars)    


    rec_loss_rgb = 'l2' 
    lat_loss_rgb = 'kld'
    beta_rgb = 1.0
      
    rec_loss_depth = 'l2' 
    lat_loss_depth = 'kld'
    beta_depth = 1.0 


    vae_rgb = VAE(sess, lr=learning_rate, batch_size=batch_size, n_z=n_z, img_size=img_size, nchannels=3, recon_loss=rec_loss_rgb, lat_loss=lat_loss_rgb, scope='vae_rgb')
    vae_rgb_vars = vae_rgb.get_vae_vars()

    vae_depth = VAE(sess, lr=learning_rate, batch_size=batch_size, n_z=n_z, img_size=img_size, nchannels=1, recon_loss=rec_loss_depth, lat_loss=lat_loss_depth, scope='vae_depth')
    vae_depth_vars = vae_depth.get_vae_vars()

    saver_vae_rgb = tf.train.Saver(vae_rgb_vars)
    saver_vae_depth = tf.train.Saver(vae_depth_vars)

    
    sess.run(tf.global_variables_initializer())


    saver_ppo.restore(sess, 'ckpt/ppo/model')

    saver_vae_rgb.restore(sess, 'ckpt/vae_rgb/model')
    saver_vae_depth.restore(sess, 'ckpt/vae_depth/model')


    print("total number of trainable params: ", Policy.num_train_params())


    navg_collisions = 0
    navg_SPL = 0.0
    nsuccess = 0
    
    largest_num_collisions = 0


#-----------------------------------------------------------------------------
    
    for ep in range(nepisodes_max):

            s = env.reset()

            rgb_ep, depth_ep, actions_ep, collision_ep = [], [], [], []

            init_agent_position = env.sim.get_agent_state().position
            dist = get_dist(env)      
            init_geodesic_dist = dist
            
            rnn_state = Policy.state_init
            done = False
            success = False
            nsteps = 0
            ep_rewards = 0.0
            dist_traveled_by_agent = 0.0
 
            old_position = init_agent_position 
         
            ncollisions = 0

            collision_sequence = 0


            while (not done) and (not env.episode_over) and (not success):  

                rgb = s["rgb"]    
                depth = s["depth"]       
                pointgoal = s["pointgoal"]; pointgoal = [pointgoal[0], pointgoal[2]]
                heading = s["heading"] 

                obs, z_rgb_t, z_depth_t = perception_embedding(vae_rgb, vae_depth, rgb, depth, pointgoal, heading, S_DIM, n_z, beta_rgb, beta_depth)

                act, v_pred, rnn_state = Policy.act(obs=obs, rnn_state=rnn_state, stochastic=False)


                # multiple continuous collisions: choose random action
                if (collision_sequence >= 2):
                     print('collision sequence ', collision_sequence)  
                     act = np.random.choice([1, 2])  


                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)
                
                s2 = env.step(allowed_actions[act])
                nsteps += 1 
                new_dist = get_dist(env)
                  
                reward, done, success = get_reward(dist, new_dist, nsteps, env) 

                if (new_dist <= 0.2): 
                    s2 = env.step(SimulatorActions.STOP)
                    new_dist = get_dist(env)                      


                new_position = env.sim.get_agent_state().position
                old_ = np.array([old_position[0], old_position[2]])
                new_ = np.array([new_position[0], new_position[2]]) 
                dist_moved = np.linalg.norm(old_ - new_)
                col_ = 0
                if (dist_moved < 0.99 * config.SIMULATOR.FORWARD_STEP_SIZE and act == 0):
                         print('collision!!! ', new_position, old_position, dist_moved)
                         ncollisions += 1  
                         col_ = 1
                         collision_sequence += 1
                else:
                         collision_sequence = 0 
                collision_ep.append(col_)
                    
                a_t = np.zeros((1,1),dtype=np.int32); a_t[0,0] = act
                pg_t = np.zeros((1,2),dtype=np.float32); pg_t[0,:] = pointgoal
                h_t = np.zeros((1,1),dtype=np.float32); h_t[0,0] = heading                

                ep_rewards += reward   
                dist_traveled_by_agent += abs(new_dist - dist)

                rgb_ep.append(rgb)
                depth_ep.append(depth)
                actions_ep.append(act)
                
                s = s2
                dist = new_dist
                new_position = old_position

#----

            # save
            if (save_imgs):
               for k in range(len(rgb_ep)):
                   rgb_ = rgb_ep[k]
                   depth_ = depth_ep[k][:,:,0] 
                   #plt.imshow(rgb_); plt.axis('off'); plt.savefig('test/rgb_' + str(ep) + '_' + str(k) + '.png'); plt.clf(); plt.close()       
                   #plt.imshow(depth_); plt.axis('off'); plt.savefig('test/depth_' + str(ep) + '_' + str(k) + '.png'); plt.clf(); plt.close()
                   depth_ = np.array(depth_)
                   np.save('test/depth_' + str(ep) + '_' + str(k), depth_)    

               actions_ep = np.array(actions_ep)     
               np.save('test/actions_' + str(ep), actions_ep)
               collision_ep = np.array(collision_ep)
               np.save('test/collision_' + str(ep), collision_ep)     
#------


            final_agent_position = env.sim.get_agent_state().position 
            target_position = env.current_episode.goals[0].position 
            geo_dist_traveled_by_agent = env.sim.geodesic_distance(init_agent_position, final_agent_position)
            SPL = float(success)*init_geodesic_dist/max(dist_traveled_by_agent, init_geodesic_dist)
            final_dist_to_target = env.sim.geodesic_distance(final_agent_position, target_position)
            

            largest_num_collisions = max(ncollisions, largest_num_collisions)

            

            print(' ')
            print('episode: ', ep, ' | ep_rewards: ', ep_rewards, ' | ep_steps: ', nsteps, ' | SPL: ', SPL) 
            print('init geodesic dist to target: ', init_geodesic_dist)
            print('final agent pos: ', final_agent_position)
            print('target pos: ', target_position)
            print(' ')   
            print('total dist traveled by agent: ', dist_traveled_by_agent)
            print('geo dist traveled by agent: ', geo_dist_traveled_by_agent) 
            print('final dist to target: ', final_dist_to_target)     
            print('ncollisions: ', ncollisions)     
            print('largest_num_collisions: ', largest_num_collisions) 
            print('-'*50)


            f = open("performance/test_ppo.txt", "a+")
            f.write(str(ep) + " " + str(ep_rewards) + " " + str(nsteps) + " " + str(SPL) + " " + " " + str(init_geodesic_dist) + " " + str(final_dist_to_target) + '\n')  
            f.close()     

            navg_collisions += ncollisions
            navg_SPL += SPL
   
            if (SPL > 0.001):
                  nsuccess += 1



    navg_collisions = float(navg_collisions)/float(nepisodes_max)
    navg_SPL = navg_SPL/float(nepisodes_max) 
    nsuccess = float(nsuccess)/float(nepisodes_max)

    print('average number of collisions per episode: ', navg_collisions)
    print('average SPL: ', navg_SPL)
    print('nsuccess: ', nsuccess)
#------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    test_rl()
    print('-'*50)
    print('done! ')

