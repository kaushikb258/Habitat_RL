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


print(SimulatorActions)

pwd = "/home/kb/habitat_experiments/code/vae_128/t1/"
config=habitat.get_config(pwd + "pointnav_gibson_kb.yaml")
print(config)

map_res = int(config['TASK']['TOP_DOWN_MAP']['MAP_RESOLUTION'])

load_ckpt = True #False
ep_start = 145000 #0


EPSILON_CLIP = 0.1 # init clip value
c_2_ENTROPY = 0.01

batch_size = 64
n_z = 128 #256
gamma = 0.95
gae_lambda = 1.0
learning_rate = 1.0e-4
img_size = 192
S_DIM = 2*n_z + 3 
A_DIM = 3
nepisodes_max = int(1e6)


allowed_actions = [SimulatorActions.MOVE_FORWARD, SimulatorActions.TURN_LEFT, SimulatorActions.TURN_RIGHT]


if (load_ckpt == False):
      ep_start = 0


print('-----------------------HABITAT-------------------')
#------------------------------------------------------------------------------------
'''
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
'''
#------------------------------------------------------------------------------------

def was_there_a_collision(env, old_position, config, act):
        new_position = env.sim.get_agent_state().position
        old_ = np.array([old_position[0], old_position[2]])
        new_ = np.array([new_position[0], new_position[2]]) 
        dist_moved = np.linalg.norm(old_ - new_)
                
        if (dist_moved < 0.95 * config.SIMULATOR.FORWARD_STEP_SIZE and act == 0):
              # collision
              #print('collision ', old_position, new_position, act, dist_moved)   
              return True

        return False 

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


def train_rl():
    print('create env')
    env = habitat.Env(config)

    # save a copy of the map
    print('map res: ', map_res) 
    top_down_map = maps.get_topdown_map(env.sim, num_samples=1000000, map_resolution=(map_res, map_res))
    top_down_map, rmin, rmax, cmin, cmax = crop_map(top_down_map)
    np.save('TDM_maps/tdm', top_down_map)


    print('create tf session')
    sess = tf.Session()

    Policy = Policy_net('policy', sess, env, S_DIM, A_DIM)
    Old_Policy = Policy_net('old_policy', sess, env, S_DIM, A_DIM)
    PPO = PPOTrain(Policy, Old_Policy, sess, gamma=gamma)

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


    if (load_ckpt):
         saver_ppo.restore(sess, 'ckpt/ppo/model')

    saver_vae_rgb.restore(sess, 'ckpt/vae_rgb/model')
    saver_vae_depth.restore(sess, 'ckpt/vae_depth/model')


    print("total number of trainable params: ", Policy.num_train_params())

    
    
    for ep in range(ep_start, nepisodes_max):

            
            #if (ep < 175000):
            epsilon_ppo = EPSILON_CLIP 
            stochastic = True
            c_2 = c_2_ENTROPY  
            #else:
            #   epsilon_ppo = EPSILON_CLIP * (1.0 - float(ep-175000)/float(100000))
            #   stochastic = False
            #   c_2 = c_2_ENTROPY * (1.0 - float(ep-175000)/float(100000))
            epsilon_ppo = max(epsilon_ppo, 0.02)
            c_2 = max(c_2, 0.001)
            print('epsilon: ', epsilon_ppo, ' | c_2: ', c_2) 


            s = env.reset()

            observations, actions, rewards, v_preds = [], [], [], []

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

            agent_path = []
 

            while (not done) and (not env.episode_over) and (not success):  

                agent_path.append(env.sim.get_agent_state().position)

                rgb = s["rgb"]    
                depth = s["depth"]       
                pointgoal = s["pointgoal"]; pointgoal = [pointgoal[0], pointgoal[2]]
                heading = s["heading"] 

                # perception_embedding is the costly operation in the entire one time step (~ 3-4X slower than env.step/Policy.act)

                obs, z_rgb_t, z_depth_t = perception_embedding(vae_rgb, vae_depth, rgb, depth, pointgoal, heading, S_DIM, n_z, beta_rgb, beta_depth)

                act, v_pred, rnn_state = Policy.act(obs=obs, rnn_state=rnn_state, stochastic=stochastic)

                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)
                
                s2 = env.step(allowed_actions[act])
 
                nsteps += 1 
                new_dist = get_dist(env)
                  
                reward, done, success = get_reward(dist, new_dist, nsteps, env) 


                #-----------------------------  
                # collision penalty
                if (act == 0):
                    if (was_there_a_collision(env, old_position, config, act)): 
                          ncollisions += 1
                #          reward -= 0.025 # 0.1
                #-----------------------------  


                new_position = env.sim.get_agent_state().position
                

                if (new_dist <= 0.2): 
                    s2 = env.step(SimulatorActions.STOP)
                    new_dist = get_dist(env)                      


                a_t = np.zeros((1,1),dtype=np.int32); a_t[0,0] = act
                pg_t = np.zeros((1,2),dtype=np.float32); pg_t[0,:] = pointgoal
                h_t = np.zeros((1,1),dtype=np.float32); h_t[0,0] = heading                

                dist_traveled_by_agent += abs(new_dist - dist)


                #-----------------------------  
                # SPL reward for success (SPL bonus reward)
                #if (success):
                #     spl = float(success)*init_geodesic_dist/max(dist_traveled_by_agent, init_geodesic_dist)    
                #     reward += spl * 10.0
                #-----------------------------  


                ep_rewards += reward   
                

                observations.append(obs)
                actions.append(act)
                rewards.append(reward)
                v_preds.append(v_pred)
               


                if done or success:
                    rgb = s2["rgb"]    
                    depth = s2["depth"]       
                    pointgoal = s2["pointgoal"]; pointgoal = [pointgoal[0], pointgoal[2]]
                    heading = s2["heading"] 
                    next_obs, _, _ = perception_embedding(vae_rgb, vae_depth, rgb, depth, pointgoal, heading, S_DIM, n_z, beta_rgb, beta_depth) 
                    _, v_last, rnn_state = Policy.act(obs=next_obs, rnn_state=rnn_state, stochastic=stochastic)
                    v_preds_next = v_preds[1:] + [v_last]  
                    break
                else:
                    s = s2
                    dist = new_dist
                    old_position = new_position 


            if (ep % 50 == 0):
                    saver_ppo.save(sess, 'ckpt/ppo/model')
                    

            gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

             
            observations = np.array(observations).astype(dtype=np.float32)
            observations = np.squeeze(observations, 1)            

            actions = np.array(actions).astype(dtype=np.int32)
            rewards = np.array(rewards).astype(dtype=np.float32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
            gaes = np.array(gaes).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) / gaes.std()

            gaes = np.squeeze(gaes,2)
            gaes = np.squeeze(gaes,1)

            PPO.assign_policy_parameters()

            inp = [observations, actions, rewards, v_preds_next, gaes]

            # train
            sample_indices = np.arange(observations.shape[0]) 
            sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  

            for epoch in range(5):
                 PPO.train(obs=sampled_inp[0], actions=sampled_inp[1], rewards=sampled_inp[2], v_preds_next=sampled_inp[3], gaes=sampled_inp[4], epsilon_ppo=epsilon_ppo, c_2=c_2)
  

            final_agent_position = env.sim.get_agent_state().position 
            target_position = env.current_episode.goals[0].position 
            geo_dist_traveled_by_agent = env.sim.geodesic_distance(init_agent_position, final_agent_position)
            SPL = float(success)*init_geodesic_dist/max(dist_traveled_by_agent, init_geodesic_dist)
            final_dist_to_target = env.sim.geodesic_distance(final_agent_position, target_position)
            
            # final position of agent  
            agent_path.append(env.sim.get_agent_state().position)

            # TDM
            if (ep % 10 == 0):
                tdm_with_path = get_map_with_path(env, map_res, agent_path)
                plt.imshow(tdm_with_path); plt.tight_layout(); plt.axis('off')
                fname = 'TDM_maps/tdm_' + str(ep) + '_' + str(int(SPL*100.0)) + '.png'; plt.savefig(fname)
           


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
            print('-'*50)


            f = open("performance/performance_ppo.txt", "a+")
            f.write(str(ep) + " " + str(ep_rewards) + " " + str(nsteps) + " " + str(SPL) + " " + " " + str(init_geodesic_dist) + " " + str(final_dist_to_target) + '\n')  
            f.close()     


#------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    train_rl()
    print('-'*50)
    print('done! ')

