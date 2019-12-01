import habitat
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

from habitat.utils.visualizations import maps
from habitat.core.simulator import AgentState, SimulatorActions
from utils_kb import *



print(SimulatorActions)

pwd = "/home/kb/habitat_experiments/code/v6/"
config=habitat.get_config(pwd + "pointnav_gibson_kb.yaml")
print(config)

map_res = int(config['TASK']['TOP_DOWN_MAP']['MAP_RESOLUTION'])


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


def train_rl():
    print('create env')
    env = habitat.Env(config)
   
    hist = np.zeros((10),dtype=np.float32)

    for ep in range(1000):

            s = env.reset()

            init_agent_position = env.sim.get_agent_state().position
            dist = get_dist(env)      
            init_geodesic_dist = dist
            
            done = False
            success = False
            nsteps = 0
            ep_rewards = 0.0
            dist_traveled_by_agent = 0.0
 

            while (not done) and (not env.episode_over) and (not success):  

                rgb = s["rgb"]    
                depth = s["depth"]       
                pointgoal = s["pointgoal"]; pointgoal = [pointgoal[0], pointgoal[2]]
                heading = s["heading"] 

                act = SimulatorActions.MOVE_FORWARD                 
                s2 = env.step(act)

                nsteps += 1 
                new_dist = get_dist(env)
                  
                reward, done, success = get_reward(dist, new_dist, nsteps, env) 

                if (new_dist <= 0.2): 
                    s2 = env.step(SimulatorActions.STOP)
                    new_dist = get_dist(env)                      


                ep_rewards += reward   
                dist_traveled_by_agent += abs(new_dist - dist)

                
                if done or success:
                    rgb = s2["rgb"]    
                    depth = s2["depth"]       
                    pointgoal = s2["pointgoal"]; pointgoal = [pointgoal[0], pointgoal[2]]
                    heading = s2["heading"] 
                    break
                else:
                    s = s2
                    dist = new_dist

  
            final_agent_position = env.sim.get_agent_state().position 
            target_position = env.current_episode.goals[0].position 
            geo_dist_traveled_by_agent = env.sim.geodesic_distance(init_agent_position, final_agent_position)
            SPL = float(success)*init_geodesic_dist/max(dist_traveled_by_agent, init_geodesic_dist)
            final_dist_to_target = env.sim.geodesic_distance(final_agent_position, target_position)
            

            print(' ')
            print('episode: ', ep, ' | ep_rewards: ', ep_rewards, ' | ep_steps: ', nsteps, ' | SPL: ', SPL) 
            print('init geodesic dist to target: ', init_geodesic_dist)
            print('final agent pos: ', final_agent_position)
            print('target pos: ', target_position)
            print(' ')   
            print('total dist traveled by agent: ', dist_traveled_by_agent)
            print('geo dist traveled by agent: ', geo_dist_traveled_by_agent) 
            print('final dist to target: ', final_dist_to_target)          
            print('-'*50)


            f = open("performance/performance_forward_only_agent.txt", "a+")
            f.write(str(ep) + " " + str(ep_rewards) + " " + str(nsteps) + " " + str(SPL) + " " + " " + str(init_geodesic_dist) + " " + str(final_dist_to_target) + '\n')  
            f.close()     

            # histogram 
            paloalto = 1.0
            for k in range(10):
              low_ = float(k)*0.1
              high_ = low_ + 0.1
              if (SPL >= low_ and SPL <= high_ and paloalto == 1.0):
                 hist[k] += 1.0 
                 paloalto = 0.0  

    hist = hist/np.sum(hist)

    f = open("performance/hist_spl_forward_only_agent.txt", "a+")
    for k in range(10):
        low_ = float(k)*0.1
        high_ = low_ + 0.1
        mid_ = 0.5*(low_ + high_)    
        f.write(str(mid_) + " " + str(hist[k]) + '\n')  
    f.close()   
 
#------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    train_rl()
    print('-'*50)
    print('done! ')

