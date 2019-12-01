import habitat
import cv2
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import argparse
import time

from habitat.utils.visualizations import maps
from habitat.core.simulator import AgentState, SimulatorActions
from habitat.tasks.utils import quaternion_rotate_vector, cartesian_to_polar
from utils_kb import *

print(SimulatorActions)



pwd = "/home/kb/habitat_experiments/code/v7/"
config=habitat.get_config(pwd + "pointnav_gibson_kb.yaml")
print(config)
# config.TASK.GOAL_SENSOR_UUID = "pointgoal"

map_res = int(config['TASK']['TOP_DOWN_MAP']['MAP_RESOLUTION'])


NEPISODES = 2000


print('-----------------------HABITAT-------------------')
#------------------------------------------------------------------------------------
'''
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
'''
#------------------------------------------------------------------------------------


def save_img(s, nep, nstep):

     rgb = s["rgb"]
     depth = s["depth"][:,:,0] 

     assert rgb.shape == (256, 256, 3), print('rgb ', rgb.shape)
     assert depth.shape == (256, 256), print('depth ', depth.shape)
   

     fdepth = "imgs/depth/depth_" + str(nep) + "_" + str(nstep) 
     np.save(fdepth,depth)   

  
     frgb = "imgs/rgb/rgb_" + str(nep) + "_" + str(nstep) 
     np.save(frgb,rgb)

     '''     
     plt.imshow(rgb)
     plt.axis('off')   
     plt.savefig("imgs/rgb/rgb_" + str(nep) + "_" + str(nstep) + ".png") 
     plt.clf()
     plt.close()
     '''
    

#-------------------------------------------  
  

def collect_imgs():
    print('create env')
    env = habitat.Env(config)

    init_pos_agent = []
    target_pos = []

    for nep in range(NEPISODES):

        print('-'*50)
        print('episode: ', nep)

        time1 = time.time()

        s = env.reset()


        actions, pose3dof = [], []

        agent_position = env.sim.get_agent_state().position
        init_geodesic_dist = get_dist(env)
        print('init geodesic dist to goal: ', init_geodesic_dist)      

        nstep = 0

        

        save_img(s, nep, nstep)

        tdm, agent_grid, target_grid = get_map(env, map_res)
        ftdm = "imgs/map/map_" + str(nep)
        np.save(ftdm,tdm)
        #print('tdm shape: ', tdm.shape)
        #plt.imshow(tdm)
        #plt.axis('off')
        #plt.savefig("imgs/map/map_" + str(nep) + ".png") 
        #plt.close()


        x = agent_position[0]
        y = agent_position[2]
        angle = s["heading"]
        pose3dof.append([x,y,angle])


        init_pos_agent.append(agent_grid) 
        target_pos.append(target_grid) 
 

        prev_act = SimulatorActions.MOVE_FORWARD

        
        #while not env.episode_over:
        for kk in range(36):  

            rgb = s["rgb"]    
            depth = s["depth"]       
            pointgoal = s["pointgoal"]; pointgoal = [pointgoal[0], pointgoal[2]] 


            # one action only
            if (nep % 2 == 0):
               act = SimulatorActions.TURN_RIGHT 
            else:
               act = SimulatorActions.TURN_LEFT  


            s2 = env.step(act)
            nstep += 1
            s = s2 
            prev_act = act 

            save_img(s, nep, nstep)

            actions.append(act)  

            agent_position = env.sim.get_agent_state().position 
            x = agent_position[0]
            y = agent_position[2]
            angle = s["heading"]
            pose3dof.append([x,y,angle])


        actions = np.array(actions,dtype=np.float32)
        np.save("imgs/npy/actions_" + str(nep), actions)
        pose3dof = np.array(pose3dof,dtype=np.float32)
        np.save("imgs/npy/pose3dof_" + str(nep),pose3dof)
        print('nsteps: ', nstep)
        print('time for ep: ', time.time()-time1)


    init_pos_agent = np.array(init_pos_agent)
    target_pos = np.array(target_pos)
    np.save("imgs/npy/init_pos_agent",init_pos_agent)
    np.save("imgs/npy/target_pos",target_pos)

if __name__ == "__main__":
    collect_imgs()
    print('-'*50)
    print('done! ')
