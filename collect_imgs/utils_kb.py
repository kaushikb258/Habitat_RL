import numpy as np
import habitat
from habitat.utils.visualizations import maps
from habitat.core.simulator import AgentState, SimulatorActions




def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax



def crop_map(top_down_map):

        recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]

        rmin, rmax, cmin, cmax = bbox(top_down_map)

        padding = 12

        rmin = max(rmin-padding, 0)
        rmax = min(rmax+padding, top_down_map.shape[0])  
        cmin = max(cmin-padding, 0)
        cmax = min(cmax+padding, top_down_map.shape[1]) 
        top_down_map = top_down_map[rmin:rmax,cmin:cmax]

        top_down_map = recolor_map[top_down_map]

        return top_down_map, rmin, rmax, cmin, cmax 




def get_map(env, map_res):

     # top down map 
     top_down_map = maps.get_topdown_map(env.sim, num_samples=1000000, map_resolution=(map_res, map_res))
     top_down_map, rmin, rmax, cmin, cmax = crop_map(top_down_map)

     # target/goal
     target_position = env.current_episode.goals[0].position 
     target_position_grid = maps.to_grid(target_position[0], target_position[2], maps.COORDINATE_MIN, maps.COORDINATE_MAX, (map_res,map_res))

     # agent's current position   # to_grid converts real world (x,y) to pixel (x,y)   
     agent_position = env.sim.get_agent_state().position
     agent_position_grid = maps.to_grid(agent_position[0], agent_position[2], maps.COORDINATE_MIN, maps.COORDINATE_MAX, (map_res,map_res))

     # grid offset for map
     agent_position_grid = (agent_position_grid[0] - rmin, agent_position_grid[1] - cmin)
     target_position_grid = (target_position_grid[0] - rmin, target_position_grid[1] - cmin)

     tdm = top_down_map.copy()

     # agent pos
     m1 = 4
     tdm[agent_position_grid[0]-m1:agent_position_grid[0]+m1,agent_position_grid[1]-m1:agent_position_grid[1]+m1] = 0

     # target pos
     m2 = 7
     tdm[target_position_grid[0]-m1:target_position_grid[0]+m1,target_position_grid[1]-m2:target_position_grid[1]+m2] = 0
     tdm[target_position_grid[0]-m2:target_position_grid[0]+m2,target_position_grid[1]-m1:target_position_grid[1]+m1] = 0


     assert agent_position_grid[0] > 0 and agent_position_grid[0] < top_down_map.shape[0], print('assert1 ', agent_position_grid, top_down_map.shape)
     assert target_position_grid[0] > 0 and target_position_grid[0] < top_down_map.shape[0], print('assert2 ', target_position_grid, top_down_map.shape)
     assert agent_position_grid[1] > 0 and agent_position_grid[1] < top_down_map.shape[1], print('assert3 ', agent_position_grid, top_down_map.shape)
     assert target_position_grid[1] > 0 and target_position_grid[1] < top_down_map.shape[1], print('assert4 ', target_position_grid, top_down_map.shape)
      
     return tdm, agent_position_grid, target_position_grid



# not sure if dones must be used; we might end up masking the last value for which reward is high
# if we did succeed in reaching the goal
def get_adv_1(v_preds, next_value, rewards, dones, gamma=0.95, gae_lambda=1.0):
       T = rewards.shape[0]       
       value_preds = np.zeros((T+1),dtype=np.float32)
       value_preds[:T] = v_preds 
       value_preds[-1] = next_value
       
       returns = np.zeros((T+1),dtype=np.float32)


       gae = 0
       for step in reversed(range(T)):
            #delta = rewards[step] + gamma * value_preds[step + 1] * (1.0-dones[step]) - value_preds[step]
            #gae = delta + gamma * gae_lambda * (1.0-dones[step]) * gae
            delta = rewards[step] + gamma * value_preds[step + 1] - value_preds[step]
            gae = delta + gamma * gae_lambda * gae
            returns[step] = gae + value_preds[step]

       advantages = returns[:-1] - value_preds[:-1]
       #advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-5)

       return advantages

# KAUSHIK- BOTH THESE ADVANTAGE FUNCTIONS ARE IDENTICAL!

def get_adv(v_preds_, next_value, rewards, dones, gamma=0.95, gae_lambda=1.0):
        T = rewards.shape[0]
        v_preds = np.zeros((T+1),dtype=np.float32) 
        v_preds[:T] = v_preds_
        v_preds[-1] = next_value 
        adv = np.zeros((T),dtype=np.float32)
        for t in range(T):
           for l in range(T-t):
             delta = -v_preds[t+l] + rewards[t+l] + gamma*v_preds[t+l+1] 
             adv[t] += ((gamma*gae_lambda)**l)*delta             
        return adv 



def get_dist(env):
     agent_position = env.sim.get_agent_state().position
     target_position = env.current_episode.goals[0].position 
     dist = env.sim.geodesic_distance(agent_position, target_position)
     #dist = np.sqrt((agent_position[0]-target_position[0])**2 + (agent_position[2]-target_position[2])**2)
     return dist


