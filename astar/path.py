import numpy as np
import sys
import matplotlib.pyplot as plt
from astar import *

#----------------------------------------------------------------------------------

def get_start_and_end(tdm):
    row, col, _ = tdm.shape
    points = []
    for i in range(row):
      for j in range(col):
          cond1 = (tdm[i,j,0] == 128 and tdm[i,j,1] == 128 and tdm[i,j,2] == 128)
          cond2 = (tdm[i,j,0] == 0 and tdm[i,j,1] == 0 and tdm[i,j,2] == 0) 
          if (cond1 or cond2): 
              points.append([i,j]) 
    N = len(points)
    print('N: ', N)
    s = np.random.randint(N)
    e = np.random.randint(N)
    start = points[s]
    end = points[e]
    start = (start[0], start[1])
    end = (end[0], end[1])
    return start, end

#----------------------------------------------------------------------------------

def do_astar():

    tdm = np.load('tdm.npy')
    print(tdm.shape)
    row, col, _ = tdm.shape

    maze = []
    for i in range(row):
       m = [1 for _ in range(col)]
       maze.append(m)
    
    for i in range(row):
       for j in range(col):
          cond1 = (tdm[i,j,0] == 128 and tdm[i,j,1] == 128 and tdm[i,j,2] == 128)
          cond2 = (tdm[i,j,0] == 0 and tdm[i,j,1] == 0 and tdm[i,j,2] == 0) 
          if (cond1 or cond2):
              maze[i][j] = 0
    
    maze = np.array(maze)
 
    start, end = get_start_and_end(tdm)


    path = astar(maze, start, end)
    print(path)

    for p in path:
       r, c = p[0], p[1]
       tdm[r,c,:] = [255, 0, 0] 

 
    tdm[start[0], start[1], :] = [0, 255, 0]
    tdm[end[0], end[1], :] = [0, 0, 255]
    
    plt.imshow(tdm)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.pause(2)
    plt.close()

#----------------------------------------------------------------------------------

for _ in range(10):
       do_astar()

