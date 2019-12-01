import numpy as np
import matplotlib.pyplot as plt

alpha = 0.01


def load_and_compute_mov_avg(fname):

   a = np.loadtxt(fname)
   print(a.shape)

   b = np.zeros((a.shape[0]),dtype=np.float32)
   b[0] = a[0,3]

   for i in range(1,a.shape[0]):
      b[i] = alpha*a[i,3] + (1.0-alpha)*b[i-1]

   return a, b



fname1 = '../../v10c/performance/performance_ppo.txt'
a1, b1 = load_and_compute_mov_avg(fname1)

fname2 = '../../depth_10f/performance/performance_ppo.txt'
a2, b2 = load_and_compute_mov_avg(fname2)

fname3 = '../../vae_128/t1/performance/performance_ppo.txt'
a3, b3 = load_and_compute_mov_avg(fname3)



plt.plot(a1[:,0],b1,color='r')
plt.plot(a2[:,0],b2,color='g')
plt.plot(a3[:,0],b3,color='b')

plt.hlines(y=0.8, xmin=0, xmax=200000)
plt.hlines(y=0.6, xmin=0, xmax=200000)
plt.hlines(y=0.4, xmin=0, xmax=200000)
plt.axis([0, 200000, 0, 1])
plt.xlabel('episode #', fontsize=20)
plt.ylabel('SPL', fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
#plt.savefig('SPL_plot.png')



