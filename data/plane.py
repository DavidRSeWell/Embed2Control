#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from numpy.random import randint
from torch.utils.data import Dataset

num_t=80 # number of trajectories (i.e. number of initial states)
T=1000 # length of each trajectory sequence
u_dim=2 # control (action) dimension
w,h=40,40
x_dim=w*h
rw=1 # robot half-width

def get_params():
  return x_dim,u_dim,T

class PlaneData:
  def __init__(self, fname, env_file):
    self.cache=fname
    self.initialized=False
    self.im=plt.imread(env_file) # grayscale
    self.params=(x_dim,u_dim,T)

  def is_colliding(self,p):
    if np.any([p-rw<0, p+rw>=w]):
      return True
    # check robot body overlap with obstacle field
    return np.mean(self.im[p[0]-rw:p[0]+rw+1, p[1]-rw:p[1]+rw+1]) > 0.05

  def compute_traj(self, max_dist=1):
    # computes P,U data for single trajectory
    # all P,U share the same environment obstacles.png
    P=np.zeros((T,2),dtype=np.int) # r,c position
    U=np.zeros((T,u_dim),dtype=np.int)
    P[0,:]=[rw,randint(rw,w-rw)] # initial location
    for t in range(1,T):
      p=np.copy(P[t-1,:])
      # dr direction
      d=randint(-1,2) # direction
      nsteps=randint(max_dist+1)
      dr=d*nsteps # applied control
      for i in range(nsteps):
        p[0]+=d
        if self.is_colliding(p):
          p[0]-=d
          break
      # dc direction
      d=randint(-1,2) # direction
      nsteps=randint(max_dist+1)
      dc=d*nsteps # applied control
      for i in range(nsteps):
        p[1]+=d
        if self.is_colliding(p):
          p[1]-=d # step back
          break
      P[t,:]=p
      U[t,:]=[dr,dc]
    return P,U

  def initialize(self):
    if os.path.exists(self.cache):
      self.load()
    else:
      self.precompute()
    self.initialized=True

  def compute_data(self):
    # compute multiple trajectories
    P=np.zeros((num_t,T,2),dtype=np.int)
    U=np.zeros((num_t,T,u_dim),dtype=np.int)
    for i in range(num_t):
      P[i,:,:], U[i,:,:] = self.compute_traj(max_dist=2)
    return P,U

  def precompute(self):
    print("Precomputing P,U...")
    self.P, self.U = self.compute_data()

  def save(self):
    print("Saving P,U...")
    np.savez(self.cache, P=self.P, U=self.U)

  def load(self):
    print("Loading P,U from %s..." % (self.cache))
    D=np.load(self.cache)
    self.P, self.U = D['P'], D['U']

  def getXp(self,p):
    # return image X given true state p (position) of robot
    x=np.copy(self.im)
    x[p[0]-rw:p[0]+rw+1, p[1]-rw:p[1]+rw+1]=1. # robot is white on black background
    return x.flat

  def getX(self,i,t):
    # i=trajectory index, t=time step
    return self.getXp(self.P[i,t,:])

  def getXTraj(self,i):
    # i=traj index
    X=np.zeros((T,x_dim),dtype=np.float)
    for t in range(T):
      X[t,:]=self.getX(i,t)
    return X

  def sample(self, batch_size):
    """
    computes (x_t,u_t,x_{t+1}) pair
    returns tuple of 3 ndarrays with shape
    (batch,x_dim), (batch, u_dim), (batch, x_dim)
    """
    if not self.initialized:
      raise ValueError("Dataset not loaded - call PlaneData.initialize() first.")
    traj=randint(0,num_t,size=batch_size) # which trajectory
    tt=randint(0,T-1,size=batch_size) # time step t for each batch
    X0=np.zeros((batch_size,x_dim))
    U0=np.zeros((batch_size,u_dim),dtype=np.int)
    X1=np.zeros((batch_size,x_dim))
    for i in range(batch_size):
      t=tt[i]
      p=self.P[traj[i], t, :]
      X0[i,:]=self.getX(traj[i],t)
      X1[i,:]=self.getX(traj[i],t+1)
      U0[i,:]=self.U[traj[i], t, :]
    return (X0,U0,X1)

  def getPSpace(self):
    """
    Returns all possible positions of agent
    """
    ww=h-2*rw
    P=np.zeros((ww*ww,2)) # max possible positions
    i=0
    p=np.array([rw,rw]) # initial location
    for dr in range(ww):
      for dc in range(ww):
        if not self.is_colliding(p+np.array([dr,dc])):
          P[i,:]=p+np.array([dr,dc])
          i+=1
    return P[:i,:]

  def getXPs(self, Ps):
    X=np.zeros((Ps.shape[0],x_dim))
    for i in range(Ps.shape[0]):
      X[i,:]=self.getXp(Ps[i,:])
    return X

class PlaneDataSet(Dataset):
    @classmethod
    def load_from_path(cls, path: str, N):
        p=PlaneData(path + "plane2.npz", path + "env1.png")
        p.initialize()
        return cls(p, N)

    def __init__(self, p: PlaneData, N: int = 10000):
        self._p = p
        self.data = self.sample_data(N)
        
    def sample_data(self, n) -> dict:
        xt, ut, xt_1 = self._p.sample(n)
        return {"xt":torch.Tensor(xt), "ut":torch.Tensor(ut), "xt_1":torch.Tensor(xt_1)}
        
    def __len__(self):
        return self.data["xt"].shape[0]
    
    def __getitem__(self, idx):
        return (self.data["xt"][idx,:], self.data["ut"][idx,:], self.data["xt_1"][idx,:])

def plot_scatter_pos(P, model, save_path):
    
    N = 40
    cmaps = ["Reds", "Greens", "Oranges", "Blues"]
    q1, z1 = [], []
    q2, z2 = [], []
    q3, z3 = [], []
    q4, z4 = [], []
    for x in range(N):
        for y in range(N):
            if P.is_colliding(np.array([y,x])):
                    continue
            if x <= 20 and y < 20:
                q1.append([x,y])
                x0 = torch.Tensor(np.array(P.getXp([y,x])).reshape((1,N**2)))
                z = model.encoder(x0).detach().numpy().flatten()
                z1.append(z)

            elif x > 20 and y < 20:
                q2.append([x,y])
                x0 = torch.Tensor(np.array(P.getXp([y,x])).reshape((1,N**2)))
                z = model.encoder(x0).detach().numpy().flatten()
                z2.append(z)

            elif x <= 20 and y >= 20:
                q3.append([x,y])
                x0 = torch.Tensor(np.array(P.getXp([y,x])).reshape((1,N**2)))
                z = model.encoder(x0).detach().numpy().flatten()
                z3.append(z)

            else:
                q4.append([x,y])
                x0 = torch.Tensor(np.array(P.getXp([y,x])).reshape((1,N**2)))
                z = model.encoder(x0).detach().numpy().flatten()
                z4.append(z)
    
    q1 = np.array(q1)
    q2 = np.array(q2)
    q3 = np.array(q3)
    q4 = np.array(q4)
    
    plt.title("Simple Environment")
    plt.scatter(np.array(q1)[:,0],np.array(q1)[:,1], c=q1[:,0] + q1[:,1], cmap = cmaps[0])
    plt.scatter(np.array(q2)[:,0],np.array(q2)[:,1], c=q2[:,0] + q2[:,1], cmap = cmaps[1])
    plt.scatter(np.array(q3)[:,0],np.array(q3)[:,1], c=q3[:,0] + q3[:,1], cmap = cmaps[2])
    plt.scatter(np.array(q4)[:,0],np.array(q4)[:,1], c=q4[:,0] + q4[:,1], cmap = cmaps[3])
    ax=plt.gca()                            # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    ax.xaxis.tick_top()                     # and move the X-Axis      
    ax.yaxis.set_ticks(np.arange(0, 40, 5)) # set y-ticks
    ax.yaxis.tick_left()                    # remove right y-Ticks
    plt.savefig(save_path + "simple_env.png")   
    plt.close()
    
    plt.title("Latent Encoding")
    plt.scatter(np.array(z1)[:,0],np.array(z1)[:,1], c=q1[:,0] + q1[:,1], cmap = cmaps[0])
    plt.scatter(np.array(z2)[:,0],np.array(z2)[:,1], c=q2[:,0] + q2[:,1], cmap = cmaps[1])
    plt.scatter(np.array(z3)[:,0],np.array(z3)[:,1], c=q3[:,0] + q3[:,1], cmap = cmaps[2])
    plt.scatter(np.array(z4)[:,0],np.array(z4)[:,1], c=q4[:,0] + q4[:,1], cmap = cmaps[3])
   
    ax=plt.gca()                            # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    ax.xaxis.tick_top()                     # and move the X-Axis      
    ax.yaxis.tick_left()                    # remove right y-Ticks
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.savefig(save_path + "simple_latent_env.png")
    plt.close()

def plot_traj(P, traj, save_path):
  """
  Plot a trajectory taken by the agent
  """
  N = 40
  plt.title("Plot Traj")
  col_obj = []
  for x in range(N):
    for y in range(N):
        if P.is_colliding(np.array([y,x])):
          col_obj.append([y,x])

  col_obj = np.array(col_obj)
  plt.scatter(col_obj[:,0], col_obj[:,1], color="black")
  plt.scatter(traj[:,1], traj[:,0], color = "red")
  ax=plt.gca()                            # get the axis
  ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
  ax.xaxis.tick_top()                     # and move the X-Axis      
  ax.yaxis.set_ticks(np.arange(0, 40, 5)) # set y-ticks
  ax.yaxis.tick_left()                    # remove right y-Ticks
  plt.savefig(save_path)
  plt.close()

def plot_traj_z(P, traj, model, start_state, goal_state, save_path):
    
    N = 40
    cmaps = ["Reds", "Greens", "Oranges", "Blues"]
    q1, z1 = [], []
    q2, z2 = [], []
    q3, z3 = [], []
    q4, z4 = [], []
    for x in range(N):
        for y in range(N):
            if P.is_colliding(np.array([y,x])):
                    continue
            if x <= 20 and y < 20:
                q1.append([x,y])
                x0 = torch.Tensor(np.array(P.getXp([y,x])).reshape((1,N**2)))
                z = model.encoder(x0).detach().numpy().flatten()
                z1.append(z)

            elif x > 20 and y < 20:
                q2.append([x,y])
                x0 = torch.Tensor(np.array(P.getXp([y,x])).reshape((1,N**2)))
                z = model.encoder(x0).detach().numpy().flatten()
                z2.append(z)

            elif x <= 20 and y >= 20:
                q3.append([x,y])
                x0 = torch.Tensor(np.array(P.getXp([y,x])).reshape((1,N**2)))
                z = model.encoder(x0).detach().numpy().flatten()
                z3.append(z)

            else:
                q4.append([x,y])
                x0 = torch.Tensor(np.array(P.getXp([y,x])).reshape((1,N**2)))
                z = model.encoder(x0).detach().numpy().flatten()
                z4.append(z)
  
    q1 = np.array(q1)
    q2 = np.array(q2)
    q3 = np.array(q3)
    q4 = np.array(q4)

    start_state = torch.Tensor(np.array(P.getXp(start_state)).reshape((1,N**2)))
    goal_state = torch.Tensor(np.array(P.getXp(goal_state)).reshape((1,N**2)))
    start_z = model.encoder(start_state).detach().numpy().flatten()
    goal_z = model.encoder(goal_state).detach().numpy().flatten()

    plt.title("Latent Encoding")
    plt.scatter(np.array(z1)[:,0],np.array(z1)[:,1], c=q1[:,0] + q1[:,1], cmap = cmaps[0])
    plt.scatter(np.array(z2)[:,0],np.array(z2)[:,1], c=q2[:,0] + q2[:,1], cmap = cmaps[1])
    plt.scatter(np.array(z3)[:,0],np.array(z3)[:,1], c=q3[:,0] + q3[:,1], cmap = cmaps[2])
    plt.scatter(np.array(z4)[:,0],np.array(z4)[:,1], c=q4[:,0] + q4[:,1], cmap = cmaps[3])
    plt.scatter(traj[:,0], traj[:,1], color = "black", marker = "x")
    for i in range(traj.shape[0]):
       plt.annotate(str(i + 1), (traj[i][0], traj[i][1]))
    plt.scatter(start_z[0], start_z[1], color = "silver", marker = "x")
    plt.scatter(goal_z[0], goal_z[1], color = "gold", marker = "x")

    ax=plt.gca()                            # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    ax.xaxis.tick_top()                     # and move the X-Axis      
    ax.yaxis.tick_left()                    # remove right y-Ticks
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.savefig(save_path + "simple_latent_env.png")
    plt.close()


if __name__ == "__main__":

  test_1 = 0
  test_2 = 1
  if test_1 == 1:

    import matplotlib.animation as animation
    p=PlaneData("plane2.npz","env1.png")
    p.initialize()
    p.save()
    im=p.im
    A,B=im.shape

    # show sample tuples
    if True:
      fig, aa = plt.subplots(1,2)
      x0,u0,x1=p.sample(2)
      m1=aa[0].matshow(x0[0,:].reshape(w,w), cmap=plt.cm.gray, vmin = 0., vmax = 1.)
      aa[0].set_title('x(t)')
      m2=aa[1].matshow(x1[0,:].reshape(w,w), cmap=plt.cm.gray, vmin = 0., vmax = 1.)
      aa[1].set_title('x(t+1), u=(%d,%d)' % (u0[0,0],u0[0,1]))
      fig.tight_layout()
      def updatemat2(t):
        x0,u0,x1=p.sample(2)
        m1.set_data(x0[0,:].reshape(w,w))
        m2.set_data(x1[0,:].reshape(w,w))
        return m1,m2

      anim=animation.FuncAnimation(fig, updatemat2, frames=100, interval=1000, blit=True, repeat=True)

      Writer = animation.writers['imagemagick'] # animation.writers.avail
      writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
      anim.save('sample_obs.gif', writer=writer)

    #show trajectory
    if True:
      fig, ax = plt.subplots()
      X=p.getXTraj(0)
      mat=ax.matshow(X[0,:].reshape((A,B)), cmap=plt.cm.gray, vmin = 0., vmax = 1.)
      def updatemat(t):
        mat.set_data(X[t,:].reshape((A,B)))
        return mat,
      anim = animation.FuncAnimation(fig, updatemat, frames=T-1, interval=30, blit=True, repeat=True)
      plt.show()

  if test_2 == 2:
    import matplotlib.animation as animation
    from embed2control.base_model_experiment import get_optimal_trajectory, lqr

    p=PlaneData("plane2.npz","env1.png")
    p.initialize()
    p.save()
    im=p.im
    A,B=im.shape

    start_state = np.array([1, 1])

    goal_state = np.array([38, 38])

    images, u_optimal, pos_optimal = get_optimal_trajectory(lqr, plane_data_set._p.im, start_state, 100, goal_state)

    plot_traj(p, traj)