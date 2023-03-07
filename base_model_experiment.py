"""
Base Model Experiment
1: Learn latent encoding s.t. ||Dec(Enc(x)) - x|| is small 
2: Learn Dynamics model z_{t + 1} = f(z, u)
3: Solve Optimal Control Task (Get agent to goal state)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset, DataLoader

from embed2control.data.plane import PlaneData, PlaneDataSet, plot_scatter_pos, plot_traj, plot_traj_z
from embed2control.models import Autoencoder
from embed2control.train import VAETrainer
from embed2control.util import load_config

def plot_image(t, im):
    fig = plt.figure()
    m1=plt.matshow(im, cmap=plt.cm.gray, vmin = 0., vmax = 1.)
    plt.title(f't = {t}')
    fig.tight_layout()
    plt.show()
    plt.close()

def plot_animation(images, save_path):

    fig, ax = plt.subplots()

    ims = []
    for i, image in enumerate(images):
        im = ax.imshow(image, animated = True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000) 

    #writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    writer = animation.PillowWriter(fps=30)

    ani.save(save_path, writer=writer)

    plt.close()



def update_image(image, pos, RW = 1):
    X_t = image.copy()
    X_t[np.clip(pos[0] - RW,0,39):pos[0] + RW + 1, np.clip(pos[1] - RW,0,39): pos[1] + RW + 1] = 1 
    return X_t

def image_to_input(image):
    x = torch.Tensor(image.flatten().reshape((1, 1600)))
    return x

def draw_me(img):
    plt.pause(.01)
    plt.imshow(img)
    
def lqr(actual_state_x, desired_state_xf, Q, R, A, B):
    # We want the system to stabilize at desired_state_xf.
    x_error = actual_state_x - desired_state_xf
    print(x_error)
    # Solutions to discrete LQR problems are obtained using the dynamic 
    # programming method.
    # The optimal solution is obtained recursively, starting at the last 
    # timestep and working backwards.
    # You can play with this number
    N = 50
 
    # Create a list of N + 1 elements
    P = [None] * (N + 1)
     
    Qf = Q
 
    # LQR via Dynamic Programming
    P[N] = Qf
 
    # For i = N, ..., 1
    for i in range(N, 0, -1):
 
        # Discrete-time Algebraic Riccati equation to calculate the optimal 
        # state cost matrix
        P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
            R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)      
 
    # Create a list of N elements
    K = [None] * N
    u = [None] * N
 
    # For i = 0, ..., N - 1
    for i in range(N):
 
        # Calculate the optimal feedback gain K
        K[i] = -np.linalg.pinv(R + B.T @ P[i+1] @ B) @ B.T @ P[i+1] @ A
 
        u[i] = K[i] @ x_error.T
 
    # Optimal control input is u_star
    u_star = u[N-1]
 
    return u_star

def get_optimal_trajectory(lqr, image, pos, T, goal, A, B, Q, R):

    current_image = update_image(image, pos)
    current_pos = pos
    pos_t = [pos]
    images = np.zeros((T, 40, 40))
    images[0,:,:] = current_image
    actions = []
    for t in range(1,T):
        u = lqr(current_pos, goal, Q, R, A, B).flatten()
        u[0] = np.clip(u[0], -1, 2)
        u[1] = np.clip(u[1], -1, 2)
        
        current_pos[0] += round(u[0])
        current_pos[1] += round(u[1])
        pos_t.append(current_pos.copy())
        current_image = update_image(image, current_pos)
        images[t,:,:] = current_image
        actions.append(u)

    pos_t = np.array(pos_t)

    return images, actions, pos_t

def get_optimal_trajectory_latent(embed_model, dynamics_model, lqr, image, pos, T, goal, A, B, Q, R):

    current_image = update_image(image, pos)
    N = current_image.shape[0]
    current_pos = pos

    current_z_pos = embed_model.encoder(torch.Tensor(current_image.reshape((1,N**2)))).detach().numpy()
    pos_t = [pos]

    latent_pos_t = [current_z_pos.copy()]
    predicted_z_pos = []

    images = np.zeros((T, 40, 40))
    images[0,:,:] = current_image
    actions = []
    for t in range(1,T):
        u = lqr(current_z_pos, goal, Q, R, A, B).flatten()
        u[0] = np.clip(u[0], -1, 2)
        u[1] = np.clip(u[1], -1, 2)
        
        current_pos[0] += round(u[0])
        current_pos[1] += round(u[1])
        pred_z_pos = dynamics_model(current_z_pos, u)

        pos_t.append(current_pos.copy())
        predicted_z_pos.append(pred_z_pos)

        current_image = update_image(image, current_pos)
        current_z_pos = embed_model.encoder(torch.Tensor(current_image.reshape((1,N**2)))).detach().numpy()
        latent_pos_t.append(current_z_pos.copy())
        images[t,:,:] = current_image
        actions.append(u)

    pos_t = np.array(pos_t)
    latent_pos_t = np.array(latent_pos_t).squeeze()
    predicted_z_pos = np.array(pred_z_pos).squeeze()

    return images, actions, pos_t, latent_pos_t, predicted_z_pos

def learn_embedding_model(model, config: dict, dataset, loss, optim):

    model_config = config["Model"]

    train_error, test_error = model.train()

    print(f"Train Error = {train_error}")

    print(f"Test Error = {test_error}")
    plt.figure(figsize=(15,8))

    plt.title("Train vs test error")
    x_ = [i for i in range(len(train_error))]
    plt.scatter(x_, train_error, label = "train")
    plt.legend()
    plt.savefig(config["save_path"])
    plt.close()

    return model

def learn_linear_dynamics_model(embed_model, p, n: int = 10000):
    """
    Here we are going to use a simple linear dynamics model
    """
    embed_model.eval()
    xt, ut, xt_1 = p.sample(n)
    xt, xt_1 = torch.Tensor(xt), torch.Tensor(xt_1)
    zt = embed_model.encoder(xt).detach().numpy()
    zt_1 = embed_model.encoder(xt_1).detach().numpy()
    X = np.concatenate((zt, ut), axis =1)
    reg = LinearRegression(fit_intercept = False).fit(X, zt_1)
    
    score = reg.score(X, zt_1)
    print("Done fitting model")
    print(score)

    A = reg.coef_[:,:2]
    B = reg.coef_[:,2:]

    f = lambda z, u: A @ z.T + B @ u.T
    return f, A, B

def run_plane_experiment(config: dict):

    data_config = config["Data"]
    dynamics_model_path = data_config["dynamics_model_path"]
    embedding_model_path = data_config["embedding_model_path"]
    model_config = config["Model"] 

    data_path = data_config["path"]
    save_path = config["save_path"]
    samples = data_config["samples"]

    plane_data_set = PlaneDataSet.load_from_path(data_path, samples)

    model = Autoencoder.init_variational(**model_config)

    optim = torch.optim.Adam(model.parameters(), model_config["lr"])

    loss = nn.BCELoss()

    embedding_model = VAETrainer(model, plane_data_set, model_config["train_split"], loss, model_config["epochs"], optim, test_every = model_config["test_every"], 
    batch_size = model_config["batch_size"])

    if len(embedding_model_path) > 0 and os.path.isfile(embedding_model_path):
        print("Loading embedding model") 
        embedding_model.model.load_state_dict(torch.load(embedding_model_path))
    else:
        embedding_model = learn_embedding_model(embedding_model, config, plane_data_set, loss, optim)
        torch.save(embedding_model.model.state_dict(), save_path + "embedding_base_model")

    plot_scatter_pos(plane_data_set._p, embedding_model.model, save_path)

    start_state = np.array([1, 1])

    goal_state = np.array([38, 38])

    A = np.diag([1, 1])
    B = np.diag([1, 1])
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    R = np.array([[0.01, 0], [0.0, 0.01]]) 

    images, u_optimal, pos_optimal = get_optimal_trajectory(lqr, plane_data_set._p.im, start_state, 25, goal_state, A, B, Q, R)

    plot_traj(plane_data_set._p, pos_optimal, save_path + "optimal_traj.png")
    
    if len(dynamics_model_path) > 0 and os.path.isfile(dynamics_model_path):
        print("Loading base dynamics model z_{t + 1} = f(z, u) ") 
    else:
        dynamics_model, A, B = learn_linear_dynamics_model(embedding_model.model, plane_data_set._p, n = 100000)

    start_state = np.array([1, 1])

    images, u_optimal, pos_optimal, latent_pos_t, predicted_z_pos = get_optimal_trajectory_latent(embedding_model.model, dynamics_model, lqr,
                                                                                                   plane_data_set._p.im, start_state, 25, goal_state, A, B, Q, R) 

    plot_animation(images, save_path + "traj.gif")

    plot_traj(plane_data_set._p, pos_optimal, save_path + "optimal_traj.png")

    start_state = np.array([1, 1])
    plot_traj_z(plane_data_set._p, latent_pos_t, embedding_model.model, start_state, goal_state, save_path + "latent_optimal_traj.png")

def main(config: dict):

    print("Running Main")
    print(config)
    if config["run_base_plane_experiment"]:
        run_plane_experiment(load_config("/Users/davidsewell/Projects/Embed2Control/embed2control/configs/base_config.yaml"))

if __name__ == "__main__":
    config_path = "/Users/davidsewell/Projects/Embed2Control/embed2control/configs/main.yaml"
    config = load_config(config_path)
    main(config)