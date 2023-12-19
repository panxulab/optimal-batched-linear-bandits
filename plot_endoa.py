from bandits import *
import numpy as np
import os

folder_path = "results"
filename="end_k_3_d_2_eps_0.01.npz"
loaded_data = np.load(os.path.join(folder_path, filename))
results = loaded_data["results"]
batch_complexity = loaded_data["batch_complexity"]
results_batch = loaded_data["results_batch"]
arms = loaded_data["arms"]
M = loaded_data["M"].item() #number of agents

theta=loaded_data["theta"]
horizon=loaded_data["horizon"].item()
eps=loaded_data["epsilon"].item()

my_plot =plot_figures(results,batch_complexity,results_batch,arms,M,theta,horizon,eps)

my_plot.plot_each()

    