from plot_figures import *
import numpy as np

loaded_data = np.load("results\\end_k_3_d_2_eps_0.01.npz")

loaded_data = np.load("results\\research_eps_0.01.npz")
loaded_data = np.load("results\\random_k_3_d_2.npz")

results = loaded_data["results"]
batch_complexity = loaded_data["batch_complexity"]
results_batch = loaded_data["results_batch"]
arms = loaded_data["arms"]
M = loaded_data["M"] #number of agents
theta=loaded_data["theta"]
horizon=loaded_data["horizon"]
eps=loaded_data["epsilon"]

my_plot=plot_figures(results,batch_complexity,results_batch,arms,M,theta,horizon,eps)

my_plot.plot_results()

# my_plot.compute_batch_complexity()

if M!= 3:
    my_plot.plot_results_batch()