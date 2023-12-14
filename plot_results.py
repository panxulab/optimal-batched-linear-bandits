from bandits import *
import numpy as np
import os

folder_path = "results"
file_list = os.listdir(folder_path)

# filename="end_k_5_d_3_eps_0.1.npz"
for filename in file_list:
    loaded_data = np.load(os.path.join(folder_path, filename))
    results = loaded_data["results"]
    batch_complexity = loaded_data["batch_complexity"]
    results_batch = loaded_data["results_batch"]
    arms = loaded_data["arms"]
    M = loaded_data["M"].item() #number of agents
    theta=loaded_data["theta"]
    horizon=loaded_data["horizon"].item()
    eps=loaded_data["epsilon"].item()

    my_plot = plot_figures(results,batch_complexity,results_batch,arms,M,theta,horizon,eps)

    my_plot.plot_results()

    # my_plot.compute_batch_complexity()

    if M != 3:
        my_plot.plot_results_batch()