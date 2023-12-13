from plot_figures import *
import numpy as np
import os


folder_path = "results"


file_list = os.listdir(folder_path)


for filename in file_list:
    loaded_data = np.load(os.path.join(folder_path, filename))
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