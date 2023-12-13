import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from arms import *

class plot_figures(object):
    def __init__(self,results,batch_complexity,results_batch,arms,M,theta,horizon,eps):
        self.theta=theta
        self.results = results
        self.batch_complexity = batch_complexity
        self.results_batch = results_batch
        self.horizon=horizon
        self.K = arms.shape[1]
        self.d=arms.shape[0]
        self.eps=eps
        self.M = M #list of agents
        self.arms=arms
        self.epsilon=eps

        self.names=["E4","PhaElimD","rs-OFUL"]
        if self.M == 4:
            self.names=["E4","PhaElimD","rs-OFUL","EndOA"]
        if self.M == 5:
            self.names = ["E4","PhaElimD","rs-OFUL","EndOA","IDS"]

        self.bandits = [GaussianArm(np.dot(self.theta,self.arms[:,i]),1) for i in range(self.K)]
    def compute_batch_complexity(self):
        if self.batch_complexity is None:
            print("No results of batch complexity yet.")
            return -1
        mean=np.zeros(self.M);std=np.zeros(self.M)
        for m in range(self.M):
            batch_complexity_m=self.batch_complexity[m]
            mean[m]=np.mean(batch_complexity_m)
            std[m]=np.std(batch_complexity_m)
        print("Batch complexity:")
        for m in range(self.M):
            print("%s: mean: %f, std: %f" % (self.names[m], mean[m], std[m]))

    def plot_result_batch(self, result, ax,name,sample_interval=30):
        horizon = result.shape[1]


        batch=result[:, ::sample_interval]

        y = np.mean(batch, axis=0)
        x = np.arange(len(y))*sample_interval
        std = np.std(batch, axis=0)
        #print(len(std))

        y_up_err = y + std
        y_low_err = y - std
        ax.plot(x, y,label=name)
        ax.fill_between(x, y_low_err, y_up_err, alpha=0.3)
        ax.set_yscale('log')

    def plot_results_batch(self):
        if self.results is None:
            print("No results yet.")
            return -1
        fig, ax = plt.subplots()
        
        for m in range(self.M):
            result = self.results_batch[m]
            # [::sample_interval, ::sample_interval]
            self.plot_result_batch(result, ax,self.names[m])
        
        plt.ylim(0, self.horizon)
        plt.legend(loc='upper right')
        
        plt.xlabel("Time step")
        plt.ylabel("Batch")
        if self.M == 4: #random
            filename = f'image\\random_k_{self.K}_d_{self.d}_batch.pdf'
        elif self.M == 5:# end of opt
            filename = f'image\\end_k_{self.K}_d_{self.d}_eps_{self.eps}_batch.pdf'
        else: #research on eps, and error
            print("no need to plot batches")
            exit()
        plt.savefig('filename',dpi=200, format='pdf',bbox_inches='tight')
        plt.show()

    def plot_result(self, result, ax,name,sample_interval = 10):
        horizon = result.shape[1]
        top_mean = self.bandits[0].mean_return
        for i in range(1, self.K):
            if self.bandits[i].mean_return > top_mean:
                top_mean = self.bandits[i].mean_return
        best_case_reward = top_mean * np.arange(1, horizon+1)
        cumulated_reward = np.cumsum(result, axis=1)
        regret = best_case_reward - cumulated_reward[:,:horizon]
        
        regret=regret[:, ::sample_interval]
        y = np.mean(regret, axis=0)
        x = np.arange(len(y))*sample_interval
        std = np.std(regret, axis=0)
        #print(len(std))

        y_up_err = y + std
        y_low_err = y - std
        ax.plot(x, y,label=name)
        ax.fill_between(x, y_low_err, y_up_err, alpha=0.15)
        #plt.show()
        return np.max(regret)

    def plot_results(self):
        mpl.rcParams['path.simplify_threshold'] = 0.999
        if self.results is None:
            print("No results yet.")
            return -1
        fig, ax = plt.subplots()
        max_regret=0
        sample_interval = 10  # to make data size 1/10
        for m in range(self.M):
            result = self.results[m]
            # [::sample_interval, ::sample_interval]
            self.plot_result(result, ax,self.names[m])
        d=self.d
        if self.M == 5: #end of opt
            if d ==2 :
                y_max = 1200
            elif d == 3:
                y_max = 8000
            else: #d=5
                y_max = 20000
        elif self.M == 4: #random
            if d == 2:
                y_max = 3000
            elif d == 3:
                y_max = 8000
            else:
                y_max = 18000
        else: #research on epsilon
            y_max=1200



        plt.ylim(-100, y_max)
        plt.legend(loc='upper right')
        
        plt.xlabel("Time step",labelpad=0)
        plt.ylabel("Regret", labelpad=0)
        
        plt.subplots_adjust(left=0.095, bottom=0.08, right=1, top=1)

        # plt.subplots_adjust( left=0.1,bottom=0.1,right=1, top=1)
        if self.M == 4:
            filename = f'image\\random_k_{self.K}_d_{self.d}.pdf'
        elif self.M == 5:
            filename = f'image\\end_k_{self.K}_d_{self.d}_eps_{self.eps}.pdf'
        else:
            if self.M != 3:
                print("error in number of agents")
                exit(0)
            filename = f'image\\research_eps_{self.eps}.pdf'
        plt.savefig(filename,dpi=200, format='pdf',bbox_inches='tight')
        
        plt.show()



