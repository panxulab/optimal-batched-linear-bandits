import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
from arms import *

from tqdm import tqdm

class environment(object):#arms is a matrix
    def __init__(self, bandits,arms,agents,theta,eps=0.01):
        self.bandits = bandits
        self.theta=theta
        self.agents = agents
        self.results = None
        self.K = arms.shape[1]
        self.d=arms.shape[0]
        self.eps=eps
        self.M = len(self.agents) #list of agents
        self.arms=arms

    def reset(self):
        for i in range(self.M):
            self.agents[i].clear() #initialization

    def run(self, horizon, experiments=1):
        self.horizon=horizon
        results = np.zeros((self.M, experiments, horizon))
        results_batch = np.zeros((self.M, experiments, horizon))
        batch_complexity=np.zeros((self.M,experiments))
        for m in tqdm(range(self.M)):
            agent = self.agents[m]
            for i in tqdm(range(experiments)):
                self.reset()
                #experiment for one agent and just one time
                results[m][i],batch_complexity[m][i],results_batch[m][i]=agent.run(self.arms,self.bandits,horizon)

        self.results = results;self.batch_complexity=batch_complexity;self.results_batch=results_batch
        file_suffix = "npz"  
        if self.eps<0:# random
            file_name = f"results\\random_k_{self.K}_d_{self.d}.{file_suffix}"
        elif self.M == 3: #research on eps
            file_name = f"results\\research_eps_{self.eps}.{file_suffix}"
        else:
            file_name = f"results\\end_k_{self.K}_d_{self.d}_eps_{self.eps}.{file_suffix}"
        np.savez(file_name, results=self.results,batch_complexity=self.batch_complexity,results_batch=self.results_batch,arms=self.arms,M=self.M,theta=self.theta,horizon=horizon,epsilon=self.eps)

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
            print("%s: mean: %f, std: %f" % (self.agents[m].name, mean[m], std[m]))

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
            self.plot_result_batch(result, ax,self.agents[m].name)
        
        plt.ylim(0, self.horizon)
        plt.legend(loc='upper right')
        
        plt.xlabel("Time step")
        plt.ylabel("Batch")
        plt.savefig('image\\random_k_3_d_2_batch.pdf',dpi=200, format='pdf',bbox_inches='tight')
        # plt.show()

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
            self.plot_result(result, ax,self.agents[m].name)

        plt.ylim(-100, 1200)
        plt.legend(loc='upper right')
        
        plt.xlabel("Time step",labelpad=0)
        plt.ylabel("Regret", labelpad=0)
        
        plt.subplots_adjust(left=0.095, bottom=0.08, right=1, top=1)

        # plt.subplots_adjust( left=0.1,bottom=0.1,right=1, top=1)
        plt.savefig('image\\research_eps_0.25.pdf',dpi=200, format='pdf',bbox_inches='tight')
        
        # plt.show()

class plot_figures(object):
    '''plot figures after collecting data'''
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
        self.names = ['$E^4$',"PhaElimD","rs-OFUL","EndOA","IDS"]
        self.colors = ['r','g','b','orange','m']
        self.linestyles=['-','-.',':','-','--']
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

    def plot_result_batch(self, result, ax,name,color,linestyle,sample_interval=30):
        horizon = result.shape[1]


        batch=result[:, ::sample_interval]

        y = np.mean(batch, axis=0)
        x = np.arange(len(y))*sample_interval
        std = np.std(batch, axis=0)
        #print(len(std))

        y_up_err = y + std
        y_low_err = y - std
        ax.plot(x, y,label=name,color=color,linestyle=linestyle,alpha=0.8,linewidth=4)
        ax.fill_between(x, y_low_err, y_up_err,color=color, alpha=0.3)
        ax.set_yscale('log')

    def plot_results_batch(self):
        matplotlib.rcParams.update({'font.size': 20})
        figsize=(8, 6)
        if self.results is None:
            print("No results yet.")
            return -1
        fig, ax = plt.subplots(figsize=figsize)
        
        for m in range(self.M):
            result = self.results_batch[m]
            # [::sample_interval, ::sample_interval]
            
            self.plot_result_batch(result, ax,self.names[m],self.colors[m],self.linestyles[m])
        
        plt.ylim(1, self.horizon)
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(handles.pop(0))
        labels.append(labels.pop(0))
        plt.legend(handles, labels,loc='upper left')
        
        
        label_size=25
        if self.M == 3:
            label_size = 30
        else:
            label_size = 35
        plt.xlabel("Time step",fontsize=label_size)
        plt.ylabel("Batch",fontsize=label_size)
        if self.M == 4: #random
            filename = f'image\\random_k_{self.K}_d_{self.d}_batch.pdf'
        elif self.M == 5:# end of opt
            filename = f'image\\end_k_{self.K}_d_{self.d}_eps_{self.eps}_batch.pdf'
        else: #research on eps, and error
            print("no need to plot batches")
            exit()
        plt.savefig(filename,dpi=200, format='pdf',bbox_inches='tight')
        
        plt.close(fig)

    def plot_result(self, result, ax,name,color,linestyle,sample_interval = 10):
        if self.horizon>30000:
            sample_interval=sample_interval*4
        if self.horizon>80000:
            sample_interval=sample_interval*4
        if self.d > 10:
            sample_interval = sample_interval*4
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
        ax.plot(x, y,color=color,linestyle=linestyle,label=name,linewidth=4)
        ax.fill_between(x, y_low_err, y_up_err,color=color, alpha=0.15)
        
        

    def plot_results(self):
        matplotlib.rcParams.update({'font.size': 20})
        figsize = (8, 6)
        mpl.rcParams['path.simplify_threshold'] = 0.999
        if self.results is None:
            print("No results yet.")
            return -1
        fig, ax = plt.subplots(figsize=figsize)
        max_regret = 0
        sample_interval = 10  # to make data size 1/10
        for m in range(self.M):
            result = self.results[m]
            self.plot_result(result, ax,self.names[m],self.colors[m],self.linestyles[m])
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
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(handles.pop(0))
        labels.append(labels.pop(0))
        plt.legend(handles, labels,loc='upper left')

        label_size=25
        if self.M == 3:
            label_size = 30
        else:
            label_size = 35

        plt.xlabel("Time step",labelpad=0,fontsize=label_size)
        plt.ylabel("Regret", labelpad=0,fontsize=label_size)
        
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
        # plt.show()
        plt.close(fig)

    def plot_each(self):
        '''plot results in each independent experiment'''
        '''sanity check for high variance algorithm'''
        # matplotlib.rcParams.update({'font.size': 20})
        
        # mpl.rcParams['path.simplify_threshold'] = 0.999
        
        # fig, ax = plt.subplots(figsize=figsize)
        max_regret=0
        sample_interval = 10  # to make data size 1/10
        
        result = self.results[3]
        
        horizon = result.shape[1]
        top_mean = self.bandits[0].mean_return
        for i in range(1, self.K):
            if self.bandits[i].mean_return > top_mean:
                top_mean = self.bandits[i].mean_return
        best_case_reward = top_mean * np.arange(1, horizon+1)
        cumulated_reward = np.cumsum(result, axis=1)
        regret = best_case_reward - cumulated_reward[:,:horizon]

        plt.figure(figsize=(15, 100))

        for i in range(10):
            plt.subplot(10, 1, i + 1)
            plt.scatter(range(10000), regret[i, :], s=1)
            plt.title(f"Row {i+1}")
            plt.xlabel("T")
            plt.ylabel("Value")
        # plt.tight_layout()
        plt.show()

        
       
        