import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class environment(object):#arms is a matrix
    def __init__(self, bandits,arms,agents):
        self.bandits = bandits
        
        self.agents = agents
        self.results = None
        self.K = arms.shape[1]
        self.M = len(self.agents) #list of agents
        self.arms=arms

    def reset(self):
        for i in range(self.M):
            self.agents[i].clear() #initialization

    def run(self, horizon=10000, experiments=1):
        results = np.zeros((self.M, experiments, horizon))
        batch_complexity=np.zeros((self.M,experiments))
        for m in tqdm(range(self.M)):
            agent = self.agents[m]
            for i in tqdm(range(experiments)):
                self.reset()
                #experiment for one agent and just one time
                results[m][i],batch_complexity[m][i]=agent.run(self.arms,self.bandits,horizon)

        self.results = results;self.batch_complexity=batch_complexity
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

    def plot_result(self, result, ax,name):
        horizon = result.shape[1]
        top_mean = self.bandits[0].mean_return
        for i in range(1, self.K):
            if self.bandits[i].mean_return > top_mean:
                top_mean = self.bandits[i].mean_return
        best_case_reward = top_mean * np.arange(1, horizon+1)
        cumulated_reward = np.cumsum(result, axis=1)
        regret = best_case_reward - cumulated_reward[:,:horizon]

        y = np.mean(regret, axis=0)
        x = np.arange(len(y))
        std = np.std(regret, axis=0)
        #print(len(std))

        y_up_err = y + std
        y_low_err = y - std
        ax.plot(x, y,label=name)
        ax.fill_between(x, y_low_err, y_up_err, alpha=0.3)
        #plt.show()
        return np.max(regret)

    def plot_results(self):
        if self.results is None:
            print("No results yet.")
            return -1
        fig, ax = plt.subplots()
        max_regret=0
        for m in range(self.M):
            result = self.results[m]
            R=self.plot_result(result, ax,self.agents[m].name)
            if R>max_regret:
                max_regret=R

        plt.ylim(-100, 1.4*max_regret)
        plt.legend()
        
        plt.xlabel("Time step")
        plt.ylabel("Regret")
        plt.savefig('d:\\research\\intern\\papers\\Optimal-Batched-Linear-Bandits\\code-Batched-Linear-Bandits\\image\\random_k_50_d_20.pdf', format='pdf')
        plt.show()
        
