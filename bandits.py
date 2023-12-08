import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


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
        