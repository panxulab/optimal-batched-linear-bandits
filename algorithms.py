import numpy as np
import cvxpy as cp
from scipy.optimize import minimize

# import sys

class E3TC(object):
    """
    Explore, Estimate, Eliminate Then Commit Algorithm in our paper
    """
    def __init__(self, num_arm, dim_context,reg=1.0, gamma=10,  name='E3TC'):
        self.reg=reg
        self.num_arm = num_arm
        self.dim_context = dim_context
        self.name = name
        self.gamma=gamma
        
        self.clear()
        
    def clear(self):
        
        # DesignInv:=A^{-1}, A=\sum x_ix_i^T
        self.DesignInv = (1 / self.reg) * np.eye(self.dim_context)  
        # Vector=\sum x_iy_i
        self.Vector = np.zeros(self.dim_context)
        #estimator of theta
        self.theta = np.zeros(self.dim_context)
        self.last_cxt = 0
        self.last_reward = 0
    def initialize_distribution(self,K):
        """ Initialize a uniform distribution over K actions """
        return np.ones(K) / K
    def calculate_covariance_matrix(self,context,pi):
        '''calculate the covariance matrix'''
        self.V=np.zeros((context.shape[0], context.shape[0]))
        for i in range(context.shape[1]):
            col = context[:, i].reshape(-1, 1)
            self.V += pi[i] * (col @ col.T)
        tol = 1e-12
        if np.linalg.det(self.V) < tol:
            self.V = self.V + 0.00001 * np.eye(self.dim_context)
        else:
            self.V = self.V
        self.V_inv=np.linalg.inv(self.V)
    

    def D_optimal_design(self,context,num_pull,max_iterations=int(1e5),threshold=1e-4):
        """ Implement the D-optimal design algorithm """
        K = context.shape[1]
        pi = self.initialize_distribution(K)
        objective=-1;max_norm=10

        for iter in range(max_iterations):
            #update distribution
            self.calculate_covariance_matrix(context,pi)
            norms=[context[:,i].T@self.V_inv@context[:,i] for i in range(self.num_arm)]
            arm_with_max_norm_index=np.argmax(norms)
            max_norm=norms[arm_with_max_norm_index]
            step_size=(max_norm/self.dim_context-1)/(max_norm-1)
            if np.abs(objective-max_norm)<threshold:
                break
            #update step
            pi=(1-step_size)*pi
            pi[arm_with_max_norm_index]+=step_size
            objective=max_norm
            
            D_optimal_design_pulling_number=np.ceil(2*pi*objective*num_pull/self.dim_context)
        return D_optimal_design_pulling_number
    
    def pull_arms_D_optimal_design(self,context,bandits,sum_pull,horizon):
        num_pull=self.D_optimal_design(context,sum_pull)
        
        for k in range(self.num_arm):
            self.break_out=False
            num=0
            while num<num_pull[k]:
                if self.t==horizon:
                    self.break_out=True
                    break
                action=k
                reward = bandits[action].draw()
                self.results_for_this_agent[self.t] = bandits[action].mean_return
                self.results_batch[self.t] = self.batch_complexity + 1
                self.receive_reward(action,context[:,action],reward)
                self.update_model()
                num+=1
            if self.break_out:
                break
        return num_pull
    def receive_reward(self, arm, context, reward):
        self.last_cxt = context
        self.last_reward = reward
        # self.results_batch[self.t]=self.batch_complexity+1

    def update_model(self, num_iter=None):
        self.Vector = self.Vector + self.last_reward * self.last_cxt
        omega = np.dot(self.DesignInv, self.last_cxt)
        self.DesignInv = self.DesignInv - np.outer(omega, omega) / (1 + np.dot(omega, self.last_cxt))
        self.theta = np.dot(self.DesignInv, self.Vector)
        self.t += 1

    def track_and_stop_proportion(self,context):
        self.estimate_mean=context.T@self.theta
        self.best_arm=np.argmax(self.estimate_mean)
        best_mean=self.estimate_mean[self.best_arm]
        Delta=best_mean-self.estimate_mean
        K=context.shape[1]

        w = cp.Variable(K)  
        objective = cp.Minimize(w@ Delta)  

        constraints = [w >= 0]  
        
        

        H_w = sum([w[j] * np.outer(context[:,j], context[:,j]) for j in range(K)])+np.eye(self.dim_context)*1e-10
        # print(H_w.shape)
        for i in range(K):
            if i !=self.best_arm: #consider sub_optimal_arms
                x = context[:,i:i+1]
                c=cp.reshape(Delta[i]**2/2,(1, 1))
                constraint=cp.bmat([[H_w, x], [x.T, c ]]) >> 0
                constraints.append(constraint)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS,max_iters=100000,eps=1e-2)
        #,verbose=True
        return w.value

    def pull_arms_track_and_stop(self,context,bandits,w,horizon):
        T=horizon
        alpha=1+self.dim_context*np.log(np.log(T))**4/np.log(T)
        threshold=min((np.log(T))**(1+self.gamma),T/(2*context.shape[1]))
        
        num_pull=np.minimum(w*alpha*np.log(T),threshold)
        # num_pull=w*alpha*np.log(T)
        num_pull=np.ceil(num_pull).astype(int)
        
        for k in range(self.num_arm):
            break_out=False
            for _ in range(num_pull[k]):
                if self.t>=horizon:
                    break_out=True
                    break
                action=k
                reward = bandits[action].draw()
                self.results_for_this_agent[self.t] = bandits[action].mean_return
                self.results_batch[self.t] = self.batch_complexity+1
                self.receive_reward(action,context[:,action],reward)
                self.update_model()
            if break_out:
                break

        return num_pull
        
    def successive_elimination(self,varepsilon,context):
        self.estimate_mean=context.T@self.theta
        best_mean=np.max(self.estimate_mean)
        threshold=best_mean-2*varepsilon
        cols_to_select=np.where(self.estimate_mean>threshold)[0]
        context=context[:,cols_to_select]
        self.num_arm=context.shape[1]
        return context,cols_to_select
        
    def run(self,context,bandits,horizon):
        self.clear()
        '''batch 1 '''
        self.batch_complexity=0
        self.t=0;self.horizon=horizon
        K=context.shape[1]#num of all arms
        self.num_arm=K
        self.results_for_this_agent = np.zeros(horizon)
        self.results_batch = np.zeros(horizon)

        #Exploration
        self.pull_arms_D_optimal_design(context,bandits,np.sqrt(horizon),horizon)
        #Calculate
        w=self.track_and_stop_proportion(context)
        self.batch_complexity+=1
        '''batch 2'''
        # self.clear()
        # num_pull_D=self.pull_arms_D_optimal_design(context,bandits,np.sqrt(horizon),horizon)
        num_pull_T=self.pull_arms_track_and_stop(context,bandits,w,horizon)
        # sum_pull_second_batch=num_pull_D+num_pull_T
        #Calculate stopping statistic Z
        values=[]
        self.estimate_mean=context.T@self.theta
        self.best_arm=np.argmax(self.estimate_mean)
        best_mean=self.estimate_mean[self.best_arm]
        Delta=best_mean-self.estimate_mean
        H = self.DesignInv+np.eye(self.dim_context)*1e-10
        # for k in range(K):
        #     H+=sum_pull_second_batch[k]*np.outer(context[:,k],context[:,k])
        for k in range(K):
            arm=context[:,k]
            
            if k!=self.best_arm:
                value_Z_k=(Delta[k])**2/(2*(arm-context[:,self.best_arm]).T@H@(arm-context[:,self.best_arm]))
                values.append(value_Z_k)
        stopping_Z=np.min(values)
        #Chernoff's stopping rule
        beta=1/2*np.log(self.t**(self.dim_context/2)*horizon)
        if stopping_Z>beta:
            context=context[:,self.best_arm:self.best_arm+1]
            bandits=[bandits[self.best_arm]]
            # print("Batch complexity:3.")
        self.batch_complexity+=1
        '''batch 3'''
        if context.shape[1]>1 and self.t<horizon:
            # self.clear()
            num_pull=(np.log(horizon))**(1+self.gamma)
            self.pull_arms_D_optimal_design(context,bandits,num_pull,horizon)
            varepsilon=np.sqrt(self.dim_context*np.log(K*horizon**2)/num_pull)
            context,cols_to_select=self.successive_elimination(varepsilon,context)
            bandits=[bandits[i] for i in cols_to_select]
            self.batch_complexity+=1
        '''batch 4...'''
        l=1
        while context.shape[1]>1 and self.t<horizon:
            # self.clear()
            num_pull=horizon**(1-1/(2**l))
            self.pull_arms_D_optimal_design(context,bandits,num_pull,horizon)
            varepsilon=np.sqrt(self.dim_context*np.log(K*horizon**2)/num_pull)
            context,cols_to_select=self.successive_elimination(varepsilon,context)
            bandits=[bandits[i] for i in cols_to_select]
            l+=1;self.batch_complexity+=1

        '''Commit'''
        action=0
        while self.t<horizon:
            reward = bandits[action].draw()
            self.results_for_this_agent[self.t] = bandits[action].mean_return
            self.results_batch[self.t] = self.batch_complexity+1
            self.receive_reward(action,context[:,action],reward)
            # self.update_model()
            if self.t == horizon-1:
                self.batch_complexity+=1
            self.t+=1
        return self.results_for_this_agent,self.batch_complexity,self.results_batch

class LinUCB(object):
    def __init__(self, num_arm, dim_context, nu, reg=1.0, C=0.5,name='rs-OFUL'):
        self.nu = nu
        self.reg = reg  #regularizer
        self.num_arm = num_arm
        self.dim_context = dim_context
        self.name = name
        self.C=C
        self.clear()

    def clear(self):
        self.t = 0
        # DesignInv:=A^{-1}, A=\sum x_ix_i^T
        self.DesignInv = (1 / self.reg) * np.eye(self.dim_context)  
        # Vector=\sum x_iy_i
        self.Vector = np.zeros(self.dim_context)
        #estimator of theta
        self.theta = np.zeros(self.dim_context)
        self.last_cxt = 0
        self.last_reward = 0

    def choose_arm(self, context):
        #context=environment.arms is a list
        tol = 1e-12
        if np.linalg.det(self.DesignInv) < tol:
            cov = self.DesignInv + 0.00001 * np.eye(self.dim_context)
        else:
            cov = self.DesignInv
        
        norms = [np.dot(np.dot(context[:,i].T ,cov) , context[:,i]) for i in range(self.num_arm)]
        norms=np.sqrt(norms)

        pred = np.dot(context.T, self.theta) + self.nu * np.array(norms)
        arm_to_pull = np.argmax(pred)
        # arm_to_pull=context[:,arm_to_pull_index]
        return arm_to_pull

    def receive_reward(self, arm, context, reward):
        self.last_cxt = context
        self.last_reward = reward

    def update_model(self, num_iter=None):
        self.Vector = self.Vector + self.last_reward * self.last_cxt
        omega = np.dot(self.DesignInv, self.last_cxt)
        self.DesignInv = self.DesignInv - np.outer(omega, omega) / (1 + np.dot(omega, self.last_cxt))
        
        self.t += 1
        self.nu=np.maximum(np.sqrt(128*self.dim_context*np.log(self.t)),8/3*np.log(self.t))

    def update_model_parameter(self):
        self.theta = np.dot(self.DesignInv, self.Vector)

    def run(self,context,bandits,horizon):
        self.batch_complexity=0

        results_for_this_agent = np.zeros(horizon)
        self.results_batch = np.zeros(horizon)

        tau=self.t
        while self.t<horizon:
            self.update_model_parameter()
            self.batch_complexity+=1
            action = self.choose_arm(context)
            Omega=self.DesignInv
            while np.linalg.det(Omega)<=(1+self.C)*np.linalg.det(self.DesignInv) and self.t<horizon:
                reward = bandits[action].draw()
                results_for_this_agent[self.t] = bandits[action].mean_return
                self.results_batch[self.t] = self.batch_complexity
                self.receive_reward(action,context[:,action],reward)
                self.update_model()
        return results_for_this_agent,self.batch_complexity,self.results_batch

class SuccessiveElimination(object):
    """
    Algorithm in Amin's paper and bandit book
    """
    def __init__(self, num_arm, dim_context,reg=1.0, gamma=1.0,  name='PhaElimD'):
        self.reg=reg
        self.num_arm = num_arm
        self.dim_context = dim_context
        self.name = name
        self.gamma=gamma
        
        self.clear()
        
    def clear(self):
        # DesignInv:=A^{-1}, A=\sum x_ix_i^T
        self.DesignInv = (1 / self.reg) * np.eye(self.dim_context)  
        # Vector=\sum x_iy_i
        self.Vector = np.zeros(self.dim_context)
        #estimator of theta
        self.theta = np.zeros(self.dim_context)
        self.last_cxt = 0
        self.last_reward = 0
    def initialize_distribution(self,K):
        """ Initialize a uniform distribution over K actions """
        return np.ones(K) / K
    def calculate_covariance_matrix(self,context,pi):
        '''calculate the covariance matrix'''
        self.V=np.zeros((context.shape[0], context.shape[0]))
        for i in range(context.shape[1]):
            col = context[:, i].reshape(-1, 1)
            self.V += pi[i] * (col @ col.T)
        tol = 1e-12
        if np.linalg.det(self.V) < tol:
            self.V = self.V + 0.00001 * np.eye(self.dim_context)
        else:
            self.V = self.V
        self.V_inv=np.linalg.inv(self.V)
    

    def D_optimal_design(self,context,num_pull,max_iterations=int(1e10),threshold=1e-4):
        """ Implement the D-optimal design algorithm """
        K = context.shape[1]
        pi = self.initialize_distribution(K)
        objective=-1;max_norm=10

        for iter in range(max_iterations):
            #update distribution
            self.calculate_covariance_matrix(context,pi)
            norms=[context[:,i].T@self.V_inv@context[:,i] for i in range(self.num_arm)]
            arm_with_max_norm_index=np.argmax(norms)
            max_norm=norms[arm_with_max_norm_index]
            step_size=(max_norm/self.dim_context-1)/(max_norm-1)
            if np.abs(objective-max_norm)<threshold:
                break
            #update step
            pi=(1-step_size)*pi
            pi[arm_with_max_norm_index]+=step_size
            objective=max_norm
            
            D_optimal_design_pulling_number=np.ceil(2*pi*objective*num_pull/self.dim_context)
        return D_optimal_design_pulling_number
    
    def pull_arms_D_optimal_design(self,context,bandits,sum_pull,horizon):
        num_pull=self.D_optimal_design(context,sum_pull)
        
        
        for k in range(self.num_arm):
            self.break_out=False
            num=0
            while num<num_pull[k]:
                if self.t==horizon:
                    self.break_out=True
                    break
                action=k
                reward = bandits[action].draw()
                self.results_for_this_agent[self.t] = bandits[action].mean_return
                self.results_batch[self.t] = self.batch_complexity+1
                self.receive_reward(action,context[:,action],reward)
                self.update_model()
                num+=1
            if self.break_out:
                break
            
        return num_pull
    def receive_reward(self, arm, context, reward):
        self.last_cxt = context
        self.last_reward = reward

    def update_model(self, num_iter=None):
        self.Vector = self.Vector + self.last_reward * self.last_cxt
        omega = np.dot(self.DesignInv, self.last_cxt)
        self.DesignInv = self.DesignInv - np.outer(omega, omega) / (1 + np.dot(omega.T, self.last_cxt))
        self.theta = np.dot(self.DesignInv, self.Vector)
        self.t += 1
        
    def successive_elimination(self,varepsilon,context):
        self.estimate_mean=context.T@self.theta
        best_mean=np.max(self.estimate_mean)
        threshold=best_mean-2*varepsilon
        cols_to_select=np.where(self.estimate_mean>threshold)[0]
        context=context[:,cols_to_select]
        self.num_arm=context.shape[1]
        return context,cols_to_select
        
    def run(self,context,bandits,horizon):
        self.clear()
        self.t=0;self.batch_complexity=0
        l=1
        K=context.shape[1]
        self.results_for_this_agent = np.zeros(horizon)
        self.results_batch = np.zeros(horizon)
        while context.shape[1]>1 and self.t<horizon:
            # self.clear()
            num_pull=horizon**(1-1/(2**l))
            # num_pull=np.sqrt(horizon)*2**l
            # num_pull=np.log(horizon)**l
            self.pull_arms_D_optimal_design(context,bandits,num_pull,horizon)
            varepsilon=np.sqrt(self.dim_context*np.log(K*horizon**2)/num_pull)
            context,cols_to_select=self.successive_elimination(varepsilon,context)
            bandits=[bandits[i] for i in cols_to_select]
            l+=1;self.batch_complexity+=1
        '''Commit'''
        action=0
        if self.t<horizon:
            self.batch_complexity+=1
        while self.t<horizon:
            reward = bandits[action].draw()
            self.results_for_this_agent[self.t] = bandits[action].mean_return
            self.results_batch[self.t] = self.batch_complexity
            self.receive_reward(action,context[:,action],reward)
            self.update_model()
        return self.results_for_this_agent,self.batch_complexity,self.results_batch

class End_of_optimism_alg(object):
    """
    Optimal algorithm in end of optimism
    """
    def __init__(self,  num_arm, dim_context,reg=1.0, gamma=1.0, c=1, name='EndOA'):
        self.reg=reg
        self.num_arm = num_arm
        self.dim_context = dim_context
        self.name = name
        self.gamma=gamma
        self.c=c
        self.clear()

    def clear(self):
        # DesignInv:=A^{-1}, A=\sum x_ix_i^T
        self.DesignInv = (1 / self.reg) * np.eye(self.dim_context)  
        # Vector=\sum x_iy_i
        self.Vector = np.zeros(self.dim_context)
        #estimator of theta
        self.theta = np.zeros(self.dim_context)
        self.last_cxt = 0
        self.last_reward = 0

    def track_and_stop_proportion(self,context):
        self.estimate_mean=context.T@self.theta
        self.best_arm=np.argmax(self.estimate_mean)
        best_mean=self.estimate_mean[self.best_arm]
        Delta=best_mean-self.estimate_mean
        K=context.shape[1]

        w = cp.Variable(K)  
        objective = cp.Minimize(w@ Delta)  

        constraints = [w >= 0]  
        
        H_w = sum([w[j] * np.outer(context[:,j], context[:,j]) for j in range(K)])+np.eye(self.dim_context)*1e-10
        # print(H_w.shape)
        for i in range(K):
            if i !=self.best_arm: #consider sub_optimal_arms
                x = context[:,i:i+1]
                c=cp.reshape(Delta[i]**2/2,(1, 1))
                constraint=cp.bmat([[H_w, x], [x.T, c ]]) >> 0
                constraints.append(constraint)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS,max_iters=100000,eps=1e-2)
        #,verbose=True
        return w.value
    
    def choose_arm(self, context):
        #context=environment.arms is a list
        tol = 1e-12
        if np.linalg.det(self.DesignInv) < tol:
            cov = self.DesignInv + 0.00001 * np.eye(self.dim_context)
        else:
            cov = self.DesignInv
        
        norms = [np.dot(np.dot(context[:,i].T ,cov) , context[:,i]) for i in range(self.num_arm)]
        norms=np.sqrt(norms)

        pred = np.dot(context.T, self.theta) + self.nu * np.array(norms)
        arm_to_pull = np.argmax(pred)
        # arm_to_pull=context[:,arm_to_pull_index]
        return arm_to_pull

    def receive_reward(self, arm, context, reward):
        self.last_cxt = context
        self.last_reward = reward

    def update_model(self, num_iter=None):#UCB
        self.Vector = self.Vector + self.last_reward * self.last_cxt
        omega = np.dot(self.DesignInv, self.last_cxt)
        self.DesignInv = self.DesignInv - np.outer(omega, omega) / (1 + np.dot(omega, self.last_cxt))
        self.theta = np.dot(self.DesignInv, self.Vector)
        self.t += 1
        self.nu=np.maximum(np.sqrt(128*self.dim_context*np.log(self.t)),8/3*np.log(self.t))

    def update_epsilon(self,context,gn,K):
        eps=[context[:,k].T@self.DesignInv@context[:,k]*np.sqrt(gn) for k in range(K)]
        self.eps_n=np.max(eps)

    def run(self,context,bandits,horizon):
        results_for_this_agent = np.zeros(horizon)
        self.t=0;K=context.shape[1];d=self.dim_context;T=horizon
        '''Warmup phase'''
        for k in range(K):
            action=k
            n=0
            while n< np.sqrt(np.log(T)):
                reward = bandits[action].draw()
                results_for_this_agent[self.t] = bandits[action].mean_return
                self.receive_reward(action,context[:,action],reward)
                self.update_model()
                n+=1
        w=self.track_and_stop_proportion(context)
        w[self.best_arm]=T
        fn=2*(1+1/np.log(T))*np.log(T)+self.c*d*np.log(d*np.log(T))
        
        num_pull=w*fn
        '''Success phase'''
        record_pull=np.zeros(K)
        gn=2*(1+1/np.log(T))*np.log(np.log(T))+self.c*d*np.log(d*np.log(T))
        self.update_epsilon(context,gn,K)
        gap_mu=0
        record_pull=np.zeros(K)
        while self.t<horizon:
            break_condition=False
            for k in range(K):
                if self.t>=horizon:
                    break
                if record_pull[k]<=num_pull[k]:
                    action=k
                    record_pull[k]+=1
                    reward = bandits[action].draw()
                    results_for_this_agent[self.t] = bandits[action].mean_return
                    self.receive_reward(action,context[:,action],reward)
                    theta_old=self.theta
                    self.update_model()
                    self.update_epsilon(context,gn,K)
                    gap_mu=np.max(np.abs(self.theta-theta_old))
                    if gap_mu>2*self.eps_n:
                        break_condition=True
                        break
            if break_condition:
                break
        '''Recovery phase'''
        while self.t<horizon:#LinUCB
            action = self.choose_arm(context)
            reward = bandits[action].draw()
            results_for_this_agent[self.t] = bandits[action].mean_return
            self.receive_reward(action,context[:,action],reward)
            self.update_model()
        return results_for_this_agent,horizon,np.arange(1,horizon+1)

class IDS():
    """
    Optimism in the Face of Uncertainty for Linear bandits Algorithm
    """
    def __init__(self,  num_arm, dim_context,reg=1.0, name='IDS'):
        self.reg=reg
        self.num_arm = num_arm
        self.dim_context = dim_context
        self.name = name
        self.clear()

    def clear(self):
        self.V=self.reg * np.eye(self.dim_context)  
        self.best_arm=0
        self.DesignInv = (1 / self.reg) * np.eye(self.dim_context)  
        self.Vector = np.zeros(self.dim_context)
        self.theta = np.zeros(self.dim_context)
        self.last_cxt = 0
        self.last_reward = 0
        self.t = 0
        self.s = 0
        self.m_max=-np.inf
        

    def receive_reward(self, arm, context, reward):
        self.last_cxt = context
        self.last_reward = reward

    def update_model(self,context,num_iter=None):
        self.Vector = self.Vector + self.last_reward * self.last_cxt
        omega = np.dot(self.DesignInv, self.last_cxt)
        self.V+=np.outer(self.last_cxt,self.last_cxt)
        self.DesignInv = self.DesignInv - np.outer(omega, omega) / (1 + np.dot(omega, self.last_cxt))
        self.theta = np.dot(self.DesignInv, self.Vector)
        self.estimate_mean=context.T@self.theta
        self.best_arm=np.argmax(self.estimate_mean)
        self.best_arm_context=context[:,self.best_arm]
        self.t += 1

    def beta(self,deltaInv):
        deltaInv=max(1,deltaInv)
        beta=(1+np.sqrt(2*np.log(deltaInv)+np.log(np.linalg.det(self.V))))**2
        return beta

    def calculate_nu(self,z):
        # nu_initial = np.zeros(self.dim_context)

        # def objective(nu):
        #     return np.dot(np.dot((nu - self.theta).T, self.V), (nu - self.theta))
        
        # def linear_constraint(nu):
        #     return np.dot(nu,z-self.best_arm_context)
        
        # con = {'type': 'ineq', 'fun': linear_constraint}

        # sol = minimize(objective, nu_initial, constraints=con)

        # return sol.fun, sol.x
        nu = self.theta-np.dot(self.theta,self.best_arm_context-z)/np.dot(np.dot((self.best_arm_context - z).T, self.DesignInv), (self.best_arm_context - z))*self.DesignInv@(self.best_arm_context-z)
        m=np.dot(np.dot((nu - self.theta).T, self.V), (nu - self.theta))
        return m,nu

    
    def calculate_info_fun(self,q,context,nu_hat):
        K=context.shape[1]
        info=np.zeros(K)
        for k in range(K):
            #calculate information gain for each arm
            x=context[:,k]
            for z in range(K):
                if z != self.best_arm:
                    norm=x.T@self.DesignInv@x
                    value=np.abs(np.dot((nu_hat[z]-self.theta),x))+np.sqrt(self.beta((self.s+1)**2)*(norm))
                    info[k]+=q[z]*(value)**2
        return info
    
    def IDS_distribution(self,Delta,Info):

        def objective(mu):
            mu_full = np.zeros(K)
            mu_full[fixed_index], mu_full[k] = mu[0], mu[1]
            return (np.dot(Delta, mu_full) ** 2) / np.dot(Info, mu_full)

        def constraint(mu):
            return np.sum(mu) - 1
        
        K = len(Delta)  
        fixed_index = self.best_arm
        best_value = np.inf
        best_solution = None

        for k in range(K):
            if k == fixed_index:
                continue

            
            mu_initial = [1/2, 1/2]

            cons = ({'type': 'eq', 'fun': constraint})

            bnds = tuple((0, 1) for _ in range(2))

            sol = minimize(objective, mu_initial, method='SLSQP', bounds=bnds, constraints=cons)

            if sol.fun < best_value:
                best_value = sol.fun 
                best_solution = np.zeros(K)
                best_solution[fixed_index], best_solution[k] = sol.x[0], sol.x[1]

        return best_solution

    def run(self,context,bandits,horizon):
        results_for_this_agent = np.zeros(horizon)
        K = context.shape[1]; d = self.dim_context
        self.best_arm_context=context[:,self.best_arm]
        self.theta=self.best_arm_context
        exploit=False
        while self.t<horizon:
            if exploit == False:
                upper_reward=max([np.dot(self.theta,context[:,k])+np.sqrt(context[:,k].T@self.DesignInv@context[:,k]*self.beta((self.s+1)**2)) for k in range(K)],default="empty list")
                Delta=[upper_reward-np.dot(self.theta,context[:,k]) for k in range(K)]
                Delta_hat=np.array(Delta)

                nu_hat = np.zeros((K,d)); m = np.inf
                m_array=np.zeros(K)
                for k in range(K):
                    if k!=self.best_arm:
                        m_array[k],nu_hat[k]=self.calculate_nu(context[:,k])
                        if m_array[k] < m:
                            m = m_array[k]
                m=m/2
                if m>self.m_max:
                    self.m_max = m
            threshold=1/2*self.beta((self.t+1)*np.log(self.t+1))
            if m>=threshold:
                exploit=True
                action=self.best_arm
                reward = bandits[action].draw()
                results_for_this_agent[self.t] = bandits[action].mean_return
                self.t+=1
            else:
                exploit=False
                self.eta = self.m_max**(-1/2)*np.log(K)
                q= np.zeros(K)
                for k in range(K):
                    if k != self.best_arm:
                        q[k]=np.exp(-self.eta*m_array[k])
                q=q/(np.sum(q))
                Info=self.calculate_info_fun(q,context,nu_hat)
                

                # mu=self.IDS_distribution(Delta_hat,Info)

                # indices = np.arange(len(mu))
                # sample = np.random.choice(indices, p=mu)
                Info_ratio=[Delta_hat[k]**2/Info[k] for k in range(K)]
                sample=np.argmin(Info_ratio)

                action = sample

                reward = bandits[action].draw()
                results_for_this_agent[self.t] = bandits[action].mean_return
                self.receive_reward(action,context[:,action],reward)
                self.update_model(context)
                self.s+=1

        return results_for_this_agent,horizon,np.arange(1,horizon+1)