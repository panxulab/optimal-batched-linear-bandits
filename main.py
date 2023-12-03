import argparse
# import time

import numpy as np
# from tqdm import tqdm
# import sys

from algorithms import *
from bandits import *
from arms import *

def parse_args():
    """
    Specifies command line arguments for the program.
    """
    parser = argparse.ArgumentParser(description='bandit regret minimization')

    parser.add_argument('--seed', default=1, type=int,
                        help='Seed for random number generators')
    # default best-arm options
    parser.add_argument('--K', default=3, type=int,
                        help='number of total arms')
    parser.add_argument('--d', default=2, type=int,
                        help='number of context dimension')
    parser.add_argument('--T', default=10000, type=int,
                        help='time horizon')
    parser.add_argument('--num_sim', default=10, type=int,
                        help='number of total simulation')
    parser.add_argument('--verbose', action='store_true', help='end of optimism instance')

    parser.add_argument('--epsilon', default=0.2, type=float,
                        help='end of optimism parameter')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    #instance
    # arms=np.array([[1,0],[0,0],[0,1]])
    # theta=np.array([1,0,0])

    if args.verbose==False:
        #True:default random instance; False: defalut end of optimism instance

        #end of optimism instance
        if args.d==2:
            args.K=3
            arms=np.array([[1,1-args.epsilon,0],[0,2*args.epsilon,1]])
            theta=np.array([1,0])
            # args.K=2
            # arms=np.array([[1,0],[0,1]])
        elif args.d==3:
            args.K=5
            arms=np.array([[1,0,0,1-args.epsilon,1-args.epsilon],[0,1,0,2*args.epsilon,0],[0,0,1,0,2*args.epsilon]])
            theta=np.array([1,0,0])
        else:
            args.K=9;args.d=5
            arms=np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0,],[0,0,0,0,1],[1-args.epsilon,2*args.epsilon,0,0,0],[1-args.epsilon,0,2*args.epsilon,0,0],[1-args.epsilon,0,0,2*args.epsilon,0],[1-args.epsilon,0,0,0,2*args.epsilon]]).T
            theta=np.array([1,0,0,0,0])


    else: 
        theta = np.zeros(args.d)
        theta[0]=1
        
        arms=np.random.uniform(0, 1, (args.d, args.K))
        arms[:,0]=theta
        
    # agent=[E3TC(args.K,args.d,1),SuccessiveElimination(args.K,args.d,1),LinUCB(args.K,args.d,1),End_of_optimism_alg(args.K,args.d,1),IDS(args.K,args.d,1)]
    # agent=[E3TC(args.K,args.d,1),SuccessiveElimination(args.K,args.d,1),LinUCB(args.K,args.d,1),End_of_optimism_alg(args.K,args.d,1)]
    agent=[E3TC(args.K,args.d,1),SuccessiveElimination(args.K,args.d,1),LinUCB(args.K,args.d,1)]
    
    bandits=[GaussianArm(np.dot(theta,arms[:,i]),1) for i in range(args.K)]

    LinearBandit = environment(bandits,arms,agents=agent)

    LinearBandit.run(args.T,args.num_sim)
    LinearBandit.plot_results()
    # LinearBandit.compute_batch_complexity()
    # LinearBandit.plot_results_batch()




if __name__=='__main__':
    main()