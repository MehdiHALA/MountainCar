import math
import numpy as np
import pandas as pd
import gym



class MDP:
    def __init__(self):
        self.states = []
        self.actions = []
        self.T = {} #  "{(s,a,s') : probability of the transition(s,a,s')}"
        self.R = {} #  "{(s,a,s') : reward of the transition(s,a,s')}"


#Global variables
min_speed = -0.07
max_speed = 0.07
min_position = -1.2
max_position=0.6


force = 0.001
gravity = 0.0025

gamma = 0.9

# Environnement Discretization
N = 10                                               # Number of Intervals
pos_c = pd.cut([min_position, max_position],bins=N)  # Pos Discretization to N intervals
v_c = pd.cut([min_speed, max_speed],bins=N*2)        # Vel Discretization to 2*N intervals


#####################################################################

def fill_states(n=N):
    '''
    Function to fill the states
    Args:
        n: number of intervals (n for pos and 2*n for vel)
    Return:
        list of possible states
    '''
    return [(a,b) for a in range(0,n) for b in range(0,n*2)]


def get_interval(x,L,n=N):
    '''
    Returns the index of the interval that contains x ( goes alongside pandas.cut )
    Args:
        x: Number to which we're trying to find the interval
        L: Intervals list
        n: number of intervals
    Return:
        Index of the corresponding interval
    '''
    for i in range(0,n):
        if x in L[i]:
            return i


def proba(s,a,s2):
    '''
    Calculates probability of going from s to s2 doing the action a
    Args:
        s: first state
        a: action to do
        s2: destination state
    Return:
        probability of going from s to s2 doing the action a
    '''
    if s2[1] > s[1] and a == 2:
        return 1/2*N
    elif s2[1] < s[1] and a == 0:
        return 1/2*N
    elif s2[1] == s[1] and a == 1:
        return 1
    return 0


def reward(s,a,s2):
    '''
    Calculates the reward of going from s to s2 doing the action a
    Args:
        s: first state
        a: action to do
        s2: destination state
    Return:
        reward of going from s to s2 doing the action a
    '''
    if MountainCar.T[(s,a,s2)] != 0 and s2[0] == N-1:
        return 0
    return -1


def value_iteration(M, gamma, B):
        '''
        Args :
            M : MDP
            gamma : deprecation factor
            B : number of iterations
        Return :
            Optimal value function as "{state: V^*(state)}"
        '''
        statesNum = len(M.states)
        actionsNum = len(M.actions)
        v = [[0 for i in range(statesNum)] for i in range(B)]
        actionsValues = []
        for k in range(B):
            for s in range(statesNum):
                for a in range(actionsNum):
                    value = 0
                    for sp in range(statesNum):
                        value += M.T[(M.states[s], M.actions[a], M.states[sp])] * (M.R[(M.states[s], M.actions[a], M.states[sp])] + gamma * v[k-1][sp])
                    actionsValues.append(value)
                v[k][s] = max(actionsValues)
                actionsValues = []
        pol = {}
        for i in range(statesNum):
            pol[M.states[i]] = v[-1][i]
        return pol

def optimal_policy(M, V, gamma):
        '''
        Args :
            M : MDP
            V : Optimal value function
            gamma : deprecation factor
        Return :
            Optimal politique as "{state: action}"
        '''
        Q = {}
        q = {}
        for state in M.states:
            for action in M.actions:
                q[action] = 0
                for s in M.states:
                    q[action] += M.T[(state, action, s)] * (M.R[(state, action, s)] + gamma * V[s])
            Q[state] = q
            q = {}
        for key in Q:
            Q[key] = max(Q[key], key=Q[key].get)
        return Q


############################################################
#############       Mountain Car MDP    ###################
############################################################
MountainCar = MDP()
MountainCar.states = fill_states(N)
MountainCar.actions = [0,1,2]
# 0 : <
# 1 : -
# 2 : >
print("MDP Created")

print("Filling T...")
MountainCar.T = {(s,a,s2): proba(s,a,s2) for s in MountainCar.states for a in MountainCar.actions for s2 in MountainCar.states}
print("Filling R...")
MountainCar.R = {(s,a,s2): reward(s,a,s2) for s in MountainCar.states for a in MountainCar.actions for s2 in MountainCar.states}



print("Calculating the optimal policy")
V= value_iteration(MountainCar, gamma, 100)
pi = optimal_policy(MountainCar, V, gamma)


print("Ready, set, go")
env = gym.make('MountainCar-v0')
state = env.reset()
done = 0
for step in range(1, 201):
    p= get_interval(state[0], pos_c.categories)
    v= get_interval(state[1], v_c.categories,N*2)
    state, reward, done, info = env.step(pi[(p,v)])
    env.render()
    if done == 1:
        if (step == 200):
            print("Come back next year !")
        else:
            print("Finish Line : Car reached its goal after",step,"steps !")
        break
env.close()
