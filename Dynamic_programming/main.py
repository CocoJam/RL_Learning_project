# Declaimer this is code provided through udacity deep reinforment learning course.
# This is for self studying only no other purpose was used
import numpy as np
import copy

import check_test
from frozenlake import FrozenLakeEnv
from plot_utils import plot_values

env = FrozenLakeEnv()
print(env)
# number of current in discrete type state within the environment
stateNum = env.observation_space
actionNum = env.action_space

# the number of state and action
StateNumber = env.nS
ActionNum = env.nA

# A dictionary of given key of state with another dictoryary of action key with list tuple of
# prob of getting reward, next state, reward by taking this action and done boolean
print(env.P)

# Iterative Policy Evaluation
# To iteractively evaluate the state value function of this current Policy
# env as environment, policy is a [state, action] shape array containing the prob
# of each state to select that action.
# gemma is the discount factor of bellman
# theta is the threshold of the loop.

def policy_evaluation(env, policy, gamma=1, theta=1e-8):
    # value of the given policy's state value as V.
    V = np.zeros(env.nS)
    while True:
        # Iteractively to update V untill the differences of update is lower than threshold.
        delta = 0
        for s in range(env.nS):
            Vs = 0
            # getting the arg of action and it's associated probability withint he policy
            for a, action_prob in enumerate(policy[s]):
                # summing all actions assoicated within the policy that is within the same state.
                # under bellman to get overall expected value with at this state, hence state value.
                for prob, next_state, reward, done in env.P[s][a]:
                    # probabilitistic action with chances, reward and discounted future reward.
                    Vs += action_prob * prob * (reward + gamma * V[next_state])
            # detla to trace is there any big changes overall in V
            delta = max(delta, np.abs(V[s]-Vs))
            V[s] = Vs
        if delta < theta:
            break
    return V
# init the policy unified fashion.
random_policy = np.ones([env.nS, env.nA]) / env.nA
V = policy_evaluation(env, random_policy)

plot_values(V)

# Now converting the state value to action values based on state.
# s is the state, which the V is the previous computed State Value
def q_from_v(env, V, s, gamma=1):
    # this evaluate the value of actions based on given state
    q = np.zeros(env.nA)
    for a in range(env.nA):
        # again bellman.
        for prob, next_state, reward, done in env.P[s][a]:
            q[a] += prob * (reward + gamma * V[next_state])
    return q

# Q value computed
Q = np.zeros([env.nS, env.nA])
for s in range(env.nS):
    Q[s] = q_from_v(env, V, s)


def policy_improvement(env, V, gamma=1):
    policy = np.zeros([env.nS, env.nA]) / env.nA
    for s in range(env.nS):
        q = q_from_v(env, V, s, gamma)
        
        # OPTION 1: construct a deterministic policy 
        # policy[s][np.argmax(q)] = 1
        
        # OPTION 2: construct a stochastic policy that puts equal probability on maximizing actions
        # getting the max Arguments the select then by identity mat to give the prob of max probability.
        best_a = np.argwhere(q==np.max(q)).flatten()
        policy[s] = np.sum([np.eye(env.nA)[i] for i in best_a], axis=0)/len(best_a)
        
    return policy

def policy_iteration(env, gamma=1, theta=1e-8):
    policy = np.ones([env.nS, env.nA]) / env.nA
    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V)
        
        # OPTION 1: stop if the policy is unchanged after an improvement step
        if (new_policy == policy).all():
            break
        
        # OPTION 2: stop if the value function estimates for successive policies has converged
        # if np.max(abs(policy_evaluation(env, policy) - policy_evaluation(env, new_policy))) < theta*1e2:
        #    break;
        
        policy = copy.copy(new_policy)
    return policy, V

policy_pi, V_pi = policy_iteration(env)

def truncated_policy_evaluation(env, policy, V, max_it=1, gamma=1):
    num_it=0
    while num_it < max_it:
        for s in range(env.nS):
            v = 0
            q = q_from_v(env, V, s, gamma)
            for a, action_prob in enumerate(policy[s]):
                v += action_prob * q[a]
            V[s] = v
        num_it += 1
    return V

def truncated_policy_iteration(env, max_it=1, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA]) / env.nA
    while True:
        policy = policy_improvement(env, V)
        old_V = copy.copy(V)
        V = truncated_policy_evaluation(env, policy, V, max_it, gamma)
        if max(abs(V-old_V)) < theta:
            break
    return policy, V
policy_tpi, V_tpi = truncated_policy_iteration(env, max_it=2)

def value_iteration(env, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            V[s] = max(q_from_v(env, V, s, gamma))
            delta = max(delta,abs(V[s]-v))
        if delta < theta:
            break
    policy = policy_improvement(env, V, gamma)
    return policy, V