import tensorflow as tf
import numpy as np
import gym
from REINFORCE import REINFORCENet

env = gym.make('CartPole-v0')
env = env.unwrapped

env.seed(1)

state_space_n = 4
action_space_n = env.action_space.n

max_episodes = 10000
learning_rate = 0.01
gamma = 0.95

allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
episode = 0
episode_states, episode_actions, episode_rewards = [], [], []

reNet = REINFORCENet(state_space_n, action_space_n)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(max_episodes):
        ep_sum_rewards = 0
        state = env.reset()
        while True:
            action = reNet.generateAction(sess, state)
            new_state, reward, done, info = env.step(action)
            action_record = np.zeros(action_space_n)
            action_record[action] =1
            episode_actions.append(action_record)
            episode_states.append(state)
            episode_rewards.append(reward)
            if done:
                ep_sum_rewards = sum(episode_rewards)
                allRewards.append(ep_sum_rewards)
                total_rewards = np.sum(allRewards)
                mean_rewards = np.divide(total_rewards, i+1)
                max_Current_rewards = np.amax(allRewards)
                reNet.step(sess,episode_states, episode_actions, episode_rewards, mean_rewards, i, gamma)
                print("==========================================")
                print("Episode: ", i)
                print("Reward: ", ep_sum_rewards)
                print("Mean Reward", mean_rewards)
                print("Max reward so far: ", max_Current_rewards)
                break
            state= new_state
        episode_states, episode_actions, episode_rewards = [], [], []
        

