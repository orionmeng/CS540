import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =  20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

if __name__ == "__main__":
    env = gym.envs.make("FrozenLake-v1")
    env.reset(seed=1)

    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()[0]

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # TODO: Replace the following with Q-Learning

        while (not done):
            if random.random() < EPSILON:
                action = env.action_space.sample() # currently only performs a random action.
            else:
                action = max(range(env.action_space.n), key=lambda x: Q_table[(obs, x)])

            next_obs,reward,terminated,truncated,_ = env.step(action)
            done = terminated or truncated
            episode_reward += reward # update episode reward

            if done:
                Q_table[(obs, action)] = ((1 - LEARNING_RATE) * Q_table[(obs, action)] + LEARNING_RATE * reward)
            else:
                max_next_Q = max(Q_table[(next_obs, i)] for i in range(env.action_space.n))
                Q_table[(obs, action)] = ((1 - LEARNING_RATE) * Q_table[(obs, action)] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_Q))

            obs = next_obs

        EPSILON *= EPSILON_DECAY
            
        # END of TODO
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    #### DO NOT MODIFY ######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################