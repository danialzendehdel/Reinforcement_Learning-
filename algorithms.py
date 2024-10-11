import numpy as np
from Slippery_bandit_walk import SBW
from tqdm import tqdm
import time


def pure_exploitation(env, n_episodes=5000):

    Q = np.zeros((env.action_space.n))
    N = np.zeros((env.action_space.n))

    Qe = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes)


    name = 'pure exploitation'

    for e in tqdm(range(n_episodes), desc= 'Episode for:' + name, leave=False):

        action = np.argmax(Q)

        next_state, reward, done, truncated, info = env.step(action)

        env.render()

        intended_action = 'Left' if action == 0 else 'Right'
        actual_movement = 'Left' if info['actual_direction'] == 0 else 'Right'

        print(f"Intended Action: {intended_action}")
        print(f"Actual Movement: {actual_movement}")
        print(f"Next State: {next_state}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print("-" * 30)
        #
        # time.sleep(1)  # Pause between episodes

        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action])/N[action]


        Qe[e] = Q
        returns[e] = reward
        actions[e] = action
    tqdm().close()
    return name, returns, Qe, actions

env = SBW()
name, returns, Qe, actions = pure_exploitation(env=env)
print(f'Name: {name}\n'
      f'returns:{returns}\n'
      f'Qe: {Qe}\n'
      f'actions:{actions}')