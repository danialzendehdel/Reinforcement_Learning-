import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class SBW(gym.Env):
    def __init__(self):
        super(SBW, self).__init__()

        self.action_space = spaces.Discrete(2)  # 0: left, 1:right
        self.observation_space = spaces.Discrete(3)  # 0: hole, 1: start, 2:goal

        self.initial_state = 1
        self.state = self.initial_state
        self.goal_state = 2
        self.hole_state = 0

        self.step_count = 0

        # For rendering
        self.fig, self.ax = None, None
        self.agent_marker = None
        self.cells = ['Hole', 'Start', 'Goal']

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.initial_state
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        self.step_count += 1

        intended_direction = action

        if np.random.rand() < 0.8:
            actual_direction = intended_direction
        else:
            actual_direction = 1 - intended_direction

        if actual_direction == 0:
            self.state = self.hole_state
        else:
            self.state = self.goal_state

        if self.state == self.goal_state:
            reward = 1
            done = True
        else:
            reward = 0
            done = True

        truncated = False
        info = {
            'step_count': self.step_count,
            'intended_action': action,
            'actual_direction': actual_direction,
            'state': self.state
        }

        return self.state, reward, done, truncated, info

    def render(self, mode='human'):
        if mode == 'human':
            cells = ['[ ]', '[ ]', '[ ]']
            if self.state == self.hole_state:
                cells[0] = '[A]'
            elif self.state == self.initial_state:
                cells[1] = '[A]'
            elif self.state == self.goal_state:
                cells[2] = '[A]'
            print(' '.join(cells))


import time

def test_render():
    env = SBW()
    num_episodes = 5  # Number of episodes to run

    for episode in range(num_episodes):
        state, _ = env.reset()
        env.render()

        print(f"Episode {episode + 1}")

        # Choose an action (e.g., always move Right for this test)
        action = 1  # Right
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

        time.sleep(1)  # Pause between episodes

    
if __name__ == "__main__":
    test_render()