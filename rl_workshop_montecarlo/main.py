import random
import numpy as np


class StateAction:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

    def next_state(self, state, action):
        if action > state or action < 0:
            return None

        next_state_win = state + action
        next_state_lose = state - action

        if next_state_win == self.state_space:
            return (next_state_win, 1)

        if next_state_lose == 0:
            return (next_state_lose, 0)

        return (next_state_win, None)


class Agent:
    def __init__(self, state_space, action_space, gamma):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.value_function = np.zeros(self.state_space + 1)

    def update_value_function(self, state, action, reward, next_state):
        next_value = self.value_function[next_state]
        self.value_function[state] += 0.1 * (reward + self.gamma * next_value - self.value_function[state])

    def get_action(self, state):
        actions = np.zeros(self.action_space + 1)
        for a in range(1, min(state, self.action_space - state) + 1):
            next_state, reward = StateAction(self.state_space, self.action_space).next_state(state, a)
            actions[a] = self.value_function[next_state]
        return np.argmax(actions)


class Environment:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space

    def step(self, state, action):
        next_state, reward = StateAction(self.state_space, self.action_space).next_state(state, action)
        return next_state, reward

    def reset(self):
        return random.randint(1, self.state_space - 1)


def value_iteration(state_space, action_space, gamma):
    agent = Agent(state_space, action_space, gamma)
    for i in range(100):
        for s in range(1, state_space):
            actions = np.zeros(action_space + 1)
            for a in range(1, min(s, action_space - s) + 1):
                next_state, reward = StateAction(state_space, action_space).next_state(s, a)
                actions[a] = reward + gamma * agent.value_function[next_state]
            agent.value_function[s] = np.max(actions)

    return agent.value_function


def monte_carlo(state_space, action_space, gamma, num_episodes):
    agent = Agent(state_space, action_space, gamma)
    env = Environment(state_space, action_space)
    returns = {(s, a): [] for s in range(1, state_space) for a in range(1, min(s, action_space - s) + 1)}
    for i in range(num_episodes):
        state = env.reset()
        episode = []
        while True:
            action = agent.get_action(state)
            next_state, reward = env.step(state, action)
            episode.append((state, action, reward))
            if next_state == None:
                break
            state = next_state

        G = 0
        for s, a, r in reversed(episode):
            G = gamma * G + r
            if (s, a) not in [(x[0], x[1]) for x in episode[::-1][len(episode) - episode.index((s, a)) - 1:]]:
                returns[(s, a)].append(G)
                agent.value_function[s] = np.mean(returns[(s, a)])