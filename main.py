import gym
import random


# Klasa agenta
class Agent():
    def __init__(self, env):
        self.action_size = env.action_space.n
        self.policy = [1 / self.action_size] * self.action_size
        self.learning_rate = 1.7
        self.discount_factor = 0.99

    def select_action(self, state):
        # Wybór akcji
        action = random.choices(range(self.action_size), self.policy)[0]
        return action

    def update_policy(self, state, action, reward, next_state):
        q_value = max([self.get_q_value(next_state, a) for a in range(self.action_size)])
        td_error = reward + self.discount_factor * q_value - self.get_q_value(state, action)
        self.policy[action] += self.learning_rate * td_error

    def get_q_value(self, state, action):
        # metoda pomocnicza do obliczania wartości Q
        return self.policy[action]


# inicjalizacja środowiska i agenta
env = gym.make('CartPole-v1', render_mode="human")

agent = Agent(env)

# liczba punktów
points = 0

# pętla uczenia
for i_episode in range(6000):
    # inicjalizacja stanu początkowego
    observation = env.reset()
    for t in range(100):
        # wyświetlenie środowiska
        env.render()
        # wybór akcji przez agenta
        action = agent.select_action(observation)
        # wykonanie akcji
        next_observation, reward, done, info, *_ = env.step(action)
        agent.update_policy(observation, action, reward, next_observation)
        observation = next_observation
        if done:
            points += t
            print("Episode finished after {} timesteps".format(t + 1))
            print(i_episode)
            break

print(f'Total points {points}')
env.close()
