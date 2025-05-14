import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
import imageio
from itertools import combinations, permutations
from collections import defaultdict

GRID_SIZE = 5
CELL_SIZE = 80
FPS = 10
EPISODES = 200
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
MAX_STEPS = 30

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
PURPLE = (128, 0, 128)
BLUE = (0, 0, 255)

class LightHuntEnv:
    def __init__(self):
        self.rows = GRID_SIZE
        self.cols = GRID_SIZE
        self.lights = {'y': (3, 3), 'g': (1, 3), 'r': (3, 1)}
        self.dark_zones = [(1, 1), (2, 2)]
        self.actions = ['up', 'down', 'left', 'right']
        self.states = []
        light_ids = ['y', 'g', 'r']

        collected_combinations = ['']
        for r in range(1, len(light_ids) + 1):
            for subset in combinations(light_ids, r):
                for perm in permutations(subset):
                    collected_combinations.append(''.join(perm))

        for x in range(self.rows):
            for y in range(self.cols):
                for collected in collected_combinations:
                    self.states.append((x, y, collected))
        self.reset()

    def reset(self):
        self.agent_pos = (0, 0)
        self.collected = ''
        self.steps = 0
        self.game_over = False
        return (self.agent_pos[0], self.agent_pos[1], self.collected)

    def step(self, action):
        if self.game_over:
            return (self.agent_pos[0], self.agent_pos[1], self.collected), 0, True

        x, y = self.agent_pos
        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < self.rows - 1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < self.cols - 1:
            y += 1
        self.agent_pos = (x, y)
        self.steps += 1

        reward = -1
        for light_id, pos in self.lights.items():
            if self.agent_pos == pos and light_id not in self.collected:
                self.collected += light_id
                reward = 50
                break
        else:
            if self.agent_pos in self.dark_zones:
                reward = -20

        next_state = (self.agent_pos[0], self.agent_pos[1], self.collected)

        if len(self.collected) == 3:
            reward += 100
            self.game_over = True
        elif self.steps >= MAX_STEPS:
            self.game_over = True

        return next_state, reward, self.game_over

class QLearningAgent:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.Q = np.zeros((len(states), len(actions)))
        self.errors = []
        self.visit_counts = np.zeros(len(states))

    def select_action(self, state_idx):
        if np.random.rand() < EPSILON:
            return np.random.randint(len(self.actions))
        return np.argmax(self.Q[state_idx])

    def update(self, s_idx, a_idx, reward, next_s_idx, done):
        target = reward
        if not done:
            target += GAMMA * np.max(self.Q[next_s_idx])
        old_value = self.Q[s_idx, a_idx]
        self.Q[s_idx, a_idx] += ALPHA * (target - self.Q[s_idx, a_idx])
        self.errors.append(abs(target - old_value))
        self.visit_counts[s_idx] += 1

def draw_grid(screen, env):
    screen.fill(WHITE)
    for i in range(env.rows):
        for j in range(env.cols):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
            pos = (i, j)
            for light_id, light_pos in env.lights.items():
                if pos == light_pos and light_id not in env.collected:
                    color = YELLOW if light_id == 'y' else GREEN if light_id == 'g' else RED
                    pygame.draw.circle(screen, color, (j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2), 20)
            if pos in env.dark_zones:
                pygame.draw.rect(screen, PURPLE, rect)
            if pos == env.agent_pos:
                pygame.draw.rect(screen, BLUE, rect)
    pygame.display.flip()

def main():
    env = LightHuntEnv()
    agent = QLearningAgent(env.states, env.actions)
    rewards = []
    frames = []

    pygame.init()
    screen = pygame.display.set_mode((env.cols * CELL_SIZE, env.rows * CELL_SIZE))
    pygame.display.set_caption("Light Hunt Q-Learning")
    clock = pygame.time.Clock()

    for ep in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            s_idx = env.states.index(state)
            a_idx = agent.select_action(s_idx)
            action = env.actions[a_idx]

            next_state, reward, done = env.step(action)
            next_s_idx = env.states.index(next_state)
            agent.update(s_idx, a_idx, reward, next_s_idx, done)

            draw_grid(screen, env)
            clock.tick(FPS)

            # Сохраняем кадр
            buffer = pygame.surfarray.array3d(screen)
            frame = np.transpose(buffer, (1, 0, 2))  # Pygame → imageio формат
            frames.append(frame)

            total_reward += reward
            state = next_state

        print(f"Эпизод {ep} завершён с общей наградой {total_reward}\n")
        rewards.append(total_reward)

    # Создание GIF после всех эпизодов
    imageio.mimsave("training.gif", frames, fps=FPS)
    print("GIF сохранён как training.gif")

    # Построение графиков
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(agent.errors)
    plt.title("Q-Learning Error")
    plt.xlabel("Step")
    plt.ylabel("MAE")
    plt.grid(True)

    state_visit_matrix = np.zeros((GRID_SIZE, GRID_SIZE))
    for idx, count in enumerate(agent.visit_counts):
        x, y, _ = env.states[idx]
        state_visit_matrix[x][y] += count

    plt.subplot(1, 3, 3)
    plt.imshow(state_visit_matrix, cmap='hot', interpolation='nearest')
    plt.title("State Visit Heatmap")
    plt.colorbar(label="Visit Count")
    plt.xticks(range(GRID_SIZE))
    plt.yticks(range(GRID_SIZE))

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    print("Графики сохранены как training_metrics.png")

    pygame.quit()

if __name__ == '__main__':
    main()
