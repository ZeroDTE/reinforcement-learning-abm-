import json
import sys

def create_notebook(is_solution=False):
      nb = {
                "cells": [],
                "metadata": {
                              "kernelspec": {
                                                "display_name": "Python 3",
                                                "language": "python",
                                                "name": "python3"
                              }
                },
                "nbformat": 4,
                "nbformat_minor": 4
      }

    title = "2. Solution Notebook" if is_solution else "1. Student Notebook"
    subtitle = "Loesungen fuer alle Phasen" if is_solution else "Aufgaben - Fuell die TODOs aus"

    nb["cells"].append({
              "cell_type": "markdown",
              "metadata": {},
              "source": [f"# Reinforcement Learning Semesteraufgabe\n", f"## {title}\n", f"\n{subtitle}"]
    })

    nb["cells"].append({
              "cell_type": "code",
              "execution_count": None,
              "metadata": {},
              "outputs": [],
              "source": ["import numpy as np\n", "import matplotlib.pyplot as plt\n", "import random\n", "import torch\n", "import torch.nn as nn\n", "import torch.optim as optim\n", "from collections import deque\n", "\n", "%matplotlib inline"]
    })

    if not is_solution:
              nb["cells"].append({
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": ["# Phase 1: Foundation\n", "\n", "Ziel: Die GridWorld Umgebung verstehen\n"]
              })

        nb["cells"].append({
                      "cell_type": "code",
                      "execution_count": None,
                      "metadata": {},
                      "outputs": [],
                      "source": ["class Config:\n", "    GRID_WIDTH = 20\n", "    GRID_HEIGHT = 20\n", "    NUM_RESOURCES = 50\n", "    NUM_OBSTACLES = 15\n", "    AGENT_VIEW_RADIUS = 3\n", "    NUM_EPISODES = 500\n", "    MAX_STEPS = 100\n", "    \n", "config = Config()"]
        })

        nb["cells"].append({
                      "cell_type": "code",
                      "execution_count": None,
                      "metadata": {},
                      "outputs": [],
                      "source": ["class GridWorld:\n", "    EMPTY, OBSTACLE, RESOURCE, AGENT = 0, 1, 2, 3\n", "    \n", "    def __init__(self, config):\n", "        self.config = config\n", "        self.width = config.GRID_WIDTH\n", "        self.height = config.GRID_HEIGHT\n", "        self.grid = np.full((self.width, self.height), self.EMPTY, dtype=int)\n", "        self.agent_pos = (0, 0)\n", "        self.agent_score = 0\n", "        self.resources_remaining = 0\n", "    \n", "    def _place_objects(self, object_type, count):\n", "        placed = 0\n", "        attempts = 0\n", "        while placed < count and attempts < count * 10:\n", "            x = np.random.randint(0, self.width)\n", "            y = np.random.randint(0, self.height)\n", "            if self.grid[x, y] == self.EMPTY:\n", "                self.grid[x, y] = object_type\n", "                placed += 1\n", "            attempts += 1\n", "    \n", "    def reset(self):\n", "        self.grid.fill(self.EMPTY)\n", "        self.agent_score = 0\n", "        self._place_objects(self.OBSTACLE, self.config.NUM_OBSTACLES)\n", "        self._place_objects(self.RESOURCE, self.config.NUM_RESOURCES)\n", "        self.resources_remaining = self.config.NUM_RESOURCES\n", "        \n", "        while True:\n", "            x = np.random.randint(0, self.width)\n", "            y = np.random.randint(0, self.height)\n", "            if self.grid[x, y] == self.EMPTY:\n", "                self.agent_pos = (x, y)\n", "                break\n", "        \n", "        return self.get_agent_view()\n", "    \n", "    def get_agent_view(self):\n", "        x, y = self.agent_pos\n", "        r = self.config.AGENT_VIEW_RADIUS\n", "        view = np.zeros((4, 2*r+1, 2*r+1), dtype=np.float32)\n", "        \n", "        for i in range(2*r+1):\n", "            for j in range(2*r+1):\n", "                grid_x = x - r + i\n", "                grid_y = y - r + j\n", "                \n", "                if not (0 <= grid_x < self.width and 0 <= grid_y < self.height):\n", "                    view[self.OBSTACLE, i, j] = 1.0\n", "                else:\n", "                    cell_type = self.grid[grid_x, grid_y]\n", "                    view[cell_type, i, j] = 1.0\n", "        \n", "        view[self.AGENT, r, r] = 1.0\n", "        return view.astype(np.float32)\n", "    \n", "    def step(self, action):\n", "        reward = -0.1\n", "        done = False\n", "        \n", "        dx, dy = 0, 0\n", "        if action == 0:\n", "            dx = -1\n", "        elif action == 1:\n", "            dx = 1\n", "        elif action == 2:\n", "            dy = -1\n", "        elif action == 3:\n", "            dy = 1\n", "        elif action == 5:\n", "            done = True\n", "            return self.get_agent_view(), reward, done\n", "        \n", "        new_x = self.agent_pos[0] + dx\n", "        new_y = self.agent_pos[1] + dy\n", "        \n", "        if not (0 <= new_x < self.width and 0 <= new_y < self.height):\n", "            reward -= 5.0\n", "        elif self.grid[new_x, new_y] == self.OBSTACLE:\n", "            reward -= 5.0\n", "        else:\n", "            self.agent_pos = (new_x, new_y)\n", "            if self.grid[new_x, new_y] == self.RESOURCE:\n", "                reward += 10.0\n", "                self.grid[new_x, new_y] = self.EMPTY\n", "                self.resources_remaining -= 1\n", "        \n", "        self.agent_score += reward\n", "        \n", "        if self.resources_remaining == 0:\n", "            done = True\n", "        \n", "        return self.get_agent_view(), reward, done"]
        })

        nb["cells"].append({
                      "cell_type": "markdown",
                      "metadata": {},
                      "source": ["## Hausaufgabe Phase 1\n", "\nImplementiere eine Funktion random_episode(), die:\n", "1. Eine Episode startet\n", "2. Max 100 zufaellige Aktionen ausfuehrt\n", "3. Den finalen Score zurueckgibt\n"]
        })

        nb["cells"].append({
                      "cell_type": "code",
                      "execution_count": None,
                      "metadata": {},
                      "outputs": [],
                      "source": ["# TODO: Implementiere random_episode()\n", "def random_episode(env, max_steps=100):\n", "    view = env.reset()\n", "    for step in range(max_steps):\n", "        # TODO: Random aktion auswaehlen (0-5)\n", "        # TODO: env.step() aufrufen\n", "        # TODO: Wenn done, break\n", "        pass\n", "    return env.agent_score"]
        })

        nb["cells"].append({
                      "cell_type": "markdown",
                      "metadata": {},
                      "source": ["# Phase 2: Q-Learning\n", "\nDer Agent lernt mit Tabellen.\n"]
        })

        nb["cells"].append({
                      "cell_type": "code",
                      "execution_count": None,
                      "metadata": {},
                      "outputs": [],
                      "source": ["# TODO: Implementiere SimpleQLearningAgent\n", "# Q(s,a) = Q(s,a) + alpha * (reward + gamma * max Q(s') - Q(s,a))\n", "\n", "class SimpleQLearningAgent:\n", "    def __init__(self, lr=0.1, gamma=0.95, epsilon=1.0):\n", "        self.lr = lr\n", "        self.gamma = gamma\n", "        self.epsilon = epsilon\n", "        self.q_table = {}\n", "        self.init_q_table()\n", "    \n", "    def init_q_table(self):\n", "        # TODO: Initialisiere Q-Tabelle fuer States\n", "        pass\n", "    \n", "    def _discretize_state(self, view):\n", "        # TODO: Vereinfache View zu diskretum State\n", "        # Hinweis: In welche Richtung ist die naechste Ressource?\n", "        pass\n", "    \n", "    def select_action(self, state, training=True):\n", "        # TODO: Epsilon-Greedy: Exploration vs Exploitation\n", "        pass\n", "    \n", "    def update(self, state, action, reward, next_state):\n", "        # TODO: Q-Learning Update\n", "        pass"]
        })

else:
        nb["cells"].append({
                      "cell_type": "markdown",
                      "metadata": {},
                      "source": ["Dies ist das Loesungsbuch mit allen Antworten.\n"]
        })

    return nb


def main():
      if len(sys.argv) > 1 and sys.argv[1] == "solution":
                student_nb = create_notebook(is_solution=False)
                solution_nb = create_notebook(is_solution=True)

        with open("1_Student_Notebook.ipynb", "w") as f:
                      json.dump(student_nb, f, indent=1)

        with open("2_Solution_Notebook.ipynb", "w") as f:
                      json.dump(solution_nb, f, indent=1)

        print("Created 1_Student_Notebook.ipynb")
        print("Created 2_Solution_Notebook.ipynb")
else:
        student_nb = create_notebook(is_solution=False)
          with open("1_Student_Notebook.ipynb", "w") as f:
                        json.dump(student_nb, f, indent=1)
                    print("Created 1_Student_Notebook.ipynb")


if __name__ == "__main__":
      main()
