import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import random
import time
import os
from collections import deque

# CONFIGURATION 
class Config:
    GRID_WIDTH = 40
    GRID_HEIGHT = 40
    NUM_RESOURCES = 150
    NUM_OBSTACLES = 40
    AGENT_VIEW_RADIUS = 7
    AGENT_VIEW_SIZE = (2 * AGENT_VIEW_RADIUS + 1)

# MODEL ARCHITECTURE 
class DQN_Architecture(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN_Architecture, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        dummy_input_size = Config.AGENT_VIEW_SIZE
        o = self.conv_layers(torch.zeros(1, input_channels, dummy_input_size, dummy_input_size))
        conv_out_size = int(np.prod(o.size()))
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_layers(x)
        x = x.view(batch_size, -1) 
        return self.fc_layers(x)

# ENVIRONMENT
class GridWorld:
    EMPTY, OBSTACLE, RESOURCE, AGENT = 0, 1, 2, 3
    
    def __init__(self, config, agents, step_penalty):
        self.config = config
        self.width = config.GRID_WIDTH
        self.height = config.GRID_HEIGHT
        self.agents = agents
        self.step_penalty = step_penalty
        self.grid = np.full((self.width, self.height), self.EMPTY, dtype=int)
        self.resources_remaining = 0
        
    def _place_objects(self, object_type, count):
        for _ in range(count):
            while True:
                x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
                if self.grid[x, y] == self.EMPTY:
                    self.grid[x, y] = object_type
                    break

    def reset(self):
        self.grid.fill(self.EMPTY)
        self._place_objects(self.OBSTACLE, self.config.NUM_OBSTACLES)
        self._place_objects(self.RESOURCE, self.config.NUM_RESOURCES)
        self.resources_remaining = self.config.NUM_RESOURCES
        for agent in self.agents:
            agent.reset()
            while True:
                x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
                if self.grid[x, y] == self.EMPTY:
                    agent.pos = (x, y); break

    def get_agent_view(self, agent_pos, exclude_self=None):
        x, y = agent_pos
        r = self.config.AGENT_VIEW_RADIUS
        padded_canvas = np.zeros((4, self.width + 2*r, self.height + 2*r), dtype=np.float32)
        padded_canvas[self.OBSTACLE, :, :] = 1.0 
        
        full_state = np.zeros((4, self.width, self.height), dtype=np.float32)
        full_state[self.EMPTY] = (self.grid == self.EMPTY)
        full_state[self.OBSTACLE] = (self.grid == self.OBSTACLE)
        full_state[self.RESOURCE] = (self.grid == self.RESOURCE)
        for ag in self.agents:
            if ag.active and ag is not exclude_self:
                full_state[self.AGENT, ag.pos[0], ag.pos[1]] = 1

        start_x, start_y = r, r
        padded_canvas[:, start_x:start_x+self.width, start_y:start_y+self.height] = full_state
        view = padded_canvas[:, x:x+2*r+1, y:y+2*r+1]
        return torch.FloatTensor(view).unsqueeze(0)

    def step(self, agent_idx, action):
        agent = self.agents[agent_idx]
        if not agent.active: return
        
        if action == 5: 
            agent.active = False; return

        dx, dy = 0, 0
        if action == 0: dx = -1
        elif action == 1: dx = 1
        elif action == 2: dy = -1
        elif action == 3: dy = 1

        px, py = agent.pos
        npx, npy = px + dx, py + dy
        
        if not (0 <= npx < self.width and 0 <= npy < self.height) or self.grid[npx, npy] == self.OBSTACLE:
            agent.score += -5.0
            return
        
        occupied = any(ag.active and ag.pos == (npx, npy) for ag in self.agents if ag is not agent)
        if occupied:
            agent.score += -2.0
            return
            
        px, py = npx, npy
        agent.score += self.step_penalty
        
        if self.grid[px, py] == self.RESOURCE:
            agent.score += 10.0
            self.grid[px, py] = self.EMPTY
            self.resources_remaining -= 1
    
        agent.pos = (px, py)

# AGENTS 
class BaseAgent:
    def __init__(self, agent_id): self.agent_id = agent_id; self.pos = (0,0); self.score = 0; self.active = True
    def reset(self): self.score = 0; self.active = True
    def select_action(self, view): raise NotImplementedError

class RandomAgent(BaseAgent):
    def select_action(self, view): return random.randint(0, 5)

class HeuristicAgent(BaseAgent):
    def select_action(self, view):
        view = view.squeeze(0).numpy(); r = Config.AGENT_VIEW_RADIUS
        res = np.argwhere(view[GridWorld.RESOURCE] == 1)
        if len(res) > 0:
            center = np.array([r, r]); dists = np.linalg.norm(res - center, axis=1)
            closest = res[np.argmin(dists)]; dx, dy = closest - center
            if abs(dx) > abs(dy): return 0 if dx < 0 else 1
            else: return 2 if dy < 0 else 3
        return random.randint(0, 4)

class TrainedAgent(BaseAgent):
    def __init__(self, agent_id, model_path):
        super().__init__(agent_id)
        self.net = DQN_Architecture(4, 6)
        self.pos_history = deque(maxlen=4)
        self.stuck_counter = 0
        
        # Load Model Safely
        if os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.net.eval()
        else:
            st.error(f"Weights file not found: {model_path}")

    def select_action(self, view):
        v_np = view.squeeze(0).numpy(); r = Config.AGENT_VIEW_RADIUS
        neighbors = [(0, r-1, r), (1, r+1, r), (2, r, r-1), (3, r, r+1)]
        random.shuffle(neighbors)
        for action, nx, ny in neighbors:
            if v_np[GridWorld.RESOURCE, nx, ny] == 1: return action

        with torch.no_grad():
            q_values = self.net(view)
            
        self.pos_history.append(self.pos)
        if len(self.pos_history) >= 3 and self.pos_history[-1] == self.pos_history[-3]:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            
        if self.stuck_counter > 2:
            self.stuck_counter = 0
            return random.randint(0, 3) 
        return q_values.max(1)[1].item()

# EB UI HELPER 
def render_to_image(env):
    # Create an RGB Array for visualization
    # Colors (R, G, B)
    c_bg = [30, 30, 30]
    c_obs = [100, 100, 100]
    c_res = [255, 215, 0] # Gold
    
    # Initialize grid image
    img = np.full((env.width, env.height, 3), c_bg, dtype=np.uint8)
    
    # Draw Static Objects
    img[env.grid == GridWorld.OBSTACLE] = c_obs
    img[env.grid == GridWorld.RESOURCE] = c_res
    
    # Draw Agents
    agent_colors = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], 
        [255, 0, 255], [255, 165, 0], [0, 255, 255]
    ]
    
    for i, ag in enumerate(env.agents):
        if ag.active:
            x, y = ag.pos
            # Ensure within bounds
            if 0 <= x < env.width and 0 <= y < env.height:
                img[x, y] = agent_colors[i % len(agent_colors)]
                
    # Upscale for better visibility (Zoom 10x)
    return img.repeat(10, axis=0).repeat(10, axis=1)

# MAIN STREAMLIT APP 
def main():
    st.set_page_config(page_title="AI Agent Simulation", layout="wide")
    st.title(" Multi-Agent Reinforcement Learning Demo")

    # Sidebar Controls
    st.sidebar.header("Settings")
    penalty = st.sidebar.slider("Step Penalty", -5.0, 0.0, -2.0, 0.1)
    speed = st.sidebar.slider("Simulation Speed (Delay)", 0.01, 0.5, 0.05)
    
    # Initialization
    if 'env' not in st.session_state:
        config = Config()
        dqn = TrainedAgent(0, "drqn_weights_2.pth")
        simple = HeuristicAgent(1)
        rnd = RandomAgent(2)
        
        agents = [dqn, simple, rnd]
        st.session_state.agent_names = ["DRQN (AI)", "Heuristic", "Random"]
        st.session_state.env = GridWorld(config, agents, step_penalty=penalty)
        st.session_state.env.reset()
        st.session_state.running = False

    # Layout
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Leaderboard")
        metrics_placeholder = st.empty()
        
        if st.button("Start / Stop Simulation"):
            st.session_state.running = not st.session_state.running
        
        if st.button("Reset Environment"):
            st.session_state.env.step_penalty = penalty
            st.session_state.env.reset()
            st.session_state.running = False

    with col1:
        game_placeholder = st.empty()

    # Simulation Loop
    if st.session_state.running:
        env = st.session_state.env
        
        # Run a batch of steps to make it smoother
        if env.resources_remaining > 0:
            indices = list(range(len(env.agents)))
            random.shuffle(indices)
            
            for i in indices:
                agent = env.agents[i]
                if not agent.active: continue
                view = env.get_agent_view(agent.pos, exclude_self=None) # Simplified for demo
                action = agent.select_action(view)
                env.step(i, action)
        else:
            st.session_state.running = False
            st.success("Simulation Finished!")

        # Update Visuals
        img = render_to_image(env)
        game_placeholder.image(img, caption=f"Resources Left: {env.resources_remaining}", use_column_width=True)
        
        # Update Leaderboard
        scores = {name: ag.score for name, ag in zip(st.session_state.agent_names, env.agents)}
        sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        metrics_placeholder.table(sorted_scores)
        
        time.sleep(speed)
        st.rerun()
    else:
        # Static render when paused
        img = render_to_image(st.session_state.env)
        game_placeholder.image(img, caption="Paused", use_column_width=True)

if __name__ == "__main__":
    main()