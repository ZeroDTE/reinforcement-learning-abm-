import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import random
import time
import os
from collections import deque
from PIL import Image, ImageDraw

# --- CONFIGURATION ---
class Config:
    GRID_WIDTH = 40
    GRID_HEIGHT = 40
    # Defaults
    DEFAULT_RESOURCES = 150
    DEFAULT_OBSTACLES = 40
    AGENT_VIEW_RADIUS = 7
    AGENT_VIEW_SIZE = (2 * AGENT_VIEW_RADIUS + 1)
    
    # Colors (R, G, B)
    COLOR_BG = (30, 30, 30)
    COLOR_GRID = (40, 40, 40)
    COLOR_OBSTACLE = (100, 100, 100)
    COLOR_RESOURCE = (255, 215, 0) # Gold
    COLOR_TEXT = (255, 255, 255)
    
    # Agent Colors
    AGENT_COLORS = [
        (255, 50, 50),    # Red
        (50, 255, 50),    # Green
        (50, 100, 255),   # Blue
        (255, 50, 255),   # Magenta
        (255, 165, 0),    # Orange
        (0, 255, 255)     # Cyan
    ]

# --- MODEL ARCHITECTURE ---
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

# --- ENVIRONMENT ---
class GridWorld:
    EMPTY, OBSTACLE, RESOURCE, AGENT = 0, 1, 2, 3
    
    def __init__(self, config, agents, step_penalty, num_resources):
        self.config = config
        self.width = config.GRID_WIDTH
        self.height = config.GRID_HEIGHT
        self.agents = agents
        self.step_penalty = step_penalty
        self.initial_resources = num_resources
        
        self.grid = np.full((self.width, self.height), self.EMPTY, dtype=int)
        self.resources_remaining = 0
        self.current_step = 0
        
    def _place_objects(self, object_type, count):
        count = int(count)
        for _ in range(count):
            while True:
                x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
                if self.grid[x, y] == self.EMPTY:
                    self.grid[x, y] = object_type
                    break

    def reset(self):
        self.grid.fill(self.EMPTY)
        self._place_objects(self.OBSTACLE, self.config.DEFAULT_OBSTACLES)
        self._place_objects(self.RESOURCE, self.initial_resources)
        self.resources_remaining = self.initial_resources
        self.current_step = 0
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
        
        # Wall Collision
        if not (0 <= npx < self.width and 0 <= npy < self.height) or self.grid[npx, npy] == self.OBSTACLE:
            agent.score += -5.0
            return
        
        # Agent Collision
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

# --- AGENTS ---
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

class BFSAgent(BaseAgent):
    def select_action(self, view):
        v = view.squeeze(0).numpy(); r = Config.AGENT_VIEW_RADIUS; start = (r, r)
        queue = deque([(start, [])]); visited = {start}
        while queue:
            (cr, cc), path = queue.popleft()
            if v[GridWorld.RESOURCE, cr, cc] == 1: return path[0] if path else 4
            for act, (dr, dc) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
                nr, nc = cr+dr, cc+dc
                if 0<=nr<v.shape[1] and 0<=nc<v.shape[2] and (nr,nc) not in visited:
                    if v[GridWorld.OBSTACLE, nr, nc] == 0:
                        visited.add((nr, nc)); queue.append(((nr, nc), path + [act]))
        return random.randint(0, 4)

class TrainedAgent(BaseAgent):
    def __init__(self, agent_id, model_path):
        super().__init__(agent_id)
        self.net = DQN_Architecture(4, 6)
        self.pos_history = deque(maxlen=4)
        self.stuck_counter = 0
        self.model_loaded = False
        
        if os.path.exists(model_path):
            try:
                self.net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                self.net.eval()
                self.model_loaded = True
            except Exception as e:
                st.error(f"Error loading model: {e}")
        else:
            st.error(f"Weights file not found: {model_path}")

    def select_action(self, view):
        if not self.model_loaded: return random.randint(0, 4)
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

# --- HIGH QUALITY RENDERER (PIL) ---
def draw_game_state(env, cell_size=20):
    # Calculate image dimensions
    w = env.width * cell_size
    h = env.height * cell_size
    
    # Create canvas
    img = Image.new("RGB", (w, h), Config.COLOR_BG)
    draw = ImageDraw.Draw(img)
    
    # 1. Draw Grid Lines (Subtle)
    for x in range(0, w, cell_size):
        draw.line([(x, 0), (x, h)], fill=Config.COLOR_GRID, width=1)
    for y in range(0, h, cell_size):
        draw.line([(0, y), (w, y)], fill=Config.COLOR_GRID, width=1)
        
    # 2. Draw Obstacles (Rectangles)
    # Optimized: Get indices of obstacles
    obs_x, obs_y = np.where(env.grid == GridWorld.OBSTACLE)
    for x, y in zip(obs_x, obs_y):
        rect = [y * cell_size, x * cell_size, (y+1) * cell_size - 1, (x+1) * cell_size - 1]
        draw.rectangle(rect, fill=Config.COLOR_OBSTACLE)
        
    # 3. Draw Resources (Circles)
    res_x, res_y = np.where(env.grid == GridWorld.RESOURCE)
    padding = 3
    for x, y in zip(res_x, res_y):
        rect = [y * cell_size + padding, x * cell_size + padding, 
                (y+1) * cell_size - padding, (x+1) * cell_size - padding]
        draw.ellipse(rect, fill=Config.COLOR_RESOURCE)

    # 4. Draw Agents (Rects with Borders)
    for i, ag in enumerate(env.agents):
        if ag.active:
            x, y = ag.pos
            color = Config.AGENT_COLORS[i % len(Config.AGENT_COLORS)]
            rect = [y * cell_size + 2, x * cell_size + 2, 
                    (y+1) * cell_size - 2, (x+1) * cell_size - 2]
            draw.rectangle(rect, fill=color, outline=(255,255,255), width=2)
            
    return img

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="Agent Showdown", layout="wide", initial_sidebar_state="expanded")

    # CUSTOM CSS TO REMOVE PADDING & MAKE IT LOOK LIKE A DASHBOARD
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }
                h1 { margin-bottom: 0px; }
        </style>
        """, unsafe_allow_html=True)

    st.title("🤖 AI Agent Simulation")

    # --- SIDEBAR SETTINGS ---
    st.sidebar.header("🧠 Brain Config")
    model_option = st.sidebar.selectbox(
        "AI Strategy",
        ("Min Cost (Optimized)", "Max Profit (Greedy)"),
        index=0
    )
    model_filename = "drqn_weights_2.pth" if model_option == "Min Cost (Optimized)" else "drqn_weights.pth"

    st.sidebar.divider()
    st.sidebar.header("🏁 Competitors")
    use_simple = st.sidebar.checkbox("Simple Heuristic (Green)", value=True)
    use_random = st.sidebar.checkbox("Random Walker (Blue)", value=True)
    use_bfs = st.sidebar.checkbox("BFS Pathfinder (Magenta)", value=False)
    
    st.sidebar.divider()
    st.sidebar.header("⚙️ World Settings")
    penalty = st.sidebar.number_input("Step Penalty", -5.0, 0.0, -2.0, 0.1)
    res_count = st.sidebar.number_input("Resource Count", 10, 500, 150, 10)
    max_steps = st.sidebar.number_input("Max Steps", 50, 1000, 250, 50)
    
    st.sidebar.divider()
    speed = st.sidebar.select_slider("Animation Speed", options=["Slow", "Medium", "Fast", "Turbo"], value="Fast")
    speed_map = {"Slow": 0.3, "Medium": 0.1, "Fast": 0.01, "Turbo": 0.0}
    
    # Function to Reset/Init
    def init_game():
        st.session_state.game_over = False
        st.session_state.winner = None
        
        # Create Agents
        agent_list = []
        names = []
        
        # 1. AI
        dqn = TrainedAgent(0, model_filename)
        agent_list.append(dqn)
        names.append(f"AI ({model_option})")
        
        id_counter = 1
        if use_simple:
            agent_list.append(HeuristicAgent(id_counter))
            names.append("Heuristic")
            id_counter += 1
        if use_random:
            agent_list.append(RandomAgent(id_counter))
            names.append("Random")
            id_counter += 1
        if use_bfs:
            agent_list.append(BFSAgent(id_counter))
            names.append("BFS Robot")
            id_counter += 1
            
        st.session_state.agent_names = names
        
        # Create Env
        config = Config()
        st.session_state.env = GridWorld(config, agent_list, step_penalty=penalty, num_resources=res_count)
        st.session_state.env.reset()

    # Buttons
    if st.sidebar.button("RESTART SIMULATION", type="primary", use_container_width=True):
        st.session_state.running = False
        init_game()
        st.rerun()

    # Initialize if first time
    if 'env' not in st.session_state:
        init_game()
        st.session_state.running = False

    # --- MAIN LAYOUT ---
    col_game, col_stats = st.columns([3, 1])

    with col_stats:
        # Dashboard Look
        st.subheader("📊 Live Stats")
        
        # Control Buttons
        c1, c2 = st.columns(2)
        if c1.button("▶ START", use_container_width=True):
            st.session_state.running = True
        if c2.button("⏸ PAUSE", use_container_width=True):
            st.session_state.running = False
            
        st.markdown("---")
        
        # Dynamic Metrics
        step_col, res_col = st.columns(2)
        step_metric = step_col.empty()
        res_metric = res_col.empty()
        
        st.markdown("### 🏆 Leaderboard")
        leaderboard_pl = st.empty()
        
        status_pl = st.empty()

    with col_game:
        canvas = st.empty()

    # --- SIMULATION LOOP ---
    env = st.session_state.env
    
    # We loop while running is true
    # Streamlit "while" loop trick to avoid full page reloads if possible for smoother anim
    
    if st.session_state.running and not st.session_state.game_over:
        
        # GAME LOGIC
        if env.resources_remaining > 0 and env.current_step < max_steps:
            env.current_step += 1
            indices = list(range(len(env.agents)))
            random.shuffle(indices)
            
            for i in indices:
                agent = env.agents[i]
                if not agent.active: continue
                view = env.get_agent_view(agent.pos, exclude_self=None) 
                action = agent.select_action(view)
                env.step(i, action)
        else:
            st.session_state.running = False
            st.session_state.game_over = True

        # RENDER
        img = draw_game_state(env, cell_size=18) # Slightly smaller cells for fit
        canvas.image(img, use_column_width=True, clamp=True)
        
        # UPDATE STATS
        step_metric.metric("Step", f"{env.current_step}/{max_steps}")
        res_metric.metric("Resources", env.resources_remaining)
        
        # Sort Scores
        scores = []
        for i, ag in enumerate(env.agents):
            status = "🟢" if ag.active else "🔴"
            scores.append({"Name": st.session_state.agent_names[i], "Score": round(ag.score, 1), "Status": status})
        
        sorted_scores = sorted(scores, key=lambda x: x["Score"], reverse=True)
        leaderboard_pl.dataframe(sorted_scores, hide_index=True, use_container_width=True)

        if st.session_state.game_over:
            if env.resources_remaining == 0:
                status_pl.success(" RESOURCES CLEARED! ")
            else:
                status_pl.error("MAX STEPS REACHED")
        else:
            time.sleep(speed_map[speed])
            st.rerun()

    else:
        # STATIC RENDER (PAUSED OR GAME OVER)
        img = draw_game_state(env, cell_size=18)
        canvas.image(img, use_column_width=True)
        
        step_metric.metric("Step", f"{env.current_step}/{max_steps}")
        res_metric.metric("Resources", env.resources_remaining)
        
        scores = []
        for i, ag in enumerate(env.agents):
            status = "🟢" if ag.active else "🔴"
            scores.append({"Name": st.session_state.agent_names[i], "Score": round(ag.score, 1), "Status": status})
        sorted_scores = sorted(scores, key=lambda x: x["Score"], reverse=True)
        leaderboard_pl.dataframe(sorted_scores, hide_index=True, use_container_width=True)
        
        if st.session_state.game_over:
            if env.resources_remaining == 0:
                status_pl.success("✨ RESOURCES CLEARED! ✨")
            else:
                status_pl.error(" MAX STEPS REACHED")

if __name__ == "__main__":
    main()