from flask import Flask, render_template, jsonify, request
import numpy as np
import torch
import torch.nn as nn
import random
import os
from collections import deque

app = Flask(__name__)

# --- CONFIGURATION & CLASSES (COPIED FROM YOUR CODE) ---
class Config:
    GRID_WIDTH = 40
    GRID_HEIGHT = 40
    AGENT_VIEW_RADIUS = 7
    AGENT_VIEW_SIZE = (2 * AGENT_VIEW_RADIUS + 1)

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
            nn.Linear(conv_out_size, 128), nn.ReLU(), nn.Linear(128, num_actions)
        )
    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_layers(x)
        x = x.view(batch_size, -1) 
        return self.fc_layers(x)

class GridWorld:
    EMPTY, OBSTACLE, RESOURCE, AGENT = 0, 1, 2, 3
    def __init__(self, config, agents, step_penalty, num_resources, num_obstacles):
        self.config = config; self.width = 40; self.height = 40
        self.agents = agents; self.step_penalty = step_penalty
        self.initial_res = num_resources; self.initial_obs = num_obstacles
        self.grid = np.full((40, 40), 0, dtype=int)
        self.resources_remaining = 0
    
    def reset(self):
        self.grid.fill(0)
        self._place(1, self.initial_obs) # Obstacles
        self._place(2, self.initial_res) # Resources
        self.resources_remaining = self.initial_res
        for ag in self.agents:
            ag.reset()
            while True:
                x,y = random.randint(0,39), random.randint(0,39)
                if self.grid[x,y] == 0: ag.pos = (x,y); break

    def _place(self, obj, count):
        for _ in range(int(count)):
            while True:
                x,y = random.randint(0,39), random.randint(0,39)
                if self.grid[x,y] == 0: self.grid[x,y] = obj; break

    def get_view(self, pos, exclude=None):
        x, y = pos; r = 7
        padded = np.zeros((4, self.width+14, self.height+14), dtype=np.float32)
        padded[1, :, :] = 1.0 # Wall bg
        
        fs = np.zeros((4, 40, 40), dtype=np.float32)
        fs[0] = (self.grid == 0); fs[1] = (self.grid == 1); fs[2] = (self.grid == 2)
        for ag in self.agents:
            if ag.active and ag is not exclude: fs[3, ag.pos[0], ag.pos[1]] = 1
        
        padded[:, 7:47, 7:47] = fs
        return torch.FloatTensor(padded[:, x:x+15, y:y+15]).unsqueeze(0)

    def step(self, idx, act):
        ag = self.agents[idx]; 
        if not ag.active: return
        if act == 5: ag.active = False; return
        
        dx, dy = 0, 0
        if act == 0: dx=-1
        elif act == 1: dx=1
        elif act == 2: dy=-1
        elif act == 3: dy=1
        
        px, py = ag.pos; npx, npy = px+dx, py+dy
        
        if not (0<=npx<40 and 0<=npy<40) or self.grid[npx,npy] == 1:
            ag.score -= 5.0; return
        
        if any(a.active and a.pos==(npx,npy) for a in self.agents if a is not ag):
            ag.score -= 2.0; return
            
        ag.score += self.step_penalty
        if self.grid[npx,npy] == 2:
            ag.score += 10.0; self.grid[npx,npy] = 0; self.resources_remaining -= 1
        ag.pos = (npx,npy)

# --- AGENTS ---
class BaseAgent:
    def __init__(self, id, name, color): 
        self.id=id; self.name=name; self.color=color; self.pos=(0,0); self.score=0; self.active=True
    def reset(self): self.score=0; self.active=True
    def act(self, view): return random.randint(0,4)

class TrainedAgent(BaseAgent):
    def __init__(self, id, name, color, model_path):
        super().__init__(id, name, color)
        self.net = DQN_Architecture(4, 6)
        if os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.net.eval()
        self.hist = deque(maxlen=4); self.stuck=0
    def act(self, view):
        # Reflex
        v = view.squeeze(0).numpy(); r=7
        for a,nx,ny in [(0,r-1,r),(1,r+1,r),(2,r,r-1),(3,r,r+1)]:
            if v[2,nx,ny]==1: return a
        with torch.no_grad(): q = self.net(view)
        
        self.hist.append(self.pos)
        if len(self.hist)>=3 and self.hist[-1]==self.hist[-3]: self.stuck+=1
        else: self.stuck=0
        if self.stuck>2: self.stuck=0; return random.randint(0,3)
        return q.max(1)[1].item()

class HeuristicAgent(BaseAgent):
    def act(self, view):
        v = view.squeeze(0).numpy(); r=7
        res = np.argwhere(v[2]==1)
        if len(res)>0:
            c = np.array([r,r]); d = np.linalg.norm(res-c, axis=1)
            cl = res[np.argmin(d)]; dx,dy = cl-c
            if abs(dx)>abs(dy): return 0 if dx<0 else 1
            else: return 2 if dy<0 else 3
        return random.randint(0,4)

class BFSAgent(BaseAgent):
    def act(self, view):
        v = view.squeeze(0).numpy(); r=7; start=(r,r)
        q = deque([(start, [])]); visited = {start}
        while q:
            (cr,cc), path = q.popleft()
            if v[2, cr,cc] == 1: return path[0] if path else 4
            for i,(dr,dc) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
                nr,nc = cr+dr, cc+dc
                if 0<=nr<v.shape[1] and 0<=nc<v.shape[2] and (nr,nc) not in visited and v[1,nr,nc]==0:
                    visited.add((nr,nc)); q.append(((nr,nc), path+[i]))
        return random.randint(0,4)

# --- GLOBAL STATE ---
# Note: In production servers, global state is risky, but for this demo on 1 worker it works.
simulation = {
    "env": None,
    "running": False,
    "config": Config()
}

def init_env(model_file, use_bfs, penalty, res_count):
    agents = []
    # 1. AI
    agents.append(TrainedAgent(0, "AI Agent", "#FF0000", model_file))
    # 2. Simple
    agents.append(HeuristicAgent(1, "Heuristic", "#00FF00"))
    # 3. Random
    agents.append(BaseAgent(2, "Random", "#0000FF"))
    # 4. BFS (Optional)
    if use_bfs:
        agents.append(BFSAgent(3, "BFS Robot", "#FF00FF"))
        
    simulation["env"] = GridWorld(Config(), agents, penalty, res_count, 40)
    simulation["env"].reset()

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reset', methods=['POST'])
def reset():
    data = request.json
    model = "drqn_weights_2.pth" if data['model'] == 'cost' else "drqn_weights.pth"
    init_env(model, data['use_bfs'], float(data['penalty']), int(data['res']))
    simulation["running"] = False
    return jsonify({"status": "ready"})

@app.route('/toggle', methods=['POST'])
def toggle():
    simulation["running"] = not simulation["running"]
    return jsonify({"running": simulation["running"]})

@app.route('/step')
def step():
    env = simulation["env"]
    if not env: return jsonify({"error": "not_init"})
    
    if simulation["running"] and env.resources_remaining > 0:
        indices = list(range(len(env.agents)))
        random.shuffle(indices)
        for i in indices:
            ag = env.agents[i]
            view = env.get_view(ag.pos)
            action = ag.act(view)
            env.step(i, action)
    
    # Pack data for Frontend
    grid_data = [] # Sparse representation to save bandwidth
    # 1. Obstacles
    ox, oy = np.where(env.grid == 1)
    for x, y in zip(ox, oy): grid_data.append({"t": 1, "x": int(x), "y": int(y)})
    # 2. Resources
    rx, ry = np.where(env.grid == 2)
    for x, y in zip(rx, ry): grid_data.append({"t": 2, "x": int(x), "y": int(y)})
    
    agents_data = []
    for ag in env.agents:
        agents_data.append({
            "name": ag.name,
            "color": ag.color,
            "x": ag.pos[0],
            "y": ag.pos[1],
            "score": round(ag.score, 1),
            "active": ag.active
        })

    return jsonify({
        "grid": grid_data,
        "agents": agents_data,
        "res_left": int(env.resources_remaining),
        "game_over": env.resources_remaining == 0
    })

if __name__ == '__main__':
    # Start with defaults
    init_env("drqn_weights_2.pth", False, -2.0, 150)
    app.run(host='0.0.0.0', port=5000)