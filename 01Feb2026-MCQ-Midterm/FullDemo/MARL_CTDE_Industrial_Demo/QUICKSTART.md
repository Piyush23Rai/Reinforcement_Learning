# Quick Start Guide

## Installation

```bash
# 1. Extract the zip file
unzip MARL_CTDE_Industrial_Demo.zip
cd MARL_CTDE_Industrial_Demo

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Running Demos

### Option 1: Run Everything
```bash
python main.py --all
```

### Option 2: Run Specific Demo

**Retail Demo** (Multi-warehouse inventory with MADDPG):
```bash
python main.py --retail
# Or directly:
python retail/demo.py
```

**Banking Demo** (Transaction routing with MAPPO):
```bash
python main.py --banking
# Or directly:
python banking/demo.py
```

### Option 3: Run Full Training

For more episodes and thorough training:

**Retail**:
```bash
python retail/train.py
```

**Banking**:
```bash
python banking/train.py
```

## Understanding the Output

### Training Progress
```
Training episodes: 50%|██████
Episode 500/1000 | Avg Reward: -45.32 | Avg Cost: -52.18 | Exploration: 0.85
```

- **Episode X/Y**: Current episode / total episodes
- **Avg Reward**: Average reward over last 100 episodes
- **Avg Cost**: Negative rewards (cost minimization)
- **Exploration**: Epsilon value (0 = exploit, 1 = explore)

### Evaluation Results
```
Evaluation Results:
  Avg Reward per Agent: 125.45
  Avg Cost per Agent: 23.56
  Avg Stockout: 2.34
```

### Generated Files
- `{retail,banking}/results/training_*.png` - Training curves
- `{retail,banking}/results/models/` - Trained agent models
- `{retail,banking}/results/metrics.json` - Performance metrics

## Key Files to Understand

### 1. **Core Algorithms** (`core/`)
- `maddpg.py` - Multi-Agent DDPG implementation
- `mappo.py` - Multi-Agent PPO implementation
- `qmix.py` - QMIX implementation
- `replay_buffer.py` - Experience storage

### 2. **Retail Demo** (`retail/`)
- `environment.py` - Multi-warehouse environment
- `train.py` - Full training script
- `demo.py` - Quick demo

### 3. **Banking Demo** (`banking/`)
- `environment.py` - Transaction routing environment
- `train.py` - Full training script (optional)
- `demo.py` - Quick demo

### 4. **Documentation** (`docs/`)
- `MARL_THEORY.md` - Theoretical background
- `CTDE_PATTERNS.md` - Architecture patterns
- `ALGORITHM_DETAILS.md` - Algorithm specifics

## Customization

### Changing Algorithm

**For Retail** (in `retail/train.py`):
```python
from core.maddpg import create_maddpg_agents  # MADDPG (default)
from core.mappo import create_mappo_agents    # Switch to MAPPO
from core.qmix import create_qmix_agents      # Switch to QMIX

agents = create_maddpg_agents(...)  # Choose algorithm here
```

### Changing Configuration

```python
config = {
    'num_agents': 5,           # Number of agents
    'num_episodes': 1000,      # Training episodes
    'learning_rate': 0.001,    # Adjust learning speed
    'batch_size': 64,          # Larger = smoother but slower
    'gamma': 0.99,             # Discount factor (0-1)
    'epsilon_decay': 0.995,    # Exploration decay rate
}
```

### Custom Environment

Create your own environment following this interface:

```python
class CustomEnv:
    def reset(self):
        """Return initial observations (num_agents, obs_dim)"""
        pass
    
    def step(self, actions):
        """Execute actions, return obs, rewards, dones, info"""
        return observations, rewards, dones, info
    
    @property
    def observation_space_size(self):
        return obs_dim
    
    @property  
    def action_space_size(self):
        return action_dim
```

## Troubleshooting

### "Module not found" Error
```bash
# Make sure you're in the project root directory
cd MARL_CTDE_Industrial_Demo

# Install dependencies
pip install -r requirements.txt

# Test import
python -c "import torch; print(torch.__version__)"
```

### Training Diverges (Loss increases)
```python
# In config, reduce learning rate
'learning_rate': 0.0001,  # Reduce from 0.001

# Or increase batch size
'batch_size': 128,  # Increase from 64
```

### Memory Issues
```python
# Reduce buffer size
'buffer_size': 5000,  # Reduce from 10000

# Reduce batch size
'batch_size': 32,  # Reduce from 64

# Fewer agents
'num_agents': 3,  # Reduce from 5
```

### GPU/CUDA Issues
```bash
# Check if PyTorch can find GPU
python -c "import torch; print(torch.cuda.is_available())"

# If False, the code will automatically use CPU
# CPU is slower but still works fine for demos
```

## Learning Path

1. **Start here**: `python main.py --list` (see available demos)
2. **Run demo**: `python retail/demo.py` (quick 5-10 min run)
3. **Read theory**: `docs/MARL_THEORY.md` (understand concepts)
4. **Full training**: `python retail/train.py` (1-2 hours)
5. **Customize**: Modify environment or algorithm
6. **Experiment**: Change hyperparameters and observe results

## Expected Results

### Retail (Multi-Warehouse)
- Initial inventory costs: ~150-200 per episode
- After training: ~80-100 per episode
- Improvement: 40-50%
- Training time: ~5-10 minutes (demo), ~1 hour (full)

### Banking (Transaction Routing)  
- Initial latency: ~500ms
- After training: ~150-180ms
- Improvement: 65-70%
- Training time: ~2-5 minutes (demo), ~30 minutes (full)

## Next Steps

1. **Study CTDE**: Read `docs/MARL_THEORY.md`
2. **Understand algorithms**: Compare MADDPG, MAPPO, QMIX
3. **Modify environments**: Change warehouse network topology or channels
4. **Experiment with hyperparameters**: Learning rate, batch size, network size
5. **Implement custom domain**: Create your own MARL problem

## Getting Help

- **Theory questions**: See `docs/MARL_THEORY.md`
- **Algorithm details**: Check algorithm-specific .md files
- **Code questions**: Read docstrings in source files
- **Stuck?**: Review example demos and compare with your code

## Performance Tips

1. **GPU acceleration**: Install PyTorch with CUDA support for 10x speedup
2. **Larger networks**: Increase hidden layer dimensions for better learning
3. **More agents**: Scales well up to 10-20 agents with QMIX
4. **Longer training**: More episodes = better convergence (diminishing returns after 1000)
5. **Better hyperparameters**: Use grid search or random search for tuning

---

**Happy learning! 🚀**

For more information, see the main README.md and documentation in the docs/ folder.
