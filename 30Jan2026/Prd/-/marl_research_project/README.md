# 🚀 Multi-Agent Reinforcement Learning Research Project

## Complete Production-Ready Setup for MARL Research

This project provides everything you need to run **CTDE**, **MADDPG**, **QMIX**, and other MARL algorithms in a production-like environment for research purposes.

---

## 📋 Table of Contents

1. [Quick Start (5 minutes)](#-quick-start-5-minutes)
2. [Detailed Setup Options](#-detailed-setup-options)
3. [Running Experiments](#-running-experiments)
4. [Available Algorithms](#-available-algorithms)
5. [Environments](#-environments)
6. [Production Deployment](#-production-deployment)
7. [Experiment Tracking](#-experiment-tracking)
8. [Troubleshooting](#-troubleshooting)

---

## ⚡ Quick Start (5 minutes)

### Option A: Minimal Setup (Recommended for beginners)

```bash
# 1. Create environment
conda create -n marl python=3.10 -y
conda activate marl

# 2. Install essentials
pip install torch pettingzoo[all] agilerl wandb matplotlib tqdm

# 3. Run your first MADDPG experiment!
python scripts/maddpg_from_scratch.py
```

### Option B: Full Research Setup

```bash
# Run the automated setup script
chmod +x setup.sh
./setup.sh
```

---

## 🛠 Detailed Setup Options

### Option 1: PettingZoo + AgileRL (Easiest - MADDPG)

**Best for:** Learning, quick experiments, continuous actions

```bash
# Install
pip install pettingzoo[all] agilerl supersuit gymnasium

# Verify
python -c "from pettingzoo.mpe import simple_spread_v3; print('✓ Ready!')"
```

**Run MADDPG:**
```bash
python scripts/train_maddpg.py --env simple_spread --episodes 5000
```

---

### Option 2: PyMARL + SMAC (QMIX on StarCraft)

**Best for:** QMIX research, StarCraft experiments, value decomposition

```bash
# Clone PyMARL2 (optimized version)
git clone https://github.com/hijkzzz/pymarl2.git
cd pymarl2

# Install dependencies
conda create -n pymarl python=3.8 -y
conda activate pymarl
bash install_dependecies.sh  # Downloads StarCraft II automatically

# Run QMIX training
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3m
```

**StarCraft Maps:**
| Map | Agents | Difficulty |
|-----|--------|------------|
| `3m` | 3 | Easy |
| `8m` | 8 | Easy |
| `2s3z` | 5 | Medium |
| `3s5z` | 8 | Hard |
| `27m_vs_30m` | 27 | Super Hard |

---

### Option 3: EPyMARL (Extended PyMARL)

**Best for:** Research benchmarking, multiple algorithms

```bash
# Clone EPyMARL
git clone https://github.com/uoe-agents/epymarl.git
cd epymarl
pip install -r requirements.txt

# Run MAPPO
python src/main.py --config=mappo --env-config=gymma \
    with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v3"
```

---

### Option 4: Ray RLlib (Production Scale)

**Best for:** Distributed training, production deployment

```bash
# Install
pip install "ray[rllib]" torch pettingzoo[all]

# Run distributed training
python scripts/rllib_mappo.py
```

---

### Option 5: TorchRL (Cutting-Edge Research)

**Best for:** Custom research, PyTorch integration

```bash
# Install
pip install torchrl vmas "pettingzoo[mpe]==1.24.3"

# Run
python scripts/torchrl_maddpg.py
```

---

## 🎯 Running Experiments

### MADDPG on PettingZoo Environments

```bash
# Simple Spread (3 cooperative agents)
python scripts/train_maddpg.py --env simple_spread --episodes 5000

# Simple Tag (predator-prey)
python scripts/train_maddpg.py --env simple_tag --episodes 10000

# With Weights & Biases logging
python scripts/train_maddpg.py --env simple_spread --wandb
```

### QMIX on StarCraft

```bash
cd pymarl2

# Easy scenarios
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3m

# Hard scenarios
python3 src/main.py --config=qmix --env-config=sc2 with env_args.map_name=3s5z

# With TensorBoard
python3 src/main.py --config=qmix --env-config=sc2 \
    with env_args.map_name=3m use_tensorboard=True save_model=True
```

### Run the From-Scratch Demo

```bash
# No external dependencies needed!
python scripts/maddpg_from_scratch.py
```

---

## 🤖 Available Algorithms

| Algorithm | Type | Framework | Best For |
|-----------|------|-----------|----------|
| **MADDPG** | Actor-Critic | AgileRL, TorchRL | Continuous actions, mixed settings |
| **QMIX** | Value Decomposition | PyMARL | Cooperative, discrete actions |
| **VDN** | Value Decomposition | PyMARL | Simple cooperative tasks |
| **MAPPO** | Policy Gradient | EPyMARL, RLlib | General cooperative |
| **IPPO** | Policy Gradient | EPyMARL | Independent learning |
| **COMA** | Actor-Critic | PyMARL | Credit assignment |

---

## 🎮 Environments

### PettingZoo MPE (Multi-Agent Particle Environment)

| Environment | Agents | Type | Actions |
|-------------|--------|------|---------|
| `simple_spread` | 3 | Cooperative | Continuous |
| `simple_tag` | 4 | Mixed | Continuous |
| `simple_adversary` | 3 | Competitive | Continuous |
| `simple_push` | 2 | Competitive | Continuous |

### SMAC (StarCraft Multi-Agent Challenge)

| Map | Agents | Difficulty |
|-----|--------|------------|
| `3m` | 3 Marines | Easy |
| `2s3z` | 5 (Stalkers + Zealots) | Medium |
| `3s5z` | 8 | Hard |
| `corridor` | 6 | Super Hard |

---

## 🏭 Production Deployment

### Docker Setup

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Run training
CMD ["python", "scripts/train_maddpg.py", "--env", "simple_spread"]
```

**Build and run:**
```bash
docker build -t marl-research .
docker run --gpus all marl-research
```

### Distributed Training with Ray

```python
import ray
from ray import tune

# Start Ray cluster
ray.init(address="auto")  # Or "ray://head-ip:10001"

# Run distributed experiment
tune.run(
    "PPO",
    config=config,
    num_samples=10,
    resources_per_trial={"cpu": 4, "gpu": 1}
)
```

### Model Export for Deployment

```python
import torch

# Save only the actor (policy) for deployment
torch.save(maddpg.actors[0].state_dict(), "agent_0_policy.pt")

# Export to ONNX for cross-platform
dummy_input = torch.randn(1, obs_dim)
torch.onnx.export(maddpg.actors[0], dummy_input, "agent_0.onnx")
```

---

## 📊 Experiment Tracking

### Weights & Biases

```bash
# Setup
wandb login

# Use in training
python scripts/train_maddpg.py --wandb
```

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir=runs/

# View at http://localhost:6006
```

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch_size` |
| Training doesn't converge | Lower learning rate, increase exploration |
| SMAC installation fails | Check `SC2PATH` environment variable |
| Import errors | Check Python version (3.8-3.10 recommended) |

### Debug Commands

```python
# Check environment
env = simple_spread_v3.parallel_env()
env.reset()
print(f"Agents: {env.agents}")
print(f"Obs space: {env.observation_space('agent_0')}")

# Check GPU
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

---

## 📚 Resources

### Free Books
- **MARL Book (FREE PDF)**: https://www.marl-book.com/
- **Sutton & Barto (FREE PDF)**: http://incompleteideas.net/book/the-book-2nd.html

### GitHub Repositories
- **PyMARL**: https://github.com/oxwhirl/pymarl
- **PyMARL2**: https://github.com/hijkzzz/pymarl2
- **EPyMARL**: https://github.com/uoe-agents/epymarl
- **MARL-code-pytorch**: https://github.com/Lizhi-sjtu/MARL-code-pytorch
- **MARLToolkit**: https://github.com/jianzhnie/deep-marl-toolkit

### Documentation
- **PettingZoo**: https://pettingzoo.farama.org/
- **Ray RLlib**: https://docs.ray.io/en/latest/rllib/
- **TorchRL MARL**: https://pytorch.org/rl/tutorials/

---

## 📁 Project Structure

```
marl_research_project/
├── scripts/
│   ├── train_maddpg.py          # Main MADDPG training script
│   ├── maddpg_from_scratch.py   # Complete implementation demo
│   └── rllib_mappo.py           # RLlib distributed training
├── configs/
│   └── default.yaml             # Hyperparameter configurations
├── environments/
│   └── custom_env.py            # Custom environment templates
├── models/
│   └── checkpoints/             # Saved model weights
├── results/
│   └── plots/                   # Training curves
├── requirements.txt
├── setup.sh
├── Dockerfile
└── README.md
```

---

## 🎓 Learning Path

1. **Start**: Run `maddpg_from_scratch.py` to understand core concepts
2. **Explore**: Train on different PettingZoo environments
3. **Scale**: Try PyMARL + SMAC for QMIX experiments
4. **Deploy**: Use Ray RLlib for distributed training
5. **Research**: Implement custom algorithms with EPyMARL

---

## 📝 Citation

If you use this code for research, please cite the relevant papers:

```bibtex
@article{lowe2017multi,
  title={Multi-agent actor-critic for mixed cooperative-competitive environments},
  author={Lowe, Ryan and Wu, Yi I and Tamar, Aviv and Harb, Jean and Abbeel, OpenAI Pieter and Mordatch, Igor},
  journal={Advances in neural information processing systems},
  year={2017}
}

@article{rashid2018qmix,
  title={QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning},
  author={Rashid, Tabish and Samvelyan, Mikayel and Schroeder, Christian and Farquhar, Gregory and Foerster, Jakob and Whiteson, Shimon},
  journal={International Conference on Machine Learning},
  year={2018}
}
```

---

**Happy researching! 🚀**
