import streamlit as st
import numpy as np
import torch
import sys
from pathlib import Path
import plotly.graph_objects as go
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.maddpg import create_maddpg_agents
from core.replay_buffer import SharedReplayBuffer
from core.utils import Logger, set_seed
from environment import MultiWarehouseEnv


def initialize_demo():
    set_seed(42)
    config = {
        'num_agents': 5,
        'num_episodes': 100,
        'max_steps_per_episode': 100,
        'batch_size': 32,
        'buffer_size': 5000,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'tau': 0.001,
        'epsilon': 1.0,
        'epsilon_decay': 0.998,
        'epsilon_min': 0.05,
        'warehouse_capacity': 1000.0,
        'demand_mean': 500.0,
    }

    env = MultiWarehouseEnv(
        num_warehouses=config['num_agents'],
        warehouse_capacity=config['warehouse_capacity'],
        demand_mean=config['demand_mean'],
        topology='line'
    )

    agents = create_maddpg_agents(
        state_dim=env.observation_space_size,
        action_dim=env.action_space_size,
        num_agents=config['num_agents'],
        config=config
    )

    replay_buffer = SharedReplayBuffer(
        buffer_size=config['buffer_size'],
        num_agents=config['num_agents'],
        state_dim=env.observation_space_size,
        action_dim=env.action_space_size,
        device=agents[0].device
    )

    st.session_state.config = config
    st.session_state.env = env
    st.session_state.agents = agents
    st.session_state.replay_buffer = replay_buffer
    st.session_state.train_rewards = []
    st.session_state.train_costs = []
    st.session_state.episode_count = 0
    st.session_state.initialized = True


def create_network_figure(env):
    # Create positions for 5 warehouses in a line
    positions = {i: (i * 2, 0) for i in range(5)}

    fig = go.Figure()

    # Add edges
    for i in range(4):
        fig.add_trace(go.Scatter(
            x=[positions[i][0], positions[i+1][0]],
            y=[positions[i][1], positions[i+1][1]],
            mode='lines',
            line=dict(width=3, color='gray'),
            showlegend=False
        ))

    # Add nodes
    for i in range(5):
        inventory_level = env.inventory[i] / env.warehouse_capacity
        # Color: red for low inventory, green for high
        r = int(255 * (1 - inventory_level))
        g = int(255 * inventory_level)
        color = f'rgb({r}, {g}, 0)'

        fig.add_trace(go.Scatter(
            x=[positions[i][0]],
            y=[positions[i][1]],
            mode='markers+text',
            marker=dict(size=50, color=color, line=dict(width=2, color='black')),
            text=[f'W{i}<br>Inv: {env.inventory[i]:.0f}<br>Dem: {env.demand[i]:.0f}'],
            textposition="bottom center",
            textfont=dict(size=10),
            showlegend=False
        ))

    fig.update_layout(
        title="Warehouse Network Status",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 9]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, 2]),
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def run_training_episodes(num_episodes):
    for episode in range(num_episodes):
        obs = st.session_state.env.reset()
        ep_reward = 0
        ep_cost = 0

        for step in range(st.session_state.config['max_steps_per_episode']):
            actions = np.array([agent.select_action(obs[i], training=True)
                              for i, agent in enumerate(st.session_state.agents)])

            next_obs, rewards, dones, info = st.session_state.env.step(actions)
            ep_reward += rewards.sum()
            ep_cost += info['total_cost']

            st.session_state.replay_buffer.add(obs, actions, rewards, next_obs, dones)

            obs = next_obs
            if dones[0]:
                break

        if st.session_state.replay_buffer.is_ready(st.session_state.config['batch_size']):
            batch = st.session_state.replay_buffer.sample(st.session_state.config['batch_size'])
            for agent in st.session_state.agents:
                agent.update(batch, st.session_state.agents)

        for agent in st.session_state.agents:
            agent.decay_exploration()

        st.session_state.train_rewards.append(ep_reward)
        st.session_state.train_costs.append(ep_cost)
        st.session_state.episode_count += 1


def run_evaluation():
    eval_costs = []
    eval_stockouts = []

    for episode in range(5):
        obs = st.session_state.env.reset()

        episode_cost = 0
        episode_stockouts = []

        while True:
            actions = np.array([agent.select_action(obs[i], training=False)
                              for i, agent in enumerate(st.session_state.agents)])

            next_obs, rewards, dones, info = st.session_state.env.step(actions)

            episode_cost += info['total_cost']
            episode_stockouts.extend(info['stockout'])

            obs = next_obs
            if dones[0]:
                break

        eval_costs.append(episode_cost)
        eval_stockouts.append(sum(episode_stockouts))

    return {
        'avg_cost': np.mean(eval_costs),
        'total_stockouts': np.sum(eval_stockouts)
    }


def main():
    st.title("🎯 MARL CTDE - Multi-Warehouse Inventory Demo")
    st.markdown("**Interactive demonstration of Multi-Agent Reinforcement Learning**")
    st.markdown("*Centralized Training with Decentralized Execution (CTDE) using MADDPG*")

    if st.button("🚀 Initialize Demo") or st.session_state.get('initialized', False):
        if not st.session_state.get('initialized', False):
            with st.spinner("Initializing demo..."):
                initialize_demo()
            st.success("Demo initialized!")

        # Layout
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📊 Warehouse Network")
            fig = create_network_figure(st.session_state.env)
            st.plotly_chart(fig, use_container_width=True, key="network")

            # Training curve
            if st.session_state.train_costs:
                st.subheader("📈 Training Progress")
                fig2 = go.Figure()
                episodes = list(range(1, len(st.session_state.train_costs) + 1))
                fig2.add_trace(go.Scatter(
                    x=episodes,
                    y=st.session_state.train_costs,
                    mode='lines+markers',
                    name='Episode Cost',
                    line=dict(color='red', width=2),
                    marker=dict(size=6)
                ))
                fig2.update_layout(
                    title="Training Cost Over Episodes",
                    xaxis_title="Episode",
                    yaxis_title="Total Cost",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig2, use_container_width=True, key="training")

        with col2:
            st.subheader("🎮 Controls")

            if st.button("🏃 Run 5 Training Episodes"):
                with st.spinner("Training..."):
                    run_training_episodes(5)
                st.success("Training completed!")
                st.rerun()

            if st.button("📊 Run Evaluation"):
                with st.spinner("Evaluating..."):
                    eval_results = run_evaluation()
                st.write("**📋 Evaluation Results:**")
                st.write(f"• Average Cost: **{eval_results['avg_cost']:.2f}**")
                st.write(f"• Total Stockouts: **{eval_results['total_stockouts']:.2f}**")

            if st.button("🔄 Reset Demo"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

            st.subheader("📈 Current Stats")
            if st.session_state.train_costs:
                st.metric("Episodes Trained", st.session_state.episode_count)
                st.metric("Latest Cost", f"{st.session_state.train_costs[-1]:.2f}")
                if len(st.session_state.train_costs) > 10:
                    initial_avg = np.mean(st.session_state.train_costs[:10])
                    recent_avg = np.mean(st.session_state.train_costs[-10:])
                    improvement = (1 - recent_avg / initial_avg) * 100
                    st.metric("Improvement", f"{improvement:.1f}%")

            st.subheader("🎯 Agent Status")
            if hasattr(st.session_state.agents[0], 'epsilon'):
                st.metric("Exploration (ε)", f"{st.session_state.agents[0].epsilon:.3f}")

    else:
        st.info("👆 Click '🚀 Initialize Demo' to start the interactive demonstration!")
        st.markdown("---")
        st.markdown("### 🎓 What you'll see:")
        st.markdown("- **Warehouse Network:** 5 warehouses connected in a line")
        st.markdown("- **Real-time Inventory:** Color-coded inventory levels")
        st.markdown("- **Training Progress:** Cost reduction over episodes")
        st.markdown("- **Interactive Controls:** Train agents, run evaluation, reset")


if __name__ == "__main__":
    main()