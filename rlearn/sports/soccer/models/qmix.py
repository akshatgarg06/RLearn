import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time

class MixingNetwork(nn.Module):
    """QMIX mixer with hypernetworks for RoboCup2D"""
    def __init__(self, n_agents, state_dim, hyper_hidden=64, mixing_hidden=32):
        super().__init__()
        self.n_agents = n_agents
        
        # Hypernetwork for weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, n_agents * mixing_hidden)
        )
        
        # Hypernetwork for biases
        self.hyper_b1 = nn.Linear(state_dim, mixing_hidden)
        
        # Final layer hypernetworks
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, mixing_hidden)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, 1)
        )

    def forward(self, agent_qs, states):
        bs = states.size(0)
        
        # Generate weights and biases
        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states))
        w1 = w1.view(bs, self.n_agents, -1)
        b1 = b1.view(bs, 1, -1)
        
        # First layer mixing
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = F.relu(hidden)
        
        # Second layer
        w2 = torch.abs(self.hyper_w2(states)).view(bs, -1, 1)
        b2 = self.hyper_b2(states)).view(bs, 1, 1)
        
        # Final output
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.squeeze(1)

class QMIX:
    """Complete QMIX implementation with training logic"""
    
    def __init__(self, agent_network, config, log_dir=None, device="auto"):
        self.device = self._get_device(device)
        self.config = config
        
        # Initialize networks
        self.agent_net = agent_network.to(self.device)
        self.mixer_net = MixingNetwork(
            n_agents=config["n_agents"],
            state_dim=config["state_dim"],
            hyper_hidden=config.get("hyper_hidden", 64),
            mixing_hidden=config.get("mixing_hidden", 32)
        ).to(self.device)
        
        # Initialize trainer
        self.trainer = QMIXTrainer(
            agent_net=self.agent_net,
            mixer_net=self.mixer_net,
            config=config,
            log_dir=log_dir,
            device=self.device
        )
    
    def _get_device(self, device_str):
        """Determine the best available device"""
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)
    
    def train(self, batch):
        """Perform one training step"""
        return self.trainer.update(batch)
    
    def save(self, path):
        """Save model weights"""
        torch.save({
            "agent_net": self.agent_net.state_dict(),
            "mixer_net": self.mixer_net.state_dict(),
            "config": self.config
        }, path)
    
    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent_net.load_state_dict(checkpoint["agent_net"])
        self.mixer_net.load_state_dict(checkpoint["mixer_net"])
    
    def close(self):
        """Clean up resources"""
        self.trainer.close()

class QMIXTrainer:
    """QMIX training logic with TensorBoard logging"""
    
    def __init__(self, agent_net, mixer_net, config, log_dir=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Networks
        self.agent_net = agent_net
        self.mixer_net = mixer_net
        
        # Target networks
        self.target_agent_net = type(agent_net)(*agent_net.init_args).to(self.device)
        self.target_mixer_net = type(mixer_net)(*mixer_net.init_args).to(self.device)
        self._hard_update()
        
        # Optimizer
        params = list(self.agent_net.parameters()) + list(self.mixer_net.parameters())
        self.optimizer = optim.Adam(params, lr=config["lr"])

        # Training parameters
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.n_agents = config["n_agents"]
        self.grad_norm_clip = config.get("grad_norm_clip", 10.0)
        
        # Logging setup
        self._init_logging(log_dir)
        
        # Training stats
        self.global_step = 0
        self.start_time = time.time()
        self.loss_history = []
        self.q_history = []

    def _hard_update(self):
        """Initialize target networks with same weights"""
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer_net.load_state_dict(self.mixer_net.state_dict())

    def _init_logging(self, log_dir):
        """Initialize TensorBoard logging"""
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join("runs", f"qmix_{timestamp}")
        
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"Logging to: {log_dir}")

    def update(self, batch):
        """Perform one training update"""
        # Prepare batch
        states = batch["states"].float().to(self.device)
        obs = batch["obs"].float().to(self.device)
        actions = batch["actions"].long().to(self.device)
        rewards = batch["rewards"].float().to(self.device)
        next_states = batch["next_states"].float().to(self.device)
        next_obs = batch["next_obs"].float().to(self.device)
        dones = batch["dones"].float().to(self.device)

        # Compute Q values
        q_total = self._compute_q_values(obs, states, actions)
        
        # Compute targets
        with torch.no_grad():
            target_q = self._compute_targets(next_obs, next_states, rewards, dones)
        
        # Optimize
        loss = F.mse_loss(q_total, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.agent_net.parameters(), self.grad_norm_clip)
        torch.nn.utils.clip_grad_norm_(self.mixer_net.parameters(), self.grad_norm_clip)
        
        self.optimizer.step()
        
        # Soft update targets
        self._soft_update()
        
        # Logging
        self._log_stats(loss, q_total, target_q)
        
        return loss.item()

    def _compute_q_values(self, obs, states, actions):
        """Compute Q values through agent and mixer networks"""
        B, T, N, O = obs.shape
        S = states.size(-1)
        
        # Process through agent network
        obs_flat = obs.view(B * T * N, O)
        h0 = self.agent_net.init_hidden(B * N).to(self.device)
        q_flat, _ = self.agent_net(obs_flat, h0)
        
        # Select action Q values
        actions_flat = actions.view(B * T * N, 1)
        q_taken_flat = q_flat.gather(1, actions_flat).squeeze(1)
        agent_qs = q_taken_flat.view(B * T, N)
        
        # Mix Q values
        states_flat = states.view(B * T, S)
        return self.mixer_net(agent_qs, states_flat)

    def _compute_targets(self, next_obs, next_states, rewards, dones):
        """Compute target Q values using target networks"""
        B, T, N, O = next_obs.shape
        
        # Process through target agent network
        next_obs_flat = next_obs.view(B * T * N, O)
        h0 = self.target_agent_net.init_hidden(B * N).to(self.device)
        q_next_flat, _ = self.target_agent_net(next_obs_flat, h0)
        max_q_next_flat = q_next_flat.max(dim=1)[0]
        max_q_next = max_q_next_flat.view(B * T, N)
        
        # Mix target Q values
        next_states_flat = next_states.view(B * T, next_states.size(-1))
        target_q_total = self.target_mixer_net(max_q_next, next_states_flat)
        
        # Compute TD target
        rewards_flat = rewards.view(B * T)
        dones_flat = dones.view(B * T)
        return rewards_flat + self.gamma * (1 - dones_flat) * target_q_total

    def _soft_update(self):
        """Soft update target networks"""
        for target_param, param in zip(self.target_agent_net.parameters(), self.agent_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_mixer_net.parameters(), self.mixer_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def _log_stats(self, loss, q_values, targets):
        """Log training statistics"""
        self.writer.add_scalar("Loss/td_error", loss.item(), self.global_step)
        self.writer.add_scalar("Q/average_q", q_values.mean().item(), self.global_step)
        self.writer.add_scalar("Q/target_q", targets.mean().item(), self.global_step)
        
        # Store history
        self.loss_history.append(loss.item())
        self.q_history.append(q_values.mean().item())
        
        # Print periodically
        if self.global_step % 100 == 0:
            elapsed = time.time() - self.start_time
            print(f"Step {self.global_step}: Loss={loss.item():.4f}, "
                  f"Q={q_values.mean().item():.4f}, "
                  f"Target={targets.mean().item():.4f}, "
                  f"Time={elapsed:.2f}s")
        
        self.global_step += 1

    def close(self):
        """Clean up resources"""
        self.writer.close()
        print(f"\nTraining completed in {time.time() - self.start_time:.2f} seconds")
        print(f"Final average loss: {sum(self.loss_history)/len(self.loss_history):.4f}")
