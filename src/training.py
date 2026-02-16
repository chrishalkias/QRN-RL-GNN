from base.flexible_agent import FlexibleQRNAgent

# Usage:
agent = FlexibleQRNAgent(lr=5e-4, gamma=0.95)
agent.train(episodes=1000, topology='chain', n_range=[4,6], use_wandb=False)
agent.validate(topology='ring', n_nodes=7)