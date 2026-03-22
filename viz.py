import pickle
import neat
import gymnasium as gym

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "config-feedforward.txt"
)

# Load trained genome
with open("winner.pkl", "rb") as f:
    winner = pickle.load(f)

net = neat.nn.FeedForwardNetwork.create(winner, config)

env = gym.make("CartPole-v1", render_mode="human")
obs, _ = env.reset()

done = False
while not done:
    action = 1 if net.activate(obs)[0] > 0.5 else 0
    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated