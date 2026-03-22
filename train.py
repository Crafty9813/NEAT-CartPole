import neat
import gymnasium as gym
import numpy as np
import pickle

env = gym.make("CartPole-v1")

def eval_genomes(genomes, config):
    EPISODES = 5

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = []

        for _ in range(EPISODES):
            observation, _ = env.reset()
            fitness.append(0)
            done = False

            while not done:
                action = net.activate(observation)
                action = 1 if action[0] > 0.5 else 0

                observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                fitness[-1] += reward

        genome.fitness = np.mean(fitness)


def run_neat(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_genomes, 50)

    print("\nBest genome:\n", winner)

    return winner


if __name__ == "__main__":
    winner = run_neat("config-feedforward.txt")

    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)