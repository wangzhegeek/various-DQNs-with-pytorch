import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import argparse

from ddqn_agent import Agent as Double_DQN_Agent
from dqn_agent import Agent as DQN_agent


def run(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores


def parse_args():
    parser = argparse.ArgumentParser(description="run DQN models in gym env.")
    parser.add_argument('-model', type=str, default='Dual_DQN',
                        help='DQN or Double_DQN or Dual_DQN or Dual_DDQN')
    parser.add_argument('-n_episodes', type=int, default=20000,
                        help='Number of episodes.')
    parser.add_argument('-max_t', type=int, default=1000,
                        help='max step in one episode.')
    parser.add_argument('-eps_start', type=float, default=1.0,
                        help='initial eps.')
    parser.add_argument('-eps_end', type=float, default=0.01,
                        help='min eps.')
    parser.add_argument('-eps_decay', type=float, default=0.995,
                        help='min eps.')
    # --------- learning args ----------------------#
    parser.add_argument('-buffer_size', type=int, default=int(1e5),
                        help='Number of samples in buff.')
    parser.add_argument('-batch_size', type=int, default=256,
                        help='batch size.')
    parser.add_argument('-gamma', type=float, default=0.99,
                        help='constant number for Q learning.')
    parser.add_argument('-tau', type=float, default=1e-3,
                        help='number of local network params moving to target network.')
    parser.add_argument('-lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('-update_steps', type=int, default=4,
                        help='step frequency for network updating')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    env = gym.make('LunarLander-v2')
    env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)

    if args.model == 'DQN':
        Agent = DQN_agent
        dual_network = False
    elif args.model == 'Double_DQN':
        Agent = Double_DQN_Agent
        dual_network = False
    elif args.model == 'Dual_DQN':
        Agent = DQN_agent
        dual_network = True
    else:
        Agent = Double_DQN_Agent
        dual_network = True

    agent = Agent(state_size=8, action_size=4, seed=0, lr=args.lr, buffer_size=args.buffer_size, batch_size=args.batch_size,
                      update_step=args.update_steps, gamma=args.gamma, tau=args.tau, dual_network=dual_network)

    scores = run(args.n_episodes, args.max_t, args.eps_start, args.eps_end, args.eps_decay)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    # load the weights from file
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

    for i in range(5):
        state = env.reset()
        for j in range(300):
            action = agent.act(state)
            env.render()
            state, reward, done, _ = env.step(action)
            if done:
                break
    env.close()
