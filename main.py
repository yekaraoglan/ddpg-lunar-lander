from agent import Agent
from plot import plotLearning
import gym
import numpy as np
import argparse

def train(args):
    score_history = []
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            score += reward
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
        if i % args.model_save_freq == 0:
            agent.save_models(i)

    x = [i+1 for i in range(n_games)]
    plotLearning(x, score_history, filename=filename)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--n_games', type=int, default=1000)
    parser.add_argument('--filename', type=str, default='lunar_lander_continuous.png')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr_actor', type=float, default=0.000025)
    parser.add_argument('--lr_critic', type=float, default=0.00025)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--monitor_freq', type=int, default=25)
    parser.add_argument('--model_save_freq', type=int, default=25)
    parser.add_argument('--tau', type=float, default=0.001)

    args = parser.parse_args()

    env = gym.make('LunarLanderContinuous-v2')
    env = gym.wrappers.Monitor(env, "recording", video_callable=lambda episode_id: episode_id%args.monitor_freq==0, force=True)

    agent = Agent(alpha_a=args.lr_actor, alpha_c=args.lr_critic, input_dims=[8], tau=args.tau, env=env, gamma=args.discount,
                    batch_size=args.batch_size, layer1_size=400, layer2_size=300, n_actions=env.action_space.shape[0])

    n_games = args.n_games
    filename = args.filename

    if args.train:
        train(args)

    else:
        agent.load_models()
        env.render()
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation = observation_
            score += reward
        print('score: ', score)

    env.close()