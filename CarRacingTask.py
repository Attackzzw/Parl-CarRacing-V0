import os
import gym
import numpy as np

import paddle.fluid as fluid
import parl
from parl import layers
from parl.utils import logger
from parl.algorithms import PolicyGradient

LEARNING_RATE = 0.0008

class Model(parl.Model):
    def __init__(self, act_dim):
        act_dim = act_dim
        hid1_size = act_dim * 10

        self.fc1 = layers.fc(size=hid1_size, act='tanh')
        self.fc2 = layers.fc(size=act_dim, act='softmax')

    def forward(self, obs):  # 可直接用 model = Model(5); model(obs)调用
        out = self.fc1(obs)
        out = self.fc2(out)
        return out


class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.act_prob = self.alg.predict(obs)

        with fluid.program_guard(
                self.learn_program):  # 搭建计算图用于 更新policy网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(name='act',shape=[1], dtype='int64') 
            reward = layers.data(name='reward', shape=[], dtype='float32')
            self.cost = self.alg.learn(obs, act, reward)

    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0)  # 增加一维维度
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]                                   #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        act=[0.0,0.0,0.0]
        act[0] = np.random.choice(np.array([-0.2,0,0.2]), p=(act_prob[0]))  # 根据动作概率选取动作range(self.act_dim)
        act[1] = np.random.choice(np.array([1,1,2]), p=(act_prob[0]))  # p=(act_prob[0]) p规定选取a中每个元素的概率，默认为概率相同
        act[2] = np.random.choice(np.array([0,0.3,0.7]), p=(act_prob[0]))  # 转向、油门、刹车
        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.argmax(act_prob)  # 根据动作概率选择概率最高的动作
        return act

    def learn(self, obs, act, reward):
        # act = np.expand_dims(act, axis=0)   #(1, 1000, 3)
        print('learn:act.shape',act.shape)#(1000, 3)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int64'),
            'reward': reward.astype('float32')
        }
        print('learn_self.cost',self.cost)
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, 
            fetch_list=[self.cost])[0]  #<<<<<<<<<<<<<<<<<<<<<<<
        return cost

def run_episode(env, agent):
    # print('run_episode_is_open')
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    i=0
    while True:
        # env.render()    
        obs_list.append(obs)
        action = agent.sample(obs) # 采样动作env.action_space.sample()     <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)
        # print(sum(reward_list))
        if done:
            break
    # print('run_episode_is_close')
    return obs_list, action_list, reward_list

# 评估 agent, 跑 5 个episode，求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            obs = preprocess(obs) # from shape (210, 160, 3) to (100800,)
            action = agent.predict(obs) # 选取最优动作
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


# 根据一个episode的每个step的reward列表，计算每一个Step的Gt
def calc_reward_to_go(reward_list, gamma=0.99):
    """calculate discounted reward"""
    reward_arr = np.array(reward_list)
    for i in range(len(reward_arr) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_arr[i] += gamma * reward_arr[i + 1]
    # normalize episode rewards
    reward_arr -= np.mean(reward_arr)
    reward_arr /= np.std(reward_arr)
    return reward_arr

if __name__ =='__main__':
    # 创建环境
    env = gym.make('CarRacing-v0')
    obs_dim = 96*96*3 #env.observation_space.shape[0] #96*96*3
    act_dim = env.action_space.shape[0]  #转向、油门、刹车
    print(act_dim)
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))   # obs_dim 288, act_dim 3

    model = Model(act_dim=act_dim)
    alg = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)
    run_episode(env,agent)
    # 加载模型
    if os.path.exists('./model.ckpt'):
        agent.restore('./model.ckpt')
    for i in range(30):
        print(i)
        obs_list, action_list, reward_list = run_episode(env, agent)  #<<<<<<<<<<<<<<<<<<<
        print("Train Episode {}, Reward Sum {}.".format(i, sum(reward_list)))
        if i % 10 == 0:
            logger.info("Train Episode {}, Reward Sum {}.".format(i, sum(reward_list)))
        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_reward_to_go(reward_list)

        # agent.learn(batch_obs, batch_action, batch_reward)  #<<<<<<<<<<<<<<<<<<<<<<<
        if (i + 1) % 100 == 0:
            total_reward = evaluate(env, agent, render=False)
            logger.info('Episode {}, Test reward: {}'.format(i + 1, 
                                                total_reward))
    # save the parameters to ./model.ckpt-
    agent.save('./model.ckpt')


#轨道随机生成