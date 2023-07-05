# -*- coding: utf-8 -*-
"""
@Time ： 2023/7/5 16:12
@Auth ： xiaolongtuan
@File ：run.py
"""

import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from RL_brain import ReplayBuffer, SAC
from torch.utils.tensorboard import SummaryWriter
import os
import glob

SUMMARY_DIR = './results/models'
LOG_DIR = './results/logs'
write_test = SummaryWriter(LOG_DIR)

# -------------------------------------- #
# 参数设置
# -------------------------------------- #

num_epochs = 100  # 训练回合数
capacity = 500  # 经验池容量
min_size = 200  # 经验池训练容量
batch_size = 64
n_hiddens = 64
actor_lr = 1e-3  # 策略网络学习率
critic_lr = 1e-2  # 价值网络学习率
alpha_lr = 1e-2  # 课训练变量的学习率
target_entropy = -1
tau = 0.005  # 软更新参数
gamma = 0.9  # 折扣因子
device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')

# -------------------------------------- #
# 环境加载
# -------------------------------------- #

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="human")
n_states = env.observation_space.shape[0]  # 状态数 4
n_actions = env.action_space.n  # 动作数 2

# -------------------------------------- #
# 模型构建
# -------------------------------------- #

agent = SAC(n_states=n_states,
            n_hiddens=n_hiddens,
            n_actions=n_actions,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            target_entropy=target_entropy,
            tau=tau,
            gamma=gamma,
            device=device,
            writter=write_test
            )

# 获取文件夹下所有以.pkl结尾的文件，并按创建时间进行排序
actor_model_files = sorted(glob.glob(os.path.join(SUMMARY_DIR, 'actor*.pkl')), key=os.path.getctime)
critic1_model_files = sorted(glob.glob(os.path.join(SUMMARY_DIR, 'critic1_net*.pkl')), key=os.path.getctime)
critic2_model_files = sorted(glob.glob(os.path.join(SUMMARY_DIR, 'critic2_net*.pkl')), key=os.path.getctime)
target_critic1_model_files = sorted(glob.glob(os.path.join(SUMMARY_DIR, 'target_critic_1*.pkl')), key=os.path.getctime)
target_critic2_model_files = sorted(glob.glob(os.path.join(SUMMARY_DIR, 'target_critic_2*.pkl')), key=os.path.getctime)
index = 0
# 检查是否存在.pkl结尾的文件
if actor_model_files:
    # 获取最新创建的文件路径
    latest_actor_model_file = actor_model_files[-1]
    latest_critic1_model_file = critic1_model_files[-1]
    latest_critic2_model_file = critic2_model_files[-1]
    latest_target_critic1_model_file = target_critic1_model_files[-1]
    latest_target_critic2_model_file = target_critic2_model_files[-1]
    # 加载模型权重
    agent.actor.load_state_dict(torch.load(latest_actor_model_file))
    agent.critic_1.load_state_dict(torch.load(latest_critic1_model_file))
    agent.critic_2.load_state_dict(torch.load(latest_critic2_model_file))
    agent.target_critic_1.load_state_dict(torch.load(latest_target_critic1_model_file))
    agent.target_critic_2.load_state_dict(torch.load(latest_target_critic2_model_file))

    file_name = os.path.basename(latest_actor_model_file)  # 获取文件名
    file_name = os.path.splitext(file_name)[0]  # 去除扩展名
    digits = int(''.join(filter(str.isdigit, file_name)))  # 提取数字
    index = digits

# -------------------------------------- #
# 经验回放池
# -------------------------------------- #

buffer = ReplayBuffer(capacity=capacity)

# -------------------------------------- #
# 模型构建
# -------------------------------------- #

while True:
    state = env.reset()[0]
    epochs_return = 0  # 累计每个时刻的reward
    done = False  # 回合结束标志

    while not done:
        # 动作选择
        action = agent.take_action(state)
        # 环境更新
        next_state, reward, done, _, _ = env.step(action)
        # 将数据添加到经验池
        buffer.add(state, action, reward, next_state, done)
        # 状态更新
        state = next_state
        # 累计回合奖励
        epochs_return += reward

        # 经验池超过要求容量，就开始训练
        if buffer.size() > min_size:
            s, a, r, ns, d = buffer.sample(batch_size)  # 每次取出batch组数据
            # 构造数据集
            transition_dict = {'states': s,
                               'actions': a,
                               'rewards': r,
                               'next_states': ns,
                               'dones': d}
            # 模型训练
            agent.update(transition_dict)
    # 保存每个回合return
    write_test.add_scalar("reward", reward, index)

    if index % 100 == 0:
        torch.save(agent.actor.state_dict(), SUMMARY_DIR + "/actor_net" +
                   str(index) + ".pkl")
        torch.save(agent.critic_1.state_dict(), SUMMARY_DIR + "/critic1_net" +
                   str(index) + ".pkl")
        torch.save(agent.critic_2.state_dict(), SUMMARY_DIR + "/critic2_net" +
                   str(index) + ".pkl")
        torch.save(agent.target_critic_1.state_dict(), SUMMARY_DIR + "/target_critic_1" +
                   str(index) + ".pkl")
        torch.save(agent.target_critic_2.state_dict(), SUMMARY_DIR + "/target_critic_2" +
                   str(index) + ".pkl")
    index += 1