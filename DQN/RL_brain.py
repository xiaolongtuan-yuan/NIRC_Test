# -*- coding: utf-8 -*-
"""
@Time ： 2023/7/4 00:55
@Auth ： xiaolongtuan
@File ：RL_brain.py
"""

"""
Deep Q Network off-policy
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import glob
SUMMARY_DIR = './results/models'
LOG_DIR = './results/logs'

np.random.seed(42)
torch.manual_seed(2)
write_test = SummaryWriter(LOG_DIR)

class Network(nn.Module):
    """
    Network Structure
    两个结构一样的全连接网络，Q网络
    """

    def __init__(self,
                 n_features,
                 n_actions,
                 n_neuron=10
                 ):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=n_neuron, bias=True),
            nn.Linear(in_features=n_neuron, out_features=n_actions, bias=True),
            nn.ReLU()
        )

    def forward(self, s):
        """
        :param s: s
        :return: q
        """
        q = self.net(s)
        return q


class DeepQNetwork(nn.Module):
    """
    Q Learning Algorithm
    """

    def __init__(self,
                 n_actions,  # 动作的维度，4
                 n_features,  # 网络的输入，即observation的维度，2
                 learning_rate=0.01,
                 reward_decay=0.9,  # 奖励递减，β
                 e_greedy=0.9,  # 表示是否使用Actor决策动作的概率，ε
                 replace_target_iter=300,  # 表示间隔多少步后更行target Q网络
                 memory_size=500,  # 记忆池容量
                 batch_size=32,
                 e_greedy_increment=None):
        super(DeepQNetwork, self).__init__()

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # 记录总的学习步数
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        # 这里用pd.DataFrame创建的表格作为memory
        # 表格的行数是memory的大小，也就是transition的个数
        # 表格的列数是transition的长度，一个transition包含[s, a, r, s_]，其中a和r分别是一个数字，s和s_的长度分别是n_features
        self.memory = pd.DataFrame(np.zeros((self.memory_size, self.n_features * 2 + 2)))

        # build two network: eval_net and target_net
        self.eval_net = Network(n_features=self.n_features, n_actions=self.n_actions)
        self.target_net = Network(n_features=self.n_features, n_actions=self.n_actions)

        # 获取文件夹下所有以.pkl结尾的文件，并按创建时间进行排序
        model_files = sorted(glob.glob(os.path.join(SUMMARY_DIR, '*.pkl')), key=os.path.getctime)
        # 检查是否存在.pkl结尾的文件
        if model_files:
            # 获取最新创建的文件路径
            latest_model_file = model_files[-1]
            # 加载模型权重
            self.eval_net.load_state_dict(torch.load(latest_model_file))
            self.target_net.load_state_dict(torch.load(latest_model_file))

            file_name = os.path.basename(latest_model_file)  # 获取文件名
            file_name = os.path.splitext(file_name)[0]  # 去除扩展名
            digits = int(''.join(filter(str.isdigit, file_name)))  # 提取数字
            self.learn_step_counter = digits

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

    # 向记忆池中添加或替换记录(s, a, r, s_)
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            # hasattr用于判断对象是否包含对应的属性。
            self.memory_counter = 0

        # np.hstack() 将参数元组的元素数组按水平方向进行叠加，最终得到的是一个维度为10的向量
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory.iloc[index, :] = transition

        self.memory_counter += 1

    # 选择动作，根据eval_net Q网络得到决策函数，取其中最大值作为当前状态下的动作
    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            s = torch.FloatTensor(observation)
            actions_value = self.eval_net(s)
            action = [np.argmax(actions_value.detach().numpy())][0]
        else:
            action = np.random.randint(0, self.n_actions)  # (1-ε)概率会随机选择动作
        return action

    def _replace_target_params(self):
        # 将evaluate 网络权重复制target网络参数
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:  # 更行target网络权重
            self._replace_target_params()
            print('\ntarget params replaced\n')

        # 从记忆池中取样batch_size个作为一个batch进行训练，这里注意当记忆池中未满时，不要在全池中取样（有些数据为0），应该
        # 从0到memory_counter条中取样
        batch_memory = self.memory.sample(self.batch_size) \
            if self.memory_counter > self.memory_size \
            else self.memory.iloc[:self.memory_counter].sample(self.batch_size, replace=True)

        # run the nextwork
        s = torch.FloatTensor(batch_memory.iloc[:, :self.n_features].values)  # 前4个数据为s
        s_ = torch.FloatTensor(batch_memory.iloc[:, -self.n_features:].values)  # 后四个数据为_s,下一个状态
        q_eval = self.eval_net(s)  # Q估计值
        q_next = self.target_net(s_)  # 下一状态的Q值

        # change q_target w.r.t q_eval's action
        q_target = q_eval.clone()

        # 更新值
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory.iloc[:, self.n_features].values.astype(int)  # a
        reward = batch_memory.iloc[:, self.n_features + 1].values  # r

        q_target[batch_index, eval_act_index] = torch.FloatTensor(reward) + self.gamma * q_next.max(dim=1).values
        #  _r = r + γQ(_s,_a)

        # 梯度下降
        loss = self.loss_function(q_target, q_eval)  # r <-> _r
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        write_test.add_scalar("Loss",loss.item(),self.learn_step_counter)

        # 这个比例值会慢慢增加到设定的最大值，这里不增加直接为0.9
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
