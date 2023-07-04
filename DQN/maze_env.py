# -*- coding: utf-8 -*-
"""
@Time ： 2023/7/4 00:53
@Auth ： xiaolongtuan
@File ：maze_env.py
"""

import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
# 导入游戏图形界面标准库

UNIT = 40   # 单元格的像素
MAZE_H = 4  # 4格高
MAZE_W = 4  # 4格宽


class Maze(tk.Tk, object): # 游戏环境
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']  # 四种操作
        self.n_actions = len(self.action_space)  # 4
        self.n_features = 2
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):  # 游戏界面
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # 画迷宫格子，四条竖线和横线
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # model数据
        origin = np.array([20, 20])

        # 黑色矩形
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')

        # 黄色椭圆
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # 红色矩形，是我们控制移动的方块
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        self.canvas.pack()

    def reset(self):  # 重新开始游戏，红色方块回到起始点
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # 这里的observation的形状为(2,) 表示红方块距离黄圆块的距离
        observation = (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
        return observation

    # 与环境交互过程
    def step(self, action):
        s = self.canvas.coords(self.rect)  # 当前红方块
        base_action = np.array([0, 0])
        if action == 0:   # 上移一格
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # 下
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # 右
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # 左
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])

        next_coords = self.canvas.coords(self.rect)  # 移动后的新状态

        # 计算奖励
        if next_coords == self.canvas.coords(self.oval):  # 到达黄圆，奖励为1，结束
            reward = 1
            done = True
        elif next_coords in [self.canvas.coords(self.hell1)]:  # 到达黑方块，奖励-1，结束
            reward = -1
            done = True
        else:  # 奖励为0，继续游戏
            reward = 0
            done = False
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
        # 同样计算新状态的observation，(2,)
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        self.update()