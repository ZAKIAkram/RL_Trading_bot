import numpy as np
import matplotlib.pyplot as plt
# import gymnasium as gym
import gym
# from gymnasium import Env, spaces
from gym import Env, spaces
import pandas as pd
import random

INITIAL_BALANCE = 10000

class TradingEnv(Env):
  def __init__(self, data_path, name, initial_step=0, end_step=0, random_steps=False):
    #self.reset() maybe
    self.df = self.__read_df(data_path)
    self.action_space = spaces.Discrete(3)
    # self.action_space = spaces.Tuple((spaces.Discrete(3),
    #                                  spaces.Box(low=0, high=1, shape=(1,)))) # Buy, Sell, do Nothing
    self.observation_space = spaces.Box(low=0, high=100000, shape=(6, 6), dtype=np.float32)
    self.current_step = initial_step
    self.end_step = self.__clamp(end_step)
    self.random_steps = random_steps
    self.name = name


  def __read_df(self, data_path):
    df = pd.read_csv(data_path, quotechar='"')
    df = df.iloc[::-1]
    # self.df['Date'] = pd.to_datetime(self.df['Date'])
    # df[['Volume']] = df[['Volume']].applymap(lambda value: float(value.replace(",", "")))
    return df

  def reset(self):
    self.shares_held = 0
    self.balance = INITIAL_BALANCE
    self.current_step = 0
    if self.random_steps: 
      self.current_step = random.randint(0, self.df.shape[0] - 6)
      self.end_step = random.randint(self.current_step, len(self.df.loc[:, 'Open'].values - 6))
    self.net_worth = INITIAL_BALANCE
    self.shares_sold = 0
    self.total_shares_sold_value = 0.0
    self.last_price = random.uniform(
          self.df.loc[self.current_step, 'Open'],
          self.df.loc[self.current_step, 'Close']
      )
    return self.__getNextObs()

  def render(self):
    # print(f"balance : {self.balance}, portfolio : {self.shares_held * self.last_price}, net_worth : {self.net_worth}")
    print(f"net_worth : {self.net_worth}")

  def step(self, action):
    prev_net_worth = self.net_worth
    self.__take_action(action)
    self.current_step += 1
    if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
        self.current_step = 0
    reward = self.__compute_reward(action, prev_net_worth)
    done = self.net_worth <= 0 or self.current_step == self.end_step
    obs = self.__getNextObs()
    return obs, reward, done, {}

  def __getNextObs(self):
    frame = np.array([
        self.df.iloc[self.current_step: self.current_step + 6, self.df.columns.get_loc('Open')],
        self.df.iloc[self.current_step: self.current_step + 6, self.df.columns.get_loc('High')],
        self.df.iloc[self.current_step: self.current_step + 6, self.df.columns.get_loc('Low')],
        self.df.iloc[self.current_step: self.current_step + 6, self.df.columns.get_loc('Close')],
        self.df.iloc[self.current_step: self.current_step + 6, self.df.columns.get_loc('Volume')]
    ])
    obs = np.append(frame, [[
          self.balance,
          self.net_worth,
          self.shares_held,
          self.shares_sold,
          self.last_price,
          self.total_shares_sold_value
      ]], axis=0)
    return obs

  def __compute_reward(self, action, prev_net_worth):
    returns = (self.net_worth - prev_net_worth) / prev_net_worth
    reward = 0.0
    if action != 2:
      #penalize the bot for making a trade
      reward -= .0002
    reward += returns
    if returns == 0.0:
      reward -= .0005
    return reward
  def __take_action(self, action):
    #select a price randomly
    current_price = random.uniform(
          self.df.loc[self.current_step, 'Open'],
          self.df.loc[self.current_step, 'Close']
      )
    self.last_price = current_price
    # action, amount = action # here amount refers to the percentage of the account balance
    amount = .2
    if action == 0:
      #Buy
      #buy a certain amount of shares
      #how to calculate the amount of shares:
      quantity = amount * self.balance / current_price # how many oz of gold for example
      self.shares_held += quantity
      self.balance -= amount * self.balance
    elif action == 1:
      #Sell
      quantity = amount * self.shares_held
      self.shares_held -= quantity
      self.shares_sold += quantity
      self.total_shares_sold_value += quantity * current_price
      self.balance += quantity * current_price
    self.net_worth = self.balance + self.shares_held * current_price
  def __clamp(self, end_step):
    if 0 < end_step < self.df.shape[0] - 5:
      return end_step
    return self.df.shape[0] - 6