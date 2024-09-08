import gym  
from gym import spaces  
import numpy as np  
import pandas as pd  
import random
from gym.envs.registration import register
import matplotlib.pyplot as plt
import os

def z_score_normalize(arr):  
    mean = np.mean(arr)  
    std_dev = np.std(arr) + 1e-7
    normalized_arr = (arr - mean) / std_dev 
    return normalized_arr

def mm_normalize(arr):
    # 对数变换  
    log_transformed_data = np.log1p(arr)  

    # 最小-最大标准化  
    min_val = np.min(log_transformed_data)  
    max_val = np.max(log_transformed_data)  

    # 避免除以零的情况  
    if max_val - min_val != 0:  
        normalized_data = (log_transformed_data - min_val) / (max_val - min_val)  
    else:  
        normalized_data = np.zeros_like(log_transformed_data)  

    return normalized_data

class GoldTrader(gym.Env):  
    def __init__(self):  
        super(GoldTrader, self).__init__()  
        # Load the CSV file  
        self.csv = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Au9999price.csv'))
        self.prices = self.csv['收盘价'].values  # 假设CSV文件中有一个'price'列  
        self.date = self.csv['date']
        
        self.action_space = spaces.Discrete(3) # bug wait ail
        self.observation_space = spaces.Box(low=-1, high=np.inf, shape=(40+31,), dtype=np.float32)  # 超过30天的gold数量 + 30天内的gold数量 + 现金 + 黄金 + 过去40天价格

        self.init_balance = 10000
        
    def reset(self, seed=None, options=None):

        if seed is not None:  
            np.random.seed(seed)  
            random.seed(seed)  

        self.balance = self.init_balance  # 初始现金余额  
        self.gold_trade = np.zeros((31,))  # 初始黄金数量
        self.day =  random.randint(41, 300)
        self.yesterday_action = 0  # + 昨天买入的价格 - 昨天卖出的数量

        recent_prices = self.prices[self.day-41:self.day]  
        obs_price_changes = np.diff(recent_prices) / recent_prices[:-1]
        obs_gold_trade = z_score_normalize(self.gold_trade)
        obs = np.concatenate((obs_price_changes, obs_gold_trade)).astype(np.float32)

        self.history = []
        self.obs = obs
        info = {'date': self.date[self.day],
                'price': self.prices[self.day],
                'balance':self.balance,
                'gold':self.gold_trade.sum(),
                'posession': self.balance + self.gold_trade.sum() * self.prices[self.day],
                'action':self.yesterday_action}
        return obs, info
    
    def set_day(self, day):
        self.day = day

    def step(self, action):
        action = 0

        posession_temp = self.balance + self.gold_trade.sum() * self.prices[self.day]


        self.day += 1
        current_price = self.prices[self.day]

        reward = 0

        fee = 0   # 手续费
        buy_gold = 0
        sail_gold = 0

        if self.yesterday_action == 0:
            # buy 100% gold
            trade = self.balance
            self.balance -= trade
            buy_gold = trade / current_price
            
        if self.yesterday_action == 1:
            # wait
            pass
        
        if self.yesterday_action == 2:
            # sail 100% gold
            sail_gold_30 = self.gold_trade[0]  * current_price
            sail_gold_7 = self.gold_trade[1:23].sum() * current_price
            sail_gold_0 = self.gold_trade[23:].sum() * current_price
            self.balance += sail_gold_30 + sail_gold_7 + sail_gold_0
            fee += sail_gold_7 * 0.001  # 持有7天 - 30天 收取 0.1%
            fee += sail_gold_0 * 0.015  # 持有低于7天 收取 1.5%
            self.gold_trade = np.zeros_like(self.gold_trade)

        self.balance -= fee  # 手续费用会导致训练崩溃，agent不愿进行交易
        assert self.balance >= 0

        self.gold_trade[0] += self.gold_trade[1]
        self.gold_trade[1:30] = self.gold_trade[2:31]
        self.gold_trade[30] = buy_gold

        # TODO 可以改一下reward和始终持有黄金对比
        posession = self.balance + self.gold_trade.sum() * self.prices[self.day]
        self.yesterday_action = action
        
        reward += posession - posession_temp
        
        done = False
        truncated = False

        recent_prices = self.prices[self.day-41:self.day]  
        obs_price_changes = np.diff(recent_prices) / recent_prices[:-1]
        obs_price_changes = mm_normalize(obs_price_changes)
        obs_gold_trade = mm_normalize(self.gold_trade)
        index = 31 - np.where(self.gold_trade > 0)[0]  # 持有的天数

        # TODO 弄清操作当天的观察 还是 前一天的
        if index is None:
            cy = 0
        if index > 7 and index < 30:
            cy = 1
        if index == 31:
            cy = 0
        if index < 7:
            cy = -1

        print(action, index, cy)
        input()

        obs = np.concatenate((obs_price_changes, obs_gold_trade)).astype(np.float32)



        info = {'date': self.date[self.day],
                'price': self.prices[self.day],
                'balance':self.balance,
                'gold':self.gold_trade.sum(),
                'posession': posession,
                'action':self.yesterday_action}
        
        self.history.append(info)
        self.obs = obs
        
        return obs, reward, done, truncated, info

    def render(self, filename='trading_history.png'):  
        dates = [info['date'] for info in self.history]  
        prices = [info['price'] for info in self.history]  
        posessions = [info['posession'] for info in self.history]  
        actions = [info['action'] for info in self.history]  
        

        bar_data = self.obs
        print(bar_data[-1])

        # 创建两个子图  
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 14))  

        # 第一个子图：曲线图  
        ax1.set_ylabel('Possession', color='blue')  
        ax1.plot(dates, posessions, label='Possession', color='blue')  
        ax1.tick_params(axis='y', labelcolor='blue')  

        # 绘制黄金价格折线  
        ax1_twin = ax1.twinx()  
        ax1_twin.plot(dates, prices, label='Gold Price', color='red')  
        ax1_twin.set_xlabel('Date')  
        ax1_twin.set_ylabel('Gold Price', color='red')  
        ax1_twin.tick_params(axis='y', labelcolor='red')  

        colors = ['#FF0000', '#FFFF00', '#00FF00']
        actions_name = ['buy', 'wait', 'sell']  
        # 在每个时间点标出交易动作  
        for i, action in enumerate(actions):  
            action = int(action)  
            ax1_twin.scatter(dates[i], prices[i], color=colors[action], label=actions_name[action] if actions_name[action] not in ax1_twin.get_legend_handles_labels()[1] else "")  
        
        if len(dates) > 20:  
            ax1.set_xticks(np.linspace(0, len(dates) - 1, 20).astype(int))  
            ax1.set_xticklabels([dates[i] for i in np.linspace(0, len(dates) - 1, 20).astype(int)], rotation=45)  

        # 添加图表元素  
        ax1.set_title('Gold Price and Possession with Actions')  
        ax1.legend(loc='upper left')  
        ax1_twin.legend(loc='upper right')  
        ax1.grid()  

        # 第二个子图：柱状图  
        ax2.bar(range(len(bar_data)), bar_data, label='Bar Data', color='gray', alpha=0.5)  
        ax2.set_ylabel('Bar Data', color='gray')  
        ax2.set_xlabel('Date')  
        ax2.tick_params(axis='y', labelcolor='gray')  
        ax2.set_title('observation')  
        ax2.legend(loc='upper left')  
        ax2.grid()  

        # 调整布局  
        plt.tight_layout()  
        plt.savefig(filename)  
        plt.close()  

    def render_(self):
        print('\n-----------------------------------------------')
        print(self.history[0]['date'], '\t', self.history[-1]['date'])
        print(self.history[0]['price'], '\t', self.history[-1]['price'])
        print(self.history[0]['posession'], '\t', self.history[-1]['posession'])
        print('-----------------------------------------------\n')



if __name__ == '__main__':
    env = GoldTrader() 
    obs = env.reset()  
    for _ in range(100):  
        action = env.action_space.sample()  # Random action  
        obs, reward, done, info = env.step(action)
        # print(obs)
        env.render()