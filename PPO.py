import gym
import sys
import torch
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical  
from torch.optim import Adam
from utils.logger import Logger
from tqdm import tqdm
from GoldTrader import GoldTrader
import random

def check_nan(model):  
    for name, param in model.named_parameters():  
        if torch.isnan(param).any():  
            print(f"Parameter {name} contains NaN!")  

class DeepFeedForwardNN(nn.Module):  
    def __init__(self, in_dim, out_dim, emb_dim=128):  
        super(DeepFeedForwardNN, self).__init__()  

        # 可以根据需要增加或减少层数  
        self.layer1 = nn.Linear(in_dim, emb_dim)  
        self.layer2 = nn.Linear(emb_dim, emb_dim)  
        self.layer3 = nn.Linear(emb_dim, emb_dim)  
        self.layer4 = nn.Linear(emb_dim, emb_dim)  
        self.layer5 = nn.Linear(emb_dim, emb_dim)  
        self.layer6 = nn.Linear(emb_dim, out_dim)  

    def forward(self, obs):  
        activation1 = F.relu(self.layer1(obs))  
        activation2 = F.relu(self.layer2(activation1))  
        activation3 = F.relu(self.layer3(activation2))  
        activation4 = F.relu(self.layer4(activation3))  
        activation5 = F.relu(self.layer5(activation4))  
        output = self.layer6(activation5)  

        return output  
    

class FeatureSequenceNN(nn.Module):  
    def __init__(self, in_dim, out_dim, num_heads=8, dim_feedforward=256):  
        super(FeatureSequenceNN, self).__init__()  

        self.embedding = nn.Linear(1, dim_feedforward)
        self.attention = nn.MultiheadAttention(embed_dim=dim_feedforward, num_heads=num_heads)  
        self.linear_out = nn.Linear(dim_feedforward, out_dim)  

    def forward(self, obs):  
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        batch_size = obs.shape[0]
        x = obs.unsqueeze(0)
        x = x.permute(2, 1, 0)
        x = self.embedding(x)

        attn_output, _ = self.attention(x, x, x)  # 查询、键、值都是 x  
        attn_output = attn_output.mean(dim=0)  # 取平均以得到一个全局特征向量  
        output = self.linear_out(attn_output)  # 输出 (batch_size, out_dim)  

        return output  



class PPO:
    def __init__(self, env):

        self.t_trajectory = 200  # 每局最大步数
        self.t_batch = self.t_trajectory * 10  # 每个batch最大步数
        self.batch_update_num = 20  # 用每次采样更新的次数
        self.gamma = 0.99  # 奖励衰减
        self.clip = 0.2
        lr = 1e-3

        self.env = env
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        # print(obs_dim, act_dim)

        self.device = torch.device('cpu')
        self.actor = DeepFeedForwardNN(obs_dim, act_dim).to(self.device)
        self.critic = DeepFeedForwardNN(obs_dim, 1).to(self.device)
        # self.actor.load_state_dict(torch.load('./ppo_actor.pth'))
        # self.critic.load_state_dict(torch.load('./ppo_critic.pth'))

        self.actor_optim = Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.logger_rew = Logger('rew')

    @torch.no_grad
    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_tra_rewards= []

        bt = 0
        while bt < self.t_batch:
            tra_rewards = []
            obs, info = self.env.reset()
            self.env.set_day(200)

            for tra_t in range(self.t_trajectory):
                bt += 1
                batch_obs.append(obs)

                if np.isnan(obs).any():
                    print(obs)
                    raise ValueError           
                     
                action, log_prob = self._sample_action(torch.tensor(obs, dtype=torch.float32, device=self.device))
                obs, rew, done, truncated, info  = self.env.step(action)
                
                tra_rewards.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    print('broke up !!!')
                    break
            
            batch_tra_rewards.append(tra_rewards)
            # self.env.render_()

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float, device=self.device)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float, device=self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=self.device)
        batch_rtgs = torch.tensor(self._compute_rtgs(batch_tra_rewards), dtype=torch.float, device=self.device)

        self.logger_rew.add_value(float(np.mean([np.sum(tra_rewards) for tra_rewards in batch_tra_rewards])))
        self.logger_rew.plot()
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs
    
    def eval(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_tra_rewards= []

        start_day = random.randint(400, 600)
        start_day = 200
        tra_rewards = []
        obs, info1 = self.env.reset()
        self.env.set_day(start_day)

        for tra_t in range(self.t_trajectory):
            batch_obs.append(obs)

            if np.isnan(obs).any():
                print(obs)
                raise ValueError           
                    
            action, log_prob = self._sample_action(torch.tensor(obs, dtype=torch.float32, device=self.device))
            
            obs, rew, done, truncated, info  = self.env.step(action)
            
            
            
            tra_rewards.append(rew)
            batch_acts.append(action)
            batch_log_probs.append(log_prob)
            # env.render()

            if done:
                print('boeak up !!!')
                break
        
        batch_tra_rewards.append(tra_rewards)
        env.render()

    def learn(self, batch_num):

        logger_critic = Logger('critic_loss')
        logger_actor = Logger('actor_loss')
        
        for bi in tqdm(range(batch_num)):
            batch_obs, batch_acts, batch_log_probs, batch_rtgs = self.rollout()

            # 提前评论，为了避免“猫捉老鼠”
            V = self.critic(batch_obs).squeeze()
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            # input(A_k) 

            for i in tqdm(range(self.batch_update_num), desc='Learn'):
                
                # train critic
                V = self.critic(batch_obs).squeeze()
                
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # train actor
                _, curr_log_probs = self._sample_action(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)  # 重要性采样

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k  # 近端约束
                actor_loss = (-torch.min(surr1, surr2)).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_value_(self.actor.parameters(), clip_value=0.01)  
                self.actor_optim.step()

                logger_critic.add_value(critic_loss.item())
                logger_actor.add_value(actor_loss.item())
                logger_critic.plot()
                logger_actor.plot()

            torch.save(self.actor.state_dict(), './ppo_actor.pth')
            torch.save(self.critic.state_dict(), './ppo_critic.pth')

            self.eval()

    def _sample_action(self, batch_obs, batch_acts=None):
        """
            Return:
                - action: tensor[bs, act_dim]
                - log_prob: tensor[ba, 1]
        """
        logits = self.actor(batch_obs)  
        probs = torch.softmax(logits, dim=-1)  
        dist = Categorical(probs)  
        action = dist.sample()

        if batch_acts is None:
            log_prob = dist.log_prob(action)
        else:
            log_prob = dist.log_prob(batch_acts)

        action = np.array(action.cpu(), dtype=np.float32)

        return action, log_prob

    def _compute_rtgs(self, batch_rews):
        """
            Parameters:
                - batch_rews: [[trajectory_1], [trajectory_2] ... [trajectory_n]]
            
            
            Return:
                - batch-rewards-to-go: [r_1, r_2, ... r_t_batch]
        """
        batch_rtgs = []

        for trajectory_rews in reversed(batch_rews):
            
            batch_reward = 0

            for reward in reversed(trajectory_rews):
                batch_rtgs.insert(0, batch_reward)
                batch_reward = reward + batch_reward * self.gamma

        return batch_rtgs

    

if __name__ == '__main__':
    # print(torch.__version__)  
    # print(torch.cuda.is_available())
    torch.manual_seed(2002)

    # env = gym.make('Pendulum-v1', render_mode='human')
    env = gym.make('GoldTrader-v0')

    model = PPO(env)
    model.learn(batch_num=5000)
    