from .env import GoldTrader
from gym.envs.registration import register  


register(  
    id='GoldTrader-v0',  
    entry_point='GoldTrader.env:GoldTrader',  
)  