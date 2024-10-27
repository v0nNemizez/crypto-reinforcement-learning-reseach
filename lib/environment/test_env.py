import gymnasium as gym
import numpy as np
from gymnasium import spaces
from lib.utils.ta import calculate_rsi
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Importer TensorBoard-pakker
from torch.utils.tensorboard import SummaryWriter
import os

# Opprett loggdir
log_dir = "./tensorboard_logs"
os.makedirs(log_dir, exist_ok=True)

# Opprett TensorBoard writer
writer = SummaryWriter(log_dir)

class CryptoTradingEnv(gym.Env):
    def __init__(self, data):
        # Lagre prisdataene fra CSV
        self.data = data
        self.current_step = 0

        # Start kapital
        self.initial_balance = 100000
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.net_worth = self.initial_balance
        self.total_reward = 0
        self.max_net_worth = self.initial_balance
        self.total_trades = 0

        self.data['RSI'] = calculate_rsi(prices=self.data['close'])
        

        # Trading fee i prosent
        self.trading_fee_percent = 0.001  # 0.1%


        # Definer handlinger: [0] - hold, [1] - kjøp, [2] - selg
        self.action_space = spaces.Discrete(3)

        # Definer observasjonene: normalisert pris, beholdning, saldo, volum + ekstra markedsindikatorer
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

    def preprocess_data(self):
        self.data['RSI'] = calculate_rsi(self.data['close'])

    def reset(self, seed=None, options=None):
        # Sett seed om det er gitt
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Tilbakestill miljøet
        self.current_step = 0
        self.balance = self.initial_balance
        self.crypto_held = 0
        self.net_worth = self.initial_balance
        self.total_reward = 0
        self.total_trades = 0

        # Returner den første observasjonen og et tomt info-dictionary
        return self._get_observation(), {}

    def _get_observation(self):
        # Hent dagens pris og volum
        current_price = self.data['close'].values[self.current_step]
        volume_btc = self.data['Volume BTC'].values[self.current_step]

    # Finn maks og min for pris og volum for normalisering
        max_price = self.data['close'].max()
        min_price = self.data['close'].min()
        max_volume = self.data['Volume BTC'].max()
        min_volume = self.data['Volume BTC'].min()

    # Beregn markedsindikatorer (RSI og SMA)
        rsi = self.data['RSI'][self.current_step]

    # Normalisering:
        normalized_price = (current_price - min_price) / (max_price - min_price)
        normalized_volume = (volume_btc - min_volume) / (max_volume - min_volume)
    
    # Normaliser beholdning
        normalized_crypto_held = self.crypto_held / (self.initial_balance / max_price)
    
    # Normaliser saldo
        normalized_balance = self.balance / self.initial_balance

    # Returner de normaliserte verdiene og forholdsberegninger
        return np.array([
            normalized_price,
            normalized_volume,
            normalized_crypto_held,
            normalized_balance,
            rsi,
            
        ], dtype=np.float32)


    def step(self, action):
    # Hent dagens pris og volum
        current_price = self.data['close'].values[self.current_step]
        volume_btc = self.data['Volume BTC'].values[self.current_step]
        rsi = self.data['RSI'][self.current_step]

        previous_net_worth = self.net_worth  # Lagre for sammenligning

        # Start med en normalisert belønning på 0
        reward = 0

        # Utfør handlingen
        if action == 1:  # Kjøp
            if self.balance > 0:  # Hvis vi har penger til å kjøpe
                amount_to_buy = self.balance / current_price
                print(f"Kjøper {amount_to_buy} crypto på pris {current_price}")
                self.crypto_held += amount_to_buy
                self.balance -= amount_to_buy * current_price
                self.balance -= self.balance * self.trading_fee_percent  # Trading fee
                self.total_trades += 1

                # Gi belønning for å kjøpe når RSI < 0.3 (oversolgt)
                if rsi < 0.4:
                    reward += 10  # Stronger reward for buying when the market is oversold
                elif 0.4 <= rsi < 0.7:
                    reward -= 10  # Smaller reward in the neutral range
                else:
                    reward -= 100 # Negative reward for unfavorable conditions (overbought)


        elif action == 2:  # Selg
            if self.crypto_held > 0:  # Hvis vi har noe å selge
                print(f"Selger {self.crypto_held} crypto på pris {current_price}") 
                self.balance += self.crypto_held * current_price
                self.balance -= self.balance * self.trading_fee_percent  # Trading fee
                self.crypto_held = 0
                self.total_trades += 1

            # Gi belønning for å selge når RSI > 0.7 (overkjøpt)
            if rsi > 0.7:
                reward += 10   # Sterkere belønning for å selge når markedet er overkjøpt
            elif rsi <= 0.7 and rsi >= 0.4:
                reward += -10
            else:
                reward -= -100   # Mindre belønning ellers

        # Oppdater nettoverdi
        self.net_worth = self.balance + self.crypto_held * current_price

        #if self.net_worth < previous_net_worth:
        #    reward -= 1

        # Gi en liten straff for å holde (ikke handle)
        if action == 0:
            reward -= 0.1

        if self.net_worth > previous_net_worth:
            reward += (self.net_worth - previous_net_worth) * 0.1  # Belønning basert på vekst
        else:
            reward -= (previous_net_worth - self.net_worth) * 0.1  # Straff for å miste verdi
        
        #if self.total_trades > (self.total_timesteps / 100):  # Juster grense etter ønsket tradinghyppighet
        #    reward -= (self.total_trades - (self.total_timesteps / 100)) * 0.5  # Straff for overtrading

    # Oppdater total belønning
        self.total_reward += reward


    # Sjekk om episoden er over
        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = self.net_worth <= 0.5 * self.initial_balance  # Tidlig avslutning hvis nettoverdi er for lav

        return self._get_observation(), reward, terminated, truncated, {}


    def render(self, mode='human'):
        # Valgfri funksjon for å vise tilstanden
        current_price = self.data['close'].values[self.current_step]
        print(f'-------------------------------------')
        print(f'Step: {self.current_step}')
        print(f'Price: {current_price}')
        print(f'Crypto Held: {self.crypto_held}')
        print(f'Balance: {self.balance}')
        print(f'Net Worth: {self.net_worth}')
        print(f'Total Reward: {self.total_reward}')
        print(f'Total Trades: {self.total_trades}')
        print('RSI: %s' % self.data['RSI'][self.current_step] )
        #print(f'SMA: {self.sma}')
        print(f'-------------------------------------')
