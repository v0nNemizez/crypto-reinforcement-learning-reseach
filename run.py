# Importer nødvendige pakker
import gymnasium as gym
from stable_baselines3 import DQN
from lib.utils.pandas import open_and_convert_data
from lib.environment.test_env import CryptoTradingEnv


def load_and_run_model(model_path, data):
    # Last inn miljøet
    env = CryptoTradingEnv(data)
    
    # Last inn den lagrede modellen
    model = DQN.load(model_path)
    
    # Tilbakestill miljøet og pakk ut observasjonen
    obs, _ = env.reset()

    # Forutsi handlinger og kjør modellen
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)  # Forutsi handling
        obs, reward, terminated, truncated, info = env.step(action)  # Utfør handling

        # Sjekk om episoden er over
        done = terminated or truncated
        
        env.render()  # Vis tilstanden

# Last inn data fra en CSV-fil eller en pandas DataFrame
data = open_and_convert_data('./training_data/BTC-Hourly.csv')  # Erstatt med riktig filsti

# Kall funksjonen med modellstien og dataen
load_and_run_model("./DQN_model-hourly.zip", data)
