from .pandas import open_and_convert_data, create_merged_dataframe
from stable_baselines3 import DQN
from lib.utils.pandas import open_and_convert_data
from lib.environment.test_env import CryptoTradingEnv
from stable_baselines3.common.logger import configure

def train_model(csv_path: str):
    # Last inn data fra en CSV-fil eller en pandas DataFrame
    data = create_merged_dataframe(csv_path)
    tensorboard_log_dir = "./tensorboard_logs/"
    logger = configure(tensorboard_log_dir, ["stdout", "tensorboard"])
    # Lag miljøet
    env = CryptoTradingEnv(data)
    print('Preprocessing indicator data...')
    env.preprocess_data()

# Last inn modellen (hvis relevant)
    model = DQN("MlpPolicy", env, verbose=1, 
                learning_rate=0.001, 
                buffer_size=10000,
                learning_starts=10000,
                exploration_fraction=0.4,
                device='auto',
                tensorboard_log=tensorboard_log_dir)

    model.learn(total_timesteps=10000000, log_interval=100)
    model.save('./DQN_model-hourly')

