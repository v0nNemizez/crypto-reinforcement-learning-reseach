from stable_baselines3.common.callbacks import BaseCallback


class CustomTensorBoardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomTensorBoardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log the net worth to TensorBoard
        self.logger.record("net_worth", self.training_env.get_attr('net_worth')[0])
        return True