from config.config import DataConfig
#Baseline experiments specific params
def get_config() -> DataConfig:
    return DataConfig(
        experiment_type="baseline",
        split_type="time_closed"
    )