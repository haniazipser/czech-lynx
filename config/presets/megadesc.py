from config.config import DataConfig
#Metric learning experiments specific params
def get_config() -> DataConfig:
    return DataConfig(
        experiment_type="metric_learning",
        image_size=384,
        batch_size=16,
    )