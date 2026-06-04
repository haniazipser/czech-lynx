from config.config import DataConfig
#Metric learning experiments specific params
def get_config() -> DataConfig:
    return DataConfig(
        experiment_type="metric_learning",
        warmup_epochs=3,
        weight_decay=0.01,
        warmup_start=0.1,
        warmup_end=1.0,
        arcface_m= 0.1,
        arcface_s= 48,
        lr=0.0001,
        arcface_lr=0.0001,
        batch_size = 16,
        split_type="time_open"
    )