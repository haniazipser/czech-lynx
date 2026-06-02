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
        arcface_s= 128,
        lr=1e-4,
        arcface_lr=1e-4,
        batch_size = 16
    )