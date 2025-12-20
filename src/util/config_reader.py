import yaml

def load_models_config(config_path: str = "config/model_config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config