import yaml
import os

def load_config(env: str):
    with open(os.path.join('config.yml'), 'r') as file:
        config = yaml.safe_load(file)

    common_config = config.get("common", {})
    env_config = config.get(env, {})

    merged_config = {**common_config, **env_config}
    return merged_config

def get_environment_config(env_key: str):
    env_switch = {
        "ant": "ant",
        "hopper": "hopper",
        "bipedal_walker": "bipedalWalker",
        "lunar_lander": "lunarLander",
        "half_cheetah": "halfCheetah"
    }

    # Get the environment key from the switch dictionary
    selected_env = env_switch.get(env_key.lower())

    if selected_env is None:
        raise ValueError(f"Environment '{env_key}' is not supported.")

    # Load and return the merged configuration for the selected environment
    return load_config(selected_env)