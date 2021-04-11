import yaml
import os


def read_configs() -> dict:
    """
    This functions read config file
    :return: dict
    """
    with open('configs/config.yml', 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    return config

