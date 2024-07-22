import pandas as pd
import yaml


def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def preprocess_data(config):
    raw_data_path = config['data']['raw_data_path']
    processed_data_path = config['data']['processed_data_path']

    # Load raw data
    data = pd.read_csv(raw_data_path)

    # Preprocessing steps (example)
    data['value'] = (data['value'] - data['value'].mean()) / data['value'].std()

    # Save processed data
    data.to_csv(processed_data_path, index=False)


if __name__ == '__main__':
    config = load_config()
    preprocess_data(config)
