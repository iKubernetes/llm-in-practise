import yaml
import logging

def setup_logging(log_file, level):
    logging.basicConfig(level=getattr(logging, level), filename=log_file, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Logging initialized.")

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
