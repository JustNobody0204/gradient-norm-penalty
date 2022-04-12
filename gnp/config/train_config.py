import ml_collections
from gnp.config.base_config import get_basic_config, get_optimizer_config, get_dataset_config
from gnp.config.model_config import get_model_config


def get_config():
    
    config = get_basic_config()
    config.unlock()

    config.model = get_model_config()
    config.dataset = get_dataset_config()
    config.opt = get_optimizer_config()
    # config.model.num_outputs = config.dataset.num_outputs
  
    return config.lock()