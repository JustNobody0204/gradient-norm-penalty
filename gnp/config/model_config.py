import ml_collections


def get_model_config():

    config = ml_collections.ConfigDict()

    config.model_name = "WideResnet28x10"
    # config.model_name = "VIT"
    
    # Config Model Parameters Here
    config.model_params = ml_collections.ConfigDict()
    
    # config.model_params.patches = ml_collections.ConfigDict({'size': (4, 4)})
    # config.model_params.hidden_size = 768
    # config.model_params.transformer = ml_collections.ConfigDict()
    # config.model_params.transformer.mlp_dim = 3072
    # config.model_params.transformer.num_heads = 12
    # config.model_params.transformer.num_layers = 12
    # config.model_params.transformer.attention_dropout_rate = 0.0
    # config.model_params.transformer.dropout_rate = 0.0
    
    return config