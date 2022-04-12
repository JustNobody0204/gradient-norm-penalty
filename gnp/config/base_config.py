import ml_collections


def get_basic_config():

    config = ml_collections.ConfigDict()

    config.model_folder = None
    config.seeds = 0
    config.batch_size = 128 * 2
    config.base_lr = 1e-1
    config.total_epochs = 200
    config.gradient_clipping = 5.0
    config.use_learning_rate_schedule = True
    config.l2_regularization = 1e-3
    config.use_rmsprop = False
    config.lr_schedule_type = "cosine"
    config.save_ckpt_every_n_epochs = 200
    config.warmup_steps = 0
    config.label_smoothing = 0.

    config.additional_checkpoints_at_epochs = []
    config.also_eval_on_training_set = False
    config.compute_top_5_error_rate = False
    config.evaluate_every = 1
    config.inner_group_size = None
    config.no_weight_decay_on_bn = False
    config.asam = False
    config.use_dual_in_adam = False

    config.gnp = ml_collections.ConfigDict()
    config.gnp.r = 0.0
    config.gnp.sync_perturbations = False
    config.gnp.alpha = 0.0
    config.gnp.norm_perturbations = True

    config.ema_decay = 0.
    config.retrain = True
    config.logging = ml_collections.ConfigDict()
    config.logging.tensorboard_logging_frequency = 1
    config.logging.basic_logger_level = "debug"
    config.logging.logger_sys_output = False
    config.write_config_to_json = True

    config.from_pretrained_checkpoint = False
    config.efficientnet_checkpoint_path = None

    return config.lock()

def get_dataset_config():

    config = ml_collections.ConfigDict()
    config.dataset_name = "cifar10"
    config.image_level_augmentations = "basic"
    config.batch_level_augmentations = "none"
    config.image_size = 32
    config.num_classes = 10
    config.num_channels = 3

    return config


def get_optimizer_config():

    config = ml_collections.ConfigDict()

    config.opt_type = "SGD"

    config.opt_params = ml_collections.ConfigDict()
    config.opt_params.nesterov = True
    config.opt_params.beta = 0.9

    # config.opt_name = "Adam"
    # config.opt_params = ml_collections.ConfigDict()
    # config.opt_params.grad_norm_clip = 1.0
    # config.opt_params.weight_decay = 0.3

    return config
