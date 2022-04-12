import os
import logging
from absl import flags
import sys
from tensorflow.io import gfile
import json

FLAGS = flags.FLAGS
logger = logging.getLogger(__name__)

def init_logger():
    config = FLAGS.config
    logging_level_list = {"debug" : logging.DEBUG,
                        "info"  : logging.INFO,
                        "warning" : logging.WARNING,
                        "error" : logging.ERROR,
                        "critical" : logging.CRITICAL}
    assert config.logging.basic_logger_level in logging_level_list.keys()
    current_logging_level = logging_level_list[config.logging.basic_logger_level]
    root_logger = logging.getLogger()
    root_logger.setLevel(current_logging_level)

    if config.logging.logger_sys_output:
        handler_sysout = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler_sysout.setFormatter(formatter)
        root_logger.addHandler(handler_sysout)

    root_logger.debug(f"Initilize logger, logger level {config.logging.basic_logger_level}.")


def create_model_folder():
    config = FLAGS.config
    model_folder_name = os.path.join(f"{FLAGS.working_dir}",
                                f"{config.model.model_name}",
                                f"{config.dataset.dataset_name}",
                                f"{config.dataset.image_level_augmentations}_{config.dataset.batch_level_augmentations}",
                                f"lr_{config.base_lr}",
                                f"bs_{config.batch_size}",
                                f"wd_{config.l2_regularization}",
                                f"grad_clip_{config.gradient_clipping}",
                                f"opt_{config.opt.opt_type}",
                                f"r_{config.gnp.r}",
                                f"alpha_{config.gnp.alpha}",
                                f"epoch_{config.total_epochs}",
                                f"run_seeds_{config.seeds}")

    logging.info(f"Model folder at {model_folder_name}")

    if not gfile.exists(model_folder_name):
        logger.info(f"Creating model folder at {model_folder_name}")
        gfile.makedirs(model_folder_name)
    else:
        if config.retrain:
            logger.info(f"Retraining. Deleting and creating the model folder {model_folder_name}")
            gfile.rmtree(model_folder_name)
            gfile.makedirs(model_folder_name)


    return model_folder_name


def write_config_to_json():
    config = FLAGS.config
    if config.write_config_to_json:
        logger.info(f"Writing json to the model foler ...")
        with open(os.path.join(config.model_folder, "config.json"), "w") as f:
            json.dump(config.to_json(), f)
    return