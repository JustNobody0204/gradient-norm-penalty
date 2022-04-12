from absl import flags
from absl import app
from ml_collections.config_flags import config_flags
import jax
import flax
import tensorflow as tf

from gnp.main import utli
from gnp.models import load_model
from tensorflow.io import gfile
from gnp.training import training
from gnp.ds_pipeline.get_dataset import get_dataset_pipeline
from gnp.optimizer.get_optimizer import get_optimizer
from gnp.training import flax_training

FLAGS = flags.FLAGS
WORK_DIR = flags.DEFINE_string('working_dir', None,
                               'Directory to store logs and model data.')
config_flags.DEFINE_config_file(
    "config",
    None,
    'File path to the training hyperparameter configuration.'
)
flags.mark_flags_as_required(["config", "working_dir"])


def main(_):

    tf.config.experimental.set_visible_devices([], 'GPU')

    model_folder = utli.create_model_folder()
    FLAGS.config.model_folder = model_folder
    batch_size = FLAGS.config.batch_size 

    ds = get_dataset_pipeline()

    module, params, state = load_model.get_model(FLAGS.config.model.model_name,
                                        batch_size, FLAGS.config.dataset.image_size,
                                        FLAGS.config.dataset.num_classes, FLAGS.config.dataset.num_channels)

    optimizer = training.utli.create_optimizer(params, 0.0)

    flax_training.train(module, optimizer, state, ds, FLAGS.config.model_folder,
                        FLAGS.config.total_epochs)


if __name__ == '__main__':
  app.run(main)
