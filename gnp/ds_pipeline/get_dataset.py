from absl import flags
from gnp.ds_pipeline.datasets import dataset_source

FLAGS = flags.FLAGS

def get_dataset_pipeline():

    if FLAGS.config.dataset.dataset_name == 'cifar10':
        image_size = None
        ds = dataset_source.Cifar10(
            FLAGS.config.batch_size,
            FLAGS.config.dataset.image_level_augmentations,
            FLAGS.config.dataset.batch_level_augmentations,
            image_size=image_size
        )
    elif FLAGS.config.dataset.dataset_name == 'cifar100':
        image_size = None
        ds = dataset_source.Cifar100(
            FLAGS.config.batch_size, FLAGS.config.dataset.image_level_augmentations,
            FLAGS.config.dataset.batch_level_augmentations, image_size=image_size
        )
    else:
        raise ValueError('Dataset not recognized.')

    if FLAGS.config.dataset.dataset_name == 'cifar10' or FLAGS.config.dataset.dataset_name == 'cifar100':
        if image_size is None:
            FLAGS.config.dataset.image_size = 32
            FLAGS.config.dataset.num_channels = 3
            FLAGS.config.dataset.num_classes = 10 if FLAGS.config.dataset.dataset_name == 'cifar10' else 100
    else:
        raise ValueError('Dataset not recognized.')

    return ds

