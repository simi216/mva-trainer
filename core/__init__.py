from . import utils
from . import reconstruction
from . import components
from . import base_classes
from . import evaluation
from .DataPlotting import DataPlotter
from .DataLoader import DataPreprocessor
from .Configs import *
from .RootPreprocessor import RootPreprocessor, PreprocessorConfig, preprocess_root_file, preprocess_root_directory

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    print("Could not set TensorFlow GPU memory growth.")
    pass

