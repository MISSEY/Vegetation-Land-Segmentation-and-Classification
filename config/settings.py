"""
Change settings here
"""

from pathlib import Path
from config import config
import os

def get_project_root()-> Path:
    return Path(__file__).parent.parent

underscore = '_'

# while training on cluster

check_point_output_directory = '/netscratch/smishra/thesis/output/output' + underscore + \
                               str(config.train_image_size) + underscore + \
                               str(config.model_name) + underscore + \
                               str(config._version_name) + underscore + \
                               str(config.experiment_name) + str(config.experiment_value) + underscore + \
                               str(config.learning_rate) + underscore + \
                               str(35)
data_directory_cluster = '/netscratch/smishra/thesis/vegetation_dataset'

weights_directory = '/netscratch/smishra/output'



data_directory = os.path.join(get_project_root(), config._data_)

raw_shape_directory = os.path.join(data_directory,config._raw_shape_directory)

