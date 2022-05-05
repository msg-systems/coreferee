import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

warnings.filterwarnings("ignore", message=r"\[W007\]", category=UserWarning)

import coreferee.manager

coreferee.manager.CorefereeBroker.set_extensions()
