import argparse
import os
import sys
import pkg_resources
from spacy.util import run_command
from .training.train import TrainingManager
from .manager import COMMON_MODELS_PACKAGE_NAMEPART

DOWNLOAD_URL = "https://github.com/msg-systems/coreferee/raw/master/models"

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command")

train_parser = subparsers.add_parser(
    "train",
    help="Train models for a language. Must be executed from the root directory of the checked-out repository. Type *python -m coreferee train -h* for more information.",
)
check_parser = subparsers.add_parser(
    "check",
    help="Check models for a language, e.g. to verify that the same performance is obtained with a new spaCy model. Loads and runs the same test documents as when running *train* but using the existing Coreferee models. Must be executed from the root directory of the checked-out repository. Type *python -m coreferee check -h* for more information.",
)
train_args = train_parser.add_argument_group("required arguments")
check_args = check_parser.add_argument_group("required arguments")
train_args.add_argument(
    "--lang",
    dest="lang",
    required=True,
    help="The ISO 639-1 code for the language to train",
)
check_args.add_argument(
    "--lang",
    dest="lang",
    required=True,
    help="The ISO 639-1 code for the language to train",
)
train_args.add_argument(
    "--loader_classes",
    dest="loader_classes",
    required=True,
    help="The class name(s) of the training data loader within *coreferee.training.loaders*. Multiple class names should be comma-separated.",
)
check_args.add_argument(
    "--loader_classes",
    dest="loader_classes",
    required=True,
    help="The class name(s) of the training data loader within *coreferee.training.loaders*. Multiple class names should be comma-separated.",
)
train_args.add_argument(
    "--data_dir",
    dest="data_dir",
    required=True,
    help="The path of the directory that contains the training data",
)
check_args.add_argument(
    "--data_dir",
    dest="data_dir",
    required=True,
    help="The path of the directory that contains the training data",
)
train_args.add_argument(
    "--log_dir",
    dest="log_dir",
    required=True,
    help="The path of the directory to which to write log files",
)
check_args.add_argument(
    "--log_dir",
    dest="log_dir",
    required=True,
    help="The path of the directory to which to write log files",
)
install_parser = subparsers.add_parser(
    "install",
    help="Install models for a language. Type *python -m coreferee install -h* for more information.",
)

install_parser.add_argument(
    "--force-reinstall",
    default=False,
    action="store_true",
    help="Forces a reinstall when models are downloaded from Github (when models are being installed from the local filesystem, a reinstall always takes place)",
)
install_parser.add_argument("lang", help="The ISO 639-1 code for the language to train")

args = parser.parse_args()
if args.command == "train":
    TrainingManager(
        __name__,
        args.lang,
        args.loader_classes,
        args.data_dir,
        args.log_dir,
        train_not_check=True,
    ).train_models()
elif args.command == "check":
    TrainingManager(
        __name__,
        args.lang,
        args.loader_classes,
        args.data_dir,
        args.log_dir,
        train_not_check=False,
    ).check_models()
elif args.command == "install":
    file_system_root = pkg_resources.resource_filename(__name__, "")
    models_dirname = "".join(
        (file_system_root, os.sep, "..", os.sep, "models", os.sep, args.lang)
    )
    if "site-packages" not in models_dirname and os.path.isdir(models_dirname):
        run_command(
            " ".join(
                (sys.executable, "-m pip install --force-reinstall", models_dirname)
            )
        )
    else:
        url = "".join(
            (DOWNLOAD_URL, "/", COMMON_MODELS_PACKAGE_NAMEPART, args.lang, ".zip")
        )
        run_command(
            " ".join(
                (
                    sys.executable,
                    "-m pip install",
                    "--force-reinstall" if args.force_reinstall else "",
                    url,
                )
            )
        )
else:
    parser.print_help()
