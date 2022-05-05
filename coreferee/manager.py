from typing import Dict, Tuple
import importlib
import os
import pickle
import traceback
from sys import exc_info

from numpy import absolute
from packaging import version
import spacy
import pkg_resources
from wasabi import Printer # type: ignore[import]
from spacy.language import Language
from spacy.tokens import Doc, Token
from thinc.api import Config
from thinc.model import Model
from .annotation import Annotator
from .data_model import FeatureTable
from .errors import (
    LanguageNotSupportedError,
    ModelNotSupportedError,
    OutdatedCorefereeModelError,
)
from .errors import VectorsModelNotInstalledError, VectorsModelHasWrongVersionError
from .tendencies import create_thinc_model, ENSEMBLE_SIZE

COMMON_MODELS_PACKAGE_NAMEPART = "coreferee_model_"

FEATURE_TABLE_FILENAME = "feature_table.bin"

THINC_MODEL_FILENAME = "model"


class CorefereeManager:
    @staticmethod
    def get_annotator(nlp: Language) -> Annotator:
        model_name = "_".join((nlp.meta["lang"], nlp.meta["name"]))
        relative_config_filename = os.sep.join(("lang", nlp.meta["lang"], "config.cfg"))
        if not pkg_resources.resource_exists(__name__, relative_config_filename):
            msg = Printer()
            msg.fail(
                "".join(
                    (
                        "Unfortunately language '",
                        nlp.meta["lang"],
                        "' is not yet supported by Coreferee.",
                    )
                )
            )
            raise LanguageNotSupportedError(nlp.meta["lang"])
        absolute_config_filename = pkg_resources.resource_filename(
            __name__, relative_config_filename
        )
        config = Config().from_disk(absolute_config_filename)
        for config_entry_name, config_entry in config.items():
            if (
                nlp.meta["name"] == config_entry["model"]
                and version.parse(nlp.meta["version"])
                >= version.parse(config_entry["from_version"])
                and version.parse(nlp.meta["version"])
                <= version.parse(config_entry["to_version"])
            ):
                if "vectors_model" in config_entry:
                    try:
                        vectors_nlp = spacy.load(
                            "_".join((nlp.meta["lang"], config_entry["vectors_model"]))
                        )
                    except OSError:
                        msg = Printer()
                        error_msg = "".join(
                            (
                                "spaCy Model ",
                                model_name,
                                " is only supported by Coreferee in conjunction with spaCy model ",
                                nlp.meta["lang"],
                                "_",
                                config_entry["vectors_model"],
                                ", which must be loaded using the command 'python -m spacy download ",
                                nlp.meta["lang"],
                                "_",
                                config_entry["vectors_model"],
                                "'.",
                            )
                        )
                        msg.fail(error_msg)
                        raise VectorsModelNotInstalledError(error_msg)
                    if version.parse(vectors_nlp.meta["version"]) < version.parse(
                        config_entry["from_version"]
                    ) or version.parse(vectors_nlp.meta["version"]) > version.parse(
                        config_entry["to_version"]
                    ):
                        msg = Printer()
                        error_msg = "".join(
                            (
                                "spaCy model ",
                                model_name,
                                " is only supported by Coreferee in conjunction with spaCy model ",
                                nlp.meta["lang"],
                                "_",
                                config_entry["vectors_model"],
                                " between versions ",
                                config_entry["from_version"],
                                " and ",
                                config_entry["to_version"],
                                " inclusive.",
                            )
                        )
                        msg.fail(error_msg)
                        raise VectorsModelHasWrongVersionError(error_msg)
                else:
                    vectors_nlp = nlp
                return get_annotator(
                    nlp=nlp,
                    vectors_nlp=vectors_nlp,
                    config_entry_name=config_entry_name,
                )
        msg = Printer()
        error_msg = "".join(
            (
                "spaCy model ",
                nlp.meta["lang"],
                "_",
                nlp.meta["name"],
                " version ",
                nlp.meta["version"],
                " is not supported by Coreferee. Please examine /coreferee/lang/",
                nlp.meta["lang"],
                "/config.cfg to see the supported models/versions.",
            )
        )
        msg.fail(error_msg)
        raise ModelNotSupportedError(error_msg)


@Language.factory("coreferee")
class CorefereeBroker:
    def __init__(self, nlp: Language, name: str):
        self.nlp = nlp
        self.pid = os.getpid()
        self.annotator = CorefereeManager().get_annotator(nlp)

    def __call__(self, doc: Doc) -> Doc:
        try:
            self.annotator.annotate(doc)
        except:
            msg = Printer()
            msg.warn("Unexpected error in Coreferee annotating document, skipping ....")
            exception_info_parts = exc_info()
            msg.warn(exception_info_parts[0])
            msg.warn(exception_info_parts[1])
            traceback.print_tb(exception_info_parts[2])
        return doc

    def __getstate__(self) -> Dict[str, str]:
        return self.nlp.meta

    def __setstate__(self, meta: Dict[str, str]):
        nlp_name = "_".join((meta["lang"], meta["name"]))
        self.nlp = spacy.load(nlp_name)
        self.annotator = CorefereeManager().get_annotator(self.nlp)
        self.pid = os.getpid()
        CorefereeBroker.set_extensions()

    @staticmethod
    def set_extensions() -> None:
        if not Doc.has_extension("coref_chains"):
            Doc.set_extension("coref_chains", default=None)
        if not Token.has_extension("coref_chains"):
            Token.set_extension("coref_chains", default=None)


def get_annotator(
    *, nlp: Language, vectors_nlp: Language, config_entry_name: str
) -> Annotator:
    model_package_name = "".join(
        (
            COMMON_MODELS_PACKAGE_NAMEPART,
            nlp.meta["lang"],
            ".",
            config_entry_name,
        )
    )
    try:
        importlib.import_module(model_package_name)
    except ModuleNotFoundError:
        msg = Printer()
        error_msg = "".join(
            (
                "Please load the Coreferee models for language '",
                nlp.meta["lang"],
                "' with the command 'python -m coreferee install ",
                nlp.meta["lang"],
                "'.",
            )
        )
        msg.fail(error_msg)
        raise ModelNotSupportedError(error_msg)
    this_feature_table_filename = pkg_resources.resource_filename(
        model_package_name, FEATURE_TABLE_FILENAME
    )
    with open(this_feature_table_filename, "rb") as feature_table_file:
        feature_table = pickle.load(feature_table_file)
    absolute_thinc_model_filename = pkg_resources.resource_filename(
        model_package_name, THINC_MODEL_FILENAME
    )
    if not os.path.isfile(absolute_thinc_model_filename):
        msg = Printer()
        error_msg = "".join(
            (
                "The Coreferee model loaded for config entry '",
                config_entry_name,
                "' is outdated. Please issue the command 'python -m coreferee install ",
                nlp.meta["lang"],
                "' to install the latest version.",
            )
        )
        msg.fail(error_msg)
        raise OutdatedCorefereeModelError(error_msg)
    thinc_model = create_thinc_model()
    thinc_model.from_disk(absolute_thinc_model_filename)
    return Annotator(nlp, vectors_nlp, feature_table, thinc_model)
