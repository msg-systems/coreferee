from typing import List
from os import sep
from threading import Lock
import pkg_resources
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from thinc.api import Config
from .errors import LanguageNotSupportedError


def debug_structures(doc: Doc) -> None:
    for token in doc:
        print(
            token.i,
            token.text,
            token.lemma_,
            token.pos_,
            token.tag_,
            token.dep_,
            token.ent_type_,
            token.head.i,
            list(token.children),
        )


language_to_nlps = {}
lock = Lock()


def get_nlps(language_name: str, *, add_coreferee: bool = True) -> List[Language]:
    """Returns a list of *nlp* objects to use when testing the functionality for *language*.
    The list contains the latest versions of the Spacy models named in the config file.
    Note that if this method is called with *add_coreferee=False*, this setting will apply
    to all future calls within the same process space. This means that *add_coreferee=False*
    is only appropriate during development of rules tests and before any smoke tests are
    required."""
    with lock:
        if language_name not in language_to_nlps:
            relative_config_filename = sep.join(("lang", language_name, "config.cfg"))
            if not pkg_resources.resource_exists("coreferee", relative_config_filename):
                raise LanguageNotSupportedError(language_name)
            absolute_config_filename = pkg_resources.resource_filename(
                __name__, relative_config_filename
            )
            config = Config().from_disk(absolute_config_filename)
            nlps = []
            for config_entry in config:
                # At present we presume there will never be an entry in the config file that
                # specifies a model name that can no longer be loaded. This seems a reasonable
                # assumption, but if it no longer applies this code will need to be changed in the
                # future.
                nlp = spacy.load("_".join((language_name, config[config_entry]["model"])))
                if add_coreferee:
                    nlp.add_pipe("coreferee")
                nlps.append(nlp)
                nlp.meta["matches_train_version"] = nlp.meta["version"] == config[config_entry]["train_version"]
            nlps = sorted(nlps, key=lambda nlp: (nlp.meta["name"], nlp.meta["version"]))
            language_to_nlps[language_name] = nlps
        return language_to_nlps[language_name]
