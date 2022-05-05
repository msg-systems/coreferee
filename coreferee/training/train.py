from typing import Dict, List, Tuple, cast, Callable
import os
import bisect
import shutil
import sys
import time
import pickle
from datetime import datetime
from random import Random
from tqdm import tqdm  # type:ignore[import]
from packaging import version
import pkg_resources
import spacy
from spacy.tokens import Doc
from spacy.language import Language
from thinc.api import Config, prefer_gpu
from thinc.loss import SequenceCategoricalCrossentropy
from thinc.model import Model
from thinc.optimizers import Adam
from thinc.types import Floats2d
from .loaders import GenericLoader
from ..annotation import Annotator
from ..data_model import FeatureTable, Mention
from ..manager import COMMON_MODELS_PACKAGE_NAMEPART, get_annotator
from ..manager import FEATURE_TABLE_FILENAME, THINC_MODEL_FILENAME
from ..rules import RulesAnalyzerFactory
from ..tendencies import TendenciesAnalyzer, generate_feature_table, create_thinc_model
from ..tendencies import DocumentPairInfo, ENSEMBLE_SIZE
from ..errors import LanguageNotSupportedError, ModelNotSupportedError


class TrainingManager:
    def __init__(
        self,
        root_path: str,
        lang: str,
        loader_classes: str,
        data_dir: str,
        log_dir: str,
        *,
        train_not_check: bool
    ):
        self.file_system_root = pkg_resources.resource_filename(root_path, "")
        relative_config_filename = os.sep.join(("lang", lang, "config.cfg"))
        if not pkg_resources.resource_exists(root_path, relative_config_filename):
            raise LanguageNotSupportedError(lang)
        self.config = Config().from_disk(
            os.sep.join((self.file_system_root, relative_config_filename))
        )
        loader_classnames = loader_classes.split(",")
        self.loaders = []
        for loader_classname in sorted(loader_classnames):
            class_ = getattr(
                sys.modules["coreferee.training.loaders"], loader_classname
            )
            self.loaders.append(class_())
        self.lang = lang
        self.models_dirname = os.sep.join((self.file_system_root, "..", "models", lang))
        if not os.path.isdir(self.models_dirname):
            self.set_up_models_dir()

        self.relevant_config_entry_names = []
        self.train_not_check = train_not_check
        self.nlp_dict: Dict[str, Language] = {}
        for config_entry_name, config_entry in self.config.items():
            this_model_dir = "".join(
                (
                    self.models_dirname,
                    os.sep,
                    "".join((COMMON_MODELS_PACKAGE_NAMEPART, self.lang)),
                    os.sep,
                    config_entry_name,
                )
            )
            if not os.path.isdir(this_model_dir) or not train_not_check:
                model_name = "_".join((lang, config_entry["model"]))
                if not self.load_model(
                    model_name,
                    config_entry_name,
                    config_entry["from_version"],
                    config_entry["to_version"],
                ):
                    print("Skipping config entry", config_entry_name, "as specified version range does not match.")
                    continue
                if "vectors_model" in config_entry:
                    vectors_model_name = "_".join((lang, config_entry["vectors_model"]))
                    self.load_model(
                        vectors_model_name,
                        config_entry_name,
                        config_entry["from_version"],
                        config_entry["to_version"],
                        is_vector_model=True,
                    )
                self.relevant_config_entry_names.append(config_entry_name)
            else:
                print("Skipping config entry", config_entry_name, "as model exists")

        self.log_dir = log_dir
        if ".." in log_dir:
            print(".. not permitted in log_dir")
            sys.exit(1)
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.isdir(data_dir):
            print("Data directory", data_dir, "not found.")
            sys.exit(1)
        self.data_dir = data_dir

        temp_dir = os.sep.join((self.log_dir, "temp"))
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
        time.sleep(1)
        os.mkdir(temp_dir)

    def load_model(
        self,
        name,
        config_entry_name,
        from_version,
        to_version,
        *,
        is_vector_model=False
    ) -> bool:
        if name not in self.nlp_dict:
            print("Loading model", name, "...")
            try:
                nlp = spacy.load(name)
            except OSError:
                if is_vector_model:
                    print(
                        "Config entry",
                        config_entry_name,
                        "specifies a vectors model",
                        name,
                        "that cannot be loaded.",
                    )
                else:
                    print(
                        "Config entry",
                        config_entry_name,
                        "specifies a model",
                        name,
                        "that cannot be loaded.",
                    )
                sys.exit(1)
        else:
            nlp = self.nlp_dict[name]
        if version.parse(nlp.meta["version"]) < version.parse(
            from_version
        ) or version.parse(nlp.meta["version"]) > version.parse(to_version):
            return False
        self.nlp_dict[name] = nlp
        return True

    def set_up_models_dir(self):
        os.mkdir(self.models_dirname)
        package_dirname = "".join((COMMON_MODELS_PACKAGE_NAMEPART, self.lang))
        os.mkdir(os.sep.join((self.models_dirname, package_dirname)))
        setup_cfg_filename = os.sep.join((self.models_dirname, "setup.cfg"))
        with open(setup_cfg_filename, "w") as setup_cfg_file:
            self.writeln(setup_cfg_file, "[metadata]")
            self.writeln(setup_cfg_file, "name = ", package_dirname.replace("_", "-"))
            self.writeln(setup_cfg_file, "version = 1.0.0")
            self.writeln(setup_cfg_file)
            self.writeln(setup_cfg_file, "[options]")
            self.writeln(setup_cfg_file, "packages = find:")
            self.writeln(setup_cfg_file, "include_package_data = True")
            self.writeln(setup_cfg_file)
            self.writeln(setup_cfg_file, "[options.package_data]")
            self.writeln(setup_cfg_file, "* = feature_table.bin, model")
        pyproject_toml_filename = os.sep.join((self.models_dirname, "pyproject.toml"))
        with open(pyproject_toml_filename, "w") as pyproject_toml_file:
            self.writeln(pyproject_toml_file, "[build-system]")
            self.writeln(pyproject_toml_file, "requires = [")
            self.writeln(pyproject_toml_file, '  "setuptools",')
            self.writeln(pyproject_toml_file, '  "wheel",')
            self.writeln(pyproject_toml_file, "]")
            self.writeln(pyproject_toml_file, 'build-backend = "setuptools.build_meta"')
        init_py_filename = os.sep.join(
            (self.models_dirname, package_dirname, "__init__.py")
        )
        with open(init_py_filename, "w") as init_py_file:
            self.writeln(init_py_file)

    @staticmethod
    def writeln(file, *args):
        file.write("".join(("".join([str(arg) for arg in args]), "\n")))

    def log_incorrect_annotation(
        self, temp_log_file, token, correct_referred_token, incorrect_referred_token
    ):
        doc = token.doc
        self.writeln(temp_log_file, "Incorrect annotation:")
        start_token_index = min(correct_referred_token.i, incorrect_referred_token.i)
        sentence_start_index = doc._.coref_chains.temp_sent_starts[
            doc[start_token_index]._.coref_chains.temp_sent_index
        ]
        if token._.coref_chains.temp_sent_index + 1 == len(
            doc._.coref_chains.temp_sent_starts
        ):
            self.writeln(temp_log_file, doc[sentence_start_index:])
            self.writeln(
                temp_log_file, "Tokens from ", sentence_start_index, " to the end:"
            )
            self.writeln(temp_log_file, doc[sentence_start_index:])
        else:
            sentence_end_index = doc._.coref_chains.temp_sent_starts[
                token._.coref_chains.temp_sent_index + 1
            ]
            self.writeln(
                temp_log_file,
                "Tokens ",
                sentence_start_index,
                " to ",
                sentence_end_index,
                ":",
            )
            self.writeln(temp_log_file, doc[sentence_start_index:sentence_end_index])
        self.writeln(temp_log_file, "Referring pronoun: ", token, " at index ", token.i)
        for potential_referred in token._.coref_chains.temp_potential_referreds:
            if hasattr(potential_referred, "true_in_training"):
                self.writeln(
                    temp_log_file,
                    "Training referred mentions: ",
                    potential_referred.pretty_representation,
                )
        self.writeln(
            temp_log_file,
            "Annotated referred mentions: ",
            [chain.pretty_representation for chain in token._.coref_chains],
        )
        self.writeln(temp_log_file)

    def train_thinc_model(
        self,
        document_pair_infos: List[DocumentPairInfo],
        test_docs: List[Doc],
        nlp: Language,
        vectors_nlp: Language,
        feature_table: FeatureTable,
    ) -> Model:
        print()
        print("Generating model ...")
        model = create_thinc_model()
        print()
        optimizer = Adam(0.001)
        epoch = 1
        last_epoch_accuracy = 0.0
        last_epoch_model = model.to_bytes()
        while True:
            print("Epoch", epoch)
            batches = model.ops.minibatch(1, document_pair_infos, shuffle=True)
            loss_calc = SequenceCategoricalCrossentropy(normalize=True)
            losses = []
            for batch_number, X in enumerate(tqdm(batches)):
                Y = []
                for x in X:
                    Y.extend(x.training_outputs)
                if epoch == 1 and batch_number == 0:
                    model.initialize(X=X, Y=Y)  # type: ignore[arg-type]
                Yh, backprop = cast(
                    Tuple[List[Floats2d], Callable], model.begin_update(X)
                )
                grads, loss = loss_calc(Yh, Y)
                losses.append(loss.tolist())  # type: ignore[attr-defined]
                backprop(grads)
                model.finish_update(optimizer)
            print(
                "Average absolute loss:",
                round(sum(losses) / len(losses), 6),
            )
            print()
            annotator = Annotator(nlp, vectors_nlp, feature_table, model)
            correct_counter = incorrect_counter = 0
            for test_doc in tqdm(test_docs):
                for token in test_doc:
                    token._.coref_chains.chains = []
                annotator.annotate(test_doc, used_in_training=True)
                for token in test_doc:
                    if hasattr(token._.coref_chains, "temp_potential_referreds"):
                        for (
                            potential_referred
                        ) in token._.coref_chains.temp_potential_referreds:
                            if hasattr(potential_referred, "true_in_training"):
                                for chain in token._.coref_chains:
                                    if Mention(token, False) not in chain:
                                        continue
                                    if potential_referred in chain:
                                        correct_counter += 1
                                    else:
                                        incorrect_counter += 1
            accuracy = round(
                100 * correct_counter / (correct_counter + incorrect_counter), 2
            )
            print("Accuracy: ", "".join((str(accuracy), "%")))
            if accuracy < last_epoch_accuracy:
                print("Saving model from epoch", epoch - 1)
                model = create_thinc_model()
                model.from_bytes(last_epoch_model)
                return model
            else:
                last_epoch_accuracy = accuracy
                last_epoch_model = model.to_bytes()
                epoch += 1

    def load_documents(self, nlp, rules_analyzer):
        docs = []
        for loader in self.loaders:
            docs.extend(loader.load(self.data_dir, nlp, rules_analyzer))
        return docs

    def train_or_check(self, config_entry_name: str, config_entry, temp_log_file):
        self.writeln(temp_log_file, "Config entry name: ", config_entry_name)
        nlp_name = "_".join((self.lang, config_entry["model"]))
        nlp = self.nlp_dict[nlp_name]
        self.writeln(
            temp_log_file, "Spacy model: ", nlp_name, " version ", nlp.meta["version"]
        )
        if self.train_not_check and config_entry["train_version"] != nlp.meta["version"]:
            raise ModelNotSupportedError("Declared train_version does not match loaded spaCy version")
        if "vectors_model" in config_entry:
            vectors_nlp_name = "_".join((self.lang, config_entry["vectors_model"]))
            vectors_nlp = self.nlp_dict[vectors_nlp_name]
            self.writeln(
                temp_log_file,
                "Spacy vectors model: ",
                vectors_nlp_name,
                " version ",
                vectors_nlp.meta["version"],
            )
        else:
            vectors_nlp = nlp
            self.writeln(temp_log_file, "Main model is being used as vectors model")

        rules_analyzer = RulesAnalyzerFactory().get_rules_analyzer(nlp)
        docs = self.load_documents(nlp, rules_analyzer)
        rand = Random(0.47)
        for _ in range(100):
            bisection = int(rand.random() * len(docs))
            docs = docs[bisection:] + docs[:bisection]

        # Separate into training and test for first run
        total_words = 0
        docs_to_total_words_position = []
        for doc in docs:
            docs_to_total_words_position.append(total_words)
            total_words += len(doc)
        split_index = bisect.bisect_right(
            docs_to_total_words_position, total_words * 0.8
        )
        training_docs = docs[:split_index]
        test_docs = docs[split_index:]
        self.writeln(temp_log_file, "Loaders:", self.loaders)
        self.writeln(temp_log_file, "Data directory:", self.data_dir)
        self.writeln(temp_log_file, "Total words: ", total_words)
        self.writeln(
            temp_log_file,
            "Training docs: ",
            len(training_docs),
            "; test docs: ",
            len(test_docs),
        )

        if self.train_not_check:
            feature_table = generate_feature_table(docs, nlp)
            self.writeln(temp_log_file, "Feature table: ", feature_table.__dict__)

            print()
            tendencies_analyzer = TendenciesAnalyzer(
                rules_analyzer, vectors_nlp, feature_table
            )
            prefer_gpu()
            print("Creating Document Pair Infos ...")
            document_pair_infos = []
            for training_doc in tqdm(training_docs):
                dpi = DocumentPairInfo.from_doc(
                    training_doc, tendencies_analyzer, ENSEMBLE_SIZE, is_train=True
                )
                if len(dpi.candidates.dataXd) > 0:
                    document_pair_infos.append(dpi)

            model = self.train_thinc_model(
                document_pair_infos,
                test_docs,
                nlp,
                vectors_nlp,
                feature_table,
            )
            annotator = Annotator(nlp, vectors_nlp, feature_table, model)
        else:
            annotator = get_annotator(
                nlp=nlp, vectors_nlp=vectors_nlp, config_entry_name=config_entry_name
            )
        self.writeln(temp_log_file)
        correct_counter = incorrect_counter = 0
        print("Analysing test documents...")
        for test_doc in tqdm(test_docs):
            for token in test_doc:
                token._.coref_chains.chains = []
            annotator.annotate(test_doc, used_in_training=True)
            self.writeln(temp_log_file, "test_doc ", test_doc[:100], "... :")
            self.writeln(temp_log_file)
            self.writeln(temp_log_file, "Coref chains:")
            self.writeln(temp_log_file)
            for chain in test_doc._.coref_chains:
                self.writeln(temp_log_file, chain.pretty_representation)
            self.writeln(temp_log_file)
            self.writeln(temp_log_file, "Incorrect annotations:")
            self.writeln(temp_log_file)
            for token in test_doc:
                if hasattr(token._.coref_chains, "temp_potential_referreds"):
                    for (
                        potential_referred
                    ) in token._.coref_chains.temp_potential_referreds:
                        if hasattr(potential_referred, "true_in_training"):
                            for chain in token._.coref_chains:
                                if Mention(token, False) not in chain:
                                    continue
                                if potential_referred in chain:
                                    correct_counter += 1
                                else:
                                    incorrect_counter += 1
                                    self.log_incorrect_annotation(
                                        temp_log_file,
                                        token,
                                        token.doc[potential_referred.root_index],
                                        token.doc[chain.mentions[0].root_index],
                                    )
        if len(test_docs) > 0:
            accuracy = round(
                100 * correct_counter / (correct_counter + incorrect_counter), 2
            )
            self.writeln(temp_log_file)
            self.writeln(
                temp_log_file,
                "Correct: ",
                correct_counter,
                "; Incorrect: ",
                incorrect_counter,
                " (",
                accuracy,
                "%)",
            )
            print(
                "".join(
                    (
                        "Correct: ",
                        str(correct_counter),
                        "; Incorrect: ",
                        str(incorrect_counter),
                        " (",
                        str(accuracy),
                        "%)",
                    )
                )
            )
        if self.train_not_check:
            this_model_dir = os.sep.join(
                (
                    self.models_dirname,
                    "".join((COMMON_MODELS_PACKAGE_NAMEPART, self.lang)),
                    config_entry_name,
                )
            )
            os.mkdir(this_model_dir)
            init_py_filename = os.sep.join((this_model_dir, "__init__.py"))
            with open(init_py_filename, "w") as init_py_file:
                self.writeln(init_py_file)
            feature_table_filename = os.sep.join(
                (this_model_dir, FEATURE_TABLE_FILENAME)
            )
            with open(feature_table_filename, "wb") as feature_table_file:
                pickle.dump(feature_table, feature_table_file)
            thinc_model_filename = "".join(
                (this_model_dir, os.sep, THINC_MODEL_FILENAME)
            )
            model.to_disk(thinc_model_filename)

    def train_models(self):
        assert self.train_not_check
        for config_entry_name in self.relevant_config_entry_names:
            config_entry = self.config[config_entry_name]
            print("Processing", config_entry_name, "...")
            temp_log_filename = "".join(
                (self.log_dir, os.sep, "temp", os.sep, config_entry_name, ".log")
            )
            with open(temp_log_filename, "w", encoding="utf-8") as temp_log_file:
                self.train_or_check(config_entry_name, config_entry, temp_log_file)
        timestamp = datetime.now().isoformat(timespec="microseconds")
        sanitized_timestamp = "".join([ch for ch in timestamp if ch.isalnum()])
        zip_filename = "".join(
            (
                self.log_dir,
                os.sep,
                "train_log_",
                self.lang,
                "_",
                sanitized_timestamp,
                ".zip",
            )
        )
        shutil.make_archive(zip_filename, "zip", os.sep.join((self.log_dir, "temp")))
        temp_dir = os.sep.join((self.log_dir, "temp"))
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
        zip_filename = "".join(
            (
                self.models_dirname,
                os.sep,
                "..",
                os.sep,
                COMMON_MODELS_PACKAGE_NAMEPART,
                self.lang,
            )
        )
        if os.path.isfile(".".join((zip_filename, "zip"))):
            os.remove(".".join((zip_filename, "zip")))
        build_dir = os.sep.join((self.models_dirname, "build"))
        if os.path.isdir(build_dir):
            shutil.rmtree(build_dir)
        shutil.make_archive(zip_filename, "zip", self.models_dirname)

    def check_models(self):
        assert not self.train_not_check
        for config_entry_name in self.relevant_config_entry_names:
            config_entry = self.config[config_entry_name]
            print("Checking", config_entry_name, "...")
            temp_log_filename = "".join(
                (self.log_dir, os.sep, "temp", os.sep, config_entry_name, ".log")
            )
            with open(temp_log_filename, "w", encoding="utf-8") as temp_log_file:
                self.train_or_check(config_entry_name, config_entry, temp_log_file)
        timestamp = datetime.now().isoformat(timespec="microseconds")
        sanitized_timestamp = "".join([ch for ch in timestamp if ch.isalnum()])
        zip_filename = "".join(
            (
                self.log_dir,
                os.sep,
                "check_log_",
                self.lang,
                "_",
                sanitized_timestamp,
                ".zip",
            )
        )
        shutil.make_archive(zip_filename, "zip", os.sep.join((self.log_dir, "temp")))
        temp_dir = os.sep.join((self.log_dir, "temp"))
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
