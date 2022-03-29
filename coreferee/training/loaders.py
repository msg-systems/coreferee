from typing import List, Dict, Set, cast
from xml.sax import make_parser
from xml.sax.handler import ContentHandler, feature_namespaces
import os
from sys import maxsize
import bisect
from abc import ABC, abstractmethod
from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from ..data_model import Mention
from ..rules import RulesAnalyzer


class GenericLoader(ABC):
    @abstractmethod
    def load(
        self, directory_name: str, nlp: Language, rules_analyzer: RulesAnalyzer
    ) -> List[Doc]:
        """Loads training data from *directory_name* to produce a list of documents parsed using
        the spacy model *nlp*. Each document goes through *RulesAnalyzer.initialize()*.
        Wherever an anaphor points to a referred mention in the training data, the
        mention within *token._.coref_chains.temp_potential_referreds* is annotated with
        *true_in_training=True*."""


class ParCorHandler(ContentHandler):
    def __init__(self):
        super().__init__()
        self._current_tag = ""
        self._working_word = None
        self.words = []
        self.words.append(".")
        self.corefs = {}

    def startElement(self, tag, attributes):
        if tag == "word":
            self._current_tag = "word"
        else:
            self._current_tag = ""
        if tag == "markable":
            coref_class = attributes.getValue("coref_class")
            if coref_class != "empty":
                label = coref_class[4:]
                span_value = attributes.getValue("span")
                if "," in span_value:
                    span_value = span_value[span_value.index(",") + 1 :]
                span_values = span_value.split("..")
                start = int(span_values[0][5:])
                end = int(span_values[1][5:] if len(span_values) > 1 else start)
                if label not in self.corefs:
                    self.corefs[label] = []
                self.corefs[label].append((start, end))

    def endElement(self, tag):
        if self._working_word is not None:
            self.words.append(self._working_word)
            self._working_word = None

    def characters(self, content):
        content = content.strip()
        if self._current_tag == "word" and len(content) > 0:
            if self._working_word is None:
                self._working_word = content
            else:
                self._working_word = "".join((self._working_word, content))


class ParCorLoader(GenericLoader):
    @staticmethod
    def load_file(
        words_filename: os.DirEntry,
        coref_level_filename: str,
        nlp: Language,
        rules_analyzer: RulesAnalyzer,
        parser,
    ) -> Doc:
        parcor_handler = ParCorHandler()
        parser.setContentHandler(parcor_handler)
        parser.parse(words_filename)
        parser.parse(coref_level_filename)
        doc = nlp(" ".join(word for word in parcor_handler.words))
        rules_analyzer.initialize(doc)
        lookup = []
        spacy_token_iterator = enumerate(doc)
        for parcor_token in parcor_handler.words:
            this_parcor_token_lookup = []
            while len(parcor_token) > 0:
                spacy_token_index, spacy_token = next(
                    spacy_token_iterator, (None, None)
                )
                if spacy_token_index is None or not parcor_token.startswith(
                    spacy_token.text  # type:ignore[union-attr]
                ):
                    break
                this_parcor_token_lookup.append(spacy_token_index)
                parcor_token = parcor_token[len(spacy_token) :]  # type:ignore[arg-type]
            assert (
                len(this_parcor_token_lookup) > 0
            ), "Unmatched parcor and spacy tokens"
            lookup.append(this_parcor_token_lookup)

        for parcor_spans in parcor_handler.corefs.values():
            thinned_parcor_spans = (
                []
            )  # only those spans that are relevant to the types of
            # coreference we are learning
            for parcor_span in parcor_spans:
                holmes_span = doc[
                    lookup[parcor_span[0]][0] : lookup[parcor_span[1]][-1] + 1
                ]
                if rules_analyzer.is_independent_noun(
                    holmes_span.root
                ) or rules_analyzer.is_potential_anaphor(holmes_span.root):
                    thinned_parcor_spans.append(parcor_span)
            thinned_parcor_spans.sort(key=lambda span: span[0])
            for index, parcor_span in enumerate(thinned_parcor_spans):
                holmes_span = doc[
                    lookup[parcor_span[0]][0] : lookup[parcor_span[1]][-1] + 1
                ]
                include_dependent_siblings = (
                    len(holmes_span.root._.coref_chains.temp_dependent_siblings) > 0
                    and holmes_span.root._.coref_chains.temp_dependent_siblings[-1].i
                    <= lookup[parcor_span[1]][-1]
                )
                working_referent = Mention(holmes_span.root, include_dependent_siblings)
                marked = False
                if index > 0:
                    previous_parcor_span = thinned_parcor_spans[index - 1]
                    previous_holmes_span = doc[
                        lookup[previous_parcor_span[0]][0] : lookup[
                            previous_parcor_span[1]
                        ][-1]
                        + 1
                    ]
                    if hasattr(
                        previous_holmes_span.root._.coref_chains,
                        "temp_potential_referreds",
                    ):
                        for (
                            mention
                        ) in (
                            previous_holmes_span.root._.coref_chains.temp_potential_referreds
                        ):
                            if mention == working_referent:
                                mention.true_in_training = True
                                marked = True
                                continue
                if not marked and index < len(thinned_parcor_spans) - 1:
                    next_parcor_span = thinned_parcor_spans[index + 1]
                    next_holmes_span = doc[
                        lookup[next_parcor_span[0]][0] : lookup[next_parcor_span[1]][-1]
                        + 1
                    ]
                    if hasattr(
                        next_holmes_span.root._.coref_chains, "temp_potential_referreds"
                    ):
                        for (
                            mention
                        ) in (
                            next_holmes_span.root._.coref_chains.temp_potential_referreds
                        ):
                            if mention == working_referent:
                                mention.true_in_training = True
                                continue
        return doc

    def load(
        self, directory_name: str, nlp: Language, rules_analyzer: RulesAnalyzer
    ) -> List[Doc]:
        parser = make_parser()
        parser.setFeature(feature_namespaces, 0)
        docs = []
        words_filenames = [
            w for w in os.scandir(directory_name) if w.path.endswith("words.xml")
        ]
        words_filenames.sort(key=lambda entry: entry.name)
        for words_filename in words_filenames:
            coref_data_filename = "".join(
                (words_filename.name[:-10], "_coref_level.xml")
            )
            coref_data_full_filename = os.sep.join(
                (directory_name, coref_data_filename)
            )
            if not os.path.isfile(coref_data_full_filename):
                raise RuntimeError(" ".join((coref_data_full_filename, "not found.")))
            print("Loading", words_filename.path)
            docs.append(
                self.load_file(
                    words_filename,
                    coref_data_full_filename,
                    nlp,
                    rules_analyzer,
                    parser,
                )
            )
        return docs


class PolishCoreferenceCorpusANNLoader(GenericLoader):
    @staticmethod
    def load_file(
        doc: Doc, ann_file_lines: List[str], rules_analyzer: RulesAnalyzer
    ) -> None:
        rules_analyzer.initialize(doc)
        token_char_start_indexes = [token.idx for token in doc]
        mention_numbers_to_spans = {}
        mention_numbers_to_set_numbers: Dict[str, int] = {}
        for index, ann_file_line in enumerate(ann_file_lines):
            words = ann_file_line.split()
            if words[0].startswith("T"):
                assert words[1] == "Mention"
                end_word = words[3]
                end_index = 3
                while ";" in end_word:
                    end_index += 1
                    end_word = words[end_index]
                span = doc[
                    bisect.bisect_left(
                        token_char_start_indexes, int(words[2])
                    ) : bisect.bisect_left(token_char_start_indexes, int(end_word))
                ]
                mention_numbers_to_spans[words[0]] = span
            if words[0] == "*" and words[1] == "Coref":
                lowest_already_defined_set_number = maxsize
                for mention_number in (words[ref] for ref in range(2, len(words))):
                    if (
                        mention_number in mention_numbers_to_set_numbers
                        and mention_numbers_to_set_numbers[mention_number]
                        < lowest_already_defined_set_number
                    ):
                        lowest_already_defined_set_number = (
                            mention_numbers_to_set_numbers[mention_number]
                        )
                if lowest_already_defined_set_number < maxsize:
                    for mention_number in (words[ref] for ref in range(2, len(words))):
                        if (
                            mention_number in mention_numbers_to_set_numbers
                            and mention_numbers_to_set_numbers[mention_number]
                            > lowest_already_defined_set_number
                        ):
                            # an intermediate set, so redefine it as part of the lowest set
                            for working_mention_number in (
                                m
                                for m in mention_numbers_to_set_numbers
                                if mention_numbers_to_set_numbers[m]
                                == mention_numbers_to_set_numbers[mention_number]
                            ):
                                mention_numbers_to_set_numbers[
                                    working_mention_number
                                ] = lowest_already_defined_set_number
                    this_set_number = lowest_already_defined_set_number
                else:
                    this_set_number = index
                for mention_number in (words[ref] for ref in range(2, len(words))):
                    mention_numbers_to_set_numbers[mention_number] = this_set_number
        for set_number in sorted(list(set(mention_numbers_to_set_numbers.values()))):
            spans = []
            for mention_number in sorted(
                [
                    m
                    for m in mention_numbers_to_set_numbers
                    if mention_numbers_to_set_numbers[m] == set_number
                ],
                key=lambda m: int(m[1:]),
            ):
                span_to_check = mention_numbers_to_spans[mention_number]
                if rules_analyzer.is_independent_noun(
                    span_to_check.root
                ) or rules_analyzer.is_potential_anaphor(span_to_check.root):
                    spans.append(span_to_check)
            for index, span in enumerate(spans):
                include_dependent_siblings = (
                    len(span.root._.coref_chains.temp_dependent_siblings) > 0
                    and span.root._.coref_chains.temp_dependent_siblings[-1].i
                    < span.end
                )
                working_referent = Mention(span.root, include_dependent_siblings)
                marked = False
                if index > 0:
                    previous_span = spans[index - 1]
                    if hasattr(
                        previous_span.root._.coref_chains, "temp_potential_referreds"
                    ):
                        for (
                            mention
                        ) in previous_span.root._.coref_chains.temp_potential_referreds:
                            if mention == working_referent:
                                mention.true_in_training = True
                                marked = True
                                continue
                if not marked and index < len(spans) - 1:
                    next_span = spans[index + 1]
                    if hasattr(
                        next_span.root._.coref_chains, "temp_potential_referreds"
                    ):
                        for (
                            mention
                        ) in next_span.root._.coref_chains.temp_potential_referreds:
                            if mention == working_referent:
                                mention.true_in_training = True
                                continue

    def load(
        self, directory_name: str, nlp: Language, rules_analyzer: RulesAnalyzer
    ) -> List[Doc]:
        txt_file_contents = []
        ann_file_lines_list = []
        txt_filenames = [
            t for t in os.scandir(directory_name) if t.path.endswith(".txt")
        ]
        txt_filenames.sort(key=lambda entry: entry.name)
        for index, txt_filename in enumerate(txt_filenames):
            with open(txt_filename, "r", encoding="UTF8") as txt_file:
                txt_file_contents.append("".join(txt_file.readlines()))
            ann_filename = "".join((txt_filename.path[:-4], ".ann"))
            with open(ann_filename, "r", encoding="UTF8") as ann_file:
                ann_file_lines_list.append(ann_file.readlines())
        docs = nlp.pipe(txt_file_contents)
        docs_to_return = []
        for index, doc in enumerate(docs):
            if index % 10 == 0:
                print("Loaded", index, "documents")
            self.load_file(doc, ann_file_lines_list[index], rules_analyzer)
            docs_to_return.append(doc)
        return docs_to_return


class LitBankANNLoader(GenericLoader):
    @staticmethod
    def load_file(
        doc: Doc, ann_file_lines: list, rules_analyzer: RulesAnalyzer
    ) -> None:
        rules_analyzer.initialize(doc)
        token_char_start_indexes = [token.idx for token in doc]
        mention_labels_to_span_sets: Dict[str, Set[Span]] = {}
        for index, ann_file_line in enumerate(ann_file_lines):
            words = ann_file_line.split()
            if words[0].startswith("T"):  # normally always true
                span = doc[
                    bisect.bisect_left(
                        token_char_start_indexes, int(words[2])
                    ) : bisect.bisect_left(token_char_start_indexes, int(words[3]))
                ]
            if "-" in words[1]:
                if words[1] in mention_labels_to_span_sets:
                    working_span_set = mention_labels_to_span_sets[words[1]]
                else:
                    working_span_set = set()
                    mention_labels_to_span_sets[words[1]] = working_span_set
                working_span_set.add(span)
        for span_set in mention_labels_to_span_sets.values():
            spans = list(
                filter(
                    lambda span: rules_analyzer.is_independent_noun(span.root)
                    or rules_analyzer.is_potential_anaphor(span.root),
                    span_set,
                )
            )
            spans.sort(key=lambda span: span.start)
            for index, span in enumerate(spans):
                include_dependent_siblings = (
                    len(span.root._.coref_chains.temp_dependent_siblings) > 0
                    and span.root._.coref_chains.temp_dependent_siblings[-1].i
                    < span.end
                )
                working_referent = Mention(span.root, include_dependent_siblings)
                marked = False
                if index > 0:
                    previous_span = spans[index - 1]
                    if hasattr(
                        previous_span.root._.coref_chains, "temp_potential_referreds"
                    ):
                        for (
                            mention
                        ) in previous_span.root._.coref_chains.temp_potential_referreds:
                            if mention == working_referent:
                                mention.true_in_training = True
                                marked = True
                                continue
                if not marked and index < len(spans) - 1:
                    next_span = spans[index + 1]
                    if hasattr(
                        next_span.root._.coref_chains, "temp_potential_referreds"
                    ):
                        for (
                            mention
                        ) in next_span.root._.coref_chains.temp_potential_referreds:
                            if mention == working_referent:
                                mention.true_in_training = True
                                continue

    def load(
        self, directory_name: str, nlp: Language, rules_analyzer: RulesAnalyzer
    ) -> List[Doc]:
        txt_file_contents = []
        ann_file_lines_list = []
        txt_filenames = [
            t for t in os.scandir(directory_name) if t.path.endswith(".txt")
        ]
        txt_filenames.sort(key=lambda entry: entry.name)
        for index, txt_filename in enumerate(txt_filenames):
            with open(txt_filename, "r", encoding="UTF8") as txt_file:
                txt_file_contents.append("".join(txt_file.readlines()))
            ann_filename = "".join((txt_filename.path[:-4], ".ann"))
            with open(ann_filename, "r", encoding="UTF8") as ann_file:
                ann_file_lines_list.append(ann_file.readlines())
        docs = nlp.pipe(txt_file_contents)
        docs_to_return = []
        for index, doc in enumerate(docs):
            if index % 10 == 0:
                print("Loaded", index, "documents")
            self.load_file(doc, ann_file_lines_list[index], rules_analyzer)
            docs_to_return.append(doc)
        return docs_to_return


class ConllLoader(GenericLoader):
    @staticmethod
    def load_file(
        conll_filename: os.DirEntry, nlp: Language, rules_analyzer: RulesAnalyzer
    ) -> List[Doc]:
        with open(conll_filename, "r", encoding="UTF8") as conll_file:
            split_conll_lines = [
                l.split() for l in conll_file.readlines() if len(l.split()) > 10
            ]
        part_ids = sorted(list({l[1] for l in split_conll_lines}))
        docs = []
        for part_id in part_ids:
            this_part_split_conll_lines = [
                l for l in split_conll_lines if l[1] == part_id
            ]
            if nlp.meta["lang"] in ("fr"):
                # Tokens ending an apostrophes have to be merged with following tokens in French,
                # otherwise parsing errors will result
                corrected_this_part_split_conll_lines: List[List[str]] = []
                index = 0
                while index < len(this_part_split_conll_lines):
                    conll_token = this_part_split_conll_lines[index][3].lstrip("/")
                    if (
                        index + 1 < len(this_part_split_conll_lines)
                        and len(conll_token) > 0
                        and len(this_part_split_conll_lines[index + 1][3]) > 0
                        and conll_token[-1] in ("'")
                    ):
                        this_part_split_conll_lines[index][
                            3
                        ] += this_part_split_conll_lines[index + 1][3].lstrip("/")
                        if this_part_split_conll_lines[index + 1][-1] not in ("-", "_"):
                            if this_part_split_conll_lines[index][-1] not in ("-", "_"):
                                this_part_split_conll_lines[index][-1] += (
                                    "|" + this_part_split_conll_lines[index + 1][-1]
                                )
                            else:
                                this_part_split_conll_lines[index][
                                    -1
                                ] = this_part_split_conll_lines[index + 1][-1]
                        corrected_this_part_split_conll_lines.append(
                            this_part_split_conll_lines[index]
                        )
                        index += 2
                    else:
                        corrected_this_part_split_conll_lines.append(
                            this_part_split_conll_lines[index]
                        )
                        index += 1
                this_part_split_conll_lines = corrected_this_part_split_conll_lines
            conll_tokens = [l[3].lstrip("/") for l in this_part_split_conll_lines]
            doc = nlp(" ".join(conll_tokens))
            rules_analyzer.initialize(doc)
            conll_to_spacy_lookup = (
                []
            )  # indexes correspond to conll token indexes, entries are lists of spaCy tokens
            spacy_token_iterator = enumerate(token for token in doc)
            for conll_token in conll_tokens:
                this_conll_token_lookup = []
                while len(conll_token) > 0:
                    spacy_token_index, spacy_token = next(
                        spacy_token_iterator, (None, None)
                    )
                    if spacy_token_index is None:
                        break
                    spacy_token = cast(Token, spacy_token)
                    if not conll_token.startswith(spacy_token.text):
                        break
                    if spacy_token.pos_ == "SPACE":
                        continue
                    this_conll_token_lookup.append(spacy_token_index)
                    conll_token = conll_token[len(spacy_token) :]
                conll_to_spacy_lookup.append(this_conll_token_lookup)
            working_spans = (
                {}
            )  # // from chain index numbers to spaCy start token indexes
            chains: Dict[
                str, List[Span]
            ] = {}  # from chain index numbers to lists of spaCy spans
            for conll_token_index, chain_markers in enumerate(
                l[-1] for l in this_part_split_conll_lines
            ):
                if chain_markers in ("-", "_"):
                    continue
                for chain_marker in chain_markers.split("|"):
                    chain_index = "".join([d for d in chain_marker if d.isdigit()])
                    if "(" in chain_marker:
                        working_spans[chain_index] = conll_to_spacy_lookup[
                            conll_token_index
                        ][0]
                    if (
                        ")" in chain_marker and chain_index in working_spans
                    ):  # sometimes errors in OntoNotes -> not the case
                        this_span = doc[
                            working_spans[chain_index] : conll_to_spacy_lookup[
                                conll_token_index
                            ][-1]
                            + 1
                        ]
                        del working_spans[chain_index]
                        if rules_analyzer.is_independent_noun(
                            this_span.root
                        ) or rules_analyzer.is_potential_anaphor(this_span.root):
                            if chain_index in chains:
                                chains[chain_index].append(this_span)
                            else:
                                chains[chain_index] = [this_span]
            for chain in (c for c in chains.values() if len(c) > 1):
                chain.sort(key=lambda span: span[0])  # type: ignore[arg-type, return-value]
                for span_index, span in enumerate(chain):
                    include_dependent_siblings = (
                        len(span.root._.coref_chains.temp_dependent_siblings) > 0
                        and span.root._.coref_chains.temp_dependent_siblings[-1] in span
                    )
                    working_referent = Mention(span.root, include_dependent_siblings)
                    if span_index > 0:
                        previous_span = chain[span_index - 1]
                        if (
                            hasattr(
                                previous_span.root._.coref_chains,
                                "temp_potential_referreds",
                            )
                            and Mention.number_of_training_mentions_marked_true(
                                previous_span.root
                            )
                            == 0
                        ):
                            for (
                                mention
                            ) in (
                                previous_span.root._.coref_chains.temp_potential_referreds
                            ):
                                if mention == working_referent:
                                    mention.true_in_training = True
                                    continue
                    if span_index < len(chain) - 1:
                        next_span = chain[span_index + 1]
                        if (
                            hasattr(
                                next_span.root._.coref_chains,
                                "temp_potential_referreds",
                            )
                            and Mention.number_of_training_mentions_marked_true(
                                next_span.root
                            )
                            == 0
                        ):
                            for (
                                mention
                            ) in next_span.root._.coref_chains.temp_potential_referreds:
                                if mention == working_referent:
                                    mention.true_in_training = True
                                    continue
            docs.append(doc)
        return docs

    def load(
        self, directory_name: str, nlp: Language, rules_analyzer: RulesAnalyzer
    ) -> List[Doc]:
        filenames = [c for c in os.scandir(directory_name) if c.path.endswith("conll")]
        filenames.sort(key=lambda entry: entry.name)
        docs = []
        if len(filenames) > 0:
            print("Loading CONLL docs from", directory_name, "...")
            for conll_filename in filenames:
                print("Loading", conll_filename.name)
                docs.extend(self.load_file(conll_filename, nlp, rules_analyzer))
            print()
        return docs
