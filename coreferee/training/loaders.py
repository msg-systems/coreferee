# Copyright 2021 msg systems ag
# Modifications Copyright 2021 Valentin-Gabriel Soumah

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import xml.sax
import os
import re
from sys import maxsize
import bisect
from abc import ABC, abstractmethod
from spacy.language import Language
from spacy.tokens import Doc
from ..data_model import Mention
from ..rules import RulesAnalyzer


class GenericLoader(ABC):
    @abstractmethod
    def load(
        self, directory_name: str, nlp: Language, rules_analyzer: RulesAnalyzer
    ) -> list:
        """Loads training data from *directory_name* to produce a list of documents parsed using
        the spacy model *nlp*. Each document goes through *RulesAnalyzer.initialize()*.
        Wherever an anaphor points to a referred mention in the training data, the
        mention within *token._.coref_chains.temp_potential_referreds* is annotated with
        *true_in_training=True*."""


class ParCorHandler(xml.sax.ContentHandler):
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
        words_filename: str,
        coref_level_filename: str,
        nlp: Language,
        rules_analyzer: RulesAnalyzer,
        parser,
    ) -> None:
        parcor_handler = ParCorHandler()
        parser.setContentHandler(parcor_handler)
        parser.parse(words_filename)
        parser.parse(coref_level_filename)
        doc = nlp(" ".join(word for word in parcor_handler.words))
        rules_analyzer.initialize(doc)
        lookup = []
        spacy_token_iterator = enumerate(token for token in doc)
        for parcor_token in parcor_handler.words:
            this_parcor_token_lookup = []
            while len(parcor_token) > 0:
                spacy_token_index, spacy_token = next(
                    spacy_token_iterator, (None, None)
                )
                if spacy_token_index is None or not parcor_token.startswith(
                    spacy_token.text
                ):
                    break
                this_parcor_token_lookup.append(spacy_token_index)
                parcor_token = parcor_token[len(spacy_token) :]
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
    ) -> list:
        parser = xml.sax.make_parser()
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)
        docs = []
        for words_filename in (
            w for w in os.scandir(directory_name) if w.path.endswith("words.xml")
        ):
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
        doc: Doc, ann_file_lines: list, rules_analyzer: RulesAnalyzer
    ) -> None:
        rules_analyzer.initialize(doc)
        token_char_start_indexes = [token.idx for token in doc]
        mention_numbers_to_spans = {}
        mention_numbers_to_set_numbers = {}
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
    ) -> list:
        txt_file_contents = []
        ann_file_lines_list = []
        for index, txt_filename in enumerate(
            t for t in os.scandir(directory_name) if t.path.endswith(".txt")
        ):
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
        mention_labels_to_span_sets = {}
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
    ) -> list:
        txt_file_contents = []
        ann_file_lines_list = []
        for index, txt_filename in enumerate(
            t for t in os.scandir(directory_name) if t.path.endswith(".txt")
        ):
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


class DEMOCRATConllLoader(GenericLoader):
    """
    Loader of the conll file format of DEMOCRAT corpus.
    Should also work for any Conll file with annotated coreference
    after a few adaptations on the token separators
    """

    @staticmethod
    def load_file(
        doc: Doc, mentions: dict, rules_analyzer: RulesAnalyzer, verbose: bool = False
    ) -> None:
        rules_analyzer.initialize(doc)
        token_start_end_indexes = {
            (token.idx, token.idx + len(token.text) - 1): token.i for token in doc
        }
        mention_labels_to_span_sets = {}
        for index, mention_span in enumerate(mentions):
            mention_label = mentions[mention_span]
            for token_start_end_index in token_start_end_indexes:
                token_start_char_index, token_end_char_index = token_start_end_index
                if token_start_char_index <= mention_span[0] <= token_end_char_index:
                    if token_start_char_index != mention_span[0] and verbose:
                        print(
                            "Spacy Missed a token limit at the beginning of token",
                            token_start_char_index,
                            mention_span,
                        )
                        i = token_start_end_indexes[token_start_end_index]
                        print(doc[i], doc[i - 5 : i], doc[i : i + 5])

                    mention_start_index = token_start_end_indexes[token_start_end_index]
                if token_start_char_index <= mention_span[1] <= token_end_char_index:
                    if token_end_char_index != mention_span[1] and verbose:
                        print(
                            "Spacy Missed token limit at the end of token",
                            token_start_char_index,
                            mention_span,
                        )
                        i = token_start_end_indexes[token_start_end_index]
                        print(
                            doc[i],
                            doc.text[token_start_char_index : token_end_char_index + 1],
                            doc[i - 5 : i],
                            doc[i : i + 5],
                        )
                    mention_end_index = token_start_end_indexes[token_start_end_index]
                    break

            span = doc[mention_start_index:mention_end_index]
            if mention_label in mention_labels_to_span_sets:
                mention_labels_to_span_sets[mention_label].add(span)
            else:
                mention_labels_to_span_sets[mention_label] = {span}

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
                                if verbose:
                                    print(
                                        "COREF BEFORE",
                                        previous_span.root,
                                        doc[working_referent.root_index],
                                        doc[
                                            previous_span.start
                                            - 5 : working_referent.root_index
                                            + 5
                                        ],
                                        sep=" | ",
                                    )
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
                                if verbose:
                                    print(
                                        "COREF AFTER",
                                        doc[working_referent.root_index],
                                        next_span.root,
                                        doc[
                                            working_referent.root_index
                                            - 5 : next_span.end
                                            + 5
                                        ],
                                        sep=" | ",
                                    )

    def turn_spans_to_mentions(
        self, coreference_spans: dict, j=0, txt="", verbose=False
    ):
        mentions = {}
        for i, coreference_span in enumerate(coreference_spans):
            start_char, end_char = coreference_span
            for reference in coreference_spans[coreference_span]:
                mention_label = re.search(r"\d+", reference).group(0)
                if reference.startswith("(") and reference.endswith(")"):
                    mentions[coreference_span] = mention_label
                elif reference.startswith("("):
                    mentions[(start_char, str(i + len(mentions)))] = mention_label

                elif reference.endswith(")"):
                    opened_coreference = False
                    for browsed_mention_span, browsed_mention_label in reversed(
                        mentions.items()
                    ):
                        if browsed_mention_label == mention_label and isinstance(
                            browsed_mention_span[1], str
                        ):

                            new_mention_span = list(browsed_mention_span)
                            new_mention_span[1] = end_char
                            new_mention_span = tuple(new_mention_span)
                            if new_mention_span in mentions and verbose:
                                print(
                                    "DUPLICATE",
                                    new_mention_span,
                                    mentions[new_mention_span],
                                    j,
                                    txt[
                                        new_mention_span[0]
                                        - 10 : new_mention_span[1]
                                        + 10
                                    ],
                                    sep=" | ",
                                )

                            mentions[new_mention_span] = mention_label
                            del mentions[browsed_mention_span]
                            opened_coreference = True
                            break
                    if not opened_coreference:
                        print(
                            txt[coreference_span[0] - 10 : coreference_span[1] + 10],
                            mention_label,
                            j,
                            txt,
                            sep=" | ",
                        )
                        raise LookupError(
                            f"closed coreference with no beginning : {coreference_span}|{mention_label}|{i}"
                        )
        return mentions

    def load(
        self,
        directory_name: str,
        nlp: Language,
        rules_analyzer: RulesAnalyzer,
        verbose=False,
        return_spans=False,
    ) -> list:
        txt_file_contents = []
        doc_mentions_spans = []

        for filename in os.scandir(directory_name):
            if not filename.path.endswith("conll"):
                continue
            with open(filename, "r", encoding="UTF8") as conll_file:
                for line in conll_file:
                    if line.startswith("#begin document"):
                        j = re.search(r"part (\d+)", line).group(1)
                        token_start, token_end = 0, -1
                        first_token = True
                        tokens_list = []
                        coreference_spans_dict = {}
                    elif line.startswith("#end document"):
                        txt_file_contents.append("".join(tokens_list))
                        doc_mentions_spans.append(
                            self.turn_spans_to_mentions(
                                coreference_spans_dict,
                                j,
                                "".join(tokens_list),
                                verbose=verbose,
                            )
                        )

                    elif line != "\n":
                        columns = line.split(" " * 10)
                        token = columns[3]

                        coreference_labels = columns[-1].strip("\n")
                        coreference_labels = (
                            coreference_labels.split("|")
                            if coreference_labels != "_"
                            else []
                        )
                        sep = " "
                        if (
                            token in (".", ",", ")", "'")
                            or first_token
                            or tokens_list[-1].endswith("'")
                            or re.match(r"\-\w+", token)
                            or tokens_list[-1].endswith("-")
                            or token.startswith("-")
                        ):
                            sep = ""

                        token_start = token_end + len(sep) + 1
                        token_end = token_start + len(token) - 1
                        tokens_list.append(sep + token)
                        coreference_spans_dict[
                            (token_start, token_end)
                        ] = coreference_labels
                        first_token = False

        docs = nlp.pipe(txt_file_contents)
        docs_to_return = []
        for index, doc in enumerate(docs):
            if index % 10 == 0:
                print("Loaded", index, "documents")
            self.load_file(
                doc, doc_mentions_spans[index], rules_analyzer, verbose=verbose
            )
            docs_to_return.append(doc)
        if return_spans:
            return docs_to_return, doc_mentions_spans
        return docs_to_return
