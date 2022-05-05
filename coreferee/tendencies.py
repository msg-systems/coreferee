from typing import List, Tuple, Callable, cast, Union, Dict, Set
from copy import copy
from dataclasses import dataclass
from thinc.model import Model
from thinc.layers import Relu, concatenate, chain, clone
from thinc.layers import Linear, noop, tuplify
from thinc.backends import Ops, get_current_ops
from thinc.types import Floats2d, Ints1d, Ragged
from spacy.tokens import Token, Doc
from spacy.language import Language
from .data_model import FeatureTable, Mention
from .rules import RulesAnalyzerFactory, RulesAnalyzer

ENSEMBLE_SIZE = 5


class TendenciesAnalyzer:
    def __init__(
        self,
        rules_analyzer: RulesAnalyzer,
        vectors_nlp: Language,
        feature_table: FeatureTable,
    ):
        self.rules_analyzer = rules_analyzer
        self.vectors_nlp = vectors_nlp
        if self.vectors_nlp.vocab[rules_analyzer.random_word].has_vector:
            self.vector_length = len(
                self.vectors_nlp.vocab[rules_analyzer.random_word].vector
            )
        else:
            self.vector_length = len(vectors_nlp(rules_analyzer.random_word)[0].vector)
        assert self.vector_length > 0
        self.feature_table = feature_table

    def get_feature_map(
        self, token_or_mention: Union[Token, Mention], doc: Doc
    ) -> List[Union[int, float]]:
        """Returns a binary list representing the features from *self.feature_table* that
        the token or any of the tokens within the mention has. The list is also
        added as *token._.coref_chains.temp_feature_map* or *mention.temp_feature_map*.
        """

        def convert_to_oneshot(reference_list, actual_list):
            """
            Returns a list of the same length as 'reference_list' where positions corresponding to
            entries in 'reference_list' that are also contained within 'actual_list' have the
            value '1' and other positions have the value '0'.
            """
            return [
                1 if reference in actual_list else 0 for reference in reference_list
            ]

        def get_oneshot_for_token_and_siblings(prop, func):
            """Executes a logical AND between the values for the respective siblings."""
            oneshot = convert_to_oneshot(prop, func(token))
            for sibling in siblings:
                sibling_oneshot = convert_to_oneshot(prop, func(sibling))
                oneshot = [
                    1 if (entry == 1 or sibling_oneshot[index] == 1) else 0
                    for (index, entry) in enumerate(oneshot)
                ]
            return oneshot

        siblings = []
        if isinstance(token_or_mention, Token):
            if hasattr(token_or_mention._.coref_chains, "temp_feature_map"):
                return token_or_mention._.coref_chains.temp_feature_map
            token = token_or_mention
        else:
            if hasattr(token_or_mention, "temp_feature_map"):
                return token_or_mention.temp_feature_map  # type:ignore[attr-defined]
            token = doc[token_or_mention.root_index]
            if len(token_or_mention.token_indexes) > 1:
                siblings = [doc[i] for i in token_or_mention.token_indexes[1:]]

        feature_map = convert_to_oneshot(self.feature_table.tags, [token.tag_])

        feature_map.extend(
            get_oneshot_for_token_and_siblings(
                self.feature_table.morphs, lambda token: token.morph
            )
        )

        feature_map.extend(
            convert_to_oneshot(self.feature_table.ent_types, [token.ent_type_])
        )

        feature_map.extend(
            get_oneshot_for_token_and_siblings(
                self.feature_table.lefthand_deps_to_children,
                lambda token: [
                    child.dep_ for child in token.children if child.i < token.i
                ],
            )
        )

        feature_map.extend(
            get_oneshot_for_token_and_siblings(
                self.feature_table.righthand_deps_to_children,
                lambda token: [
                    child.dep_ for child in token.children if child.i > token.i
                ],
            )
        )

        if token.dep_ != self.rules_analyzer.root_dep and token.i < token.head.i:
            feature_map.extend(
                convert_to_oneshot(
                    self.feature_table.lefthand_deps_to_parents, [token.dep_]
                )
            )
        else:
            feature_map.extend(
                convert_to_oneshot(self.feature_table.lefthand_deps_to_parents, [])
            )

        if token.dep_ != self.rules_analyzer.root_dep and token.i > token.head.i:
            feature_map.extend(
                convert_to_oneshot(
                    self.feature_table.righthand_deps_to_parents, [token.dep_]
                )
            )
        else:
            feature_map.extend(
                convert_to_oneshot(self.feature_table.righthand_deps_to_parents, [])
            )

        if token.dep_ != self.rules_analyzer.root_dep:
            feature_map.extend(
                convert_to_oneshot(self.feature_table.parent_tags, [token.head.tag_])
            )
        else:
            feature_map.extend(convert_to_oneshot(self.feature_table.parent_tags, []))

        if token.dep_ != self.rules_analyzer.root_dep:
            feature_map.extend(
                convert_to_oneshot(self.feature_table.parent_morphs, token.head.morph)
            )
        else:
            feature_map.extend(convert_to_oneshot(self.feature_table.parent_morphs, []))

        if token.dep_ != self.rules_analyzer.root_dep:
            feature_map.extend(
                convert_to_oneshot(
                    self.feature_table.parent_lefthand_deps_to_children,
                    [
                        child.dep_
                        for child in token.head.children
                        if child.i < token.head.i
                    ],
                )
            )
        else:
            feature_map.extend(
                convert_to_oneshot(
                    self.feature_table.parent_lefthand_deps_to_children, []
                )
            )

        if token.dep_ != self.rules_analyzer.root_dep:
            feature_map.extend(
                convert_to_oneshot(
                    self.feature_table.parent_righthand_deps_to_children,
                    [
                        child.dep_
                        for child in token.head.children
                        if child.i > token.head.i
                    ],
                )
            )
        else:
            feature_map.extend(
                convert_to_oneshot(
                    self.feature_table.parent_righthand_deps_to_children, []
                )
            )

        if isinstance(token_or_mention, Token):
            token_or_mention._.coref_chains.temp_feature_map = feature_map
        else:
            token_or_mention.temp_feature_map = feature_map  # type:ignore[attr-defined]
        return feature_map

    def get_position_map(
        self, token_or_mention: Union[Token, Mention], doc: Doc
    ) -> List[Union[int, float]]:
        """Returns a list of numbers representing the position, depth, etc. of the token or mention
        within its sentence. The list is also added as *token._.coref_chains.temp_position_map*
        or *mention.temp_position_map*.
        """

        if isinstance(token_or_mention, Token):
            if hasattr(token_or_mention._.coref_chains, "temp_position_map"):
                return token_or_mention._.coref_chains.temp_position_map
            token = token_or_mention
        else:
            if hasattr(token_or_mention, "temp_position_map"):
                return token_or_mention.temp_position_map  # type:ignore[attr-defined]
            token = doc[token_or_mention.root_index]

        # This token is the nth word within its sentence
        position_map = [
            token.i
            - token.doc._.coref_chains.temp_sent_starts[
                token._.coref_chains.temp_sent_index
            ]
        ]

        # This token is at depth n from the root
        position_map.append(len(list(token.ancestors)))

        # This token is n verbs from the root
        position_map.append(
            len(
                [
                    ancestor
                    for ancestor in token.ancestors
                    if ancestor.pos_ in self.rules_analyzer.verb_pos
                ]
            )
        )

        # This token is the nth token at its depth within its sentence
        position_map.append(
            len(
                [
                    1
                    for token_in_sentence in token.sent
                    if token_in_sentence.i < token.i
                    and len(list(token_in_sentence.ancestors))
                    == len(list(token.ancestors))
                ]
            )
        )

        # This token is the nth child of its parents
        if token.dep_ != self.rules_analyzer.root_dep:
            position_map.append(
                sorted([child.i for child in token.head.children]).index(token.i)
            )
        else:
            position_map.append(-1)

        # Number of dependent siblings, or -1 if the method was passed a mention that is within
        # a coordination phrase but only covers one token within that phrase
        if token._.coref_chains.temp_governing_sibling is not None or (
            len(token._.coref_chains.temp_dependent_siblings) > 0
            and not (
                isinstance(token_or_mention, Mention)
                and len(token_or_mention.token_indexes) > 1
            )
        ):
            position_map.append(-1)
        else:
            position_map.append(len(token._.coref_chains.temp_dependent_siblings))

        position_map.append(
            1 if token._.coref_chains.temp_governing_sibling is not None else 0
        )

        if isinstance(token_or_mention, Token):
            token_or_mention._.coref_chains.temp_position_map = position_map
        else:
            token_or_mention.temp_position_map = (  # type:ignore[attr-defined]
                position_map
            )
        return position_map

    def get_compatibility_map(
        self, referred: Mention, referring: Token
    ) -> List[Union[int, float]]:
        """Returns a list of numbers representing the interaction between *referred* and
        *referring*. It will already have been established that coreference between the two is
        possible; the compatibility map assists the neural network in ascertaining how likely
        it is. The list is also added as *referred.temp_compatibility_map*.
        """
        doc = referring.doc
        referred_root = doc[referred.root_index]

        if hasattr(referred, "temp_compatibility_map"):
            return referred.temp_compatibility_map  # type:ignore[attr-defined]

        # Referential distance in words (may be negative in the case of cataphora)
        compatibility_map = cast(
            List[Union[int, float]], [referring.i - referred_root.i]
        )

        # Referential distance in sentences
        compatibility_map.append(
            referring._.coref_chains.temp_sent_index
            - referred_root._.coref_chains.temp_sent_index
        )

        # Whether the referred mention, its lefthand sibling or its head is among the ancestors
        # of the referring element
        compatibility_map.append(
            1
            if referred_root in referring.ancestors
            or (
                referred_root.dep_ != self.rules_analyzer.root_dep
                and referred_root.head in referring.ancestors
            )
            or referred_root._.coref_chains.temp_governing_sibling is not None
            and (
                referred_root._.coref_chains.temp_governing_sibling
                in referring.ancestors
                or (
                    referred_root._.coref_chains.temp_governing_sibling.dep_
                    != self.rules_analyzer.root_dep
                    and referred_root._.coref_chains.temp_governing_sibling.head
                    in referring.ancestors
                )
            )
            else 0
        )

        # The cosine similarity of the two objects' heads' vectors
        if (
            referred_root.dep_ != self.rules_analyzer.root_dep
            and referring.dep_ != self.rules_analyzer.root_dep
        ):
            referred_head_lexeme = self.vectors_nlp.vocab[referred_root.head.lemma_]
            referring_head_lexeme = self.vectors_nlp.vocab[referring.head.lemma_]
            if referred_head_lexeme.has_vector and referring_head_lexeme.has_vector:
                compatibility_map.append(
                    referred_head_lexeme.similarity(referring_head_lexeme)
                )
            elif referred_root.has_vector and referring.has_vector:  # _sm models
                compatibility_map.append(referred_root.similarity(referring))
            else:
                compatibility_map.append(-1)
        else:
            compatibility_map.append(-1)

        # The number of common true values in the feature maps of *referred.root* and *referring*.
        referred_feature_map = self.get_feature_map(referred, referring.doc)
        referring_feature_map = self.get_feature_map(
            Mention(referring, False), referring.doc
        )
        compatibility_map.append(
            [
                1 if (entry == 1 and referring_feature_map[index] == 1) else 0
                for (index, entry) in enumerate(referred_feature_map)
            ].count(1)
        )

        referred.temp_compatibility_map = compatibility_map  # type:ignore[attr-defined]
        return compatibility_map

    def score(self, doc: Doc, thinc_ensemble: Model) -> None:
        """Scores all possible anaphoric pairs in *doc*. The scores are never referenced
        outside this method because the possible pairs on each anaphor are sorted within
        this method with the more likely interpretations at the front of the list.
        """
        document_pair_info = DocumentPairInfo.from_doc(doc, self, ENSEMBLE_SIZE)
        if len(document_pair_info.candidates.dataXd) > 0:
            scores = thinc_ensemble.predict([document_pair_info])
            referring_scores_iterator = iter(scores)
            for referring in (
                t for t in doc if hasattr(t._.coref_chains, "temp_potential_referreds")
            ):
                referring_scores = next(referring_scores_iterator)
                mention_scores_iterator = iter(referring_scores)
                for potential_referred in (
                    p for p in referring._.coref_chains.temp_potential_referreds
                ):
                    ensemble_scores = next(mention_scores_iterator)
                    potential_referred.temp_score = sum(ensemble_scores) / len(
                        ensemble_scores
                    )
                is_last = False
                try:
                    next(mention_scores_iterator)
                except StopIteration:
                    is_last = True
                assert (
                    is_last
                ), "Mismatch between potential referreds and neural network output."
            is_last = False
            try:
                next(referring_scores_iterator)
            except StopIteration:
                is_last = True
            assert (
                is_last
            ), "Mismatch between referring anaphors and neural network output."
            for referring in (
                t for t in doc if hasattr(t._.coref_chains, "temp_potential_referreds")
            ):
                referring._.coref_chains.temp_potential_referreds.sort(
                    key=lambda potential_referred: (
                        potential_referred.temp_is_uncertain,
                        0 - potential_referred.temp_score,
                    )
                )


def generate_feature_table(docs: list, nlp: Language) -> FeatureTable:

    rules_analyzer = RulesAnalyzerFactory().get_rules_analyzer(nlp)
    tags: Set[str] = set()
    morphs: Set[str] = set()
    ent_types: Set[str] = set()
    lefthand_deps_to_children: Set[str] = set()
    righthand_deps_to_children: Set[str] = set()
    lefthand_deps_to_parents: Set[str] = set()
    righthand_deps_to_parents: Set[str] = set()
    parent_tags: Set[str] = set()
    parent_morphs: Set[str] = set()
    parent_lefthand_deps_to_children: Set[str] = set()
    parent_righthand_deps_to_children: Set[str] = set()

    for doc in docs:
        for token in (
            token
            for token in doc
            if rules_analyzer.is_independent_noun(token)
            or rules_analyzer.is_potential_anaphor(token)
        ):
            tags.add(token.tag_)
            morphs.update(token.morph)
            ent_types.add(token.ent_type_)
            lefthand_deps_to_children.update(
                (child.dep_ for child in token.children if child.i < token.i)
            )
            righthand_deps_to_children.update(
                (child.dep_ for child in token.children if child.i > token.i)
            )
            if token.dep_ != rules_analyzer.root_dep:
                if token.i < token.head.i:
                    lefthand_deps_to_parents.add(token.dep_)
                else:
                    righthand_deps_to_parents.add(token.dep_)
                parent_tags.add(token.head.tag_)
                parent_morphs.update(token.head.morph)
                parent_lefthand_deps_to_children.update(
                    (
                        child.dep_
                        for child in token.head.children
                        if child.i < token.head.i
                    )
                )
                parent_righthand_deps_to_children.update(
                    (
                        child.dep_
                        for child in token.head.children
                        if child.i > token.head.i
                    )
                )

    return FeatureTable(
        tags=sorted(list(tags)),
        morphs=sorted(list(morphs)),
        ent_types=sorted(list(ent_types)),
        lefthand_deps_to_children=sorted(list(lefthand_deps_to_children)),
        righthand_deps_to_children=sorted(list(righthand_deps_to_children)),
        lefthand_deps_to_parents=sorted(list(lefthand_deps_to_parents)),
        righthand_deps_to_parents=sorted(list(righthand_deps_to_parents)),
        parent_tags=sorted(list(parent_tags)),
        parent_morphs=sorted(list(parent_morphs)),
        parent_lefthand_deps_to_children=sorted(list(parent_lefthand_deps_to_children)),
        parent_righthand_deps_to_children=sorted(
            list(parent_righthand_deps_to_children)
        ),
    )


@dataclass
class DocumentPairInfo:
    """Coreference 'questions' for a document, i.e. the referring expressions and
    their potential antecedents
    doc = "the dog bit the man . He bit it back."
    referrers = [6, 8]
    all_antecedents = [1, 5]
    antecedents = [[0, 1], [0, 1]]
    """

    doc: Doc
    referrers: Ints1d

    # Antecedents lists all antecedents for the whole batch. Each antecedent
    # can have multiple words, so they're packed into a ragged.
    # This handles cases like "the dog and the lion bit the man. He bit them back",
    # where 'them' refers to '{dog, lion}'.
    antecedents: Ragged

    # Candidates is aligned with referrer_words, and the items point into antecedents.
    # So let's say the ith referrer can have the potential antecedents:
    # [lion, animal, [lion, animal]]. Those might be items 0, 1, 2 in the antecedents
    # list. The candidates[i] would then be [0, 1, 2]
    # The two layers of indirection are necessary because otherwise we would have to have a nested
    # list: each referrer can have multiple candidates, and each candidate can be an antecedent
    # that contains multiple words.
    candidates: Ragged

    # A list specifying which referrer the candidate at each position points to.
    referrers2candidates_pointers: Ints1d

    static_infos: Floats2d
    training_outputs: List[Floats2d]

    @classmethod
    def from_doc(
        cls,
        doc: Doc,
        tendencies_analyzer: TendenciesAnalyzer,
        ensemble_size: int,
        ops=None,
        is_train: bool = False,
    ) -> "DocumentPairInfo":
        if ops is None:
            ops = get_current_ops()

        referrers_list: List[int] = []
        antecedents_list: List[List[int]] = []
        candidates_list: List[List[int]] = []
        static_infos_list: List[List[Union[float, int]]] = []
        training_outputs_list: List[List[float]] = []
        candidates2antecedents: Dict[Tuple[int, ...], int] = {}
        for token in doc:
            if not hasattr(token._.coref_chains, "temp_potential_referreds"):
                continue
            _set_vectors(tendencies_analyzer.vectors_nlp, ops, token)
            temp_potential_referreds = cast(
                List[Mention], token._.coref_chains.temp_potential_referreds
            )
            if is_train and Mention.number_of_training_mentions_marked_true(token) == 0:
                continue
            referrers_list.append(token.i)
            candidates_list.append([])
            temp_potential_referreds.sort(key=lambda m: m.root_index)
            for mention in temp_potential_referreds:
                if is_train and hasattr(mention, "spanned_in_training"):
                    continue
                # spanned in training - X->Y and Y->Z; we do want to present X->Z
                # as neither correct nor incorrect and so remove it from the
                # training data

                token_indexes = tuple(mention.token_indexes)
                if token_indexes in candidates2antecedents:
                    candidates_list[-1].append(candidates2antecedents[token_indexes])
                else:
                    candidates2antecedents[token_indexes] = len(antecedents_list)
                    candidates_list[-1].append(len(antecedents_list))
                    antecedents_list.append(mention.token_indexes)
                    for token_index in token_indexes:
                        _set_vectors(
                            tendencies_analyzer.vectors_nlp, ops, token.doc[token_index]
                        )
                static_info = copy(tendencies_analyzer.get_feature_map(token, doc))
                static_info.extend(tendencies_analyzer.get_position_map(token, doc))
                static_info.extend(tendencies_analyzer.get_feature_map(mention, doc))
                static_info.extend(tendencies_analyzer.get_position_map(mention, doc))
                static_info.extend(
                    tendencies_analyzer.get_compatibility_map(mention, token)
                )
                static_infos_list.append(static_info)
                if is_train:
                    training_outputs_list.append(
                        [1.0] * ensemble_size
                        if hasattr(mention, "true_in_training")
                        else [0.0] * ensemble_size
                    )
        candidates = (
            _list2ragged(ops, candidates_list)
            if len(candidates_list) > 0
            else _empty_Ragged(ops, "i")
        )
        if is_train:
            if len(candidates_list) > 0:
                cumsums = ops.xp.cumsum(candidates.lengths)[:-1]
                training_outputs = ops.asarray1f(training_outputs_list)
                training_outputs = ops.xp.split(training_outputs, cumsums.tolist())
            else:
                training_outputs = [ops.alloc2f(0, 0)]
        else:
            training_outputs = None
        return cls(
            doc=doc,
            referrers=ops.asarray1i(referrers_list),
            antecedents=_list2ragged(ops, antecedents_list)
            if len(antecedents_list) > 0
            else _empty_Ragged(ops, "i"),
            candidates=candidates,
            referrers2candidates_pointers=ops.asarray1i(
                [
                    item
                    for sublist in [
                        [index] * len(entries)
                        for index, entries in enumerate(candidates_list)
                    ]
                    for item in sublist
                ]
            ),
            static_infos=ops.asarray2f(static_infos_list),
            training_outputs=training_outputs,
        )


def create_thinc_model() -> Model[List["DocumentPairInfo"], Tuple]:
    def create_vector_squeezer() -> Model[Floats2d, Floats2d]:
        """Generates part of the network that accepts a full-width vector and squeezes
        it down to 3 neurons to feed into the rest of the network. This is intended
        to force the network to learn succinct, relevant information about the vectors
        and also to reduce the overall importance of the vectors compared to the other
        map inputs during training.
        """
        return chain(
            Relu(24),
            Relu(3),
        )

    with Model.define_operators(
        {"|": concatenate, ">>": chain, "**": clone, "&": tuplify}
    ):

        referrers = get_referrers()
        antecedents = get_antecedents()
        static_inputs = get_static_inputs()

        ensemble_members = []
        for _ in range(ENSEMBLE_SIZE):

            inputs: Model[List["DocumentPairInfo"], Floats2d] = concatenate(
                referrers >> create_vector_squeezer(),
                antecedents >> create_vector_squeezer(),
                static_inputs,
            )

            model: Model[List["DocumentPairInfo"], Floats2d] = chain(
                inputs,
                Relu(639),
                Relu(20),
                Linear(1),
            )

            ensemble_members.append(model)

        ensemble: Model[List["DocumentPairInfo"], Tuple] = concatenate(
            *ensemble_members
        )
        return chain(noop() & ensemble, apply_softmax_sequences())


def apply_softmax_sequences() -> Model[
    Tuple[List["DocumentPairInfo"], Floats2d], Floats2d
]:
    return Model("apply_softmax_sequences", apply_softmax_sequences_forward)


def apply_softmax_sequences_forward(
    model: Model, inputs: Tuple[List["DocumentPairInfo"], Floats2d], is_train: bool
) -> Tuple[List[Floats2d], Callable]:
    def backprop(
        d_softmax_sequences: List[Floats2d],
    ) -> Tuple[List["DocumentPairInfo"], Floats2d]:

        d_softmax_inputs = model.ops.xp.concatenate(d_softmax_sequences)
        backpropped_softmax_inputs = model.ops.backprop_softmax_sequences(
            d_softmax_inputs, softmax_output, lengths
        )
        return dpis, backpropped_softmax_inputs

    dpis, predictions = inputs
    lengths = model.ops.xp.concatenate(
        [model.ops.asarray1i(dpi.candidates.lengths) for dpi in dpis]
    )
    softmax_output = model.ops.softmax_sequences(predictions, lengths)
    cumsums = model.ops.xp.cumsum(lengths)[:-1]
    return model.ops.xp.split(softmax_output, cumsums.tolist()), backprop


def get_referrers() -> Model[List["DocumentPairInfo"], List[Floats2d]]:
    return Model("get_referrers", referrers_forward)


def referrers_forward(
    model: Model[List["DocumentPairInfo"], List[Floats2d]],
    document_pair_infos: List["DocumentPairInfo"],
    is_train: bool,
) -> Tuple[Floats2d, Callable]:
    def backprop(d_vectors: Floats2d) -> List["DocumentPairInfo"]:
        return []

    vectors_to_return = []

    for document_pair_info in document_pair_infos:

        this_document_vector = model.ops.asarray2f(
            [
                document_pair_info.doc[referrer]._.coref_chains.temp_vector
                for referrer in document_pair_info.referrers.tolist()
            ]
        )[model.ops.asarray1i(document_pair_info.referrers2candidates_pointers)]

        vectors_to_return.append(this_document_vector)

    return model.ops.xp.concatenate(vectors_to_return), backprop


def get_referrer_heads() -> Model[List["DocumentPairInfo"], Floats2d]:
    return Model("get_referrer_heads", referrer_heads_forward)


def referrer_heads_forward(
    model: Model, document_pair_infos: List["DocumentPairInfo"], is_train: bool
) -> Tuple[Floats2d, Callable]:
    def backprop(d_vectors: Floats2d) -> List["DocumentPairInfo"]:
        return []

    vectors_to_return = []

    for document_pair_info in document_pair_infos:

        this_document_vector = model.ops.asarray2f(
            [
                document_pair_info.doc[referrer]._.coref_chains.temp_head_vector
                for referrer in document_pair_info.referrers.tolist()
            ]
        )[model.ops.asarray1i(document_pair_info.referrers2candidates_pointers)]

        vectors_to_return.append(this_document_vector)

    return model.ops.xp.concatenate(vectors_to_return), backprop


def get_antecedents() -> Model[List["DocumentPairInfo"], List[Floats2d]]:
    return Model("get_antecedents", antecedents_forward)


def antecedents_forward(
    model: Model, document_pair_infos: List["DocumentPairInfo"], is_train: bool
) -> Tuple[Floats2d, Callable]:
    def backprop(d_vectors: Floats2d) -> List["DocumentPairInfo"]:
        return []

    vectors_to_return = []

    for document_pair_info in document_pair_infos:

        this_document_vector = model.ops.asarray2f(
            [
                model.ops.asarray1f(
                    [
                        document_pair_info.doc[
                            cast(int, index[0])
                        ]._.coref_chains.temp_vector
                        for index in document_pair_info.antecedents[i].dataXd.tolist()
                    ]
                ).mean(  # type: ignore
                    axis=0
                )
                for i in range(len(document_pair_info.antecedents))
            ]
        )[model.ops.asarray1i(cast(Ints1d, document_pair_info.candidates.dataXd))]

        vectors_to_return.append(this_document_vector)

    return model.ops.xp.concatenate(vectors_to_return), backprop


def get_antecedent_heads() -> Model[List["DocumentPairInfo"], Floats2d]:
    return Model("get_antecedent_heads", antecedent_heads_forward)


def antecedent_heads_forward(
    model: Model, document_pair_infos: List["DocumentPairInfo"], is_train: bool
) -> Tuple[Floats2d, Callable]:
    def backprop(d_vectors: Floats2d) -> List["DocumentPairInfo"]:
        return []

    vectors_to_return = []

    for document_pair_info in document_pair_infos:

        this_document_vector = model.ops.asarray2f(
            # We only examine the head of the first element within the coordinated phrase
            # because other elements will not have the true semantic head as their
            # syntactic head
            [
                document_pair_info.doc[
                    cast(int, document_pair_info.antecedents[i].dataXd.tolist()[0][0])
                ]._.coref_chains.temp_head_vector
                for i in range(len(document_pair_info.antecedents))
            ]
        )[model.ops.asarray1i(cast(Ints1d, document_pair_info.candidates.dataXd))]

        vectors_to_return.append(this_document_vector)

    return model.ops.xp.concatenate(vectors_to_return), backprop


def get_static_inputs() -> Model[List["DocumentPairInfo"], Floats2d]:
    return Model("get_static_inputs", static_inputs_forward)


def static_inputs_forward(
    model: Model, document_pair_infos: List["DocumentPairInfo"], is_train: bool
) -> Tuple[Floats2d, Callable]:
    def backprop(d_vectors: Floats2d) -> List["DocumentPairInfo"]:
        return []

    return (
        model.ops.xp.concatenate(
            [model.ops.asarray2f(d.static_infos) for d in document_pair_infos]
        ),
        backprop,
    )


def _list2ragged(ops: Ops, items: List[List[int]]) -> Ragged:
    return Ragged(
        ops.xp.concatenate([ops.asarray1i(x) for x in items]),
        lengths=ops.asarray1i([len(x) for x in items]),
    )


def _empty_Ragged(ops: Ops, dtype: str) -> Ragged:
    return Ragged(ops.xp.zeros((0,), dtype=dtype), ops.alloc1i(0))


def _set_vectors(vectors_nlp: Language, ops: Ops, token: Token) -> None:
    if hasattr(token._.coref_chains, "temp_vector"):
        return
    if (not vectors_nlp.vocab[token.lemma_].has_vector) and len(token.vector) > 0:
        token._.coref_chains.temp_vector = token.vector
    else:
        token._.coref_chains.temp_vector = vectors_nlp.vocab[token.lemma_].vector
    if token != token.head:
        if (not vectors_nlp.vocab[token.head.lemma_].has_vector) and len(
            token.head.vector
        ) > 0:
            token._.coref_chains.temp_head_vector = token.head.vector
        else:
            token._.coref_chains.temp_head_vector = vectors_nlp.vocab[
                token.head.lemma_
            ].vector
    else:
        token._.coref_chains.temp_head_vector = (
            ops.alloc1f(len(token._.coref_chains.temp_vector)) + 0.0
        )
