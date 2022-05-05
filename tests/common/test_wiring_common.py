from logging import debug
from xml.dom.minidom import Document
import pytest
import spacy
from coreferee.rules import RulesAnalyzerFactory
from coreferee.tendencies import *
from coreferee.test_utils import get_nlps
from thinc.backends import get_current_ops

ops = get_current_ops()

nlps = get_nlps('en')
train_version_mismatch = False
for nlp in nlps:
    if not nlp.meta["matches_train_version"]:
        train_version_mismatch = True
train_version_mismatch_message = "Loaded model version does not match train model version"

@pytest.fixture
def setup_simple_example():
    nlp = spacy.load("en_core_web_sm")
    rules_analyzer = RulesAnalyzerFactory.get_rules_analyzer(nlp)
    doc = nlp("Sarah's sister flew to Silicon Valley via Berlin. She loved it.")
    rules_analyzer.initialize(doc)
    feature_table = generate_feature_table([doc], nlp)
    tendencies_analyzer = TendenciesAnalyzer(rules_analyzer, nlp, feature_table)
    return DocumentPairInfo.from_doc(doc, tendencies_analyzer, 5), nlp


@pytest.fixture(params=["en_core_web_sm", "en_core_web_md"])
def setup_three_sentences_with_conjunction(request):
    nlp = spacy.load(request.param)
    rules_analyzer = RulesAnalyzerFactory.get_rules_analyzer(nlp)
    doc = nlp("People. Richard and the man said they were entering the big house. He.")
    rules_analyzer.initialize(doc)
    feature_table = generate_feature_table([doc], nlp)
    tendencies_analyzer = TendenciesAnalyzer(rules_analyzer, nlp, feature_table)
    return (DocumentPairInfo.from_doc(doc, tendencies_analyzer, 5), nlp)


@pytest.fixture
def setup_training_doc():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Sarah's sister flew to Silicon Valley via Berlin. She loved it.")
    rules_analyzer = RulesAnalyzerFactory().get_rules_analyzer(nlp)
    rules_analyzer.initialize(doc)
    feature_table = generate_feature_table([doc], nlp)
    tendencies_analyzer = TendenciesAnalyzer(rules_analyzer, nlp, feature_table)
    # linguistically nonsensical labels that only serve to test wiring
    doc[10]._.coref_chains.temp_potential_referreds[2].true_in_training = True
    doc[12]._.coref_chains.temp_potential_referreds[0].true_in_training = True
    return DocumentPairInfo.from_doc(doc, tendencies_analyzer, 5, is_train=True)


@pytest.fixture
def setup_training_doc_with_spanned():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Sarah's sister flew to Silicon Valley via Berlin. She loved it.")
    rules_analyzer = RulesAnalyzerFactory().get_rules_analyzer(nlp)
    rules_analyzer.initialize(doc)
    feature_table = generate_feature_table([doc], nlp)
    tendencies_analyzer = TendenciesAnalyzer(rules_analyzer, nlp, feature_table)
    # linguistically nonsensical labels that only serve to test wiring
    doc[10]._.coref_chains.temp_potential_referreds[1].spanned_in_training = True
    doc[10]._.coref_chains.temp_potential_referreds[2].true_in_training = True
    return DocumentPairInfo.from_doc(doc, tendencies_analyzer, 5, is_train=True)

@pytest.mark.skipif(train_version_mismatch, reason=train_version_mismatch_message)
def test_dpi_normal(setup_simple_example):
    document_pair_info, _ = setup_simple_example
    assert list(document_pair_info.referrers) == [10, 12]
    assert list(document_pair_info.antecedents.dataXd) == [0, 2, 6, 8]
    assert list(document_pair_info.antecedents.lengths) == [1, 1, 1, 1]
    assert list(document_pair_info.candidates.dataXd) == [0, 1, 2, 3, 2, 3]
    assert list(document_pair_info.candidates.lengths) == [4, 2]
    assert list(document_pair_info.referrers2candidates_pointers) == [0, 0, 0, 0, 1, 1]
    assert document_pair_info.training_outputs is None
    for index in range(6):
        pointed_to_referrer = document_pair_info.referrers2candidates_pointers[index]
        referrer = document_pair_info.referrers[pointed_to_referrer]
        referrer_feature_map = document_pair_info.doc[
            referrer
        ]._.coref_chains.temp_feature_map
        assert list(document_pair_info.static_infos[index][:33]) == list(
            referrer_feature_map
        )
        referrer_position_map = document_pair_info.doc[
            referrer
        ]._.coref_chains.temp_position_map
        assert list(document_pair_info.static_infos[index][33:40]) == list(
            referrer_position_map
        )
        working_antecedent_index = index if index < 4 else index - 4
        working_mention = document_pair_info.doc[
            referrer
        ]._.coref_chains.temp_potential_referreds[working_antecedent_index]
        antecedent_feature_map = working_mention.temp_feature_map
        assert list(document_pair_info.static_infos[index][40:73]) == list(
            antecedent_feature_map
        )
        antecedent_position_map = working_mention.temp_position_map
        assert list(document_pair_info.static_infos[index][73:80]) == list(
            antecedent_position_map
        )
        compatibility_map = working_mention.temp_compatibility_map
        assert list(document_pair_info.static_infos[index][80:]) == list(
            compatibility_map
        )


def test_dpi_training(setup_training_doc):
    document_pair_info = setup_training_doc

    assert list(document_pair_info.referrers) == [10, 12]
    assert list(document_pair_info.antecedents.dataXd) == [0, 2, 6, 8]
    assert list(document_pair_info.antecedents.lengths) == [1, 1, 1, 1]
    assert list(document_pair_info.candidates.dataXd) == [0, 1, 2, 3, 2, 3]
    assert list(document_pair_info.candidates.lengths) == [4, 2]
    assert list(document_pair_info.referrers2candidates_pointers) == [0, 0, 0, 0, 1, 1]
    assert document_pair_info.training_outputs[0].tolist() == [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert document_pair_info.training_outputs[1].tolist() == [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]


def test_dpi_training_with_spanned(setup_training_doc_with_spanned):
    document_pair_info = setup_training_doc_with_spanned

    assert list(document_pair_info.referrers) == [10]
    assert list(document_pair_info.antecedents.dataXd) == [0, 6, 8]
    assert list(document_pair_info.antecedents.lengths) == [1, 1, 1]
    assert list(document_pair_info.candidates.dataXd) == [0, 1, 2]
    assert list(document_pair_info.candidates.lengths) == [3]
    assert list(document_pair_info.referrers2candidates_pointers) == [0, 0, 0]
    assert document_pair_info.training_outputs[0].tolist() == [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]


def test_antecedent_vectors_simple(setup_three_sentences_with_conjunction):
    document_pair_info, _ = setup_three_sentences_with_conjunction
    vectors, _ = antecedents_forward(get_antecedents(), [document_pair_info], False)
    assert list(vectors[0]) == list(document_pair_info.doc[0].vector)


def test_antecedent_vectors_mean(setup_three_sentences_with_conjunction):
    document_pair_info, _ = setup_three_sentences_with_conjunction
    vectors, _ = antecedents_forward(get_antecedents(), [document_pair_info], False)
    assert list(vectors[1]) == list(
        ops.xp.mean(
            [document_pair_info.doc[2].vector, document_pair_info.doc[5].vector], axis=0
        )
    )


def test_antecedent_head_vectors(setup_three_sentences_with_conjunction):
    document_pair_info, nlp = setup_three_sentences_with_conjunction
    vector_size = 96 if nlp.meta["name"].endswith("sm") else 300
    vectors, _ = antecedent_heads_forward(
        get_antecedent_heads(), [document_pair_info], False
    )
    assert list(vectors[0]) == list(ops.xp.zeros(vector_size))
    if nlp.meta["name"].endswith("sm"):
        assert list(vectors[1]) == list(document_pair_info.doc[6].vector)
        assert list(vectors[2]) == list(document_pair_info.doc[6].vector)
        assert list(vectors[3]) == list(document_pair_info.doc[2].vector)
    else:
        assert list(vectors[1]) == list(
            nlp.vocab[document_pair_info.doc[6].lemma_].vector
        )
        assert list(vectors[2]) == list(
            nlp.vocab[document_pair_info.doc[6].lemma_].vector
        )
        assert list(vectors[3]) == list(
            nlp.vocab[document_pair_info.doc[2].lemma_].vector
        )


def test_referrer_vectors(setup_three_sentences_with_conjunction):
    document_pair_info, nlp = setup_three_sentences_with_conjunction
    vectors, _ = referrers_forward(get_referrers(), [document_pair_info], False)
    if nlp.meta["name"].endswith("sm"):
        assert list(vectors[0]) == list(document_pair_info.doc[7].tensor)
        assert list(vectors[2]) == list(document_pair_info.doc[14].tensor)
    else:
        assert list(vectors[0]) == list(
            nlp.vocab[document_pair_info.doc[7].lemma_].vector
        )
        assert list(vectors[2]) == list(
            nlp.vocab[document_pair_info.doc[14].lemma_].vector
        )


def test_referrer_head_vectors(setup_three_sentences_with_conjunction):
    document_pair_info, nlp = setup_three_sentences_with_conjunction
    vector_size = 96 if nlp.meta["name"].endswith("sm") else 300
    vectors, _ = referrer_heads_forward(
        get_referrer_heads(), [document_pair_info], False
    )
    if nlp.meta["name"].endswith("sm"):
        assert list(vectors[0]) == list(document_pair_info.doc[9].tensor)
    else:
        assert list(vectors[0]) == list(
            nlp.vocab[document_pair_info.doc[9].lemma_].vector
        )
    assert list(vectors[2]) == list(ops.xp.zeros(vector_size))


def test_softmax_sequences(setup_simple_example):
    document_pair_info, _ = setup_simple_example

    grouped_outputs, _ = apply_softmax_sequences_forward(
        apply_softmax_sequences(),
        (
            [document_pair_info],
            ops.asarray2f(
                [
                    [0.0] * 8,
                    [1.0] * 8,
                    [2.0] * 8,
                    [3.0] * 8,
                    [4.0] * 8,
                    [5.0] * 8,
                ]
            ),
        ),
        False,
    )

    first_candidate_grouped_outputs = grouped_outputs[0]
    assert (
        ops.xp.mean(first_candidate_grouped_outputs[0])
        == first_candidate_grouped_outputs[0][0]
    )
    assert ops.xp.sum(first_candidate_grouped_outputs[:, 0]) == 1.0


@pytest.mark.skipif(train_version_mismatch, reason=train_version_mismatch_message)
def test_generate_feature_table(setup_simple_example):

    document_pair_info, nlp = setup_simple_example
    feature_table = generate_feature_table([document_pair_info.doc], nlp)
    assert feature_table.__dict__ == {
        "tags": ["NN", "NNP", "PRP"],
        "morphs": [
            "Case=Acc",
            "Case=Nom",
            "Gender=Fem",
            "Gender=Neut",
            "Number=Sing",
            "Person=3",
            "PronType=Prs",
        ],
        "ent_types": ["", "GPE", "LOC", "PERSON"],
        "lefthand_deps_to_children": ["compound", "poss"],
        "righthand_deps_to_children": ["case"],
        "lefthand_deps_to_parents": ["nsubj", "poss"],
        "righthand_deps_to_parents": ["dobj", "pobj"],
        "parent_tags": ["IN", "NN", "VBD"],
        "parent_morphs": ["Number=Sing", "Tense=Past", "VerbForm=Fin"],
        "parent_lefthand_deps_to_children": ["nsubj", "poss"],
        "parent_righthand_deps_to_children": ["dobj", "pobj", "prep", "punct"],
    }

    assert len(feature_table) == 33
