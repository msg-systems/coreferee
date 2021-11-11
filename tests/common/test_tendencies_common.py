# Copyright 2021 msg systems ag

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import warnings
import numpy as np
import spacy
import coreferee
from coreferee.rules import RulesAnalyzerFactory
from coreferee.training.model import ModelGenerator
from coreferee.test_utils import get_nlps
from coreferee.tendencies import TendenciesAnalyzer
from coreferee.data_model import Mention

class CommonTendenciesTest(unittest.TestCase):

    def setUp(self):

        warnings.filterwarnings("ignore", message=r"\[W007\]", category=UserWarning)

        nlps = get_nlps('en')
        for nlp in (nlp for nlp in nlps if nlp.meta['name'] == 'core_web_sm'):
            self.sm_nlp = nlp
        self.sm_rules_analyzer = RulesAnalyzerFactory().get_rules_analyzer(self.sm_nlp)
        sm_model_generator = ModelGenerator(self.sm_rules_analyzer, self.sm_nlp, self.sm_nlp)
        sm_doc = self.sm_nlp('Richard said he was entering the big house')
        self.sm_feature_table = sm_model_generator.generate_feature_table([sm_doc])
        self.sm_tendencies_analyzer = TendenciesAnalyzer(self.sm_rules_analyzer, self.sm_nlp,
            self.sm_feature_table)

        for nlp in (nlp for nlp in nlps if nlp.meta['name'] == 'core_web_lg'):
            self.lg_nlp = nlp
        self.lg_rules_analyzer = RulesAnalyzerFactory().get_rules_analyzer(self.lg_nlp)
        lg_model_generator = ModelGenerator(self.lg_rules_analyzer, self.lg_nlp, self.lg_nlp)
        lg_doc = self.lg_nlp('Richard said he was entering the big house')
        self.lg_feature_table = lg_model_generator.generate_feature_table([lg_doc])
        self.lg_tendencies_analyzer = TendenciesAnalyzer(self.lg_rules_analyzer, self.lg_nlp,
            self.lg_feature_table)

    def test_generate_feature_table(self):

        doc = self.sm_nlp('Richard said he was entering the big house')
        model_generator = ModelGenerator(self.sm_rules_analyzer, self.sm_nlp, self.sm_nlp)
        feature_table = model_generator.generate_feature_table([doc])
        self.assertEqual({'tags': ['NN', 'NNP', 'PRP'], 'morphs': ['Case=Nom', 'Gender=Masc', 'NounType=Prop', 'Number=Sing', 'Person=3', 'PronType=Prs'], 'ent_types': ['', 'PERSON'], 'lefthand_deps_to_children': ['amod', 'det'], 'righthand_deps_to_children': [], 'lefthand_deps_to_parents': ['nsubj'], 'righthand_deps_to_parents': ['dobj'], 'parent_tags': ['VBD', 'VBG'], 'parent_morphs': ['Aspect=Prog', 'Tense=Past', 'Tense=Pres', 'VerbForm=Fin', 'VerbForm=Part'], 'parent_lefthand_deps_to_children': ['aux', 'nsubj'], 'parent_righthand_deps_to_children': ['ccomp', 'dobj']}, feature_table.__dict__)
        self.assertEqual(26, len(feature_table))

    def test_get_feature_map_simple_mention(self):

        doc = self.sm_nlp('Richard said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        mention = Mention(doc[0], False)
        feature_map = self.sm_tendencies_analyzer.get_feature_map(mention, doc)
        self.assertEqual(len(self.sm_feature_table), len(feature_map))
        self.assertEqual(mention.temp_feature_map, feature_map)
        self.assertEqual(
            [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
            feature_map)

        feature_map = self.sm_tendencies_analyzer.get_feature_map(Mention(doc[2], False), doc)
        self.assertEqual(len(self.sm_feature_table), len(feature_map))
        self.assertEqual(
            [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
            feature_map)

    def test_get_feature_map_simple_token(self):

        doc = self.sm_nlp('Richard said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        feature_map = self.sm_tendencies_analyzer.get_feature_map(doc[0], doc)
        self.assertEqual(len(self.sm_feature_table), len(feature_map))
        self.assertEqual(doc[0]._.coref_chains.temp_feature_map, feature_map)
        self.assertEqual(
            [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
            feature_map)

        feature_map = self.sm_tendencies_analyzer.get_feature_map(doc[2], doc)
        self.assertEqual(len(self.sm_feature_table), len(feature_map))
        self.assertEqual(
            [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
            feature_map)

    def test_get_feature_map_conjunction(self):

        doc = self.sm_nlp('Richard and the man said they were entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        feature_map = self.sm_tendencies_analyzer.get_feature_map(Mention(doc[0], False), doc)
        self.assertEqual(len(self.sm_feature_table), len(feature_map))
        self.assertEqual(
            [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
            feature_map)

        feature_map = self.sm_tendencies_analyzer.get_feature_map(Mention(doc[0], True), doc)
        self.assertEqual(len(self.sm_feature_table), len(feature_map))
        self.assertEqual(
            [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
            feature_map)

        feature_map = self.sm_tendencies_analyzer.get_feature_map(Mention(doc[5], False), doc)
        self.assertEqual(len(self.sm_feature_table), len(feature_map))
        self.assertEqual(
            [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
            feature_map)

    def test_get_position_map_first_sentence_token(self):

        doc = self.sm_nlp('Richard said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        position_map = self.sm_tendencies_analyzer.get_position_map(doc[0], doc)
        self.assertEqual(doc[0]._.coref_chains.temp_position_map, position_map)
        self.assertEqual([0, 1, 1, 0, 0, 0, 0], position_map)

        position_map = self.sm_tendencies_analyzer.get_position_map(doc[2], doc)
        self.assertEqual([2, 2, 2, 0, 0, 0, 0], position_map)

        position_map = self.sm_tendencies_analyzer.get_position_map(doc[6], doc)
        self.assertEqual([6, 3, 2, 1, 1, 0, 0], position_map)

    def test_get_position_map_first_sentence_mention(self):

        doc = self.sm_nlp('Richard said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        mention = Mention(doc[0], False)
        position_map = self.sm_tendencies_analyzer.get_position_map(mention, doc)
        self.assertEqual(mention.temp_position_map, position_map)
        self.assertEqual([0, 1, 1, 0, 0, 0, 0], position_map)

        position_map = self.sm_tendencies_analyzer.get_position_map(Mention(doc[2], False), doc)
        self.assertEqual([2, 2, 2, 0, 0, 0, 0], position_map)

        position_map = self.sm_tendencies_analyzer.get_position_map(Mention(doc[6], False), doc)
        self.assertEqual([6, 3, 2, 1, 1, 0, 0], position_map)

    def test_get_position_map_second_sentence_token(self):

        doc = self.sm_nlp(
            'This is a preceding sentence. Richard said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        position_map = self.sm_tendencies_analyzer.get_position_map(doc[6], doc)
        self.assertEqual([0, 1, 1, 0, 0, 0, 0], position_map)

        position_map = self.sm_tendencies_analyzer.get_position_map(doc[8], doc)
        self.assertEqual([2, 2, 2, 0, 0, 0, 0], position_map)

    def test_get_position_map_second_sentence_mention(self):

        doc = self.sm_nlp(
            'This is a preceding sentence. Richard said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        position_map = self.sm_tendencies_analyzer.get_position_map(Mention(doc[6], False), doc)
        self.assertEqual([0, 1, 1, 0, 0, 0, 0], position_map)

        position_map = self.sm_tendencies_analyzer.get_position_map(Mention(doc[8], False), doc)
        self.assertEqual([2, 2, 2, 0, 0, 0, 0], position_map)

    def test_get_position_map_root_token(self):

        doc = self.sm_nlp('Richard said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        position_map = self.sm_tendencies_analyzer.get_position_map(doc[1], doc)
        self.assertEqual([1, 0, 0, 0, -1, 0, 0], position_map)

    def test_get_position_map_root_mention(self):

        doc = self.sm_nlp('Richard said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        position_map = self.sm_tendencies_analyzer.get_position_map(Mention(doc[1], False), doc)
        self.assertEqual([1, 0, 0, 0, -1, 0, 0], position_map)

    def test_get_position_map_conjunction_first_sentence_tokens(self):

        doc = self.sm_nlp('Peter and Jane spoke to him and her.')
        self.sm_rules_analyzer.initialize(doc)
        position_map = self.sm_tendencies_analyzer.get_position_map(doc[0], doc)
        self.assertEqual([0, 1, 1, 0, 0, -1, 0], position_map)
        position_map = self.sm_tendencies_analyzer.get_position_map(doc[2], doc)
        self.assertEqual([2, 2, 1, 1, 1, -1, 1], position_map)
        position_map = self.sm_tendencies_analyzer.get_position_map(doc[5], doc)
        self.assertEqual([5, 2, 1, 2, 0, -1, 0], position_map)
        position_map = self.sm_tendencies_analyzer.get_position_map(doc[7], doc)
        self.assertEqual([7, 3, 1, 1, 1, -1, 1], position_map)

    def test_get_position_map_conjunction_first_sentence_mentions_false(self):

        doc = self.sm_nlp('Peter and Jane spoke to him and her.')
        self.sm_rules_analyzer.initialize(doc)
        position_map = self.sm_tendencies_analyzer.get_position_map(Mention(doc[0], False), doc)
        self.assertEqual([0, 1, 1, 0, 0, -1, 0], position_map)
        position_map = self.sm_tendencies_analyzer.get_position_map(Mention(doc[2], False), doc)
        self.assertEqual([2, 2, 1, 1, 1, -1, 1], position_map)
        position_map = self.sm_tendencies_analyzer.get_position_map(Mention(doc[5], False), doc)
        self.assertEqual([5, 2, 1, 2, 0, -1, 0], position_map)
        position_map = self.sm_tendencies_analyzer.get_position_map(Mention(doc[7], False), doc)
        self.assertEqual([7, 3, 1, 1, 1, -1, 1], position_map)

    def test_get_position_map_conjunction_second_sentence_mentions_false(self):
        doc = self.sm_nlp('A preceding sentence. Peter and Jane spoke to him and her.')
        self.sm_rules_analyzer.initialize(doc)
        position_map = self.sm_tendencies_analyzer.get_position_map(Mention(doc[4], False), doc)
        self.assertEqual([0, 1, 1, 0, 0, -1, 0], position_map)
        position_map = self.sm_tendencies_analyzer.get_position_map(Mention(doc[6], False), doc)
        self.assertEqual([2, 2, 1, 1, 1, -1, 1], position_map)
        position_map = self.sm_tendencies_analyzer.get_position_map(Mention(doc[9], False), doc)
        self.assertEqual([5, 2, 1, 2, 0, -1, 0], position_map)
        position_map = self.sm_tendencies_analyzer.get_position_map(Mention(doc[11], False), doc)
        self.assertEqual([7, 3, 1, 1, 1, -1, 1], position_map)

    def test_get_position_map_conjunction_first_sentence_mentions_true(self):

        doc = self.sm_nlp('Peter and Jane spoke to him and her.')
        self.sm_rules_analyzer.initialize(doc)
        position_map = self.sm_tendencies_analyzer.get_position_map(Mention(doc[0], True), doc)
        self.assertEqual([0, 1, 1, 0, 0, 1, 0], position_map)
        position_map = self.sm_tendencies_analyzer.get_position_map(Mention(doc[5], True), doc)
        self.assertEqual([5, 2, 1, 2, 0, 1, 0], position_map)

    def test_get_position_map_conjunction_second_sentence_mentions_true(self):
        doc = self.sm_nlp('A preceding sentence. Peter and Jane spoke to him and her.')
        self.sm_rules_analyzer.initialize(doc)
        position_map = self.sm_tendencies_analyzer.get_position_map(Mention(doc[4], True), doc)
        self.assertEqual([0, 1, 1, 0, 0, 1, 0], position_map)
        position_map = self.sm_tendencies_analyzer.get_position_map(Mention(doc[9], True), doc)
        self.assertEqual([5, 2, 1, 2, 0, 1, 0], position_map)

    def compare_compatibility_map(self, expected_compatibility_map, returned_compatibility_map):
        self.assertEqual(expected_compatibility_map[0], returned_compatibility_map[0])
        self.assertEqual(expected_compatibility_map[1], returned_compatibility_map[1])
        self.assertEqual(expected_compatibility_map[2], returned_compatibility_map[2])
        self.assertAlmostEqual(expected_compatibility_map[3], returned_compatibility_map[3], 4)
        self.assertEqual(expected_compatibility_map[4], returned_compatibility_map[4])

    def test_get_compatibility_map_simple(self):

        doc = self.sm_nlp('Richard said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        self.compare_compatibility_map([2, 0, 1, 0.26251248, 3],
            self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[2]))

    def test_get_compatibility_map_coordination(self):

        doc = self.sm_nlp('Richard and Jane said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        self.compare_compatibility_map([4, 0, 1, 0.20765561, 3],
            self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[0], True), doc[4]))

    def test_get_compatibility_map_different_sentences(self):

        doc = self.sm_nlp('Richard called. He said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        self.compare_compatibility_map([3, 1, 0, 0.52525896, 6],
            self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[3]))

    def test_get_compatibility_map_same_sentence_no_governance(self):

        doc = self.sm_nlp('After Richard arrived, he said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        self.compare_compatibility_map([4, 0, 0, -0.12525316, 5],
            self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[4]))

    def test_get_compatibility_map_same_sentence_lefthand_sibling_governance(self):

        doc = self.lg_nlp('Richard said Peter and he were entering the big house')
        self.lg_rules_analyzer.initialize(doc)
        self.compare_compatibility_map([4, 0, 1, 0.15999001, 4],
            self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[4]))

    def test_get_compatibility_map_same_sentence_lefthand_sibling_no_governance(self):

        doc = self.sm_nlp('After Richard arrived, Peter and he said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        self.compare_compatibility_map([5, 0, 0, 0.32681236, 6],
            self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[1], False), doc[6]))

    def test_get_cosine_similarity_lg(self):

        doc = self.lg_nlp('After Richard arrived, he said he was entering the big house')
        self.lg_rules_analyzer.initialize(doc)
        self.compare_compatibility_map([4, 0, 0, 0.3336621, 5],
            self.lg_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[4]))

    def test_get_cosine_similarity_lg_no_vector_1(self):

        doc = self.lg_nlp('After Richard arfewfewfrived, he said he was entering the big house')
        self.lg_rules_analyzer.initialize(doc)

        self.compare_compatibility_map([4, 0, 0, 0.59521705, 5],
            self.lg_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[4]))

    def test_get_cosine_similarity_lg_no_vector_2(self):

        doc = self.lg_nlp('After Richard arrived, he saifefefwefefd he was entering the big house')
        self.lg_rules_analyzer.initialize(doc)
        self.compare_compatibility_map([4, 0, 0, 0.59521705, 3],
            self.lg_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[4]))

    def test_get_cosine_similarity_sm_root_1(self):

        doc = self.sm_nlp('Richard. He said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)

        self.compare_compatibility_map([2, 1, 0, -1, 1],
            self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[2]))

    def test_get_cosine_similarity_sm_root_2(self):

        doc = self.sm_nlp('Richard arrived. He.')
        self.sm_rules_analyzer.initialize(doc)
        self.compare_compatibility_map([3, 1, 0, -1, 1],
            self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[3]))

    def test_get_cosine_similarity_lg_root_1(self):

        doc = self.lg_nlp('Richard. He said he was entering the big house')
        self.lg_rules_analyzer.initialize(doc)

        self.compare_compatibility_map([2, 1, 0, -1, 1],
            self.lg_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[2]))

    def test_get_cosine_similarity_lg_root_2(self):

        doc = self.lg_nlp('Richard arrived. He.')
        self.lg_rules_analyzer.initialize(doc)
        self.compare_compatibility_map([3, 1, 0, -1, 1],
            self.lg_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[3]))

    def test_get_vectors_token_with_head_sm(self):

        doc = self.sm_nlp('He arrived')
        self.sm_rules_analyzer.initialize(doc)
        vectors = self.sm_tendencies_analyzer.get_vectors(doc[0], doc)
        self.assertTrue(vectors[0].any())
        self.assertTrue(vectors[1].any())
        self.assertEqual(len(vectors[0]), len(vectors[1]))
        self.assertEqual(vectors, doc[0]._.coref_chains.temp_vectors)

    def test_get_vectors_token_without_head_sm(self):

        doc = self.sm_nlp('He arrived')
        self.sm_rules_analyzer.initialize(doc)
        vectors = self.sm_tendencies_analyzer.get_vectors(doc[1], doc)
        self.assertTrue(vectors[0].any())
        self.assertFalse(vectors[1].any())
        self.assertEqual(len(vectors[0]), len(vectors[1]))
        self.assertEqual(vectors, doc[1]._.coref_chains.temp_vectors)

    def test_get_vectors_token_with_head_lg(self):

        doc = self.lg_nlp('He arrived')
        self.lg_rules_analyzer.initialize(doc)
        vectors = self.lg_tendencies_analyzer.get_vectors(doc[0], doc)
        self.assertTrue(vectors[0].any())
        self.assertTrue(vectors[1].any())
        self.assertEqual(len(vectors[0]), len(vectors[1]))
        self.assertEqual(vectors, doc[0]._.coref_chains.temp_vectors)

    def test_get_vectors_token_without_head_lg(self):

        doc = self.lg_nlp('He arrived')
        self.lg_rules_analyzer.initialize(doc)
        vectors = self.lg_tendencies_analyzer.get_vectors(doc[1], doc)
        self.assertTrue(vectors[0].any())
        self.assertFalse(vectors[1].any())
        self.assertEqual(len(vectors[0]), len(vectors[1]))
        self.assertEqual(vectors, doc[1]._.coref_chains.temp_vectors)

    def test_get_vectors_mention_with_head_sm(self):

        doc = self.sm_nlp('He arrived')
        self.sm_rules_analyzer.initialize(doc)
        mention = Mention(doc[0], False)
        vectors = self.sm_tendencies_analyzer.get_vectors(mention, doc)
        self.assertTrue(vectors[0].any())
        self.assertTrue(vectors[1].any())
        self.assertEqual(len(vectors[0]), len(vectors[1]))
        self.assertEqual(vectors, mention.temp_vectors)

    def test_get_vectors_mention_without_head_sm(self):

        doc = self.sm_nlp('He arrived')
        self.sm_rules_analyzer.initialize(doc)
        vectors = self.sm_tendencies_analyzer.get_vectors(Mention(doc[1], False), doc)
        self.assertTrue(vectors[0].any())
        self.assertFalse(vectors[1].any())
        self.assertEqual(len(vectors[0]), len(vectors[1]))

    def test_get_vectors_mention_with_head_lg(self):

        doc = self.lg_nlp('He arrived')
        self.lg_rules_analyzer.initialize(doc)
        vectors = self.lg_tendencies_analyzer.get_vectors(Mention(doc[0], False), doc)
        self.assertTrue(vectors[0].any())
        self.assertTrue(vectors[1].any())
        self.assertEqual(len(vectors[0]), len(vectors[1]))

    def test_get_vectors_mention_without_head_lg(self):

        doc = self.lg_nlp('He arrived')
        self.lg_rules_analyzer.initialize(doc)
        vectors = self.lg_tendencies_analyzer.get_vectors(Mention(doc[1], False), doc)
        self.assertTrue(vectors[0].any())
        self.assertFalse(vectors[1].any())
        self.assertEqual(len(vectors[0]), len(vectors[1]))

    def test_vectors_twoway_coordination_sm(self):
        doc = self.sm_nlp('Peter and Jane arrived')
        self.sm_rules_analyzer.initialize(doc)
        peter_vectors = self.sm_tendencies_analyzer.get_vectors(Mention(doc[0], False), doc)
        jane_vectors = self.sm_tendencies_analyzer.get_vectors(Mention(doc[2], False), doc)
        combined_vectors = self.sm_tendencies_analyzer.get_vectors(Mention(doc[0], True), doc)
        for index in range(len(peter_vectors[0])):
            self.assertAlmostEqual((peter_vectors[0][index] + jane_vectors[0][index]) / 2,
                combined_vectors[0][index])

    def test_vectors_twoway_coordination_lg(self):
        doc = self.lg_nlp('Peter and Jane arrived')
        self.lg_rules_analyzer.initialize(doc)
        peter_vectors = self.lg_tendencies_analyzer.get_vectors(Mention(doc[0], False), doc)
        jane_vectors = self.lg_tendencies_analyzer.get_vectors(Mention(doc[2], False), doc)
        combined_vectors = self.lg_tendencies_analyzer.get_vectors(Mention(doc[0], True), doc)
        for index in range(len(peter_vectors[0])):
            self.assertAlmostEqual((peter_vectors[0][index] + jane_vectors[0][index]) / 2,
                combined_vectors[0][index])

    def test_vectors_threeway_coordination_sm(self):
        doc = self.sm_nlp('Richard, Peter and Jane arrived')
        self.sm_rules_analyzer.initialize(doc)
        richard_vectors = self.sm_tendencies_analyzer.get_vectors(Mention(doc[0], False), doc)
        peter_vectors = self.sm_tendencies_analyzer.get_vectors(Mention(doc[2], False), doc)
        jane_vectors = self.sm_tendencies_analyzer.get_vectors(Mention(doc[4], False), doc)
        combined_vectors = self.sm_tendencies_analyzer.get_vectors(Mention(doc[0], True), doc)
        for index in range(len(peter_vectors[0])):
            self.assertAlmostEqual((peter_vectors[0][index] + jane_vectors[0][index] +
                richard_vectors[0][index]) / 3, combined_vectors[0][index], places=3)

    def test_vectors_threeway_coordination_lg(self):
        doc = self.lg_nlp('They spoke to Richard, Peter and Jane.')
        self.lg_rules_analyzer.initialize(doc)
        richard_vectors = self.lg_tendencies_analyzer.get_vectors(Mention(doc[3], False), doc)
        peter_vectors = self.lg_tendencies_analyzer.get_vectors(Mention(doc[5], False), doc)
        jane_vectors = self.lg_tendencies_analyzer.get_vectors(Mention(doc[7], False), doc)
        combined_vectors = self.lg_tendencies_analyzer.get_vectors(Mention(doc[3], True), doc)
        for index in range(len(peter_vectors[0])):
            self.assertAlmostEqual((peter_vectors[0][index] + jane_vectors[0][index] +
                richard_vectors[0][index]) / 3, combined_vectors[0][index], places=3)
