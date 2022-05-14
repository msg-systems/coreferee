import unittest
import warnings
import numpy as np
from coreferee.rules import RulesAnalyzerFactory
from coreferee.test_utils import get_nlps
from coreferee.tendencies import TendenciesAnalyzer, generate_feature_table
from coreferee.data_model import Mention

nlps = get_nlps('en')
train_version_mismatch = False
for nlp in nlps:
    if not nlp.meta["matches_train_version"]:
        train_version_mismatch = True
train_version_mismatch_message = "Loaded model version does not match train model version"

class CommonTendenciesTest(unittest.TestCase):

    def setUp(self):

        warnings.filterwarnings("ignore", message=r"\[W007\]", category=UserWarning)

        nlps = get_nlps('en')
        for nlp in (nlp for nlp in nlps if nlp.meta['name'] == 'core_web_sm'):
            self.sm_nlp = nlp
        self.sm_rules_analyzer = RulesAnalyzerFactory().get_rules_analyzer(self.sm_nlp)
        sm_doc = self.sm_nlp('Richard said he was entering the big house')
        self.sm_feature_table = generate_feature_table([sm_doc], self.sm_nlp)
        self.sm_tendencies_analyzer = TendenciesAnalyzer(self.sm_rules_analyzer, self.sm_nlp,
            self.sm_feature_table)

        for nlp in (nlp for nlp in nlps if nlp.meta['name'] == 'core_web_lg'):
            self.lg_nlp = nlp
        self.lg_rules_analyzer = RulesAnalyzerFactory().get_rules_analyzer(self.lg_nlp)
        lg_doc = self.lg_nlp('Richard said he was entering the big house')
        self.lg_feature_table = generate_feature_table([lg_doc], self.lg_nlp)
        self.lg_tendencies_analyzer = TendenciesAnalyzer(self.lg_rules_analyzer, self.lg_nlp,
            self.lg_feature_table)

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_get_feature_map_simple_mention(self):

        doc = self.sm_nlp('Richard said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        mention = Mention(doc[0], False)
        feature_map = self.sm_tendencies_analyzer.get_feature_map(mention, doc)
        self.assertEqual(len(self.sm_feature_table), len(feature_map))
        self.assertEqual(mention.temp_feature_map, feature_map)
        if nlp.meta['version'] == '3.2.0':            
            self.assertEqual(
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
                feature_map)
        elif nlp.meta['version'] == '3.3.0':
            self.assertEqual(
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
                feature_map)
        else:
            self.fail("Unsupported version.")

        feature_map = self.sm_tendencies_analyzer.get_feature_map(Mention(doc[2], False), doc)
        self.assertEqual(len(self.sm_feature_table), len(feature_map))
        if nlp.meta['version'] == '3.2.0':            
            self.assertEqual(
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                feature_map)
        elif nlp.meta['version'] == '3.3.0':
            self.assertEqual(
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                feature_map)

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_get_feature_map_simple_token(self):

        doc = self.sm_nlp('Richard said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        feature_map = self.sm_tendencies_analyzer.get_feature_map(doc[0], doc)
        self.assertEqual(len(self.sm_feature_table), len(feature_map))
        self.assertEqual(doc[0]._.coref_chains.temp_feature_map, feature_map)
        if nlp.meta['version'] == '3.2.0':            
            self.assertEqual(
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
                feature_map)
        elif nlp.meta['version'] == '3.3.0':
            self.assertEqual(
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
                feature_map)
        else:
            self.fail("Unsupported version.")

        feature_map = self.sm_tendencies_analyzer.get_feature_map(doc[2], doc)
        self.assertEqual(len(self.sm_feature_table), len(feature_map))
        if nlp.meta['version'] == '3.2.0':            
            self.assertEqual(
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                feature_map)
        elif nlp.meta['version'] == '3.3.0':
            self.assertEqual(
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                feature_map)

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_get_feature_map_conjunction(self):

        doc = self.sm_nlp('Richard and the man said they were entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        feature_map = self.sm_tendencies_analyzer.get_feature_map(Mention(doc[0], False), doc)
        self.assertEqual(len(self.sm_feature_table), len(feature_map))
        if nlp.meta['version'] == '3.2.0':            
            self.assertEqual(
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
                feature_map)
        elif nlp.meta['version'] == '3.3.0':            
            self.assertEqual(
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
                feature_map)
        else:
            self.fail("Unsupported version")


        feature_map = self.sm_tendencies_analyzer.get_feature_map(Mention(doc[0], True), doc)
        self.assertEqual(len(self.sm_feature_table), len(feature_map))
        if nlp.meta['version'] == '3.2.0':            
            self.assertEqual(
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
                feature_map)
        elif nlp.meta['version'] == '3.3.0':            
            self.assertEqual(
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
                feature_map)

        feature_map = self.sm_tendencies_analyzer.get_feature_map(Mention(doc[5], False), doc)
        self.assertEqual(len(self.sm_feature_table), len(feature_map))
        if nlp.meta['version'] == '3.2.0':            
            self.assertEqual(
                [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
                feature_map)
        elif nlp.meta['version'] == '3.3.0':            
            self.assertEqual(
                [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
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

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_get_compatibility_map_simple(self):

        doc = self.sm_nlp('Richard said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        if nlp.meta['version'] == '3.2.0':            
            self.compare_compatibility_map([2, 0, 1, 0.29702997, 3],
                self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[2]))
        elif nlp.meta['version'] == '3.3.0':            
            self.compare_compatibility_map([2, 0, 1, 0.34484535, 3],
                self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[2]))
        else:
            self.fail("Unsupported version")


    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_get_compatibility_map_coordination(self):

        doc = self.sm_nlp('Richard and Jane said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        if nlp.meta['version'] == '3.2.0':            
            self.compare_compatibility_map([4, 0, 1, 0.28721756, 3],
                self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[0], True), doc[4]))
        elif nlp.meta['version'] == '3.3.0':            
            self.compare_compatibility_map([4, 0, 1, 0.37224450, 3],
                self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[0], True), doc[4]))
        else:
            self.fail("Unsupported version")

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_get_compatibility_map_different_sentences(self):

        doc = self.sm_nlp('Richard called. He said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        if nlp.meta['version'] == '3.2.0':            
            self.compare_compatibility_map([3, 1, 0, 0.47986302, 6],
                self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[3]))
        elif nlp.meta['version'] == '3.3.0':            
            self.compare_compatibility_map([3, 1, 0, 0.42599782, 6],
                self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[3]))
        else:
            self.fail("Unsupported version")

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_get_compatibility_map_same_sentence_no_governance(self):

        doc = self.sm_nlp('After Richard arrived, he said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        
        if nlp.meta['version'] == '3.2.0':            
            self.compare_compatibility_map([4, 0, 0, -0.02203778, 5],
                self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[4]))
        elif nlp.meta['version'] == '3.3.0':            
            self.compare_compatibility_map([4, 0, 0, -0.00317071, 5],
                self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[4]))
        else:
            self.fail("Unsupported version")


    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_get_compatibility_map_same_sentence_lefthand_sibling_governance(self):

        doc = self.lg_nlp('Richard said Peter and he were entering the big house')
        self.lg_rules_analyzer.initialize(doc)
        if self.lg_nlp.meta['version'] == '3.2.0':            
            self.compare_compatibility_map([4, 0, 1, 0.15999001, 3],
                self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[4]))
        elif self.lg_nlp.meta['version'] == '3.3.0':            
            self.compare_compatibility_map([4, 1, 0, 0.15999001, 4],
                self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[4]))
        else:
            self.fail("Unsupported version.")

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_get_compatibility_map_same_sentence_lefthand_sibling_no_governance(self):

        doc = self.sm_nlp('After Richard arrived, Peter and he said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)
        if self.sm_nlp.meta['version'] == '3.2.0':            
            self.compare_compatibility_map([5, 0, 0, 0.29553932, 6],
                self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[1], False), doc[6]))
        elif self.sm_nlp.meta['version'] == '3.3.0':            
            self.compare_compatibility_map([5, 0, 0, 0.40949851, 6],
                self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[1], False), doc[6]))
        else:
            self.fail("Unsupported version.")

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_get_cosine_similarity_lg(self):

        doc = self.lg_nlp('After Richard arrived, he said he was entering the big house')
        self.lg_rules_analyzer.initialize(doc)
        self.compare_compatibility_map([4, 0, 0, 0.3336621, 5],
            self.lg_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[4]))

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_get_cosine_similarity_lg_no_vector_1(self):

        doc = self.lg_nlp('After Richard arfewfewfrived, he said he was entering the big house')
        self.lg_rules_analyzer.initialize(doc)

        self.compare_compatibility_map([4, 0, 0, 0.59521705, 5],
            self.lg_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[4]))

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_get_cosine_similarity_lg_no_vector_2(self):

        doc = self.lg_nlp('After Richard arrived, he saifefefwefefd he was entering the big house')
        self.lg_rules_analyzer.initialize(doc)
        self.compare_compatibility_map([4, 0, 0, 0.59521705, 5],
            self.lg_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[4]))

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_get_cosine_similarity_sm_root_1(self):

        doc = self.sm_nlp('Richard. He said he was entering the big house')
        self.sm_rules_analyzer.initialize(doc)

        self.compare_compatibility_map([2, 1, 0, -1, 1],
            self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[2]))

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_get_cosine_similarity_sm_root_2(self):

        doc = self.sm_nlp('Richard arrived. He.')
        self.sm_rules_analyzer.initialize(doc)
        self.compare_compatibility_map([3, 1, 0, -1, 1],
            self.sm_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[3]))

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_get_cosine_similarity_lg_root_1(self):

        doc = self.lg_nlp('Richard. He said he was entering the big house')
        self.lg_rules_analyzer.initialize(doc)

        self.compare_compatibility_map([2, 1, 0, -1, 1],
            self.lg_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[2]))

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_get_cosine_similarity_lg_root_2(self):

        doc = self.lg_nlp('Richard arrived. He.')
        self.lg_rules_analyzer.initialize(doc)
        self.compare_compatibility_map([3, 1, 0, -1, 1],
            self.lg_tendencies_analyzer.get_compatibility_map(Mention(doc[0], False), doc[3]))
