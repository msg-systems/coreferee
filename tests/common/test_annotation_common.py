import unittest
from coreferee.test_utils import get_nlps

class CommonAnnotationTest(unittest.TestCase):

    def setUp(self):

        self.nlps = get_nlps('en')

    def all_nlps(self, func):
        for nlp in self.nlps:
            func(nlp)

    def compare_annotations(self, doc_text, expected_coref_chains, *, excluded_nlps=[],
        alternative_expected_coref_chains=None):

        def func(nlp):

            if nlp.meta['name'] in excluded_nlps:
                return

            doc = nlp(doc_text)
            chains_representation = str(doc._.coref_chains)
            if alternative_expected_coref_chains is None:
                self.assertEqual(expected_coref_chains,
                    chains_representation, nlp.meta['name'])
            else:
                self.assertTrue(expected_coref_chains == chains_representation or
                    alternative_expected_coref_chains == chains_representation)

        self.all_nlps(func)

    def test_annotations_simple(self):
        self.compare_annotations('Richard said he had finished', '[0: [0], [2]]')

    def test_annotations_coordination(self):
        self.compare_annotations('Richard and Peter said they had finished', '[0: [0, 2], [4]]')

    def test_annotations_coordination_with_or(self):
        self.compare_annotations('There were two men. Richard or Peter said he had finished',
            '[0: [5], [9]]')

    def test_annotations_coordination_two_chains(self):
        self.compare_annotations(
            'Richard came in. He and Peter went out. They said they had finished',
            '[0: [0], [4], 1: [4, 6], [10], [12]]')

    def test_annotations_coordination_two_chains_long_gap(self):
        self.compare_annotations(
            'Richard came in. He and Peter went out. They said they had finished. Again. Again. Again. They are here.',
            '[0: [0], [4], 1: [4, 6], [10], [12], [22]]')

    def test_annotations_with_scoring(self):
        self.compare_annotations('Richard told Peter he had finished', '[0: [0], [3]]')

    def test_annotations_cataphora(self):
        self.compare_annotations('Although he was still working, Richard said hello',
            '[0: [1], [6]]')

    def test_annotations_cataphora_and_coordination(self):
        self.compare_annotations('Although they were still working, Richard and Peter said hello',
            '[0: [1], [6, 8]]')

    def test_annotations_cataphora_and_anaphora(self):
        self.compare_annotations('Although he was still working, Richard said he had finished',
            '[0: [1], [6], [8]]')

    def test_annotations_cataphora_and_anaphora_and_coordination(self):
        self.compare_annotations(
            'Although they were still working, Richard and Peter said they had finished',
            '[0: [1], [6, 8], [10]]')

    def test_annotations_nouns_simple(self):
        self.compare_annotations('Richard was here. Richard had finished', '[0: [0], [4]]')

    def test_annotations_nouns_and_pronouns_mixed_(self):
        self.compare_annotations('He was here. Richard was here. Richard had finished. He was here. Richard was here. He saw him.', '[0: [0], [22], 1: [4], [8], [12], [16], [20]]',
        alternative_expected_coref_chains='[0: [0], [12], [22], 1: [4], [8], [16], [20]]')

    def test_non_reflexive_preferred_in_reflexive_position(self):
        self.compare_annotations('Richard came in. Peter saw himself', '[0: [4], [6]]')

    def test_non_reflexive_preferred_in_reflexive_position_control_1(self):
        self.compare_annotations('Richard came in. Peter saw him', '[0: [0], [6]]')

    def test_non_reflexive_allowed_in_reflexive_position(self):
        self.compare_annotations('Richard came in. The wind saw himself', '[0: [0], [7]]')

    def test_two_pronouns_previous_noun(self):
        self.compare_annotations('Richard came in. Someone saw him. He was ready',
            '[0: [0], [6], [8]]')

    def test_sentence_referential_distance_pronoun(self):
        self.compare_annotations('Richard came in. Yes. Yes. Yes. Yes. Yes. Someone saw him.',
            '[]')

    def test_sentence_referential_distance_with_spanning_pronoun(self):
        self.compare_annotations('Richard came in. Yes. Yes. Someone saw him. Yes. Yes. Yes. Someone saw him.',
            '[0: [0], [10], [20]]')

    def test_sentence_referential_distance_noun(self):
        self.compare_annotations('Richard came in. Yes. Yes. Richard came in.',
            '[]')

    def test_sentence_referential_distance_noun_with_spanning_pronoun(self):
        self.compare_annotations('Richard came in. Yes. Someone saw him. Yes. Richard came in.',
            '[0: [0], [8], [12]]')

    def test_excluded(self):
        self.compare_annotations(
            'The person spoke to the woman. He asked whether he could help him.',
            '[0: [1], [7], [10]]')

    def test_excluded_control(self):
        self.compare_annotations(
            'The person spoke to the woman. He asked whether he could help her.',
            '[0: [1], [7], [10], 1: [5], [13]]')

    def test_retry(self):
        self.compare_annotations(
            'The dog saw the house. It saw him.',
            '[0: [1], [8], 1: [4], [6]]')

    def test_retry_control(self):
        self.compare_annotations(
            'The dog saw the person. It saw him.',
            '[0: [1], [6], 1: [4], [8]]')

    def test_no_anaphora_within_coordination(self):
        self.compare_annotations('I spoke to Richard and him', '[]')

    def test_reflexive_excluded_mix_of_coordination_and_single_member_1(self):
        self.compare_annotations(
            'Peter and Jane came in. They saw him.',
            '[0: [0, 2], [6]]')

    def test_reflexive_excluded_mix_of_coordination_and_single_member_2(self):
        self.compare_annotations(
            'Peter and Jane came in. They saw her.',
            '[0: [0, 2], [6]]')

    def test_reflexive_excluded_mix_of_coordination_and_single_member_1(self):
        self.compare_annotations(
            'Peter and Jane came in. They saw him.',
            '[0: [0, 2], [6]]')

    def test_reflexive_excluded_mix_of_coordination_and_single_member_2(self):
        self.compare_annotations(
            'Peter and Jane came in. They saw her.',
            '[0: [0, 2], [6]]')

    def test_reflexive_excluded_mix_of_coordination_and_single_member_3(self):
        self.compare_annotations(
            'Peter and Jane came in. He saw them.',
            '[0: [0], [6]]')

    def test_reflexive_excluded_mix_of_coordination_and_single_member_4(self):
        self.compare_annotations(
            'Peter and Jane came in. She saw them.',
            '[0: [2], [6]]')
