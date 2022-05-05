import unittest
from coreferee.test_utils import get_nlps

class EnglishSmokeTest(unittest.TestCase):

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

    def test_simple(self):
        self.compare_annotations('I saw a dog and it was chasing a cat', '[0: [3], [5]]')

    def test_simple_plural(self):
        self.compare_annotations('I saw dogs and they was chasing a cat', '[0: [2], [4]]')

    def test_simple_conjunction_same_word(self):
        self.compare_annotations(
            'I saw a dog and a dog and they were chasing a cat', '[0: [3, 6], [8]]')

    def test_simple_conjunction_different_words(self):
        self.compare_annotations(
            'I saw a dog and a horse and they were chasing a cat', '[0: [3, 6], [8]]')

    def test_conjunction_different_pronouns(self):
        self.compare_annotations(
            'I saw Peter and Jane, and she and he were chasing a cat', '[0: [2], [9], 1: [4], [7]]',
            excluded_nlps=['core_web_sm'])

    def test_conjunction_involving_pronoun(self):
        self.compare_annotations(
            'I saw Peter and Jane. She and Richard were chasing a cat', '[0: [4], [6]]')

    def test_different_sentence(self):
        self.compare_annotations(
            'I saw Peter. He was chasing a cat.', '[0: [2], [4]]')

    def test_proper_noun_coreference(self):
        self.compare_annotations(
            'I saw Peter. Peter was chasing a cat.', '[0: [2], [4]]')

    def test_proper_noun_coreference_multiword(self):
        self.compare_annotations(
            'I saw Peter Paul. Peter Paul was chasing a cat.', '[0: [3], [6]]')

    def test_proper_noun_coreference_multiword_only_second_repeated(self):
        self.compare_annotations(
            'I saw Peter Paul. Paul was chasing a cat.', '[0: [3], [5]]')

    def test_proper_noun_coreference_multiword_only_first_repeated(self):
        self.compare_annotations(
            'I saw Peter Paul. Peter was chasing a cat.', '[]')

    def test_common_noun_coreference(self):
        self.compare_annotations(
            'I saw a big dog. The dog was chasing a cat. It was wagging its tail',
            '[0: [4], [7], [13], [16]]', alternative_expected_coref_chains='[0: [4], [7], 1: [9], [13], [16]]')

    def test_entity_coreference(self):
        self.compare_annotations(
            'I saw Mr. Platt. Everyone liked the friendly man.',
            '[0: [3], [9]]')

    def test_reflexive_simple(self):
        self.compare_annotations(
            'The panther chased itself',
            '[0: [1], [3]]')

    def test_reflexive_coordination(self):
        self.compare_annotations(
            'The panther and the leopard chased themselves',
            '[0: [1, 4], [6]]')

    def test_reflexive_excluded_mix_of_coordination_and_single_member_1(self):
        self.compare_annotations(
            'Peter and Jane came in. They saw him.',
            '[0: [0, 2], [6]]')

    def test_reflexive_anaphor_precedes_referent(self):
        self.compare_annotations(
            'We discussed himself and Peter came in.',
            '[]')

    def test_cataphora_simple(self):
        self.compare_annotations(
            'Although he was tired, Peter went home.',
            '[0: [1], [5]]')

    def test_cataphora_with_coordination(self):
        self.compare_annotations(
            'Although they had reached their house, the man and the woman were sad',
            '[0: [1], [4], [8, 11]]')

    def test_documentation_example_1(self):
        self.compare_annotations(
            'Although he was very busy with his work, Peter had had enough of it. He and his wife decided they needed a holiday. They travelled to Spain because they loved the country very much.',
            '[0: [1], [6], [9], [16], [18], 1: [7], [14], 2: [16, 19], [21], [26], [31], 3: [29], [34]]'
        )

    def test_documentation_example_2(self):
        self.compare_annotations(
            'The woman stood up and saw Lesley. She looked up and greeted her',
            '[0: [1], [13], 1: [6], [8]]'
        )

    def test_documentation_example_3(self):
        self.compare_annotations(
            'The woman stood up and saw Lesley. She looked up and greeted him',
            '[0: [1], [8], 1: [6], [13]]'
        )

    def test_documentation_example_4(self):
        self.compare_annotations(
            'He went to Spain. He loved the country. He often told his friends about it.',
            '[0: [0], [5], [10], [13], 1: [3], [8], [16]]'
        )
