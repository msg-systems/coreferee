import unittest
from coreferee.test_utils import get_nlps, debug_structures

nlps = get_nlps("de")
train_version_mismatch = False
for nlp in nlps:
    if not nlp.meta["matches_train_version"]:
        train_version_mismatch = True
train_version_mismatch_message = "Loaded model version does not match train model version"

class GermanSmokeTest(unittest.TestCase):

    def setUp(self):

        self.nlps = get_nlps('de')

    def all_nlps(self, func):
        for nlp in self.nlps:
            func(nlp)

    def compare_annotations(self, doc_text, expected_coref_chains, *, excluded_nlps=[],
        alternative_expected_coref_chains=None):

        def func(nlp):

            if nlp.meta['name'] in excluded_nlps:
                return

            doc = nlp(doc_text)
            debug_structures(doc)
            chains_representation = str(doc._.coref_chains)
            if alternative_expected_coref_chains is None:
                self.assertEqual(expected_coref_chains,
                    chains_representation, nlp.meta['name'])
            else:
                self.assertTrue(expected_coref_chains == chains_representation or
                    alternative_expected_coref_chains == chains_representation)

        self.all_nlps(func)

    def test_simple(self):
        self.compare_annotations('Ich sah einen Hund, und er jagte eine Katze', '[0: [3], [6]]')

    def test_simple_plural(self):
        self.compare_annotations('Ich sah Hunde, und sie jagten eine Katze', '[0: [2], [5]]')

    def test_simple_conjunction_same_word(self):
        self.compare_annotations(
            'Ich sah einen Hund und einen Hund, und sie jagten eine Katze', '[0: [3, 6], [9]]',
                excluded_nlps=['core_news_lg'])

    def test_simple_conjunction_different_words(self):
        self.compare_annotations(
            'Ich sah einen Hund und ein Pferd, und sie jagten eine Katze', '[0: [3, 6], [9]]')

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_conjunction_different_pronouns(self):
        self.compare_annotations(
            'Peter und das Mädchen haben gesprochen, und dieses und er jagten ein Katze', '[0: [0], [10], 1: [3], [8]]')

    def test_conjunction_involving_pronoun(self):
        self.compare_annotations(
            'Ich sah Peter und Jana. Er und Richard jagten eine Katze', '[0: [2], [6]]')

    def test_different_sentence(self):
        self.compare_annotations(
            'Ich sah Peter. Er jagte eine Katze.', '[0: [2], [4]]')

    def test_proper_noun_coreference(self):
        self.compare_annotations(
            'Ich sah Peter. Peter jagte eine Katze.', '[0: [2], [4]]')

    def test_proper_noun_coreference_multiword(self):
        self.compare_annotations(
            'Ich sah Peter Paul. Peter Paul jagte eine Katze.', '[0: [3], [6]]')

    def test_proper_noun_coreference_multiword_only_second_repeated(self):
        self.compare_annotations(
            'Ich sah Peter Paul. Paul jagte eine Katze.', '[0: [3], [5]]')

    def test_proper_noun_coreference_multiword_only_first_repeated(self):
        self.compare_annotations(
            'Ich sah Peter Paul. Peter jagte eine Katze.', '[]')

    def test_proper_noun_coreference_with_gender_difference(self):
        self.compare_annotations(
            'Er arbeitete bei der BMW AG. Das Unternehmen gefiel ihm. Es hatte viel Erfolg auf dem deutschen Markt', '[0: [0], [10], 1: [4], [8], [12]]', excluded_nlps=["core_news_md"])

    def test_common_noun_coreference(self):
        self.compare_annotations(
            'Ich sah einen großen Hund. Der Hund jagte eine Katze. Er wedelte mit seinem Schwanz.',
            '[0: [4], [7], [12], [15]]')

    def test_entity_coreference(self):
        self.compare_annotations(
            'Peter sprach laut. Alle mochten den freundlichen Mann.',
            '[0: [0], [8]]', excluded_nlps=['core_news_sm'])

    def test_reflexive_simple(self):
        self.compare_annotations(
            'Der Gepard jagte sich',
            '[0: [1], [3]]')

    def test_reflexive_coordination(self):
        self.compare_annotations(
            'Der Gepard und der Leopard jagten sich',
            '[0: [1, 4], [6]]')

    def test_reflexive_excluded_mix_of_coordination_and_single_member_1(self):
        self.compare_annotations(
            'Peter und Jana kamen rein. Sie sahen ihn.',
            '[0: [0, 2], [6]]')

    def test_reflexive_excluded_mix_of_coordination_and_single_member_2(self):
        self.compare_annotations(
            'Peter und Jana kamen rein. Sie sahen sie.',
            '[0: [0, 2], [6]]')

    def test_reflexive_preceding(self):
        self.compare_annotations(
            'Er wollte, dass sich sein Sohn um die Sache kümmere',
            '[0: [0], [5], 1: [4], [6]]')

    def test_cataphora_simple(self):
        self.compare_annotations(
            'Obwohl er müde war, ging Peter nach Hause',
            '[0: [1], [6]]')

    def test_cataphora_with_coordination(self):
        self.compare_annotations(
            'Obwohl sie ihr Haus erreicht hatten, waren der Mann und die Frau traurig',
            '[0: [1], [12], 1: [2], [9, 12]]',
            alternative_expected_coref_chains='[0: [1], [2], [9, 12]]')

    def test_possessive_pronoun_within_threeway_coordination(self):
        self.compare_annotations(
            'Wir sahen Peter, seine Frau und seine Kinder.',
            '[0: [2], [4], [7]]')

    def test_anaphoric_preposition_within_coordination(self):
        self.compare_annotations(
            'Die Maßnahmen sowie die daraus resultierenden Probleme wurden betrachtet.',
            '[0: [1], [4]]')

    def test_pronominal_anaphor(self):
        self.compare_annotations(
            'Sie fand ein Messer und aß damit einen Apfel.',
            '[0: [3], [6]]')

    def test_documentation_example(self):
        self.compare_annotations(
            'Weil er mit seiner Arbeit sehr beschäftigt war, hatte Peter davon genug. Er und seine Frau haben entschieden, dass ihnen ein Urlaub gut tun würde. Sie sind nach Spanien gefahren, weil ihnen das Land sehr gefiel.',
            '[0: [1], [3], [10], [14], [16], 1: [4], [11], 2: [14, 17], [22], [29], [36], 3: [32], [38]]', excluded_nlps=['core_news_sm']
        )
