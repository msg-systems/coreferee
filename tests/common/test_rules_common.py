import unittest
import spacy
import coreferee
from coreferee.rules import RulesAnalyzerFactory
from coreferee.test_utils import get_nlps
from coreferee.data_model import Mention

class CommonRulesTest(unittest.TestCase):

    def setUp(self):

        self.nlps = get_nlps('en')
        self.rules_analyzers = [RulesAnalyzerFactory().get_rules_analyzer(nlp) for
            nlp in self.nlps]
        self.sm_nlp = [nlp for nlp in self.nlps if nlp.meta['name'] == 'core_web_sm'][0]
        self.sm_rules_analyzer = RulesAnalyzerFactory().get_rules_analyzer(self.sm_nlp)

    def all_nlps(self, func):
        for nlp in self.nlps:
            func(nlp)

    def compare_sent_starts(self, doc_text, expected_sent_starts):

        def func(nlp):

            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory().get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            self.assertEqual(expected_sent_starts,
                doc._.coref_chains.temp_sent_starts, nlp.meta['name'])

        self.all_nlps(func)

    def test_sent_starts(self):
        self.compare_sent_starts('My name is Charles. I am here. The weather is good',
            [0, 5, 9])

    def compare_sent_indexes(self, doc_text, expected_sent_indexes):

        def func(nlp):

            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory().get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            self.assertEqual(expected_sent_indexes,
                [token._.coref_chains.temp_sent_index for token in doc], nlp.meta['name'])

        self.all_nlps(func)

    def test_sent_index(self):
        self.compare_sent_indexes('My name is Charles. I am here. The weather is good',
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

    def compare_get_dependent_sibling_info(self, doc_text, index, expected_dependent_siblings,
        expected_governing_sibling, expected_has_or_coordination, *, excluded_nlps=[]):

        def func(nlp):


            if nlp.meta['name'] in excluded_nlps:
                return            
            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory.get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            self.assertEqual(expected_dependent_siblings, str(
                doc[index]._.coref_chains.temp_dependent_siblings), nlp.meta['name'])
            for sibling in (sibling for sibling in
                    doc[index]._.coref_chains.temp_dependent_siblings if sibling.i != index):
                self.assertEqual(doc[index], sibling._.coref_chains.temp_governing_sibling,
                    nlp.meta['name'])
            if expected_governing_sibling is None:
                self.assertEqual(None, doc[index]._.coref_chains.temp_governing_sibling,
                    nlp.meta['name'])
            else:
                self.assertEqual(doc[expected_governing_sibling],
                    doc[index]._.coref_chains.temp_governing_sibling, nlp.meta['name'])
            self.assertEqual(expected_has_or_coordination,
                doc[index]._.coref_chains.temp_has_or_coordination, nlp.meta['name'])

        self.all_nlps(func)

    def test_get_dependent_sibling_info_no_conjunction(self):
        self.compare_get_dependent_sibling_info('Richard went home', 0, '[]', None, False)

    def test_get_dependent_sibling_info_two_member_conjunction_phrase_and(self):
        self.compare_get_dependent_sibling_info('Richard and Christine went home', 0,
            '[Christine]', None, False)

    def test_get_dependent_sibling_info_two_member_conjunction_phrase_or(self):
        self.compare_get_dependent_sibling_info('Richard or Christine went home', 0,
            '[Christine]', None, True)

    def test_get_dependent_sibling_info_three_member_conjunction_phrase_with_comma_and(self):
        self.compare_get_dependent_sibling_info('Carol, Richard and Ralf had a meeting', 0,
            '[Richard, Ralf]', None, False)

    def test_get_dependent_sibling_info_three_member_conjunction_phrase_with_comma_or(self):
        self.compare_get_dependent_sibling_info('Carol, Richard or Ralf had a meeting', 0,
            '[Richard, Ralf]', None, True)

    def test_get_dependent_sibling_info_three_member_conjunction_phrase_with_and(self):
        self.compare_get_dependent_sibling_info(
            'There was a meeting with Carol and Ralf and Richard', 5,
            '[Ralf, Richard]', None, False)

    def test_get_dependent_sibling_info_three_member_conjunction_phrase_with_or(self):
        self.compare_get_dependent_sibling_info(
            'A meeting with Carol or Ralf or Richard took place', 3,
            '[Ralf, Richard]', None, True)

    def test_get_dependent_sibling_info_three_member_conjunction_phrase_with_and_and_or(self):
        self.compare_get_dependent_sibling_info(
            'There was a meeting with Carol or Ralf and Richard', 5,
            '[Ralf, Richard]', None, True, excluded_nlps=['core_web_sm'])

    def test_get_dependent_sibling_info_conjunction_itself(self):
        self.compare_get_dependent_sibling_info(
            'There was a meeting with Carol and Ralf and Richard', 6,
            '[]', None, False)

    def test_get_dependent_sibling_info_dependent_sibling(self):
        self.compare_get_dependent_sibling_info(
            'There was a meeting with Carol and Ralf and Richard', 7,
            '[]', 5, False)

    def compare_quote_array(self, doc_text, index, expected_quote_array):

        def func(nlp):

            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory().get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            self.assertEqual(expected_quote_array, doc[index]._.coref_chains.temp_quote_array,
                nlp.meta['name'])

        self.all_nlps(func)

    def test_quote_array_simple(self):
        self.compare_quote_array("He said ‘Give it back‘", 1, [0, 0, 0])
        self.compare_quote_array("He said ‘Give it back‘", 3, [0, 0, 1])

    def test_quote_array_complex(self):
        self.compare_quote_array("He said “Give it \"back\"”", 1, [0, 0, 0])
        self.compare_quote_array("He said “Give it \"back\"”", 3, [0, 1, 0])
        self.compare_quote_array("He said “Give it \"back\"”", 6, [1, 1, 0])

    def compare_potential_noun_pair(self, doc_text, referred_index, referring_index,
            expected_truth, *, excluded_nlps=[]):

        def func(nlp):
            if nlp.meta['name'] in excluded_nlps:
                return
            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory().get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            self.assertEqual(expected_truth,
                rules_analyzer.is_potential_coreferring_noun_pair(doc[referred_index],
                doc[referring_index]), nlp.meta['name'])

        self.all_nlps(func)

    def test_potential_noun_pair_proper_noun_referred(self):
        self.compare_potential_noun_pair('This is Peter. Peter is here', 2, 4, True)

    def test_potential_noun_pair_proper_noun_referred_multiword(self):
        self.compare_potential_noun_pair('This is Peter Smith. Peter Smith is here', 3, 6, True)

    def test_potential_noun_pair_proper_noun_referred_multiword_referring_end(self):
        self.compare_potential_noun_pair('This is Peter Smith. Smith is here', 3, 5, True)

    def test_potential_noun_pair_proper_noun_referred_multiword_referring_beginning(self):
        self.compare_potential_noun_pair('This is Peter Smith. Peter is here', 3, 5, False)

    def test_potential_noun_pair_referred_proper_noun_with_child(self):
        self.compare_potential_noun_pair('I spoke to big Peter. Peter is here', 4, 6, True)

    def test_potential_noun_pair_referred_proper_noun_conjunction_first_member(self):
        self.compare_potential_noun_pair('I spoke to Peter and Jane. Peter is here',
            3, 7, True)

    def test_potential_noun_pair_referred_proper_noun_conjunction_second_member(self):
        self.compare_potential_noun_pair('I spoke to Peter and Jane. Jane is here',
            5, 7, True)

    def test_potential_noun_pair_referring_back_proper_noun_conjunction_first_member(self):
        self.compare_potential_noun_pair('I spoke to Peter. Peter and Jane are here',
            3, 5, True)

    def test_potential_noun_pair_referring_back_proper_noun_conjunction_second_member(self):
        self.compare_potential_noun_pair('I spoke to Jane. Peter and Jane are here',
            3, 7, True)

    def test_potential_noun_pair_definite_common_noun_referred(self):
        self.compare_potential_noun_pair('This is a man. The man is here', 3, 6, True)

    def test_potential_noun_pair_indefinite_common_noun_referred(self):
        self.compare_potential_noun_pair('This is the man. A man is here', 3, 6, False)

    def test_potential_noun_pair_definite_plural_common_noun_referred_1(self):
        self.compare_potential_noun_pair('These are men. The men is here', 2, 5, False)

    def test_potential_noun_pair_definite_plural_common_noun_referred_2(self):
        self.compare_potential_noun_pair('These are some men. The men are here', 3, 6, True)

    def test_potential_noun_pair_definite_plural_common_noun_referred_3(self):
        self.compare_potential_noun_pair('These are the men. The men are here', 3, 6, True)

    def test_potential_noun_pair_indefinite_plural_common_noun_referred(self):
        self.compare_potential_noun_pair('These are men. Men is here', 2, 4, False)

    def test_potential_noun_pair_numbers_do_not_match(self):
        self.compare_potential_noun_pair('This is a man. The men are here.', 3, 6, False)

    def test_potential_noun_pair_entity_labels_referred(self):
        self.compare_potential_noun_pair(
            'I spoke to the boss of Lehman Brothers. The company went bust', 7, 10, True)

    def test_potential_noun_pair_entity_labels_referred_not_definite_control(self):
        self.compare_potential_noun_pair(
            'I spoke to the boss of Lehman Brothers. A company went bust', 7, 10, False)

    def test_potential_noun_pair_wrong_entity_label_referred(self):
        self.compare_potential_noun_pair(
            'I spoke to the boss of Lehman Brothers. The person went bust', 7, 10, False)

    def test_potential_noun_pair_single_letters(self):
        self.compare_potential_noun_pair(
            'This is a ©. The © is here.', 3, 6, False)

    def test_potential_noun_pair_same_governing_sibling(self):
        self.compare_potential_noun_pair(
            'The dog, the dog and the dog came home.', 4, 7, False)

    def test_potentially_independent_nouns_stored_on_token(self):
        doc = self.sm_nlp('They went to look at the space suits')
        self.sm_rules_analyzer.initialize(doc)
        self.assertFalse(doc[3]._.coref_chains.temp_potentially_referring)
        self.assertFalse(doc[6]._.coref_chains.temp_potentially_referring)
        self.assertTrue(doc[7]._.coref_chains.temp_potentially_referring)


    def compare_potential_pair(self, doc_text, referred_index, include_dependent_siblings,
        referring_index, expected_truth, consider_syntax=True, *, excluded_nlps=[]):

        def func(nlp):

            if nlp.meta['name'] in excluded_nlps:
                return
            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory.get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            assert rules_analyzer.is_independent_noun(doc[referred_index]) or \
                rules_analyzer.is_potential_anaphor(doc[referred_index])
            assert rules_analyzer.is_potential_anaphor(doc[referring_index])
            referred_mention = Mention(doc[referred_index], include_dependent_siblings)
            if consider_syntax:
                self.assertEqual(expected_truth,
                    rules_analyzer.language_independent_is_potential_anaphoric_pair(
                    referred_mention, doc[referring_index]), nlp.meta['name'])
            else:
                self.assertEqual(expected_truth, rules_analyzer.is_potential_anaphoric_pair(
                    referred_mention, doc[referring_index], False), nlp.meta['name'])
        self.all_nlps(func)


    def test_closer_within_structure_propn(self):

        self.compare_potential_pair('Richard arrived. Richard saw him.', 0, False, 5, 1)

    def test_closer_within_structure_propn_conjunction_first(self):

        self.compare_potential_pair('Richard and Peter arrived. Richard saw him.', 0, False, 7, 1)

    def test_closer_within_structure_propn_conjunction_second(self):

        self.compare_potential_pair('Richard and Peter arrived. Peter saw him.', 2, False, 7, 1)

    def test_closer_within_structure_child(self):

        self.compare_potential_pair('The dog arrived. The big dog saw him.', 1, False, 8, 2)

    def test_closer_within_structure_only_determiner(self):

        self.compare_potential_pair('The dog arrived. The dog saw him.', 1, False, 7, 1)

    def test_closer_within_structure_only_determiner_conjunction_first(self):

        self.compare_potential_pair('The dog and the cat arrived. The dog saw him.', 1, False,
            10, 1)

    def test_closer_within_structure_only_determiner_conjunction_second(self):

        self.compare_potential_pair('The dog and the cat arrived. The cat saw him.', 4, False,
            10, 1)

    def test_closer_within_structure_all_pronouns_control(self):

        self.compare_potential_pair('He arrived. He saw him.', 0, False, 5, 2)

    def test_consider_syntax_false(self):

        self.compare_potential_pair('He saw him.', 0, False, 2, 2, False)

    def test_consider_syntax_false_control(self):

        self.compare_potential_pair('He saw him', 0, False, 2, 0, True)

    def test_quotes(self):
        self.compare_potential_pair('"Richard is here", he said.', 1, False, 6, 1)

    def test_propn_subtree_beginning(self):
        doc = self.sm_nlp('Richard Hudson is here')
        self.assertEqual([], [t.i for t in self.sm_rules_analyzer.get_propn_subtree(doc[0])])
        self.assertEqual([0, 1], [t.i for t in self.sm_rules_analyzer.get_propn_subtree(doc[1])])
        self.assertEqual([], [t.i for t in self.sm_rules_analyzer.get_propn_subtree(doc[3])])

    def test_propn_subtree_middle(self):
        doc = self.sm_nlp('He spoke to Richard Hudson yesterday')
        self.assertEqual([], [t.i for t in self.sm_rules_analyzer.get_propn_subtree(doc[0])])
        self.assertEqual([], [t.i for t in self.sm_rules_analyzer.get_propn_subtree(doc[3])])
        self.assertEqual([3, 4], [t.i for t in self.sm_rules_analyzer.get_propn_subtree(doc[4])])
        self.assertEqual([], [t.i for t in self.sm_rules_analyzer.get_propn_subtree(doc[5])])

    def test_propn_subtree_end(self):
        doc = self.sm_nlp('He spoke to Richard Hudson')
        self.assertEqual([], [t.i for t in self.sm_rules_analyzer.get_propn_subtree(doc[0])])
        self.assertEqual([], [t.i for t in self.sm_rules_analyzer.get_propn_subtree(doc[3])])
        self.assertEqual([3, 4], [t.i for t in self.sm_rules_analyzer.get_propn_subtree(doc[4])])

    def test_propn_subtree_with_coordination(self):
        doc = self.sm_nlp('Richard Hudson and Peter Jones are here')
        self.assertEqual([], [t.i for t in self.sm_rules_analyzer.get_propn_subtree(doc[0])])
        self.assertEqual([0, 1], [t.i for t in self.sm_rules_analyzer.get_propn_subtree(doc[1])])
        self.assertEqual([], [t.i for t in self.sm_rules_analyzer.get_propn_subtree(doc[2])])
        self.assertEqual([], [t.i for t in self.sm_rules_analyzer.get_propn_subtree(doc[3])])
        self.assertEqual([3, 4], [t.i for t in self.sm_rules_analyzer.get_propn_subtree(doc[4])])

    def compare_potentially_referring(self, doc_text, expected_per_indexes, *,
        excluded_nlps=[]):

        def func(nlp):

            if nlp.meta['name'] in excluded_nlps:
                return
            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory.get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            per_indexes = [token.i for token in doc if
                    rules_analyzer.is_independent_noun(token)]
            self.assertEqual(expected_per_indexes, per_indexes, nlp.meta['name'])

        self.all_nlps(func)

    def test_has_morph_without_value(self):

        def func(nlp):

            doc = nlp('they')
            rules_analyzer = RulesAnalyzerFactory().get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            self.assertEqual(True, rules_analyzer.has_morph(doc[0], 'Number'), nlp.meta['name'])
            self.assertEqual(False, rules_analyzer.has_morph(doc[0], 'Other'), nlp.meta['name'])
        self.all_nlps(func)

    def test_has_morph_with_value(self):

        def func(nlp):

            doc = nlp('they')
            rules_analyzer = RulesAnalyzerFactory().get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            self.assertEqual(True, rules_analyzer.has_morph(
                doc[0], 'Number', 'Plur'), nlp.meta['name'])
            self.assertEqual(False, rules_analyzer.has_morph(
                doc[0], 'Number', 'Other'), nlp.meta['name'])
            self.assertEqual(False, rules_analyzer.has_morph(
                doc[0], 'Other', 'Other'), nlp.meta['name'])
        self.all_nlps(func)

    def compare_non_or_truths(self, doc_text, expected_trues):

        def func(nlp):

            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory().get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            non_or_truths = [token.i for token in doc
                if rules_analyzer.is_involved_in_non_or_conjunction(token)]
            self.assertEqual(expected_trues, non_or_truths, nlp.meta['name'])

        self.all_nlps(func)

    def test_is_involved_in_non_or_conjunction_and_conjunction(self):

        self.compare_non_or_truths('Richard and Christine went home', [0,2])

    def test_is_involved_in_non_or_conjunction_or_conjunction(self):

        self.compare_non_or_truths('Richard or Christine went home', [])

    def test_is_involved_in_non_or_conjunction_mixed_conjunction(self):

        self.compare_non_or_truths('Richard or Christine and Peter went home', [])

    def compare_potentially_introducing(self, doc_text, index, expected_truth, *,
            excluded_nlps=[]):

        def func(nlp):

            if nlp.meta['name'] in excluded_nlps:
                return
            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory.get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            self.assertEqual(expected_truth,
                rules_analyzer.is_potentially_introducing_noun(doc[index]),
                nlp.meta['name'])

        self.all_nlps(func)

    def test_potentially_introducing_not_noun(self):
        self.compare_potentially_introducing('I spoke to Peter', 2, False)

    def test_potentially_introducing_definite_noun(self):
        self.compare_potentially_introducing('I spoke to the man', 4, False)

    def test_potentially_introducing_indefinite_noun(self):
        self.compare_potentially_introducing('I spoke to a man', 4, True)

    def test_potentially_introducing_definite_noun_with_adjective(self):
        self.compare_potentially_introducing('I spoke to the big man', 5, True)

    def test_potentially_introducing_definite_noun_with_dependent_phrase(self):
        self.compare_potentially_introducing('I saw the man whom we had discussed', 3, True,
            excluded_nlps='core_web_sm')

    def test_potentially_introducing_common_noun_conjunction_first_member(self):
        self.compare_potentially_introducing('I spoke to a man and a woman', 4, True)

    def test_potentially_introducing_common_noun_conjunction_second_member(self):
        self.compare_potentially_introducing('I spoke to a man and a woman', 7, True)

    def test_potentially_introducing_twoway_conjunction_second_member_no_article(self):
        self.compare_potentially_introducing('I spoke to some men and women', 6, True)

    def test_potentially_introducing_threeway_conjunction_second_member_no_article(self):
        self.compare_potentially_introducing('I spoke to some men, women and children', 6,
            True)

    def test_potentially_introducing_threeway_conjunction_third_member_no_article(self):
        self.compare_potentially_introducing('I spoke to some men, women and children',
            8, True)

    def test_potentially_introducing_twoway_conjunction_second_member_no_article_control(self):
        self.compare_potentially_introducing('I spoke to the men and women', 6, False)

    def test_potentially_introducing_threeway_conjunction_second_member_no_article_control(self):
        self.compare_potentially_introducing('I spoke to the men, women and children', 6,
            False)

    def test_potentially_introducing_threeway_conjunction_third_member_no_article_control(self):
        self.compare_potentially_introducing('I spoke to the men, women and children',
            8, False)

    def compare_potentially_referring_back_noun(self, doc_text, index, expected_truth, *,
            excluded_nlps=[]):

        def func(nlp):

            if nlp.meta['name'] in excluded_nlps:
                return
            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory.get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            self.assertEqual(expected_truth,
                rules_analyzer.is_potentially_referring_back_noun(doc[index]),
                nlp.meta['name'])

        self.all_nlps(func)

    def test_potentially_referring_back_noun_not_noun(self):
        self.compare_potentially_referring_back_noun('I spoke to Peter', 2, False)

    def test_potentially_referring_back_noun_definite_noun(self):
        self.compare_potentially_referring_back_noun('I spoke to the man', 4, True)

    def test_potentially_referring_back_noun_indefinite_noun(self):
        self.compare_potentially_referring_back_noun('I spoke to a man', 4, False)

    def test_potentially_referring_back_noun_definite_noun_with_adjective(self):
        self.compare_potentially_referring_back_noun('I spoke to the big man', 5, False)

    def test_potentially_referring_back_noun_definite_noun_with_dependent_phrase(self):
        self.compare_potentially_referring_back_noun('I saw the man whom we had discussed', 3, False,
            excluded_nlps='core_web_sm')

    def test_potentially_referring_back_noun_common_noun_conjunction_first_member(self):
        self.compare_potentially_referring_back_noun('I spoke to the man and the woman', 4, True)

    def test_potentially_referring_back_noun_common_noun_conjunction_second_member(self):
        self.compare_potentially_referring_back_noun('I spoke to the man and the woman', 7, True)

    def test_potentially_referring_back_twoway_conjunction_second_member_no_article(self):
        self.compare_potentially_referring_back_noun('I spoke to the men and women', 6, True)

    def test_potentially_referring_back_threeway_conjunction_second_member_no_article(self):
        self.compare_potentially_referring_back_noun('I spoke to the men, women and children', 6,
            True)

    def test_potentially_referring_back_threeway_conjunction_third_member_no_article(self):
        self.compare_potentially_referring_back_noun('I spoke to the men, women and children',
            8, True)

    def test_potentially_referring_back_twoway_conjunction_second_member_no_article_control(self):
        self.compare_potentially_referring_back_noun('I spoke to some men and women', 6, False)

    def test_potentially_referring_back_threeway_conjunction_second_member_no_article_control(self):
        self.compare_potentially_referring_back_noun('I spoke to some men, women and children', 6,
            False)

    def test_potentially_referring_back_threeway_conjunction_third_member_no_article_control(self):
        self.compare_potentially_referring_back_noun('I spoke to some men, women and children',
            8, False)
