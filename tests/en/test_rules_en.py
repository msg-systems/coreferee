import unittest
from coreferee.rules import RulesAnalyzerFactory
from coreferee.test_utils import get_nlps
from coreferee.data_model import Mention

nlps = get_nlps('en')
train_version_mismatch = False
for nlp in nlps:
    if not nlp.meta["matches_train_version"]:
        train_version_mismatch = True
train_version_mismatch_message = "Loaded model version does not match train model version"


class EnglishRulesTest(unittest.TestCase):
    def setUp(self):

        self.nlps = get_nlps("en")
        self.rules_analyzers = [
            RulesAnalyzerFactory.get_rules_analyzer(nlp) for nlp in self.nlps
        ]

    def all_nlps(self, func):
        for nlp in self.nlps:
            func(nlp)

    def compare_get_dependent_sibling_info(
        self,
        doc_text,
        index,
        expected_dependent_siblings,
        expected_governing_sibling,
        expected_has_or_coordination,
        *,
        excluded_nlps=[]
    ):
        def func(nlp):

            if nlp.meta["name"] in excluded_nlps:
                return
            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory.get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            self.assertEqual(
                expected_dependent_siblings,
                str(doc[index]._.coref_chains.temp_dependent_siblings),
                nlp.meta["name"],
            )
            for sibling in (
                sibling
                for sibling in doc[index]._.coref_chains.temp_dependent_siblings
                if sibling.i != index
            ):
                self.assertEqual(
                    doc[index],
                    sibling._.coref_chains.temp_governing_sibling,
                    nlp.meta["name"],
                )
            if expected_governing_sibling is None:
                self.assertEqual(
                    None,
                    doc[index]._.coref_chains.temp_governing_sibling,
                    nlp.meta["name"],
                )
            else:
                self.assertEqual(
                    doc[expected_governing_sibling],
                    doc[index]._.coref_chains.temp_governing_sibling,
                    nlp.meta["name"],
                )
            self.assertEqual(
                expected_has_or_coordination,
                doc[index]._.coref_chains.temp_has_or_coordination,
                nlp.meta["name"],
            )

        self.all_nlps(func)

    def test_get_dependent_sibling_info_no_conjunction(self):
        self.compare_get_dependent_sibling_info(
            "Richard went home", 0, "[]", None, False
        )

    def test_get_dependent_sibling_info_two_member_conjunction_phrase_and(self):
        self.compare_get_dependent_sibling_info(
            "Richard and Christine went home", 0, "[Christine]", None, False
        )

    def test_get_governing_sibling_info_two_member_conjunction_phrase_and(self):
        self.compare_get_dependent_sibling_info(
            "Richard and Christine went home", 2, "[]", 0, False
        )

    def test_get_dependent_sibling_info_two_member_conjunction_phrase_or(self):
        self.compare_get_dependent_sibling_info(
            "Richard or Christine went home", 0, "[Christine]", None, True
        )

    def test_get_dependent_sibling_info_apposition_control(self):
        self.compare_get_dependent_sibling_info(
            "Richard, the developer, went home", 0, "[]", None, False
        )

    def test_get_governing_sibling_info_apposition_control(self):
        self.compare_get_dependent_sibling_info(
            "Richard, the developer, went home", 3, "[]", None, False
        )        

    def test_get_dependent_sibling_info_three_member_conjunction_phrase_with_comma_and(
        self,
    ):
        self.compare_get_dependent_sibling_info(
            "Carol, Richard and Ralf had a meeting", 0, "[Richard, Ralf]", None, False
        )

    def test_get_dependent_sibling_info_three_member_conjunction_phrase_with_comma_or(
        self,
    ):
        self.compare_get_dependent_sibling_info(
            "Carol, Richard or Ralf had a meeting", 0, "[Richard, Ralf]", None, True
        )

    def test_get_dependent_sibling_info_three_member_conjunction_phrase_with_and(self):
        self.compare_get_dependent_sibling_info(
            "There was a meeting with Carol and Ralf and Richard",
            5,
            "[Ralf, Richard]",
            None,
            False,
        )

    def test_get_dependent_sibling_info_three_member_conjunction_phrase_with_or(self):
        self.compare_get_dependent_sibling_info(
            "A meeting with Carol or Ralf or Richard took place",
            3,
            "[Ralf, Richard]",
            None,
            True,
        )

    def test_get_dependent_sibling_info_three_member_conjunction_phrase_with_and_and_or(
        self,
    ):
        self.compare_get_dependent_sibling_info(
            "There was a meeting with Carol or Ralf and Richard",
            5,
            "[Ralf, Richard]",
            None,
            True,
            excluded_nlps=['core_web_sm']
        )

    def test_get_dependent_sibling_info_conjunction_itself(self):
        self.compare_get_dependent_sibling_info(
            "There was a meeting with Carol and Ralf and Richard", 6, "[]", None, False
        )

    def test_get_dependent_sibling_info_dependent_sibling(self):
        self.compare_get_dependent_sibling_info(
            "There was a meeting with Carol and Ralf and Richard", 7, "[]", 5, False
        )

    def compare_independent_noun(
        self, doc_text, expected_per_indexes, *, alternative=[], excluded_nlps=[]
    ):
        def func(nlp):

            if nlp.meta["name"] in excluded_nlps:
                return
            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory.get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            per_indexes = [
                token.i for token in doc if rules_analyzer.is_independent_noun(token)
            ]
            self.assertTrue(
                expected_per_indexes == per_indexes or alternative == per_indexes,
                nlp.meta["name"],
            )

        self.all_nlps(func)

    def test_independent_noun_nouns(self):
        self.compare_independent_noun("They went to look at the space suits", [7])

    def test_independent_noun_proper_nouns(self):
        self.compare_independent_noun(
            "Peter and Jane went home", [0, 2], alternative=[0, 2, 4]
        )

    def test_independent_noun_numerals(self):
        self.compare_independent_noun(
            "One of the two people saw two of the three people", [0, 4, 6, 10]
        )

    def test_independent_noun_determiners(self):
        self.compare_independent_noun(
            "Those of these people said that was interesting", [0, 3, 5]
        )

    def test_independent_noun_pronouns(self):
        self.compare_independent_noun(
            "Those of these people said somebody was interesting", [0, 3, 5]
        )

    def test_independent_noun_blacklisted_phrase_beginning(self):
        self.compare_independent_noun("No wonder the issue was a problem", [3, 6])

    def test_independent_noun_blacklisted_phrase_end(self):
        self.compare_independent_noun("The issue was a problem, by the way", [1, 4])

    def test_independent_noun_blacklisted_phrase_middle_and_end(self):
        self.compare_independent_noun(
            "The issue, for example, was a problem, by the way", [1, 8]
        )

    def compare_potential_anaphor(
        self, doc_text, expected_per_indexes, *, excluded_nlps=[]
    ):
        def func(nlp):

            if nlp.meta["name"] in excluded_nlps:
                return
            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory.get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            per_indexes = [
                token.i for token in doc if rules_analyzer.is_potential_anaphor(token)
            ]
            self.assertEqual(expected_per_indexes, per_indexes, nlp.meta["name"])

        self.all_nlps(func)

    def test_third_person_pronouns(self):
        self.compare_potential_anaphor("She went outside to look at her car", [0, 6])

    def test_first_and_second_person_pronouns(self):
        self.compare_potential_anaphor("I went outside to look at your car", [])

    def test_non_pleonastic_it_and_its(self):
        self.compare_potential_anaphor(
            "I saw a star. It was shining and its brilliance was amazing.", [5, 9]
        )

    def test_non_pleonastic_it_with_conjunction(self):
        self.compare_potential_anaphor(
            "It and the man were raining. The man and it were raining",
            [0, 10],
            excluded_nlps=["core_web_trf", "core_web_md"],
        )

    def test_pleonastic_it_avalent_verbs(self):
        self.compare_potential_anaphor(
            "It was raining. It rained. It started raining. It stopped raining.", []
        )

    def test_pleonastic_it_avalent_verbs_2(self):
        self.compare_potential_anaphor(
            "It should have stopped thinking about raining.",
            [],
            excluded_nlps=["core_web_md", "core_web_sm"],
        )

    def test_pleonastic_it_case_1(self):
        self.compare_potential_anaphor("It is important that he has done it", [4, 7])

    def test_pleonastic_it_case_2(self):
        self.compare_potential_anaphor("It is believed that he has done it", [4, 7])

    def test_pleonastic_it_case_2_control(self):
        self.compare_potential_anaphor("It is believed in that part of the world", [0])

    def test_pleonastic_it_case_3(self):
        self.compare_potential_anaphor(
            "This makes it unlikely that he has done it", [5, 8]
        )

    def test_pleonastic_it_case_4(self):
        self.compare_potential_anaphor(
            "It is in everyone's interest that attempting it should succeed",
            [8],
            excluded_nlps="core_web_sm",
        )

    def compare_ancestor(self, doc_text, index, expected_index, *, excluded_nlps=[]):
        def func(nlp):

            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory.get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            ancestor = rules_analyzer.get_ancestor_spanning_any_preposition(doc[index])
            if ancestor is None:
                self.assertEqual(expected_index, None)
            else:
                self.assertEqual(expected_index, ancestor.i)

        self.all_nlps(func)

    def test_ancestor_simple(self):
        self.compare_ancestor("The dog chased the cat", 4, 2)

    def test_ancestor_preposition(self):
        self.compare_ancestor("The dog chased the cat into the house", 7, 2)

    def test_ancestor_root(self):
        self.compare_ancestor("The dog chased the cat", 2, None)

    def test_ancestor_preposition_is_root(self):
        self.compare_ancestor("into the house", 2, None)

    def compare_potential_pair(
        self,
        doc_text,
        referred_index,
        include_dependent_siblings,
        referring_index,
        expected_truth,
        *,
        excluded_nlps=[]
    ):
        def func(nlp):

            if nlp.meta["name"] in excluded_nlps:
                return
            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory.get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            assert rules_analyzer.is_independent_noun(
                doc[referred_index]
            ) or rules_analyzer.is_potential_anaphor(doc[referred_index])
            assert rules_analyzer.is_potential_anaphor(doc[referring_index])
            referred_mention = Mention(doc[referred_index], include_dependent_siblings)
            self.assertEqual(
                expected_truth,
                rules_analyzer.is_potential_anaphoric_pair(
                    referred_mention, doc[referring_index], True
                ),
                nlp.meta["name"],
            )

        self.all_nlps(func)

    def test_potential_pair_trivial_singular(self):
        self.compare_potential_pair("I saw a man. He was walking", 3, False, 5, 2)

    def test_potential_pair_trivial_plural(self):
        self.compare_potential_pair(
            "I saw a man and a woman. They were walking", 3, True, 8, 2
        )

    def test_potential_pair_plurals_with_coordination_first(self):
        self.compare_potential_pair(
            "I saw men and women. They were walking", 2, False, 6, 0
        )

    def test_potential_pair_plurals_with_coordination_second(self):
        self.compare_potential_pair(
            "I saw men and women. They were walking", 4, False, 6, 0
        )

    def test_potential_pair_plurals_with_coordination_second(self):
        self.compare_potential_pair(
            "I saw men and women. They were walking", 4, False, 6, 0
        )

    def test_potential_pair_plurals_with_coordination_first(self):
        self.compare_potential_pair(
            "I saw a man and a woman. They were walking", 3, False, 8, 0
        )

    def test_potential_pair_different_pronouns_1(self):
        self.compare_potential_pair("I saw him and her friend", 2, False, 4, 0)

    def test_potential_pair_different_pronouns_2(self):
        self.compare_potential_pair("I saw him and their friend", 2, False, 4, 0)

    def test_potential_pair_different_pronouns_control(self):
        self.compare_potential_pair("I saw him and his friend", 2, False, 4, 2)

    def test_potential_pair_plural_referred_singular_referring(self):
        self.compare_potential_pair("I saw the men. He was there", 3, False, 5, 0)

    def test_potential_pair_and_conjunction_referred_singular_referring(self):
        self.compare_potential_pair(
            "I saw the man and the woman. He was there", 3, True, 8, 0
        )

    def test_potential_pair_and_conjunction_referred_singular_referring_control(self):
        self.compare_potential_pair(
            "I saw the man and the woman. He was there", 3, False, 8, 2
        )

    def test_potential_pair_they_singular_antecedent(self):
        self.compare_potential_pair("I saw the house. They were there", 3, False, 5, 0)

    def test_potential_pair_they_singular_antecedent_person(self):
        self.compare_potential_pair("I saw the judge. They were there", 3, False, 5, 2)

    def test_potential_pair_they_singular_antecedent_male_person(self):
        self.compare_potential_pair(
            "I saw the gentleman. They were there", 3, False, 5, 1
        )

    def test_potential_pair_they_singular_antecedent_female_person(self):
        self.compare_potential_pair("I saw the lady. They were there", 3, False, 5, 1)

    def test_potential_pair_they_singular_antecedent_proper_name_person(self):
        self.compare_potential_pair("I spoke to Jenny. They were there", 3, False, 5, 1)

    def test_potential_pair_they_singular_antecedent_proper_name_non_person(self):
        self.compare_potential_pair(
            "I worked for Peters. They were there", 3, False, 5, 1
        )

    def test_potential_pair_it_singular_antecedent_singular_proper_name_person(self):
        self.compare_potential_pair("I spoke to Jenny. It was there", 3, False, 5, 0)

    def test_potential_pair_it_singular_antecedent_plural_proper_name_person(self):
        self.compare_potential_pair("I spoke to Peters. It was there", 3, False, 5, 0)

    def test_potential_pair_it_singular_antecedent_proper_name_non_person(self):
        self.compare_potential_pair(
            "I worked for Skateboards plc. It was there", 4, False, 6, 2
        )

    def test_potential_pair_he_she_antecedent_non_person_noun(self):
        self.compare_potential_pair("I saw the house. She was there", 3, False, 5, 0)

    def test_potential_pair_he_she_antecedent_person_noun(self):
        self.compare_potential_pair("I spoke to Jenny. She was there", 3, False, 5, 2)

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_potential_pair_he_she_antecedent_non_person_proper_noun(self):
        self.compare_potential_pair(
            "I worked for Skateboards plc. She was there", 4, False, 6, 1
        )

    def test_potential_pair_it_exclusively_person_antecedent(self):
        self.compare_potential_pair("I saw the lady. It was there", 3, False, 5, 0)

    def test_potential_pair_it_name_antecedent(self):
        self.compare_potential_pair("I saw Peter. It was there", 2, False, 4, 0)

    def test_potential_pair_it_exclusively_person_antecedent_control(self):
        self.compare_potential_pair("I saw the house. It was there", 3, False, 5, 2)

    def test_potential_pair_he_exclusively_female_antecedent(self):
        self.compare_potential_pair("I saw the woman. He was there", 3, False, 5, 0)

    def test_potential_pair_he_exclusively_female_name_antecedent(self):
        self.compare_potential_pair("I saw Jane. He was there", 2, False, 4, 1)

    def test_potential_pair_he_exclusively_female_name_antecedent_control(self):
        self.compare_potential_pair("I saw Jane. She was there", 2, False, 4, 2)

    def test_potential_pair_he_exclusively_female_name_compound_antecedent(self):
        self.compare_potential_pair("I saw Mrs. Jones. He was there", 3, False, 5, 1)

    def test_potential_pair_he_exclusively_female_name_compound_antecedent_control(
        self,
    ):
        self.compare_potential_pair("I saw Mrs. Jones. She was there", 3, True, 5, 2)

    def test_potential_pair_he_exclusively_female_antecedent_control(self):
        self.compare_potential_pair("I saw the person. He was there", 3, False, 5, 2)

    def test_potential_pair_she_exclusively_male_antecedent(self):
        self.compare_potential_pair(
            "I saw the gentleman. She was there", 3, False, 5, 0
        )

    def test_potential_pair_she_exclusively_male_antecedent_control(self):
        self.compare_potential_pair("I saw the person. She was there", 3, False, 5, 2)

    def test_potential_pair_she_exclusively_male_name_antecedent(self):
        self.compare_potential_pair("I saw Peter. She was there", 2, False, 4, 1)

    def test_potential_pair_she_exclusively_male_name_antecedent_control(self):
        self.compare_potential_pair("I saw Peter. He was there", 2, False, 4, 2)

    def test_potential_pair_she_exclusively_male_name_compound_antecedent(self):
        self.compare_potential_pair("I saw Peter Jones. She was there", 3, False, 5, 1)

    def test_potential_pair_she_exclusively_male_name_compound_antecedent_control(self):
        self.compare_potential_pair("I saw Peter Jones. He was there", 3, True, 5, 2)

    def test_potential_pair_person_word_non_capitalized(self):
        self.compare_potential_pair("I saw a job. He was there", 3, False, 5, 0)

    def test_potential_pair_person_word_capitalized(self):
        self.compare_potential_pair("I saw Job. He was there", 2, False, 4, 1)

    def test_potential_pair_person_word_non_capitalized_exclusively_person_word(self):
        self.compare_potential_pair("I saw a copt. He was there", 3, False, 5, 0, excluded_nlps=["core_web_md"])

    def test_potential_pair_person_word_capitalized_exclusively_person_word(self):
        self.compare_potential_pair("I saw a Copt. He was there", 3, False, 5, 2)

    def test_potential_pair_antecedent_in_prepositional_phrase_in_question(self):
        self.compare_potential_pair("In which room was it?", 2, False, 4, 0)

    def compare_potential_reflexive_pair(
        self,
        doc_text,
        referred_index,
        include_dependent_siblings,
        referring_index,
        expected_truth,
        expected_reflexive_truth,
        is_reflexive_anaphor_truth,
        *,
        excluded_nlps=[]
    ):
        def func(nlp):

            if nlp.meta["name"] in excluded_nlps:
                return
            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory.get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            assert rules_analyzer.is_independent_noun(
                doc[referred_index]
            ) or rules_analyzer.is_potential_anaphor(doc[referred_index])
            assert rules_analyzer.is_potential_anaphor(doc[referring_index])
            referred_mention = Mention(doc[referred_index], include_dependent_siblings)
            self.assertEqual(
                expected_truth,
                rules_analyzer.is_potential_anaphoric_pair(
                    referred_mention, doc[referring_index], True
                ),
                nlp.meta["name"],
            )
            self.assertEqual(
                expected_reflexive_truth,
                rules_analyzer.is_potential_reflexive_pair(
                    referred_mention, doc[referring_index]
                ),
                nlp.meta["name"],
            )
            self.assertEqual(
                is_reflexive_anaphor_truth,
                rules_analyzer.is_reflexive_anaphor(doc[referring_index]),
                nlp.meta["name"],
            )

        self.all_nlps(func)

    def test_reflexive_in_wrong_situation_different_sentence(self):
        self.compare_potential_reflexive_pair(
            "I saw the person. The other person saw himself", 3, False, 9, 1, False, 1
        )

    def test_reflexive_in_wrong_situation_different_sentence_control(self):
        self.compare_potential_reflexive_pair(
            "I saw the person. The other person saw him", 3, False, 9, 2, False, 0
        )

    def test_reflexive_in_wrong_situation_same_sentence_1(self):
        self.compare_potential_reflexive_pair(
            "I saw the person while the other person saw himself",
            3,
            False,
            9,
            1,
            False,
            1,
        )

    def test_reflexive_in_wrong_situation_same_sentence_control(self):
        self.compare_potential_reflexive_pair(
            "I saw the person while the other person saw him", 3, False, 9, 2, False, 0
        )

    def test_non_reflexive_in_wrong_situation_same_sentence(self):
        self.compare_potential_reflexive_pair(
            "The man saw him", 1, False, 3, 0, True, 0
        )

    def test_non_reflexive_in_wrong_situation_same_sentence_control(self):
        self.compare_potential_reflexive_pair(
            "The man saw himself", 1, False, 3, 2, True, 1
        )

    def test_non_reflexive_in_same_sentence_with_verb_conjunction(self):
        self.compare_potential_reflexive_pair(
            "The man saw everything and heard himself", 1, False, 6, 2, True, 1
        )

    def test_reflexive_in_right_situation_xcomp(self):
        self.compare_potential_reflexive_pair(
            "The man wanted to see himself", 1, False, 5, 2, True, 1
        )

    def test_reflexive_in_right_situation_pcomp(self):
        self.compare_potential_reflexive_pair(
            "The man thought about seeing himself", 1, False, 5, 2, True, 1
        )

    def test_reflexive_in_right_situation_within_subordinate_clause(self):
        self.compare_potential_reflexive_pair(
            "He saw the man see himself",
            3,
            False,
            5,
            2,
            True,
            True,
            excluded_nlps=["core_web_sm", "core_web_md"],
        )

    def test_reflexive_in_right_situation_within_subordinate_clause_control(self):
        self.compare_potential_reflexive_pair(
            "He saw the man see himself",
            0,
            False,
            5,
            1,
            False,
            True,
            excluded_nlps=["core_web_sm", "core_web_md"],
        )

    def test_reflexive_with_conjuction(self):
        self.compare_potential_reflexive_pair(
            "The house and the car outdid themselves", 1, True, 6, 2, True, 1
        )

    def test_reflexive_with_conjuction_control(self):
        self.compare_potential_reflexive_pair(
            "The house and the car outdid themselves", 1, False, 6, 0, True, 1
        )

    def test_reflexive_with_passive(self):
        self.compare_potential_reflexive_pair(
            "The house was outdone by itself", 1, False, 5, 2, True, 1
        )

    def test_reflexive_with_passive_and_conjunction(self):
        self.compare_potential_reflexive_pair(
            "The house and the car were outdone by themselves", 1, False, 8, 0, True, 1
        )

    def test_reflexive_with_object_antecedent(self):
        self.compare_potential_reflexive_pair(
            "He mixed the chemical with itself", 3, False, 5, 2, True, 1
        )
        self.compare_potential_reflexive_pair(
            "He mixed the chemical with itself", 0, False, 5, 0, True, 1
        )

    def test_reflexive_with_object_antecedent_and_coordination(self):
        self.compare_potential_reflexive_pair(
            "He mixed the chemical and the salt with themselves", 3, True, 8, 2, True, 1
        )
        self.compare_potential_reflexive_pair(
            "He mixed the chemical and the salt with themselves",
            0,
            False,
            8,
            0,
            True,
            1,
        )

    def test_reflexive_with_verb_coordination_one_subject(self):
        self.compare_potential_reflexive_pair(
            "He saw it and congratulated himself", 0, False, 5, 2, True, 1
        )

    def test_reflexive_with_verb_coordination_two_subjects(self):
        self.compare_potential_reflexive_pair(
            "He saw it, and his boss congratulated himself",
            0,
            False,
            8,
            1,
            False,
            1,
            excluded_nlps=["core_web_md"],
        )

    def test_reflexive_with_to(self):
        self.compare_potential_reflexive_pair(
            "They wanted the boy to know himself", 3, False, 6, 2, True, 1
        )

    def test_non_reflexive_in_wrong_situation_subordinate_clause(self):
        self.compare_potential_reflexive_pair(
            "Although he saw him, he was happy.", 1, False, 3, 0, True, 0
        )

    def test_reflexive_completely_within_noun_phrase_1(self):
        self.compare_potential_reflexive_pair(
            "My friend's opinion of himself was exaggerated", 1, False, 5, 2, True, 1
        )

    def test_reflexive_completely_within_noun_phrase_2(self):
        self.compare_potential_reflexive_pair(
            "The opinion of my friend about himself was exaggerated",
            4,
            False,
            6,
            2,
            True,
            1,
        )

    def test_reflexive_completely_within_noun_phrase_1_control(self):
        self.compare_potential_reflexive_pair(
            "My friend's opinion of him was exaggerated", 1, False, 5, 0, True, 0
        )

    def test_reflexive_completely_within_noun_phrase_2_control(self):
        self.compare_potential_reflexive_pair(
            "The opinion of my friend about him was exaggerated",
            4,
            False,
            6,
            0,
            True,
            0,
        )

    def test_reflexive_double_coordination_without_preposition(self):
        self.compare_potential_reflexive_pair(
            "Peter and Jane saw him and her.", 0, False, 4, 0, True, 0
        )
        self.compare_potential_reflexive_pair(
            "Peter and Jane saw him and her.", 2, False, 6, 0, True, 0
        )

    def test_reflexive_double_coordination_with_preposition(self):
        self.compare_potential_reflexive_pair(
            "Peter and Jane spoke to him and her.", 0, False, 5, 0, True, 0
        )
        self.compare_potential_reflexive_pair(
            "Peter and Jane spoke to him and her.", 2, False, 7, 0, True, 0
        )

    def test_reflexive_pronoun_precedes_referent(self):
        self.compare_potential_reflexive_pair(
            "Peter saw himself and Peter came in.", 4, False, 2, 1, False, 1
        )

    def test_reflexive_noun_phrase(self):
        self.compare_potential_reflexive_pair(
            "He had no idea whether to see him.", 0, False, 7, 0, True, 0
        )

    def test_reflexive_relative_clause_subject(self):
        self.compare_potential_reflexive_pair(
            "The man who saw him came home.", 1, False, 4, 0, True, 0
        )

    def test_reflexive_relative_clause_object_1(self):
        self.compare_potential_reflexive_pair(
            "The man he saw came home.", 1, False, 2, 0, True, 0
        )

    def test_reflexive_relative_clause_object_2(self):
        self.compare_potential_reflexive_pair(
            "The man that he saw came home.", 1, False, 3, 0, True, 0
        )

    def test_reflexive_relative_clause_subject_with_conjunction(self):
        self.compare_potential_reflexive_pair(
            "The man and the woman who saw them came home.", 1, True, 7, 0, True, 0
        )

    def test_reflexive_relative_clause_object_with_conjunction(self):
        self.compare_potential_reflexive_pair(
            "The man and the woman they saw came home.", 1, True, 5, 0, True, 0
        )

    def compare_potential_cataphoric_pair(
        self,
        doc_text,
        referred_index,
        include_dependent_siblings,
        referring_index,
        expected_truth,
        *,
        excluded_nlps=[]
    ):
        def func(nlp):

            if nlp.meta["name"] in excluded_nlps:
                return
            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory.get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            assert rules_analyzer.is_independent_noun(
                doc[referred_index]
            ) or rules_analyzer.is_potential_anaphor(doc[referred_index])
            assert rules_analyzer.is_potential_anaphor(doc[referring_index])
            assert referred_index > referring_index
            referred_mention = Mention(doc[referred_index], include_dependent_siblings)
            self.assertEqual(
                expected_truth,
                rules_analyzer.is_potential_cataphoric_pair(
                    referred_mention, doc[referring_index]
                )
                and rules_analyzer.is_potential_anaphoric_pair(
                    referred_mention, doc[referring_index], True
                )
                > 0,
                nlp.meta["name"],
            )

        self.all_nlps(func)

    def test_cataphora_simple_case(self):
        self.compare_potential_cataphoric_pair(
            "Although he was enjoying himself, Tom went home.",
            6,
            False,
            1,
            True,
            excluded_nlps=["core_web_trf"],
        )

    def test_cataphora_with_conjunction(self):
        self.compare_potential_cataphoric_pair(
            "Although they outdid themselves, the house and the car were splendid.",
            6,
            True,
            1,
            True,
        )

    def test_cataphora_with_conjunction_control(self):
        self.compare_potential_cataphoric_pair(
            "Although they outdid themselves, the house and the car were splendid.",
            6,
            False,
            1,
            False,
        )

    def test_cataphora_tokens_deeper_in_tree_1(self):
        self.compare_potential_cataphoric_pair(
            "Although all his fans talked about wanting to see him, the press had long since stopped bothering James.",
            18,
            False,
            2,
            True,
        )

    def test_cataphora_tokens_deeper_in_tree_2(self):
        self.compare_potential_cataphoric_pair(
            "Although all his fans talked about wanting to see him, the press had long since stopped bothering James.",
            18,
            False,
            9,
            True,
        )

    def test_cataphora_tokens_deeper_in_tree_conjunction_1(self):
        self.compare_potential_cataphoric_pair(
            "Although all their fans talked about wanting to see them, the press had long since stopped bothering James and Jane.",
            18,
            True,
            2,
            True,
        )

    def test_cataphora_tokens_deeper_in_tree_conjunction_2(self):
        self.compare_potential_cataphoric_pair(
            "Although all their fans talked about wanting to see them, the press had long since stopped bothering James and Jane.",
            18,
            True,
            9,
            True,
        )

    def test_cataphora_tokens_deeper_in_tree_conjunction_control(self):
        self.compare_potential_cataphoric_pair(
            "Although all their fans talked about wanting to see them, the press had long since stopped bothering James and Jane.",
            18,
            False,
            9,
            False,
        )

    def test_cataphora_wrong_structure_1(self):
        self.compare_potential_cataphoric_pair(
            "The press had long since interviewed him although all his fans talked about wanting to see James",
            16,
            False,
            6,
            False,
        )

    def test_cataphora_wrong_structure_2(self):
        self.compare_potential_cataphoric_pair(
            "All his fans talked about wanting to see James", 8, False, 1, False
        )

    def test_cataphora_wrong_structure_conjunction(self):
        self.compare_potential_cataphoric_pair(
            "The press had long since interviewed them although all their fans talked about wanting to see James and Jane",
            16,
            True,
            6,
            False,
        )

    def test_cataphora_conjunction_at_verb_level(self):
        self.compare_potential_cataphoric_pair(
            "Although he was available, Peter had time and John came home",
            9,
            False,
            1,
            False,
        )

    def test_cataphora_referred_is_pronoun(self):
        self.compare_potential_cataphoric_pair(
            "Although he was available, he came home", 5, False, 1, False
        )

    def test_cataphora_referred_is_pronoun_control(self):
        self.compare_potential_cataphoric_pair(
            "Although he was available, James came home", 5, False, 1, True
        )

    def test_cataphora_not_advcl(self):
        self.compare_potential_cataphoric_pair(
            "He was available; James came home", 4, False, 0, False
        )

    def compare_potential_referreds(
        self, doc_text, index, expected_potential_referreds, *, excluded_nlps=[]
    ):
        def func(nlp):

            if nlp.meta["name"] in excluded_nlps:
                return
            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory.get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            if expected_potential_referreds is None:
                self.assertFalse(
                    hasattr(doc[index]._.coref_chains, "temp_potential_referreds")
                )
            else:
                potential_referreds = [
                    referred.pretty_representation
                    for referred in doc[index]._.coref_chains.temp_potential_referreds
                ]
                self.assertEqual(
                    expected_potential_referreds, potential_referreds, nlp.meta["name"]
                )

        self.all_nlps(func)

    def test_potential_referreds_same_sentence(self):
        self.compare_potential_referreds("Richard came in and he sat down.", 0, None)
        self.compare_potential_referreds("Richard came in and he sat down.", 1, None)
        self.compare_potential_referreds(
            "Richard came in and he sat down.", 4, ["Richard(0)"]
        )

    def test_potential_referreds_conjunction_same_sentence(self):
        self.compare_potential_referreds(
            "Peter and Jane came in and they sat down.", 6, ["[Peter(0); Jane(2)]"]
        )

    def test_potential_referreds_maximum_sentence_referential_distance(self):
        self.compare_potential_referreds(
            "Richard came in. A man. A man. A man. A man. He sat down.",
            16,
            ["Richard(0)", "man(5)", "man(8)", "man(11)", "man(14)"],
        )

    def test_potential_referreds_over_maximum_sentence_referential_distance(self):
        self.compare_potential_referreds(
            "Richard came in. A man. A man. A man. A man. A man. He sat down.",
            19,
            ["man(5)", "man(8)", "man(11)", "man(14)", "man(17)"],
        )

    def test_potential_referreds_first_and_last_token(self):
        self.compare_potential_referreds("He came in and someone saw him", 0, None)
        self.compare_potential_referreds("He came in and someone saw him", 6, ["He(0)"])

    def test_potential_referreds_last_token(self):
        self.compare_potential_referreds(
            "Richard came in and someone saw him", 6, ["Richard(0)"]
        )

    def test_potential_referreds_cataphora_simple(self):
        self.compare_potential_referreds(
            "Although he came in, someone saw Richard", 1, ["someone(5)", "Richard(7)"]
        )

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_potential_referreds_cataphora_conjunction(self):
        self.compare_potential_referreds(
            "Although they came in, someone saw Peter and Jane",
            1,
            ["someone(5)", "[Peter(7); Jane(9)]"],
        )

    def compare_potentially_indefinite(
        self, doc_text, index, expected_truth, *, excluded_nlps=[]
    ):
        def func(nlp):

            if nlp.meta["name"] in excluded_nlps:
                return
            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory.get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            self.assertEqual(
                expected_truth,
                rules_analyzer.is_potentially_indefinite(doc[index]),
                nlp.meta["name"],
            )

        self.all_nlps(func)

    def test_potentially_indefinite_proper_noun(self):
        self.compare_potentially_indefinite("I spoke to Peter", 3, False)

    def test_potentially_indefinite_definite_noun(self):
        self.compare_potentially_indefinite("I spoke to the man", 4, False)

    def test_potentially_indefinite_indefinite_noun(self):
        self.compare_potentially_indefinite("I spoke to a man", 4, True)

    def test_potentially_indefinite_common_noun_conjunction_first_member(self):
        self.compare_potentially_indefinite("I spoke to a man and a woman", 4, True)

    def test_potentially_indefinite_common_noun_conjunction_second_member(self):
        self.compare_potentially_indefinite("I spoke to a man and a woman", 7, True)

    def test_potentially_indefinite_common_noun_conjunction_first_member_control(self):
        self.compare_potentially_indefinite(
            "I spoke to the man and the woman", 4, False
        )

    def test_potentially_indefinite_common_noun_conjunction_second_member_control(self):
        self.compare_potentially_indefinite(
            "I spoke to the man and the woman", 7, False
        )

    def compare_potentially_definite(
        self, doc_text, index, expected_truth, *, excluded_nlps=[]
    ):
        def func(nlp):

            if nlp.meta["name"] in excluded_nlps:
                return
            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory.get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            self.assertEqual(
                expected_truth,
                rules_analyzer.is_potentially_definite(doc[index]),
                nlp.meta["name"],
            )

        self.all_nlps(func)

    def test_potentially_definite_proper_noun(self):
        self.compare_potentially_definite("I spoke to Peter", 3, False)

    def test_potentially_definite_definite_noun(self):
        self.compare_potentially_definite("I spoke to the man.", 4, True)

    def test_potentially_definite_indefinite_noun(self):
        self.compare_potentially_definite("I spoke to a man", 4, False)

    def test_potentially_definite_common_noun_conjunction_first_member(self):
        self.compare_potentially_definite("I spoke to the man and a woman", 4, True)

    def test_potentially_definite_common_noun_conjunction_second_member(self):
        self.compare_potentially_definite("I spoke to a man and the woman", 7, True)

    def test_potentially_definite_common_noun_conjunction_first_member_control(self):
        self.compare_potentially_definite("I spoke to the man and a woman", 7, False)

    def test_potentially_definite_common_noun_conjunction_second_member_control(self):
        self.compare_potentially_definite("I spoke to a man and the woman", 4, False)
