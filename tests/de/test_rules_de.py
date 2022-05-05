import unittest
from coreferee.rules import RulesAnalyzerFactory
from coreferee.test_utils import get_nlps
from coreferee.data_model import Mention

nlps = get_nlps('de')
train_version_mismatch = False
for nlp in nlps:
    if not nlp.meta["matches_train_version"]:
        train_version_mismatch = True
train_version_mismatch_message = "Loaded model version does not match train model version"

class GermanRulesTest(unittest.TestCase):
    def setUp(self):

        self.nlps = get_nlps("de")
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
    ):
        def func(nlp):

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
            "Richard ging heim", 0, "[]", None, False
        )

    def test_get_dependent_sibling_info_two_member_conjunction_phrase_and(self):
        self.compare_get_dependent_sibling_info(
            "Richard und Christine gingen heim", 0, "[Christine]", None, False
        )

    def test_get_governing_sibling_info_two_member_conjunction_phrase_and(self):
        self.compare_get_dependent_sibling_info(
            "Richard und Christine gingen heim", 2, "[]", 0, False
        )

    def test_get_dependent_sibling_info_two_member_conjunction_phrase_or(self):
        self.compare_get_dependent_sibling_info(
            "Richard oder Christine ging heim", 0, "[Christine]", None, True
        )

    def test_get_dependent_sibling_info_apposition_control(self):
        self.compare_get_dependent_sibling_info(
            "Richard, der Entwickler, ging heim", 0, "[]", None, False
        )

    def test_get_governing_sibling_info_apposition_control(self):
        self.compare_get_dependent_sibling_info(
            "Richard, der Entwickler, ging heim", 3, "[]", None, False
        )

    def test_get_dependent_sibling_info_three_member_conjunction_phrase_with_comma_and(
        self,
    ):
        self.compare_get_dependent_sibling_info(
            "Carol, Richard und Ralf hatten ein Meeting",
            0,
            "[Richard, Ralf]",
            None,
            False,
        )

    def test_get_dependent_sibling_info_three_member_conjunction_phrase_with_comma_or(
        self,
    ):
        self.compare_get_dependent_sibling_info(
            "Carol, Richard oder Ralf hatte ein Meeting",
            0,
            "[Richard, Ralf]",
            None,
            True,
        )

    def test_get_dependent_sibling_info_three_member_conjunction_phrase_with_and(self):
        self.compare_get_dependent_sibling_info(
            "Es gab ein Meeting mit Carol und Ralf und Richard",
            5,
            "[Ralf, Richard]",
            None,
            False,
        )

    def test_get_dependent_sibling_info_three_member_conjunction_phrase_with_or(self):
        self.compare_get_dependent_sibling_info(
            "Ein Meeting mit Carol oder Ralf oder Richard fand statt",
            3,
            "[Ralf, Richard]",
            None,
            True,
        )

    def test_get_dependent_sibling_info_three_member_conjunction_phrase_with_and_and_or(
        self,
    ):
        self.compare_get_dependent_sibling_info(
            "Es gab ein Meeting mit Carol oder Ralf und Richard",
            5,
            "[Ralf, Richard]",
            None,
            True,
        )

    def test_get_dependent_sibling_info_conjunction_itself(self):
        self.compare_get_dependent_sibling_info(
            "Es gab ein Meeting mit Carol und Ralf und Richard", 6, "[]", None, False
        )

    def test_get_dependent_sibling_info_dependent_sibling(self):
        self.compare_get_dependent_sibling_info(
            "Es gab ein Meeting mit Carol und Ralf und Richard", 7, "[]", 5, False
        )

    def test_hyphenated_first_member(self):
        self.compare_get_dependent_sibling_info(
            "Ein Kindes- und Jugendzentrum", 1, "[]", None, False
        )

    def test_hyphenated_second_member(self):
        self.compare_get_dependent_sibling_info(
            "Ein Kindes- und Jugendzentrum", 3, "[]", None, False
        )

    def compare_independent_noun(
        self, doc_text, expected_per_indexes, *, excluded_nlps=[]
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
            self.assertEqual(expected_per_indexes, per_indexes, nlp.meta["name"])

        self.all_nlps(func)

    def test_independent_noun_simple(self):
        self.compare_independent_noun("Sie haben sich die großen Löwen angeschaut", [5])

    def test_independent_noun_conjunction(self):
        self.compare_independent_noun(
            "Sie haben sich die großen Löwen, die Tiger und die Elefanten angeschaut",
            [5, 8, 11],
        )

    def test_substituting_indefinite_pronoun(self):
        self.compare_independent_noun("Einer der Jungen ist heimgekommen", [0, 2])

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_pronoun_noun(self):
        self.compare_independent_noun(
            "Diejenigen der Jungen, die heimgekommen sind, waren müde",
            [2],
            excluded_nlps=["core_news_md", "core_news_sm"],
        )

    def test_blacklisted(self):
        self.compare_independent_noun(
            "Meines Erachtens ist der Junge zum Beispiel immer müde", [4]
        )

    def test_blacklisted_control(self):
        self.compare_independent_noun(
            "Meines Erachtens ist das ein gutes Beispiel.", [6]
        )

    def test_proper_noun_component(self):
        self.compare_independent_noun(
            "Sie fuhren in die Walt Disney World.", [6], excluded_nlps=["core_news_sm"]
        )

    def test_punctuation(self):
        self.compare_independent_noun("[ Vogel ]", [1], excluded_nlps=["core_news_sm"])

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
        self.compare_potential_anaphor(
            "Sie ist rausgegangen, um ihr Auto anzusehen", [0, 5]
        )

    def test_first_and_second_person_pronouns(self):
        self.compare_potential_anaphor("Ich weiß, dass du ihn kennst", [5])

    def test_colloquial_pronouns(self):
        self.compare_potential_anaphor(
            "Die ist rausgegangen, um den zu treffen. Die Frau war da", [0, 5]
        )

    def test_das_not_colloquial_pronoun(self):
        self.compare_potential_anaphor("Ich sah das Haus. Das war gut.", [])

    def test_formal_pronouns(self):
        self.compare_potential_anaphor(
            "Diese ist rausgegangen, um jenen zu treffen. Diese Frau war da aus jenem Grund.",
            [0, 5],
            excluded_nlps=["core_news_sm"],
        )

    def test_prepositions_1(self):
        self.compare_potential_anaphor("Sie aß es damit.", [0, 2, 3], excluded_nlps=['core_news_md'])

    def test_prepositions_2(self):
        self.compare_potential_anaphor("Und damit aß sie es.", [1, 3, 4])

    def test_prepositions_control_1(self):
        self.compare_potential_anaphor("Das ist außerdem interessant.", [])

    def test_prepositions_control_2(self):
        self.compare_potential_anaphor("Sie aß es daher.", [0, 2])

    def test_prepositions_control_with_verb_phrase(self):
        self.compare_potential_anaphor(
            "Sie aß es damit, dass sie ein Messer benutzte.", [0, 2, 6], excluded_nlps=['core_news_md']
        )

    def test_initial_prepositions(self):
        self.compare_potential_anaphor("Damit hast du es geschafft.", [3])

    def test_pleonastic_es(self):
        self.compare_potential_anaphor(
            "Es scheint nicht sehr wahrscheinlich, dass er es weiß.", [7, 8]
        )

    def test_avalent_es_1(self):
        self.compare_potential_anaphor("Es donnert.", [])

    def test_avalent_es_2(self):
        self.compare_potential_anaphor("Es hat gedonnert.", [])

    def test_avalent_es_3(self):
        self.compare_potential_anaphor("Es soll gedonnert haben.", [])

    def test_avalent_es_4(self):
        self.compare_potential_anaphor("Es hörte auf, zu donnern.", [])

    def test_avalent_es_5(self):
        self.compare_potential_anaphor("Es soll aufgehört haben, zu donnern.", [])

    def test_possessive_pronouns(self):
        self.compare_potential_anaphor(
            "Mein Haus, dein Haus, sein Haus, ihr Haus.", [6, 9]
        )

    def test_pleonastic_es_object_position_1(self):
        self.compare_potential_anaphor("Wir haben es angeregt, dass er es tut.", [6, 7])

    def test_pleonastic_es_object_position_2(self):
        self.compare_potential_anaphor("Wir haben es angeregt, es zu tun.", [5])

    def test_pleonastic_es_object_aux_position_1(self):
        self.compare_potential_anaphor(
            "Wir werden es anregen können, dass er es tut.", [7, 8]
        )

    def test_pleonastic_es_object_aux_position_2(self):
        self.compare_potential_anaphor("Wir hätten es anregen sollen, es zu tun.", [6])

    def test_pleonastic_darauf_1(self):
        self.compare_potential_anaphor(
            "Das Ergebnis kam darauf an, dass er es tut.", [7, 8]
        )

    def test_pleonastic_darauf_2(self):
        self.compare_potential_anaphor("Das Ergebnis kam darauf an, es zu tun.", [6])

    def test_pleonastic_darauf_aux_1(self):
        self.compare_potential_anaphor(
            "Es wäre darauf angekommen, dass er es tut.", [0, 6, 7]
        )

    def test_pleonastic_darauf_aux_2(self):
        self.compare_potential_anaphor("Es ist darauf angekommen, es zu tun.", [0, 5])

    def test_pleonastic_dessen_object_positions(self):
        self.compare_potential_anaphor(
            "Das war die Idee dessen, was wir taten.",
            [],
            excluded_nlps=["core_news_sm"],
        )

    def test_you_Sie_mid_sentence(self):
        self.compare_potential_anaphor("Was möchten Sie?.", [])

    def test_you_Sie_mid_sentence_control(self):
        self.compare_potential_anaphor("Was möchten sie?", [2])

    def test_possible_you_Sie_beginning_of_sentence_one_sentence(self):
        self.compare_potential_anaphor("Sie wollten nach Hause.", [0])

    def test_possible_you_Sie_beginning_of_sentence_two_sentences(self):
        self.compare_potential_anaphor("Der Tag war kalt. Sie wollten nach Hause.", [5])

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
        self.compare_potentially_indefinite("Ich sprach mit Peter", 3, False)

    def test_potentially_indefinite_definite_noun(self):
        self.compare_potentially_indefinite("Ich sprach mit dem Mann", 4, False)

    def test_potentially_indefinite_indefinite_noun(self):
        self.compare_potentially_indefinite("Ich sprach mit irgendeinem Mann", 4, True)

    def test_potentially_indefinite_common_noun_conjunction_first_member(self):
        self.compare_potentially_indefinite(
            "Ich sprach mit einem Mann und einer Frau", 4, True
        )

    def test_potentially_indefinite_common_noun_conjunction_second_member(self):
        self.compare_potentially_indefinite(
            "Ich sprach mit einem Mann und einer Frau", 7, True
        )

    def test_potentially_indefinite_common_noun_conjunction_first_member_control(self):
        self.compare_potentially_indefinite(
            "Ich sprach mit dem Mann und der Frau", 4, False
        )

    def test_potentially_indefinite_common_noun_conjunction_second_member_control(self):
        self.compare_potentially_indefinite(
            "Ich sprach mit dem Mann und der Frau", 7, False
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
        self.compare_potentially_definite("Ich sprach mit Peter", 3, False)

    def test_potentially_definite_definite_noun(self):
        self.compare_potentially_definite("Ich sprach mit jenem Mann.", 4, True)

    def test_potentially_definite_indefinite_noun(self):
        self.compare_potentially_definite("Ich sprach mit irgendeinem Mann", 4, False)

    def test_potentially_definite_common_noun_conjunction_first_member(self):
        self.compare_potentially_definite(
            "Ich sprach mit diesem Mann und einer Frau", 4, True
        )

    def test_potentially_definite_common_noun_conjunction_second_member(self):
        self.compare_potentially_definite(
            "Ich sprach mit einem Mann und jener Frau", 7, True
        )

    def test_potentially_definite_common_noun_conjunction_first_member_control(self):
        self.compare_potentially_definite(
            "Ich sprach mit dem Mann und einer Frau", 7, False
        )

    def test_potentially_definite_common_noun_conjunction_second_member_control(self):
        self.compare_potentially_definite(
            "Ich sprach mit einem Mann und der Frau", 4, False
        )

    def compare_potential_pair(
        self,
        doc_text,
        referred_index,
        include_dependent_siblings,
        referring_index,
        expected_truth,
        *,
        excluded_nlps=[],
        directly=True
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
                    referred_mention, doc[referring_index], directly
                ),
                nlp.meta["name"],
            )

        self.all_nlps(func)

    def test_potential_pair_trivial_masc(self):
        self.compare_potential_pair("Ich sah einen Mann. Er lief", 3, False, 5, 2)

    def test_potential_pair_trivial_masc_control_1(self):
        self.compare_potential_pair("Ich sah einen Mann. Sie lief", 3, False, 5, 0)

    def test_potential_pair_trivial_masc_control_2(self):
        self.compare_potential_pair(
            "Ich sah einen Mann. Dann sprach sie und weinte", 3, False, 7, 0
        )

    def test_potential_pair_trivial_masc_control_3(self):
        self.compare_potential_pair("Ich sah einen Mann. Sie liefen", 3, False, 5, 0)

    def test_potential_pair_trivial_masc_possessive(self):
        self.compare_potential_pair(
            "Ich sah einen Mann. Sein Hund lief", 3, False, 5, 2
        )

    def test_potential_pair_trivial_masc_possessive_control_1(self):
        self.compare_potential_pair("Ich sah einen Mann. Ihr Hund lief", 3, False, 5, 0)

    def test_potential_pair_trivial_fem(self):
        self.compare_potential_pair("Ich sah eine Frau. Die lief", 3, False, 5, 2)

    def test_potential_pair_trivial_fem_control_1(self):
        self.compare_potential_pair("Ich sah eine Frau. Der lief", 3, False, 5, 0)

    def test_potential_pair_trivial_fem_control_2(self):
        self.compare_potential_pair(
            "Ich sah eine Frau. Es betrachtete den Himmel", 3, False, 5, 0
        )

    def test_potential_pair_trivial_fem_control_3(self):
        self.compare_potential_pair("Ich sah eine Frau. Die liefen", 3, False, 5, 0)

    def test_potential_pair_trivial_fem_possessive(self):
        self.compare_potential_pair("Ich sah eine Frau. Ihr Hund lief", 3, False, 5, 2)

    def test_potential_pair_trivial_fem_possessive_control(self):
        self.compare_potential_pair("Ich sah eine Frau. Sein Hund lief", 3, False, 5, 0)

    def test_potential_pair_trivial_neut(self):
        self.compare_potential_pair("Ich sah ein Haus. Dieses stand", 3, False, 5, 2)

    def test_potential_pair_trivial_neut_control_1(self):
        self.compare_potential_pair("Ich sah ein Haus. Dieser stand", 3, False, 5, 0)

    def test_potential_pair_trivial_neut_control_2(self):
        self.compare_potential_pair("Ich sah ein Haus. Diese stand", 3, False, 5, 0)

    def test_potential_pair_trivial_neut_control_3(self):
        self.compare_potential_pair("Ich sah ein Haus. Diese standen", 3, False, 5, 0)

    def test_potential_pair_trivial_neut_possessive(self):
        self.compare_potential_pair("Ich sah ein Haus. Sein Dach stand", 3, False, 5, 2)

    def test_potential_pair_trivial_neut_possessive_control(self):
        self.compare_potential_pair("Ich sah ein Haus. Ihr Dach stand", 3, False, 5, 0)

    def test_potential_pair_trivial_plur_single_element(self):
        self.compare_potential_pair(
            "Ich sah einige Frauen. Sie standen", 3, False, 5, 2
        )

    def test_potential_pair_trivial_plur_single_element_control_1(self):
        self.compare_potential_pair("Ich sah einige Frauen. Er stand", 3, False, 5, 0)

    def test_potential_pair_trivial_plur_single_element_control_2(self):
        self.compare_potential_pair("Ich sah einige Frauen. Sie stand", 3, False, 5, 0)

    def test_potential_pair_trivial_plur_single_element_control_3(self):
        self.compare_potential_pair(
            "Ich sah einige Frauen. Dieses stand", 3, False, 5, 0
        )

    def test_potential_pair_trivial_plur_single_element_possessive(self):
        self.compare_potential_pair(
            "Ich sah einige Frauen. Ihr Hund stand", 3, False, 5, 2
        )

    def test_potential_pair_trivial_plur_single_element_possessive_control(self):
        self.compare_potential_pair(
            "Ich sah einige Frauen. Sein Hund stand", 3, False, 5, 0
        )

    def test_potential_pair_trivial_plur_coordination(self):
        self.compare_potential_pair(
            "Ich sah einen Mann und eine Frau. Sie standen", 3, True, 8, 2
        )

    def test_potential_pair_trivial_plur_coordination_control_1(self):
        self.compare_potential_pair(
            "Ich sah einen Mann und eine Frau. Er stand", 3, True, 8, 0
        )

    def test_potential_pair_trivial_plur_coordination_control_2(self):
        self.compare_potential_pair(
            "Ich sah einen Mann und eine Frau. Sie stand", 3, True, 8, 0
        )

    def test_potential_pair_trivial_plur_coordination_control_3(self):
        self.compare_potential_pair(
            "Ich sah einen Mann und eine Frau. Dieses stand", 3, True, 8, 0
        )

    def test_potential_pair_trivial_plur_coordination_possessive(self):
        self.compare_potential_pair(
            "Ich sah einen Mann und eine Frau. Ihr Hund stand",
            3,
            True,
            8,
            2,
            excluded_nlps=["core_news_sm"],
        )

    def test_potential_pair_trivial_plur_coordination_possessive_control(self):
        self.compare_potential_pair(
            "Ich sah einen Mann und eine Frau. Sein Hund stand",
            3,
            True,
            8,
            0,
            excluded_nlps=["core_news_sm"],
        )

    def test_potential_pair_trivial_plur_coordination_elements_plural_1(self):
        self.compare_potential_pair(
            "Ich sah einige Männer und einige Frauen. Sie standen", 3, False, 8, 0
        )

    def test_potential_pair_trivial_plur_coordination_elements_plural_2(self):
        self.compare_potential_pair(
            "Ich sah einige Männer und einige Frauen. Sie standen", 6, False, 8, 0
        )

    def test_potential_pair_trivial_sing_coordination_first_element(self):
        self.compare_potential_pair(
            "Ich sah einen Mann und eine Frau. Er stand", 3, False, 8, 2
        )

    def test_potential_pair_trivial_sing_coordination_first_element_control_1(self):
        self.compare_potential_pair(
            "Ich sah einen Mann und eine Frau. Sie stand", 3, False, 8, 0
        )

    def test_potential_pair_trivial_sing_coordination_first_element_control_2(self):
        self.compare_potential_pair(
            "Ich sah einen Mann und eine Frau. Dieses stand", 3, False, 8, 0
        )

    def test_potential_pair_trivial_sing_coordination_first_element_control_3(self):
        self.compare_potential_pair(
            "Ich sah einen Mann und eine Frau. Sie standen", 3, False, 8, 0
        )

    def test_potential_pair_trivial_sing_coordination_first_element_possessive(self):
        self.compare_potential_pair(
            "Ich sah einen Mann und eine Frau. Sein Hund stand", 3, False, 8, 2
        )

    def test_potential_pair_trivial_sing_coordination_first_element_possessive_control(
        self,
    ):
        self.compare_potential_pair(
            "Ich sah einen Mann und eine Frau. Ihr Hund stand", 3, False, 8, 0
        )

    def test_potential_pair_trivial_sing_coordination_second_element(self):
        self.compare_potential_pair(
            "Ich sah einen Mann und eine Frau. Sie stand", 6, False, 8, 2
        )

    def test_potential_pair_trivial_sing_coordination_second_element_control_1(self):
        self.compare_potential_pair(
            "Ich sah einen Mann und eine Frau. Er stand", 6, False, 8, 0
        )

    def test_potential_pair_trivial_sing_coordination_second_element_control_2(self):
        self.compare_potential_pair(
            "Ich sah einen Mann und eine Frau. Dieses stand", 6, False, 8, 0
        )

    def test_potential_pair_trivial_sing_coordination_second_element_control_3(self):
        self.compare_potential_pair(
            "Ich sah einen Mann und eine Frau. Sie standen", 6, False, 8, 0
        )

    def test_potential_pair_trivial_sing_coordination_second_element_possessive(self):
        self.compare_potential_pair(
            "Ich sah einen Mann und eine Frau. Ihr Hund stand", 6, False, 8, 2
        )

    def test_potential_pair_trivial_sing_coordination_second_element_possessive(self):
        self.compare_potential_pair(
            "Ich sah einen Mann und eine Frau. Sein Hund stand", 6, False, 8, 0
        )

    def test_potential_pair_person_neut_1(self):
        self.compare_potential_pair("Ich sah ein Kind. Er stand", 3, False, 5, 2)

    def test_potential_pair_person_neut_2(self):
        self.compare_potential_pair("Ich sah ein Kind. Sie stand", 3, False, 5, 2)

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_potential_pair_person_neut_3(self):
        self.compare_potential_pair(
            "Ich sah ein Kind. Dann lächelte es und weinte", 3, False, 7, 2
        )

    def test_potential_pair_person_neut_control(self):
        self.compare_potential_pair("Ich sah ein Kind. diese standen", 3, False, 5, 0)

    def test_potential_pair_male_neut_1(self):
        self.compare_potential_pair("Ich sah ein Mannsbild. Er stand", 3, False, 5, 2)

    def test_potential_pair_male_neut_2(self):
        self.compare_potential_pair(
            "Ich sah ein Mannsbild. Dieses stand", 3, False, 5, 2
        )

    def test_potential_pair_male_neut_control_1(self):
        self.compare_potential_pair("Ich sah ein Mannsbild. Sie stand", 3, False, 5, 0)

    def test_potential_pair_male_neut_control_2(self):
        self.compare_potential_pair(
            "Ich sah ein Mannsbild. Sie standen", 3, False, 5, 0
        )

    def test_potential_pair_fem_neut_1(self):
        self.compare_potential_pair("Ich sah ein Mädchen. Sie stand", 3, False, 5, 2)

    def test_potential_pair_fem_neut_2(self):
        self.compare_potential_pair("Ich sah ein Mädchen. Dieses stand", 3, False, 5, 2)

    def test_potential_pair_fem_neut_control_1(self):
        self.compare_potential_pair("Ich sah ein Mädchen. Er stand", 3, False, 5, 0)

    def test_potential_pair_fem_neut_control_2(self):
        self.compare_potential_pair("Ich sah ein Mädchen. Sie standen", 3, False, 5, 0)

    def test_potential_pair_diminutive_1(self):
        self.compare_potential_pair(
            "Ich sah ein Schlüsselein. Er stand", 3, False, 5, 2
        )

    def test_potential_pair_diminutive_2(self):
        self.compare_potential_pair(
            "Ich sah ein Schlüsselein. Sie stand", 3, False, 5, 2
        )

    def test_potential_pair_diminutive_3(self):
        self.compare_potential_pair(
            "Ich sah ein Schlüsselein. Dieses stand", 3, False, 5, 2
        )

    def test_potential_pair_diminutive_control(self):
        self.compare_potential_pair(
            "Ich sah ein Schlüsselein. Sie standen", 3, False, 5, 0
        )

    def test_potential_pair_male_name(self):
        self.compare_potential_pair("Ich sah Peter. Er stand", 2, False, 4, 2)

    def test_potential_pair_male_name_control_1(self):
        self.compare_potential_pair("Ich sah Peter. Sie stand", 2, False, 4, 0)

    def test_potential_pair_male_name_control_2(self):
        self.compare_potential_pair("Ich sah Peter. Dieses stand", 2, False, 4, 0)

    def test_potential_pair_male_name_control_3(self):
        self.compare_potential_pair("Ich sah Peter. Sie standen", 2, False, 4, 0)

    def test_potential_pair_female_name(self):
        self.compare_potential_pair("Ich sah Petra. Sie stand", 2, False, 4, 2)

    def test_potential_pair_female_name_control_1(self):
        self.compare_potential_pair("Ich sah Petra. Er stand", 2, False, 4, 0)

    def test_potential_pair_female_name_control_2(self):
        self.compare_potential_pair(
            "Ich sah Petra. Dieses stand", 2, False, 4, 0, excluded_nlps=["core_news_sm"]
        )

    def test_potential_pair_female_name_control_3(self):
        self.compare_potential_pair("Ich sah Petra. Sie standen", 2, False, 4, 0)

    def test_potential_pair_male_female_name_1(self):
        self.compare_potential_pair("Ich sah Carol. Er stand", 2, False, 4, 2)

    def test_potential_pair_male_female_name_2(self):
        self.compare_potential_pair("Ich sah Carol. Sie stand", 2, False, 4, 2)

    def test_potential_pair_male_female_name_control_1(self):
        self.compare_potential_pair("Ich sah Carol. Dieses stand", 2, False, 4, 0)

    def test_potential_pair_male_female_name_control_2(self):
        self.compare_potential_pair("Ich sah Carol. Sie standen", 2, False, 4, 0)

    def test_potential_pair_masc_dative_anaphor_1(self):
        self.compare_potential_pair(
            "Ich sah einen Mann. Mit ihm ging ich in die Stadt", 3, False, 6, 2
        )

    def test_potential_pair_masc_dative_anaphor_2(self):
        self.compare_potential_pair(
            "Ich sah ein Kind. Mit ihm ging ich in die Stadt", 3, False, 6, 2
        )

    def test_potential_pair_masc_dative_anaphor_control_1(self):
        self.compare_potential_pair(
            "Ich sah eine Frau. Mit ihm ging ich in die Stadt", 3, False, 6, 0
        )

    def test_potential_pair_masc_dative_anaphor_control_2(self):
        self.compare_potential_pair(
            "Ich sah einige Frauen. Mit ihm ging ich in die Stadt", 3, False, 6, 0
        )

    def test_potential_pair_fem_acc_anaphor_1(self):
        self.compare_potential_pair(
            "Ich sah eine Frau. Ich sah sie",
            3,
            False,
            7,
            2,
            excluded_nlps=["core_news_sm"],
        )

    def test_potential_pair_fem_acc_anaphor_2(self):
        self.compare_potential_pair(
            "Ich sah einige Frauen. Ich sah sie",
            3,
            False,
            7,
            2,
            excluded_nlps=["core_news_sm"],
        )

    def test_potential_pair_fem_acc_anaphor_control_1(self):
        self.compare_potential_pair("Ich sah einen Mann. Ich sah sie", 3, False, 7, 0)

    def test_potential_pair_fem_acc_anaphor_control_2(self):
        self.compare_potential_pair("Ich sah ein Haus. Ich sah sie", 3, False, 7, 0)

    def test_potential_pair_proav_1(self):
        self.compare_potential_pair(
            "Sie nahm einen Löffel und aß damit", 3, False, 6, 2
        )

    def test_potential_pair_proav_2(self):
        self.compare_potential_pair("Sie nahm eine Gabel und aß damit", 3, False, 6, 2)

    def test_potential_pair_proav_3(self):
        self.compare_potential_pair("Sie nahm ein Messer und aß damit", 3, False, 6, 2)

    def test_potential_pair_proav_4(self):
        self.compare_potential_pair(
            "Sie nahm einige Messer und aß damit", 3, False, 6, 2
        )

    def test_potential_pair_proav_srd_1(self):
        self.compare_potential_pair(
            "Sie nahm einige Messer. Sie aß damit", 3, False, 7, 2
        )

    def test_potential_pair_proav_srd_2(self):
        self.compare_potential_pair(
            "Sie nahm einige Messer. Dann passierte etwas. Sie aß damit",
            3,
            False,
            11,
            0,
        )

    def test_potential_pair_proav_male_anaphor(self):
        self.compare_potential_pair(
            "Er nahm einige Messer und aß damit", 0, False, 6, 0
        )

    def test_potential_pair_proav_female_anaphor(self):
        self.compare_potential_pair(
            "Sie nahm einige Messer und aß damit", 0, False, 6, 0
        )

    def test_potential_pair_proav_male_name(self):
        self.compare_potential_pair(
            "Richard nahm einen Löffel und aß damit", 0, False, 6, 0
        )

    def test_potential_pair_proav_female_name(self):
        self.compare_potential_pair(
            "Petra nahm einen Löffel und aß damit", 0, False, 6, 0
        )

    def test_potential_pair_proav_person(self):
        self.compare_potential_pair(
            "Peter nahm einen Löffel und aß damit", 0, False, 6, 0
        )

    def test_potential_pair_proav_place(self):
        self.compare_potential_pair(
            "München ist schön und verdient Geld damit", 0, False, 6, 0
        )

    def test_potential_pair_proav_company(self):
        self.compare_potential_pair(
            "BMW nahm einen Löffel und aß damit", 0, False, 6, 0
        )

    def test_potential_pair_proav_cataphoric(self):
        self.compare_potential_pair("Er aß damit und nahm einen Löffel", 6, False, 2, 0)

    def test_potential_pair_proav_preceding_word(self):
        self.compare_potential_pair("Er aß das Fleisch damit", 3, False, 4, 0)

    def test_potential_pair_possessive_in_genitive_phrase_simple(self):
        self.compare_potential_pair("Der Mann seines Freundes", 1, False, 2, 0)

    def test_potential_pair_possessive_in_genitive_phrase_simple_not_directly(self):
        self.compare_potential_pair(
            "Der Mann seines Freundes", 1, False, 2, 2, directly=False
        )

    def test_potential_pair_possessive_in_genitive_phrase_coordination_child(self):
        self.compare_potential_pair(
            "Der Mann und der Mann seines Freundes und seines Freundes",
            4,
            False,
            8,
            0,
            excluded_nlps="core_news_sm",
        )

    def test_potential_pair_possessive_in_genitive_phrase_control(self):
        self.compare_potential_pair("Der Mann mit seinem Freund", 1, False, 3, 2)

    def test_potential_pair_possessive_in_genitive_phrase_double_simple(self):
        self.compare_potential_pair(
            "Der Mann seines Freundes seines Freundes", 1, False, 4, 0
        )

    def test_potential_pair_possessive_in_genitive_phrase_double_control_1(self):
        self.compare_potential_pair(
            "Der Mann mit dem Freund seines Freundes", 1, False, 5, 2
        )

    def test_potential_pair_possessive_in_genitive_phrase_double_control_1(self):
        self.compare_potential_pair(
            "Der Mann des Freundes mit seinem Freund", 1, False, 5, 2
        )

    def test_potential_pair_possessive_in_genitive_phrase_double_coordination_everywhere_1(
        self,
    ):
        self.compare_potential_pair(
            "Der Mann und der Mann seines Freundes und seines Freundes seines Freundes und seines Freundes",
            1,
            False,
            13,
            2,
        )

    def test_potential_pair_possessive_in_genitive_phrase_double_coordination_everywhere_2(
        self,
    ):
        self.compare_potential_pair(
            "Der Mann und der Mann seines Freundes und seines Freundes seines Freundes und seines Freundes",
            4,
            False,
            13,
            0,
        )

    def test_potential_pair_neuter_subject_personal_verb(self):
        self.compare_potential_pair(
            "Das Haus war da. Es sagte, alles OK.", 1, False, 5, 1
        )

    def test_potential_pair_neuter_subject_personal_verb_control_conjunction(self):
        self.compare_potential_pair(
            "Das Haus und das Haus waren da. Sie sagten, alles OK.", 1, True, 8, 1
        )

    def test_potential_pair_neuter_subject_personal_verb_control_1(self):
        self.compare_potential_pair("Peter war da. Er sagte, alles OK.", 0, False, 4, 2)

    def test_potential_pair_neuter_subject_personal_verb_control_2(self):
        self.compare_potential_pair("Petra war da. Sie sagte, alles OK.", 0, False, 4, 2)

    def test_potential_pair_neuter_subject_personal_verb_control_3(self):
        self.compare_potential_pair(
            "Der Mann kam hinein. Er sagte, alles OK.", 1, False, 5, 1
        )

    def test_potential_pair_neuter_subject_personal_verb_control_conjunction_1(self):
        self.compare_potential_pair(
            "Peter und das Haus waren da. Sie sagten, alles OK.", 0, True, 7, 2
        )

    def test_potential_pair_neuter_subject_personal_verb_control_conjunction_2(self):
        self.compare_potential_pair(
            "Das Haus und Peter waren da. Sie sagten, alles OK.", 1, True, 7, 2
        )

    def test_potential_pair_gender_not_marked_on_anaphoric_pronoun(self):
        self.compare_potential_pair("Peter kam rein. Ich folgte ihm", 0, False, 6, 2)

    def test_potential_pair_sie_gender_not_marked(self):
        self.compare_potential_pair(
            "Es gab Hunde. Jemand verkaufte sie.", 2, False, 6, 2, excluded_nlps=["core_news_sm"]
        )

    def test_potential_pair_antecedent_in_prepositional_phrase_in_question(self):
        self.compare_potential_pair("In welchem Raum war er?", 2, False, 4, 0)

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
            "Ich sah den Menschen. Der Mensch sah sich", 3, False, 8, 0, False, 2
        )

    def test_reflexive_in_wrong_situation_different_sentence_control(self):
        self.compare_potential_reflexive_pair(
            "Ich sah den Menschen. Der andere Mensch sah ihn", 3, False, 9, 2, False, 0
        )

    def test_reflexive_in_wrong_situation_same_sentence_1(self):
        self.compare_potential_reflexive_pair(
            "Ich sah den Menschen, während der andere Mensch sich selbst sah.",
            3,
            False,
            9,
            0,
            False,
            2,
        )

    def test_reflexive_in_wrong_situation_same_sentence_control(self):
        self.compare_potential_reflexive_pair(
            "Ich sah den Mann, während der andere Mensch ihn sah",
            3,
            False,
            9,
            2,
            False,
            0,
        )

    def test_non_reflexive_in_wrong_situation_same_sentence(self):
        self.compare_potential_reflexive_pair(
            "Der Mann sah ihn.", 1, False, 3, 0, True, 0
        )

    def test_non_reflexive_in_wrong_situation_same_sentence_control(self):
        self.compare_potential_reflexive_pair(
            "Der Mann sah sich.", 1, False, 3, 2, True, 2
        )

    def test_non_reflexive_in_same_sentence_with_verb_conjunction(self):
        self.compare_potential_reflexive_pair(
            "Der Mann hörte alles und sah sich.", 1, False, 6, 2, True, 2
        )

    def test_reflexive_in_right_situation_modal(self):
        self.compare_potential_reflexive_pair(
            "Der Mann wollte sich sehen.", 1, False, 3, 2, True, 2
        )

    def test_reflexive_in_right_situation_zu_clause(self):
        self.compare_potential_reflexive_pair(
            "Der Mann dachte darüber nach, sich zu sehen.", 1, False, 6, 2, True, 2
        )

    def test_reflexive_in_right_situation_within_subordinate_clause(self):
        self.compare_potential_reflexive_pair(
            "Er wusste, dass der Mann sich selbst sah.", 5, False, 6, 2, True, 2
        )

    def test_reflexive_in_right_situation_within_subordinate_clause_control(self):
        self.compare_potential_reflexive_pair(
            "Er wusste, dass der Mann sich selbst sah.", 0, False, 6, 0, False, 2
        )

    def test_reflexive_in_right_situation_within_subordinate_clause_anaphor_first(self):
        self.compare_potential_reflexive_pair(
            "Er wusste, dass sich der Mann selbst sah.", 6, False, 4, 2, True, 2
        )

    def test_reflexive_in_right_situation_within_subordinate_clause_anaphor_first_control(
        self,
    ):
        self.compare_potential_reflexive_pair(
            "Er wusste, dass sich der Mann selbst sah.", 0, False, 4, 0, False, 2
        )

    def test_reflexive_with_conjuction(self):
        self.compare_potential_reflexive_pair(
            "Das Haus und das Auto haben sich selbst übertroffen",
            1,
            True,
            6,
            2,
            True,
            2,
        )

    def test_reflexive_with_conjuction_control(self):
        self.compare_potential_reflexive_pair(
            "Das Haus und das Auto haben sich selbst übertroffen",
            1,
            False,
            6,
            0,
            True,
            2,
        )

    def test_reflexive_with_passive(self):
        self.compare_potential_reflexive_pair(
            "Das Haus wurde durch sich selbst übertroffen", 1, False, 4, 2, True, 2
        )

    def test_reflexive_with_passive_and_conjunction(self):
        self.compare_potential_reflexive_pair(
            "Das Haus und das Auto wurden durch sich selbst übertroffen",
            1,
            True,
            7,
            2,
            True,
            2,
        )

    def test_reflexive_with_object_antecedent(self):
        self.compare_potential_reflexive_pair(
            "Er mischte die Chemikalie mit sich.", 3, False, 5, 2, True, 2
        )
        self.compare_potential_reflexive_pair(
            "Er mischte die Chemikalie mit sich.", 0, False, 5, 2, True, 2
        )

    def test_reflexive_with_object_antecedent_and_coordination(self):
        self.compare_potential_reflexive_pair(
            "Er mischte die Chemikalie und das Salz mit sich.", 3, True, 8, 2, True, 2
        )
        self.compare_potential_reflexive_pair(
            "Er mischte die Chemikalie und das Salz mit sich.", 0, False, 8, 2, True, 2
        )

    def test_reflexive_with_verb_coordination_one_subject(self):
        self.compare_potential_reflexive_pair(
            "Er sah es und gratulierte sich.", 0, False, 5, 2, True, 2
        )

    def test_reflexive_with_verb_coordination_two_subjects(self):
        self.compare_potential_reflexive_pair(
            "Er sah es, und sein Chef gratulierte sich.", 0, False, 8, 0, False, 2
        )

    def test_reflexive_pronoun_before_referent(self):
        self.compare_potential_reflexive_pair(
            "Sie wollten, dass sich der Junge kennt.", 6, False, 4, 2, True, 2
        )

    def test_reflexive_pronoun_before_referent_control(self):
        self.compare_potential_reflexive_pair(
            "Sie wollten, dass sich der Junge wegen des Erfolgs kennt.",
            9,
            False,
            4,
            0,
            False,
            2,
        )

    def test_reflexive_with_(self):
        self.compare_potential_reflexive_pair(
            "Sie wollten, dass sich der Junge kennt.", 6, False, 4, 2, True, 2
        )

    def test_reflexive_with_referent_within_noun_phrase(self):
        self.compare_potential_reflexive_pair(
            "Sie diskutierten die Möglichkeit, dass sich ein Individuum selbst sieht.",
            8,
            False,
            6,
            2,
            True,
            2,
        )

    def test_non_reflexive_in_wrong_situation_subordinate_clause(self):
        self.compare_potential_reflexive_pair(
            "Obwohl er ihn sah, war er glücklich.", 1, False, 2, 0, True, 0
        )

    def test_reflexive_completely_within_noun_phrase_1(self):
        self.compare_potential_reflexive_pair(
            "Die Meinung meines Freundes über sich selbst war übertrieben.",
            3,
            False,
            5,
            2,
            True,
            2,
        )

    def test_reflexive_completely_within_noun_phrase_1_control(self):
        self.compare_potential_reflexive_pair(
            "Die Meinung meines Freundes über ihn war übertrieben.",
            1,
            False,
            5,
            0,
            True,
            0,
        )

    def test_reflexive_referred_in_prepositional_phrase_control(self):
        self.compare_potential_reflexive_pair(
            "Aus dieser Überlegung ergab sich ein Problem.", 2, False, 4, 0, False, 2
        )

    def test_reflexive_double_coordination_without_preposition(self):
        self.compare_potential_reflexive_pair(
            "Wolfgang und Marie sahen ihn und sie.", 0, False, 4, 0, True, 0
        )
        self.compare_potential_reflexive_pair(
            "Wolfgang und Marie sahen ihn und sie.",
            2,
            False,
            6,
            0,
            True,
            False,
            excluded_nlps=["core_news_md", "core_news_sm"]
        )

    def test_reflexive_double_coordination_with_preposition(self):
        self.compare_potential_reflexive_pair(
            "Wolfgang und Marie sprachen mit ihm und ihr.", 0, False, 5, 0, True, 0
        )
        self.compare_potential_reflexive_pair(
            "Wolfgang und Marie sprachen mit ihm und ihr.", 2, False, 7, 0, True, 0
        )

    def test_reflexive_relative_clause_subject(self):
        self.compare_potential_reflexive_pair(
            "Der Mann, der ihn sah, kam heim.", 1, False, 4, 0, True, 0
        )

    def test_reflexive_relative_clause_object_1(self):
        self.compare_potential_reflexive_pair(
            "Der Mann, den er sah, kam heim.", 1, False, 4, 0, True, 0
        )

    def test_reflexive_relative_clause_with_conjunction(self):
        self.compare_potential_reflexive_pair(
            "Der Mann und die Frau, die sie sahen, kamen heim.", 1, True, 7, 0, True, 0
        )

    def compare_potential_noun_pair(
        self,
        doc_text,
        referred_index,
        referring_index,
        expected_truth,
        *,
        excluded_nlps=[]
    ):
        def func(nlp):
            if nlp.meta["name"] in excluded_nlps:
                return
            doc = nlp(doc_text)
            rules_analyzer = RulesAnalyzerFactory().get_rules_analyzer(nlp)
            rules_analyzer.initialize(doc)
            self.assertEqual(
                expected_truth,
                rules_analyzer.is_potential_coreferring_noun_pair(
                    doc[referred_index], doc[referring_index]
                ),
                nlp.meta["name"],
            )

        self.all_nlps(func)

    def test_potential_noun_pair_proper_noun_test(self):
        self.compare_potential_noun_pair(
            "von Bach über Beethoven, Brahms, Brückner.", 5, 7, False
        )
