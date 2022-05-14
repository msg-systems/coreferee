import unittest
from coreferee.rules import RulesAnalyzerFactory
from coreferee.test_utils import get_nlps
from coreferee.data_model import Mention

nlps = get_nlps("pl")
train_version_mismatch = False
for nlp in nlps:
    if not nlp.meta["matches_train_version"]:
        train_version_mismatch = True
train_version_mismatch_message = (
    "Loaded model version does not match train model version"
)


class PolishRulesTest(unittest.TestCase):
    def setUp(self):

        self.nlps = get_nlps("pl")
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
            "Richard poszedł do domu", 0, "[]", None, False
        )

    def test_get_dependent_sibling_info_two_member_conjunction_phrase_and(self):
        self.compare_get_dependent_sibling_info(
            "Richard i Christine poszli do domu", 0, "[Christine]", None, False
        )

    def test_get_dependent_sibling_info_two_member_conjunction_phrase_with(self):
        self.compare_get_dependent_sibling_info(
            "Richard z synem poszli do domu", 0, "[synem]", None, False,
            excluded_nlps=["core_news_sm"]
        )

    def test_get_dependent_sibling_info_two_member_conjunction_phrase_verb_anaphor_with(
        self,
    ):
        self.compare_get_dependent_sibling_info(
            "Tomek przyjechał. Wyszedł z Anną",
            3,
            "[Anną]",
            None,
            False,
            excluded_nlps=["core_news_md", "core_news_sm"],
        )

    def test_get_dependent_sibling_info_two_member_conjunction_phrase_verb_anaphor_with_control_1(
        self,
    ):
        self.compare_get_dependent_sibling_info(
            "Tomek przyjechał. Wyszedł codziennie z Anną", 3, "[]", None, False
        )

    def test_get_dependent_sibling_info_two_member_conjunction_phrase_verb_anaphor_with_control_2(
        self,
    ):
        self.compare_get_dependent_sibling_info(
            "Tomek przyjechał. On wyszedł z Anną", 3, "[]", None, False
        )

    def test_get_dependent_sibling_info_two_member_conjunction_phrase_verb_anaphor_with_and(
        self,
    ):
        self.compare_get_dependent_sibling_info(
            "Tomek przyjechał. Wyszedł z Anną i Agnieszką",
            3,
            "[Anną, Agnieszką]",
            None,
            False,
            excluded_nlps=["core_news_md"],
        )

    def test_get_dependent_sibling_info_two_member_conjunction_phrase_with_same_parent(
        self,
    ):
        self.compare_get_dependent_sibling_info(
            "Widział psa z psem.", 1, "[psem]", None, False
        )

    def test_get_dependent_sibling_info_two_member_conjunction_phrase_with_same_parent_contrl(
        self,
    ):
        self.compare_get_dependent_sibling_info(
            "Widział psa już z psem.", 1, "[]", None, False
        )

    def test_get_governing_sibling_info_two_member_conjunction_phrase_and(self):
        self.compare_get_dependent_sibling_info(
            "Richard i Christine poszli do domu", 2, "[]", 0, False
        )

    def test_get_dependent_sibling_info_three_member_conjunction_phrase_with_comma_and(
        self,
    ):
        self.compare_get_dependent_sibling_info(
            "Carol, Richard i Ralf mieli zebranie.",
            0,
            "[Richard, Ralf]",
            None,
            False,
            excluded_nlps=["core_news_md", "core_news_sm"],
        )

    def test_get_dependent_sibling_info_conjunction_itself(self):
        self.compare_get_dependent_sibling_info(
            "Zebranie z Carolem i Ralfem i Richardem miało miejsce wczoraj.",
            3,
            "[]",
            None,
            False,
        )

    def test_get_dependent_sibling_info_dependent_sibling(self):
        self.compare_get_dependent_sibling_info(
            "Zebranie z Carolem i Ralfem i Richardem miało miejsce wczoraj.",
            4,
            "[]",
            0,
            False,
        )

    def test_get_dependent_sibling_other_instrumental(self):
        self.compare_get_dependent_sibling_info(
            "Rozmawiali o opinii nad tą ustawą.", 5, "[]", None, False
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
        self.compare_independent_noun("Pooglądali sobie wielkie lwy.", [3])

    def test_independent_noun_conjunction(self):
        self.compare_independent_noun(
            "Pooglądali sobie wielkie lwy, węże, i słonie", [3, 5, 8],
            excluded_nlps=['core_news_sm']
        )

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_substituting_indefinite_pronoun(self):
        self.compare_independent_noun("Jeden z chłopców przyszedł do domu", [2, 5])

    def test_blacklisted(self):
        self.compare_independent_noun(
            "Moim zdaniem bywa chłopiec na przykład zawsze zmęczony", [3]
        )

    def test_blacklisted_control(self):
        self.compare_independent_noun("Moim zdaniem jest to dobry przykład.", [5])

    def test_punctuation(self):
        self.compare_independent_noun("[Enter]", [1])

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
        self.compare_potentially_indefinite("Rozmawiałem z Piotrem", 2, False)

    def test_potentially_indefinite_common_noun(self):
        self.compare_potentially_indefinite("Rozmawiałem z bratem", 2, True)

    def test_potentially_indefinite_common_noun_jakis(self):
        self.compare_potentially_indefinite(
            "Rozmawiałem z jakimś bratem", 3, True, excluded_nlps=["core_news_md"]
        )

    def test_potentially_indefinite_definite_common_noun(self):
        self.compare_potentially_indefinite("Rozmawiałem z tym bratem", 3, False,
        excluded_nlps=["core_news_sm"])

    def test_potentially_indefinite_common_noun_with_possessive_pronoun(self):
        self.compare_potentially_indefinite("Rozmawiałem z naszym bratem", 3, False)

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
        self.compare_potentially_definite("Rozmawiałem z Piotrem", 2, False)

    def test_potentially_definite_common_noun(self):
        self.compare_potentially_definite("Rozmawiałem z bratem", 2, True)

    def test_potentially_definite_definite_common_noun(self):
        self.compare_potentially_definite("Rozmawiałem z tym bratem", 3, True)

    def test_potentially_definite_common_noun_with_possessive_pronoun(self):
        self.compare_potentially_definite("Rozmawiałem z naszym bratem", 3, True)

    def test_potentially_definite_common_noun_jakis(self):
        self.compare_potentially_definite("Rozmawiałem z jakimś bratem", 3, False)

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
            "Kobieta powołała go i pokazała mu jego samochód i swój samochód.",
            [2, 5, 6, 9],
        )

    def test_third_person_pronouns_full_forms(self):
        self.compare_potential_anaphor(
            "Kobieta powołała jego i pokazała jemu jego samochód i swój samochód.",
            [2, 5, 6, 9],
        )

    def test_first_and_second_person_pronouns(self):
        self.compare_potential_anaphor("Ja wiem, że ty go znasz", [5])

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_reflexive_non_clitic(self):
        self.compare_potential_anaphor(
            "Wtedy umył sobie zęby i siebie zobaczył w lustrze.", [1, 2, 5, 6]
        )

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_reflexive_clitic(self):
        self.compare_potential_anaphor(
            "Wtedy umył sobie zęby i zobaczył się w lustrze.", [1, 2, 5]
        )

    def test_verb_imperfective_past(self):
        self.compare_potential_anaphor("Jechała do domu.", [0])

    def test_verb_perfective_past(self):
        self.compare_potential_anaphor("Poszła do domu.", [0])

    def test_verb_imperfective_present(self):
        self.compare_potential_anaphor("Idzie do domu.", [0])

    def test_verb_imperfective_present_control_1(self):
        self.compare_potential_anaphor("Dziecko idzie do domu.", [])

    def test_verb_imperfective_present_control_2(self):
        self.compare_potential_anaphor("Dziecko zostaje przyjęte do domu.", [])

    def test_verb_perfective_future(self):
        self.compare_potential_anaphor("Pójdzie do domu.", [0])

    def test_verb_imperfective_future(self):
        self.compare_potential_anaphor("Będzie jechać do domu.", [0])

    def test_verb_imperfective_future_control(self):
        self.compare_potential_anaphor("Dziecko będzie jechać do domu.", [])

    def test_verb_frequentative_present_non_motion_verb(self):
        self.compare_potential_anaphor("Bywa w domu.", [0])

    def test_verb_frequentative_present_motion_verb(self):
        self.compare_potential_anaphor("Chodzi do szkoły.", [0])

    def test_verb_conditional(self):
        self.compare_potential_anaphor("Wtedy poszedłby do szkoły.", [1])

    def test_verb_conditional_split(self):
        self.compare_potential_anaphor("By poszedł do szkoły.", [1], excluded_nlps=['core_news_md'])

    def test_verb_modal_and_control_imperfective_infinitive(self):
        self.compare_potential_anaphor("Chce iść do szkoły.", [0])

    def test_verb_modal_and_control_imperfective_infinitive_with_conjunction(self):
        self.compare_potential_anaphor("Chce i będzie iść do szkoły.", [0, 2])

    def test_verb_modal_and_control_imperfective_infinitive_control(self):
        self.compare_potential_anaphor("Dziecko chce iść do szkoły.", [])

    def test_verb_modal_and_control_perfective_infinitive(self):
        self.compare_potential_anaphor("Chciałby pójść do szkoły.", [0])

    def test_verb_auxiliary(self):
        self.compare_potential_anaphor("Został powitany w nowej szkole.", [0])

    def test_verb_control_1_2_persons(self):
        self.compare_potential_anaphor(
            "Mam, masz, ma, mamy, macie, mają idee.", [4, 10]
        )

    def test_verb_impersonal_present(self):
        self.compare_potential_anaphor("Okazuje się, że to nieprawda.", [])

    def test_verb_impersonal_past(self):
        self.compare_potential_anaphor("Okazało się, że to nieprawda.", [])

    def test_verb_impersonal_past_control(self):
        self.compare_potential_anaphor("Okazała się, że to nieprawda.", [0])

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

    def test_third_person_pronouns(self):
        self.compare_potential_pair(
            "Wyszła, żeby rzucić okiem na jej samochód.", 0, False, 6, 2
        )

    def test_third_person_verbs(self):
        self.compare_potential_pair("Wyszła. Krzyknęła.", 0, False, 2, 2)

    def test_masculine_pronoun(self):
        self.compare_potential_pair(
            "Chłopiec wszedł. On był szczęśliwy.", 0, False, 3, 2
        )

    def test_masculine_pronoun_control_gender(self):
        self.compare_potential_pair(
            "Chłopiec wszedł. Ona była szczęśliwa.", 0, False, 3, 0
        )

    def test_masculine_pronoun_control_number(self):
        self.compare_potential_pair(
            "Chłopiec wszedł. One były szczęśliwe.", 0, False, 3, 0
        )

    def test_masculine_verb_marked(self):
        self.compare_potential_pair("Chłopiec wszedł. Szczęśliwy był.", 0, False, 4, 2)

    def test_masculine_verb_not_marked(self):
        self.compare_potential_pair("Chłopiec wszedł. Szczęśliwy jest.", 0, False, 4, 2)

    def test_masculine_verb_control_gender(self):
        self.compare_potential_pair("Chłopiec wszedł. Potem wyszło.", 0, False, 4, 0)

    def test_masculine_verb_control_number(self):
        self.compare_potential_pair("Chłopiec wszedł. Szczęśliwi byli.", 0, False, 4, 0)

    def test_masculine_reflexive_possessive(self):
        self.compare_potential_pair("Chłopiec zobaczył swojego psa.", 0, False, 2, 2)

    def test_masculine_nonreflexive_possessive(self):
        self.compare_potential_pair("Chłopiec zobaczył jego psa.", 0, False, 2, 2,
        excluded_nlps=["core_news_sm"])

    def test_masculine_nonreflexive_possessive_control(self):
        self.compare_potential_pair("Chłopiec zobaczył też jej psa.", 0, False, 3, 0)

    def test_feminine_pronoun(self):
        self.compare_potential_pair(
            "Dziewczyna weszła. Ona była szczęśliwa.", 0, False, 3, 2
        )

    def test_feminine_pronoun_control_gender(self):
        self.compare_potential_pair(
            "Dziewczyna weszła. On był szczęśliwy.", 0, False, 3, 0
        )

    def test_feminine_pronoun_control_number(self):
        self.compare_potential_pair(
            "Dziewczyna weszła. Oni byli szczęśliwi.", 0, False, 3, 0
        )

    def test_feminine_verb_marked(self):
        self.compare_potential_pair(
            "Dziewczyna weszła. Szczęśliwa była.", 0, False, 4, 2
        )

    def test_feminine_verb_not_marked(self):
        self.compare_potential_pair(
            "Dziewczyna weszła. Szczęśliwa jest.", 0, False, 4, 2
        )

    def test_feminine_verb_control_gender(self):
        self.compare_potential_pair(
            "Dziewczyna weszła. Szczęśliwy był.", 0, False, 4, 0
        )

    def test_feminine_verb_control_number(self):
        self.compare_potential_pair("Dziewczyna weszła. Potem wyszły.", 0, False, 4, 0)

    def test_feminine_reflexive_possessive(self):
        self.compare_potential_pair("Dziewczyna zobaczyła swojego psa.", 0, False, 2, 2)

    def test_feminine_nonreflexive_possessive(self):
        self.compare_potential_pair("Dziewczyna zobaczyła też jej psa.", 0, False, 3, 2)

    def test_feminine_nonreflexive_possessive_control(self):
        self.compare_potential_pair("Dziewczyna zobaczyła jego psa.", 0, False, 2, 0)

    def test_neuter_pronoun_1(self):
        self.compare_potential_pair(
            "Dziecko weszło. Ono było szczęśliwe.", 0, False, 3, 2
        )

    def test_neuter_pronoun_2(self):
        self.compare_potential_pair("Dziecko weszło. Widzieli go.", 0, False, 4, 2, excluded_nlps=['core_news_sm'])

    def test_neuter_pronoun_3(self):
        self.compare_potential_pair("Dziecko weszło. Pomogli mu.", 0, False, 4, 2, excluded_nlps=['core_news_sm'])

    def test_neuter_pronoun_control_gender(self):
        self.compare_potential_pair(
            "Dziecko weszło. Ona była szczęśliwa.", 0, False, 3, 0
        )

    def test_neuter_pronoun_control_number(self):
        self.compare_potential_pair(
            "Dziecko weszło. Oni byli szczęśliwi.", 0, False, 3, 0
        )

    def test_neuter_verb_marked(self):
        self.compare_potential_pair("Dziecko weszło. Potem wyszło.", 0, False, 4, 2)

    def test_neuter_verb_not_marked(self):
        self.compare_potential_pair("Dziecko weszło. Szczęśliwe jest.", 0, False, 4, 2)

    def test_neuter_verb_control_gender(self):
        self.compare_potential_pair("Dziecko weszło. Szczęśliwy był.", 0, False, 4, 0)

    def test_neuter_verb_control_number(self):
        self.compare_potential_pair("Dziecko weszło. Szczęśliwi byli.", 0, False, 4, 0)

    def test_neuter_reflexive_possessive(self):
        self.compare_potential_pair("Dziecko zobaczyło swojego psa.", 0, False, 2, 2)

    def test_neuter_nonreflexive_possessive(self):
        self.compare_potential_pair("Dziecko zobaczyło jego psa.", 0, False, 2, 2)

    def test_virile_pronoun(self):
        self.compare_potential_pair(
            "Faceci weszli. Oni byli szczęśliwi.", 0, False, 3, 2
        )

    def test_virile_pronoun_control_gender(self):
        self.compare_potential_pair(
            "Faceci weszli. One były szczęśliwe.", 0, False, 3, 0
        )

    def test_virile_pronoun_control_number(self):
        self.compare_potential_pair("Faceci weszli. On był szczęśliwy.", 0, False, 3, 0)

    def test_virile_verb_marked(self):
        self.compare_potential_pair("Faceci weszli. Szczęśliwi byli.", 0, False, 4, 2)

    def test_virile_verb_not_marked(self):
        self.compare_potential_pair("Faceci weszli. Szczęśliwi są.", 0, False, 4, 2,
        excluded_nlps=["core_news_sm"])

    def test_virile_verb_control_gender(self):
        self.compare_potential_pair("Faceci weszli. Szczęśliwe były.", 0, False, 4, 0,
        excluded_nlps=["core_news_sm"])

    def test_virile_verb_control_number(self):
        self.compare_potential_pair("Faceci weszli. Szczęśliwa była.", 0, False, 4, 0,
        excluded_nlps=["core_news_sm"])

    def test_virile_reflexive_possessive(self):
        self.compare_potential_pair("Faceci zobaczyli swojego psa.", 0, False, 2, 2)

    def test_virile_nonreflexive_possessive(self):
        self.compare_potential_pair(
            "Faceci zobaczyli ich psa.", 0, False, 2, 2, excluded_nlps=["core_news_md"]
        )

    def test_nonvirile_pronoun_1(self):
        self.compare_potential_pair(
            "Kobiety weszły. One były szczęśliwe.", 0, False, 3, 2
        )

    def test_nonvirile_pronoun_2(self):
        self.compare_potential_pair("Psy weszły. One były szczęśliwe.", 0, False, 3, 2)

    def test_nonvirile_pronoun_3(self):
        self.compare_potential_pair("Domy weszły. One były szczęśliwe.", 0, False, 3, 2)

    def test_nonvirile_pronoun_4(self):
        self.compare_potential_pair(
            "Dzieci weszły. One były szczęśliwe.", 0, False, 3, 2
        )

    def test_nonvirile_pronoun_control_gender(self):
        self.compare_potential_pair(
            "Kobiety weszły. Oni byli szczęśliwi.", 0, False, 3, 0
        )

    def test_nonvirile_pronoun_control_number(self):
        self.compare_potential_pair(
            "Kobiety weszły. Ona była szczęśliwa.", 0, False, 3, 0
        )

    def test_nonvirile_verb_marked_1(self):
        self.compare_potential_pair("Kobiety weszły. Szczęśliwe były.", 0, False, 4, 2,
        excluded_nlps=["core_news_sm"])

    def test_nonvirile_verb_marked_2(self):
        self.compare_potential_pair("Psy weszły. Szczęśliwe były.", 0, False, 4, 2)

    def test_nonvirile_verb_marked_3(self):
        self.compare_potential_pair("Domy weszły. Szczęśliwe były.", 0, False, 4, 2)

    def test_nonvirile_verb_marked_4(self):
        self.compare_potential_pair("Dzieci weszły. Szczęśliwe były.", 0, False, 4, 2)

    def test_nonvirile_verb_not_marked(self):
        self.compare_potential_pair("Kobiety weszly. Szczęśliwe są.", 0, False, 4, 2,
        excluded_nlps=["core_news_sm"])

    def test_nonvirile_verb_control_gender(self):
        self.compare_potential_pair("Kobiety weszły. Szczęśliwi byli.", 0, False, 4, 0)

    def test_nonvirile_verb_control_number(self):
        self.compare_potential_pair("Kobiety weszły. Szczęśliwa była.", 0, False, 4, 0,
        excluded_nlps=["core_news_sm"])

    def test_nonvirile_reflexive_possessive(self):
        self.compare_potential_pair("Kobiety zobaczyły swojego psa.", 0, False, 2, 2)

    def test_nonvirile_nonreflexive_possessive(self):
        self.compare_potential_pair(
            "Kobiety zobaczyły ich psa.", 0, False, 2, 2, 
            excluded_nlps=["core_news_md", "core_news_sm"]
        )

    def test_male_name(self):
        self.compare_potential_pair(
            "Krzysiek widzi niebo. On jest szczęśliwy.", 0, False, 4, 2
        )

    def test_male_name_control(self):
        self.compare_potential_pair(
            "Krzysiek widzi niebo. Ona jest szczęśliwa.", 0, False, 4, 0
        )

    def test_female_name(self):
        self.compare_potential_pair(
            "Anna widzi niebo. Ona jest szczęśliwa.", 0, False, 4, 2
        )

    def test_female_name_control(self):
        self.compare_potential_pair(
            "Anna widzi niebo. On jest szczęśliwy.", 0, False, 4, 0
        )

    def test_coordinated_phrase_two_personal_masculine(self):
        self.compare_potential_pair(
            "Są syn i ojciec. Oni są szczęśliwi.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_personal_masculine_control(self):
        self.compare_potential_pair(
            "Są syn i ojciec. One są szczęśliwe.", 1, True, 5, 0
        )

    def test_coordinated_phrase_two_animal_masculine(self):
        self.compare_potential_pair("Są pies i lew. One są szczęśliwe.", 1, True, 5, 2,
        excluded_nlps=["core_news_sm"]
)

    def test_coordinated_phrase_two_animal_masculine_control(self):
        self.compare_potential_pair("Są pies i lew. Oni są szczęśliwi.", 1, True, 5, 0,
        excluded_nlps=["core_news_sm"])

    def test_coordinated_phrase_two_object_masculine(self):
        self.compare_potential_pair(
            "Są dom i samochód. One są szczęśliwe.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_object_masculine_control(self):
        self.compare_potential_pair(
            "Są dom i samochód. Oni są szczęśliwi.", 1, True, 5, 0
        )

    def test_coordinated_phrase_two_object_masculine_z(self):
        self.compare_potential_pair(
            "Są dom z samochodem. One są szczęśliwe.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_object_masculine_z_control(self):
        self.compare_potential_pair(
            "Są dom z samochodem. Oni są szczęśliwi.", 1, True, 5, 0
        )

    def test_coordinated_phrase_two_object_feminine(self):
        self.compare_potential_pair(
            "Są mama i córka. One są szczęśliwe.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_object_feminine_control(self):
        self.compare_potential_pair(
            "Są mama i córka. Oni są szczęśliwi.", 1, True, 5, 0
        )

    def test_coordinated_phrase_two_object_neuter(self):
        self.compare_potential_pair(
            "Są dziecko i dziecko. One są szczęśliwe.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_object_neuter_control(self):
        self.compare_potential_pair(
            "Są dziecko i dziecko. Oni są szczęśliwi.", 1, True, 5, 0
        )

    def test_coordinated_phrase_two_plural_personal_masculine(self):
        self.compare_potential_pair(
            "Są synowie i ojciec. Oni są szczęśliwi.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_plural_personal_masculine_control(self):
        self.compare_potential_pair(
            "Są synowie i ojciec. One są szczęśliwe.", 1, True, 5, 0
        )

    def test_coordinated_phrase_two_personal_masculine_mixed(self):
        self.compare_potential_pair("Są syn i córka. Oni są szczęśliwi.", 1, True, 5, 2)

    def test_coordinated_phrase_two_personal_masculine_mixed(self):
        self.compare_potential_pair("Są syn i córki. One są szczęśliwe.", 1, True, 5, 0)

    def test_coordinated_phrase_two_plural_personal_masculine_mixed(self):
        self.compare_potential_pair(
            "Są synowie i córki. Oni są szczęśliwi.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_plural_personal_masculine_mixed(self):
        self.compare_potential_pair(
            "Są synowie i córka. One są szczęśliwe.", 1, True, 5, 0
        )

    def test_coordinated_phrase_two_plural_animal_masculine(self):
        self.compare_potential_pair("Są psy i lwy. One są szczęśliwe.", 1, True, 5, 2)

    def test_coordinated_phrase_two_plural_animal_masculine_control(self):
        self.compare_potential_pair("Są psy i lwy. Oni są szczęśliwi.", 1, True, 5, 0,
        excluded_nlps=["core_news_sm"])

    def test_coordinated_phrase_two_masculine_animal_and_object(self):
        self.compare_potential_pair("Są pies i dom. One są szczęśliwe.", 1, True, 5, 2)

    def test_coordinated_phrase_two_masculine_animal_and_object_control(self):
        self.compare_potential_pair("Są pies i dom. Oni są szczęśliwi.", 1, True, 5, 0)

    def test_coordinated_phrase_two_feminine(self):
        self.compare_potential_pair(
            "Są kobieta i córka. One są szczęśliwe.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_feminine_control(self):
        self.compare_potential_pair(
            "Są kobieta i córka. Oni są szczęśliwi.", 1, True, 5, 0
        )

    def test_coordinated_phrase_two_neuter(self):
        self.compare_potential_pair(
            "Są dziecko i dziecko. One są szczęśliwe.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_neuter_control(self):
        self.compare_potential_pair(
            "Są dziecko i dziecko. Oni są szczęśliwi.", 1, True, 5, 0
        )

    def test_coordinated_phrase_two_masculine_animal_feminine_mix_1(self):
        self.compare_potential_pair(
            "Są pies i kobieta. One są szczęśliwe.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_masculine_animal_feminine_mix_2(self):
        self.compare_potential_pair(
            "Są pies i kobieta. Oni są szczęśliwi.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_masculine_animal_feminine_mix_3(self):
        self.compare_potential_pair(
            "Są psy i kobiety. One są szczęśliwe.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_masculine_animal_feminine_mix_4(self):
        self.compare_potential_pair(
            "Są psy i kobiety. Oni są szczęśliwi.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_masculine_animal_feminine_z_mix_1(self):
        self.compare_potential_pair(
            "Są psy z kobietami. One są szczęśliwe.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_masculine_animal_feminine_z_mix_2(self):
        self.compare_potential_pair(
            "Są psy z kobietami. Oni są szczęśliwi.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_masculine_object_feminine_mix_1(self):
        self.compare_potential_pair(
            "Są dom i kobieta. One są szczęśliwe.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_masculine_object_feminine_mix_2(self):
        self.compare_potential_pair(
            "Są dom i kobieta. Oni są szczęśliwi.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_masculine_object_feminine_mix_3(self):
        self.compare_potential_pair(
            "Są domy i kobiety. One są szczęśliwe.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_masculine_object_feminine_mix_4(self):
        self.compare_potential_pair(
            "Są domy i kobiety. Oni są szczęśliwi.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_masculine_animal_neuter_mix_1(self):
        self.compare_potential_pair(
            "Są pies i dziecko. One są szczęśliwe.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_masculine_animal_neuter_mix_2(self):
        self.compare_potential_pair(
            "Są pies i dziecko. Oni są szczęśliwi.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_masculine_animal_neuter_mix_3(self):
        self.compare_potential_pair(
            "Są psy i dzieci. One są szczęśliwe.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_masculine_animal_neuter_mix_4(self):
        self.compare_potential_pair(
            "Są psy i dzieci. Oni są szczęśliwi.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_masculine_object_neuter_mix_1(self):
        self.compare_potential_pair(
            "Są dom i dziecko. One są szczęśliwe.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_masculine_object_neuter_mix_2(self):
        self.compare_potential_pair(
            "Są dom i dziecko. Oni są szczęśliwi.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_masculine_object_neuter_mix_3(self):
        self.compare_potential_pair(
            "Są domy i dzieci. One są szczęśliwe.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_masculine_object_neuter_mix_4(self):
        self.compare_potential_pair(
            "Są domy i dzieci. Oni są szczęśliwi.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_feminine_neuter_mix_1(self):
        self.compare_potential_pair(
            "Są kobieta i dziecko. One są szczęśliwe.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_feminine_neuter_mix_2(self):
        self.compare_potential_pair(
            "Są kobieta i dziecko. Oni są szczęśliwi.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_feminine_neuter_mix_3(self):
        self.compare_potential_pair(
            "Są kobiety i dzieci. One są szczęśliwe.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_feminine_neuter_mix_4(self):
        self.compare_potential_pair(
            "Są kobiety i dzieci. Oni są szczęśliwi.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_feminine_neuter_mix_verb_virile(self):
        self.compare_potential_pair(
            "Przyszli kobiety i dzieci. Oni byli szczęśliwi.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_feminine_neuter_mix_verb_virile_control(self):
        self.compare_potential_pair(
            "Przyszli kobiety i dzieci. One były szczęśliwe.", 1, True, 5, 0,
            excluded_nlps=["core_news_sm"]
        )

    def test_coordinated_phrase_two_feminine_neuter_mix_verb_nonvirile(self):
        self.compare_potential_pair(
            "Przyszły kobiety i dzieci. One były szczęśliwe.", 1, True, 5, 2
        )

    def test_coordinated_phrase_two_feminine_neuter_mix_verb_nonvirile_control(self):
        self.compare_potential_pair(
            "Przyjechały kobiety i dzieci. Oni byli szczęśliwi.", 1, True, 5, 0,
            excluded_nlps=["core_news_sm"]
        )

    def test_coordinated_phrase_two_plural_personal_masculine_mixed_only_feminine(self):
        self.compare_potential_pair(
            "Są synowie i dziewczyny. One są szczęśliwe.", 3, False, 5, 2,
            excluded_nlps=["core_news_sm"]
        )

    def test_coordinated_phrase_two_plural_personal_masculine_mixed_only_feminine_z(
        self,
    ):
        self.compare_potential_pair(
            "Są synowie z dziewczynami. One są szczęśliwe.", 3, False, 5, 2
        )

    def test_coordinated_phrase_two_plural_personal_masculine_mixed_only_feminine_control(
        self,
    ):
        self.compare_potential_pair(
            "Są synowie i dziewczyny. Oni są szczęśliwi.", 3, False, 5, 0
        )

    def test_coordinated_phrase_two_plural_personal_masculine_mixed_only_masculine(
        self,
    ):
        self.compare_potential_pair(
            "Są synowie i dziewczyny. Oni są szczęśliwi.", 1, False, 5, 0
        )

    def test_coordinated_phrase_two_plural_personal_masculine_mixed_only_masculine_or(
        self,
    ):
        self.compare_potential_pair(
            "Są synowie lub dziewczyny. Oni są szczęśliwi.", 1, False, 5, 2
        )

    def test_coordinated_phrase_two_plural_personal_masculine_mixed_only_masculine_z(
        self,
    ):
        self.compare_potential_pair(
            "Są synowie z dziewczynami. Oni są szczęśliwi.", 1, False, 5, 0
        )

    def test_coordinated_phrase_singular_or_1(self):
        self.compare_potential_pair(
            "Jest syn albo córka. On jest szczęśliwy.", 1, True, 5, 2
        )

    def test_coordinated_phrase_singular_or_2(self):
        self.compare_potential_pair(
            "Jest syn albo córka. Ona jest szczęśliwa.",
            1,
            True,
            5,
            2,
            excluded_nlps=["core_news_md"],
        )

    def test_coordinated_phrase_singular_or_control_1(self):
        self.compare_potential_pair(
            "Jest syn albo córka. Ono jest szczęśliwe.", 1, True, 5, 0
        )

    def test_coordinated_phrase_singular_or_control_2(self):
        self.compare_potential_pair(
            "Jest syn albo córka. Oni są szczęśliwi.", 1, True, 5, 0
        )

    def test_coordinated_phrase_singular_or_control_3(self):
        self.compare_potential_pair(
            "Jest syn albo córka. One są szczęśliwe.", 1, True, 5, 0
        )

    def test_potential_pair_possessive_in_genitive_phrase_simple_nonreflexive_1(self):
        self.compare_potential_pair("Mąż jego kolegi przemówił", 0, False, 1, 0,
        excluded_nlps=["core_news_sm"]
    )

    def test_potential_pair_possessive_in_genitive_phrase_simple_nonreflexive_2(self):
        self.compare_potential_pair("Mąż swojego kolegi przemówił", 0, False, 1, 0)

    def test_potential_pair_possessive_in_genitive_phrase_simple_not_directly(self):
        self.compare_potential_pair(
            "Mąż jego kolegi przemówił", 0, False, 1, 2, directly=False
        )

    def test_potential_pair_possessive_in_genitive_phrase_coordination_head_nonreflexive(
        self,
    ):
        self.compare_potential_pair(
            "Mąż z mężem jego kolegi przemówili", 0, False, 3, 2, excluded_nlps=['core_news_sm']
        )

    def test_potential_pair_possessive_in_genitive_phrase_coordination_head_reflexive(
        self,
    ):
        self.compare_potential_pair(
            "Przyszedł mąż z mężem swojego kolegi", 1, False, 4, 0
        )

    def test_potential_pair_possessive_in_genitive_phrase_control(self):
        self.compare_potential_pair("Mąż z jego kolegą przemówili", 0, False, 2, 2)

    def test_potential_pair_possessive_in_genitive_phrase_double_simple(self):
        self.compare_potential_pair(
            "Przyszedł mąż jego kolegi jego kolegi", 1, False, 4, 0
        )

    def test_potential_pair_possessive_in_genitive_phrase_double_control_1(self):
        self.compare_potential_pair(
            "Przyszedł mąż z kolegą jego kolegi", 1, False, 4, 2
        )

    def test_potential_pair_possessive_in_genitive_phrase_double_control_2(self):
        self.compare_potential_pair(
            "Przyszedł mąż kolegi z jego kolegą",
            1,
            False,
            4,
            2,
            excluded_nlps=["core_news_md", "core_news_sm"],
        )

    def test_potential_pair_non_personal_subject_personal_verb(self):
        self.compare_potential_pair(
            "Dom stał. Powiedział, wszystko dobrze.", 0, False, 3, 1
        )

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_potential_pair_non_personal_subject_personal_verb_control_conjunction(
        self,
    ):
        self.compare_potential_pair(
            "Dom i dom stoją. One powiedziały, wszystko dobrze", 0, True, 5, 1, 
            excluded_nlps=['core_news_md', 'core_news_sm']
        )

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_potential_pair_non_personal_subject_personal_verb_control_z_conjunction(
        self,
    ):
        self.compare_potential_pair(
            "Dom z domem stoją. One powiedziały, wszystko dobrze", 0, True, 5, 1, 
            excluded_nlps=['core_news_md', 'core_news_sm']
        )

    def test_potential_pair_non_personal_subject_personal_verb_noun_not_recognised(
        self,
    ):
        self.compare_potential_pair(
            "Mężczyzna był. Powiedział, wszystko dobrze.", 0, False, 3, 1
        )

    def test_potential_pair_non_personal_subject_personal_verb_control_1(self):
        self.compare_potential_pair(
            "Piotr był. Powiedział, wszystko dobrze.", 0, False, 3, 2
        )

    def test_potential_pair_non_personal_subject_personal_verb_control_2(self):
        self.compare_potential_pair(
            "Anna była. Powiedziała, wszystko dobrze.", 0, False, 3, 2
        )

    def test_potential_pair_non_personal_subject_personal_verb_control_conjunction_1(
        self,
    ):
        self.compare_potential_pair(
            "Piotr i dom byli. Powiedzieli, wszystko dobrze.", 0, True, 5, 2
        )

    def test_potential_pair_non_personal_subject_personal_verb_control_conjunction_2(
        self,
    ):
        self.compare_potential_pair(
            "Dom i Piotr byli. Powiedzieli, wszystko dobrze.", 0, True, 5, 2
        )

    def test_potential_pair_problem_sentence_1(self):
        self.compare_potential_pair(
            "Mówią o kryzysie polskiej rodziny dotkniętej plagą rozwodów i alkoholizmem. Bywało, że krzyczał.",
            9,
            False,
            11,
            0,
        )

    def test_potential_pair_two_singular_verb_anaphors(self):
        self.compare_potential_pair(
            "Buduje i cieszy się. On jest szczęśliwy", 0, True, 5, 2
        )

    def test_potential_pair_two_singular_verb_anaphors_control(self):
        self.compare_potential_pair(
            "Buduje i cieszy się. Oni są szczęśliwi", 0, True, 5, 0
        )

    def test_potential_pair_singular_verb_anaphor_with_comitative_phrase_simple(self):
        self.compare_potential_pair(
            "Piotr był szczęśliwy. Kupił z żoną nowy dom.", 0, False, 4, 2
        )

    def test_potential_pair_plural_verb_anaphor_with_comitative_phrase_simple(self):
        self.compare_potential_pair(
            "Piotr był szczęśliwy. Kupili z żoną nowy dom.", 0, False, 4, 2,
            excluded_nlps=['core_news_sm']
        )

    def test_potential_pair_plural_verb_anaphor_with_comitative_phrase_coordination(
        self,
    ):
        self.compare_potential_pair(
            "Piotr był szczęśliwy. Kupili z koleżanką i znajomą nowy dom.",
            0,
            False,
            4,
            2,
        )

    def test_potential_pair_plural_verb_anaphor_with_comitative_phrase_coordination_everywhere(
        self,
    ):
        self.compare_potential_pair(
            "Piotr i Janek byli szczęśliwy. Kupili z koleżanką i znajomą nowy dom.",
            0,
            True,
            6,
            2,
        )

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
            "Widziałem człowieka. Człowiek widział siebie.", 1, False, 5, 0, False, 2
        )

    def test_reflexive_in_wrong_situation_different_sentence_control(self):
        self.compare_potential_reflexive_pair(
            "Widziałem człowieka. Drugi człowiek widział go.", 1, False, 6, 2, False, 0,
            excluded_nlps=["core_news_sm"]

        )

    def test_reflexive_in_wrong_situation_same_sentence_1(self):
        self.compare_potential_reflexive_pair(
            "Widziałem człowieka, dopóki drugi człowiek siebie widział.",
            1,
            False,
            6,
            0,
            False,
            2,
        )

    def test_reflexive_in_wrong_situation_same_sentence_control(self):
        self.compare_potential_reflexive_pair(
            "Widziałem człowieka, dopóki drugi człowiek go widział.",
            1,
            False,
            6,
            2,
            False,
            0,
        )

    def test_non_reflexive_in_wrong_situation_same_sentence(self):
        self.compare_potential_reflexive_pair(
            "Człowiek widział go.", 0, False, 2, 0, True, 0,
            excluded_nlps=['core_news_sm']
        )

    def test_non_reflexive_in_wrong_situation_same_sentence_control(self):
        self.compare_potential_reflexive_pair(
            "Człowiek widział siebie.", 0, False, 2, 2, True, 2
        )

    def test_non_reflexive_in_wrong_situation_same_sentence_instr(self):
        self.compare_potential_reflexive_pair(
            "Człowiek poszedł z nim.", 0, False, 3, 0, True, 0, excluded_nlps=["core_news_sm"]
        )

    def test_non_reflexive_in_wrong_situation_same_sentence_instr_control(self):
        self.compare_potential_reflexive_pair(
            "Człowiek poszedł ze sobą.", 0, False, 3, 2, True, 2
        )

    def test_non_reflexive_in_same_sentence_with_verb_conjunction(self):
        self.compare_potential_reflexive_pair(
            "Człowiek słyszał wszystko i widział siebie.", 0, False, 5, 2, True, 2
        )

    def test_reflexive_in_right_situation_modal(self):
        self.compare_potential_reflexive_pair(
            "Człowiek chciał siebie wiedzieć.", 0, False, 2, 2, True, 2
        )

    def test_reflexive_in_right_situation_zu_clause(self):
        self.compare_potential_reflexive_pair(
            "Człowiek myślał o tym, by siebie wiedzieć.", 0, False, 6, 2, True, 2
        )

    def test_reflexive_in_right_situation_within_subordinate_clause(self):
        self.compare_potential_reflexive_pair(
            "Wiedział, że człowiek widział siebie.", 3, False, 5, 2, True, 2
        )

    def test_reflexive_in_right_situation_within_subordinate_clause_control(self):
        self.compare_potential_reflexive_pair(
            "Piotr wiedział, że człowiek widział siebie.", 0, False, 6, 0, False, 2
        )

    def test_reflexive_in_right_situation_within_subordinate_clause_anaphor_first(self):
        self.compare_potential_reflexive_pair(
            "Piotr wiedział, że sobie człowiek dom budował.", 5, False, 4, 2, True, 2
        )

    def test_reflexive_in_right_situation_within_subordinate_clause_anaphor_first_control(
        self,
    ):
        self.compare_potential_reflexive_pair(
            "Piotr wiedział, że sobie człowiek dom budował.", 0, False, 4, 0, False, 2
        )

    def test_reflexive_with_conjuction(self):
        self.compare_potential_reflexive_pair(
            "Dom i samochód przewyższały siebie", 0, True, 4, 2, True, 2
        )

    def test_reflexive_with_passive(self):
        self.compare_potential_reflexive_pair(
            "Dom był przewyższany przez siebie", 0, False, 4, 2, True, 2
        )

    def test_reflexive_with_passive_and_conjunction(self):
        self.compare_potential_reflexive_pair(
            "Dom i samochód były przewyższane przez siebie", 0, True, 6, 2, True, 2
        )

    def test_reflexive_with_object_antecedent(self):
        self.compare_potential_reflexive_pair(
            "Chłopiec przemieszczał substancje ze sobą.", 2, False, 4, 2, True, 2
        )
        self.compare_potential_reflexive_pair(
            "Chłopiec przemieszczał substancje ze sobą.", 0, False, 4, 2, True, 2
        )

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_reflexive_with_object_antecedent_and_coordination(self):
        self.compare_potential_reflexive_pair(
            "Chłopiec przemieszczał substancje i sól ze sobą.", 2, True, 6, 2, True, 2
        )
        self.compare_potential_reflexive_pair(
            "Chłopiec przemieszczał substancje i sól ze sobą.", 0, False, 6, 2, True, 2
        )

    def test_reflexive_with_object_antecedent_control_preceding(self):
        self.compare_potential_reflexive_pair(
            "Chłopiec przemieszczał ze sobą substancje.", 4, False, 3, 0, False, 2
        )
        self.compare_potential_reflexive_pair(
            "Chłopiec przemieszczał ze sobą substancje.", 0, False, 3, 2, True, 2
        )

    def test_reflexive_with_verb_coordination_one_subject(self):
        self.compare_potential_reflexive_pair(
            "On to zobaczył i pogratulował siebie.", 0, False, 5, 2, True, 2
        )

    def test_reflexive_with_verb_coordination_two_subjects(self):
        self.compare_potential_reflexive_pair(
            "On to zobaczył i jego szef pogratulował siebie.", 0, False, 7, 0, False, 2
        )

    def test_reflexive_with_to(self):
        self.compare_potential_reflexive_pair(
            "Chcieli, żeby siebie chłopiec znał", 4, False, 3, 2, True, 2
        )

    def test_non_reflexive_in_wrong_situation_subordinate_clause(self):
        self.compare_potential_reflexive_pair(
            "Pomimo, że go zobaczył, był szczęśliwy", 4, False, 3, 0, True, 0
        )

    def test_reflexive_completely_within_noun_phrase_1(self):
        self.compare_potential_reflexive_pair(
            "Opinia mojego przyjaciela o sobie była przesadna", 2, False, 4, 2, True, 2,
        )

    def test_reflexive_completely_within_noun_phrase_1_control(self):
        self.compare_potential_reflexive_pair(
            "Opinia mojego przyjaciela o nim była przesadna", 2, False, 4, 0, True, 0,
            excluded_nlps=["core_news_md"]
        )

    def test_reflexive_double_coordination_without_preposition(self):
        self.compare_potential_reflexive_pair(
            "Piotr i Agnieszka widzieli jego i ją", 0, False, 4, 0, True, 0
        )
        self.compare_potential_reflexive_pair(
            "Piotr i Agnieszka widzieli jego i ją", 2, False, 6, 0, True, False,
            excluded_nlps=["core_news_md", "core_news_sm"]

        )

    def test_reflexive_double_coordination_with_preposition(self):
        self.compare_potential_reflexive_pair(
            "Piotr i Agnieszka rozmawiali z nim i z nią", 0, False, 5, 0, True, 0
        )
        self.compare_potential_reflexive_pair(
            "Piotr i Agnieszka rozmawiali z nim i z nią",
            2,
            False,
            8,
            0,
            True,
            False,
            excluded_nlps=["core_news_md", "core_news_sm"]
        )

    def test_reflexive_posessive_same_noun_phrase(self):
        self.compare_potential_reflexive_pair(
            "Był zajęty swoją pracą", 3, False, 2, 0, False, 2
        )

    def test_reflexive_with_cataphora_control(self):
        self.compare_potential_reflexive_pair(
            "Ponieważ był zajęty swoją pracą, Janek miał jej dość.",
            4,
            False,
            3,
            0,
            False,
            2,
        )

    def test_reflexive_with_non_reflexive_possessive_pronoun(self):
        self.compare_potential_reflexive_pair(
            "Janek zjadł jego kolację.", 0, False, 2, 2, True, 1
        )

    def test_reflexive_relative_clause_subject(self):
        self.compare_potential_reflexive_pair(
            "Mężczyzna, który go widział, przyjechał do domu.", 0, False, 3, 0, True, 0
        )

    def test_reflexive_relative_clause_object(self):
        self.compare_potential_reflexive_pair(
            "Mężczyzna, którego widział, przyjechał do domu.", 0, False, 3, 0, True, 0
        )

    def test_reflexive_relative_clause_subject_with_conjunction(self):
        self.compare_potential_reflexive_pair(
            "Mężczyzna i kobieta, którzy ich widzieli, przyjechali do domu.",
            0,
            True,
            5,
            0,
            True,
            0,
            excluded_nlps=["core_news_sm"]
        )

    def test_reflexive_relative_clause_object_with_conjunction(self):
        self.compare_potential_reflexive_pair(
            "Mężczyzna i kobieta, których widzieli, przyjechali do domu.",
            0,
            True,
            5,
            0,
            True,
            0,
            excluded_nlps=["core_news_md"],
        )

    def compare_potentially_introducing(
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
                rules_analyzer.is_potentially_introducing_noun(doc[index]),
                nlp.meta["name"],
            )

        self.all_nlps(func)

    def test_potentially_introducing_with_preposition(self):
        self.compare_potentially_introducing("On mieszka również z facetem", 4, True)

    @unittest.skipIf(train_version_mismatch, train_version_mismatch_message)
    def test_potentially_introducing_with_ten_control(self):
        self.compare_potentially_introducing(
            "On mieszka z tym kolegą", 4, False, excluded_nlps=['core_news_md', 'core_news_sm']
        )

    def test_potentially_introducing_with_ten_and_relative_clause(self):
        self.compare_potentially_introducing(
            "On mieszka z tym kolegą, którego znasz", 4, True,
            excluded_nlps=["core_news_sm"]
        )

    def compare_potentially_referring_back_noun(
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
                rules_analyzer.is_potentially_referring_back_noun(doc[index]),
                nlp.meta["name"],
            )

        self.all_nlps(func)

    def test_potentially_referring_back_noun_with_ten(self):
        self.compare_potentially_referring_back_noun(
            "Mieszka z tym kolegą", 3, True, excluded_nlps=["core_news_md", "core_news_sm"]
        )

    def test_potentially_referring_back_noun_with_ten_and_relative_clause_control(self):
        self.compare_potentially_referring_back_noun(
            "Mieszka z tym kolegą, którego znasz",
            3,
            False,
            excluded_nlps=["core_news_md"],
        )

    def test_get_dependent_sibling_info_apposition_control(self):
        self.compare_get_dependent_sibling_info(
            "Richard, wielki informatyk, poszedł do domu", 0, "[]", None, False
        )

    def test_get_governing_sibling_info_apposition_control(self):
        self.compare_get_dependent_sibling_info(
            "Richard, wielki informatyk, poszedł do domu", 3, "[]", None, False
        )
