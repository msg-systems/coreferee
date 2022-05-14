import unittest
from coreferee.test_utils import get_nlps


class PolishSmokeTest(unittest.TestCase):
    def setUp(self):

        self.nlps = get_nlps("pl")

    def all_nlps(self, func):
        for nlp in self.nlps:
            func(nlp)

    def compare_annotations(
        self,
        doc_text,
        expected_coref_chains,
        *,
        excluded_nlps=[],
        alternative_expected_coref_chains=None
    ):
        def func(nlp):

            if nlp.meta["name"] in excluded_nlps:
                return

            doc = nlp(doc_text)
            chains_representation = str(doc._.coref_chains)
            if alternative_expected_coref_chains is None:
                self.assertEqual(
                    expected_coref_chains, chains_representation, nlp.meta["name"]
                )
            else:
                self.assertTrue(
                    expected_coref_chains == chains_representation
                    or alternative_expected_coref_chains == chains_representation
                )

        self.all_nlps(func)

    def test_simple_verb(self):
        self.compare_annotations("Widziałem psa i polował na kota", "[0: [1], [3]]")

    def test_simple_pronoun(self):
        self.compare_annotations("Widziałem psa i on polował na kota", "[0: [1], [3]]")

    def test_simple_verb_plural(self):
        self.compare_annotations("Widziałem psy i polowały na kota", "[0: [1], [3]]")

    def test_simple_pronoun_plural(self):
        self.compare_annotations(
            "Widziałem psy, i one polowały na kota", "[0: [1], [4]]"
        )

    def test_simple_verb_conjunction_same_word(self):
        self.compare_annotations(
            "Widziałem psa i psa, i polowały na kota", "[0: [1, 3], [6]]"
        )

    def test_simple_pronoun_conjunction_same_word(self):
        self.compare_annotations(
            "Widziałem psa i psa, i one polowały na kota", "[0: [1, 3], [6]]"
        )

    def test_simple_verb_conjunction_different_words(self):
        self.compare_annotations(
            "Widziałem psa i konia, i polowały na kota", "[0: [1, 3], [6]]"
        )

    def test_simple_pronoun_conjunction_different_words(self):
        self.compare_annotations(
            "Widziałem psa i konia, i one polowały na kota", "[0: [1, 3], [6]]"
        )

    def test_simple_pronoun_conjunction_z_same_word(self):
        self.compare_annotations(
            "Widziałem psa z psem, i one polowały na kota", "[0: [1, 3], [6]]"
        )

    def test_simple_verb_conjunction_z_different_words(self):
        self.compare_annotations(
            "Widziałem psa z koniem, i polowały na kota",
            "[0: [1, 3], [6]]",
            excluded_nlps=["core_news_md"],
        )

    def test_simple_pronoun_conjunction_z_different_words(self):
        self.compare_annotations(
            "Widziałem psa z koniem, i one polowały na kota",
            "[0: [1, 3], [6]]",
            excluded_nlps=["core_news_md"],
        )

    def test_conjunction_different_pronouns(self):
        self.compare_annotations(
            "Widziałem Piotra i Agnieszkę. Ona i on polowali na kota",
            "[0: [1], [7], 1: [3], [5]]",
        )

    def test_conjunction_z_different_pronouns(self):
        self.compare_annotations(
            "Widziałem Piotra z Agnieszką, i ona z nim polowali na kota",
            "[0: [1], [8], 1: [3], [6]]",
        )

    def test_conjunction_involving_pronoun(self):
        self.compare_annotations(
            "Widziałem Piotra i Agnieszkę. On z Richardem polowali na kota",
            "[0: [1], [5], 1: [1, 3], [8]]",
            alternative_expected_coref_chains="[0: [1], [5]]",
        )

    def test_z_conjunction_with_verb_anaphor_singular_verb(self):
        self.compare_annotations(
            "Widziałem Piotra i Agnieszkę. Polował z Richardem na kota", "[0: [1], [5]]"
        )

    def test_z_conjunction_with_verb_anaphor_nonvirile_verb(self):
        self.compare_annotations(
            "Widziałem Piotra i Agnieszkę. Polowały z koleżanką na kota. Szczęśliwe były.",
            "[0: [3], [5], 1: [5, 7], [12]]",
        )

    def test_different_sentence_verb(self):
        self.compare_annotations("Piotra widziałem. Polował na kota", "[0: [0], [3]]")

    def test_different_sentence_pronoun(self):
        self.compare_annotations(
            "Piotra widziałem. On polował na kota", "[0: [0], [3]]"
        )

    def test_proper_noun_coreference(self):
        self.compare_annotations(
            "Widziałem Piotra. Piotr polował na kota", "[0: [1], [3]]"
        )

    def test_proper_noun_coreference_multiword(self):
        self.compare_annotations(
            "Widziałem Richarda Piotra. Richard Piotr polował na kota.", "[0: [1], [4]]"
        )

    def test_proper_noun_coreference_multiword_only_second_repeated(self):
        self.compare_annotations(
            "Widziałem Richarda Piotra. Piotr polował na kota.", "[0: [1], [4]]"
        )

    def test_proper_noun_coreference_multiword_only_first_repeated(self):
        self.compare_annotations(
            "Widziałem Richarda Piotra. Richard polował na kota.", "[]"
        )

    def test_proper_noun_coreference_with_gender_difference(self):
        self.compare_annotations(
            "Pracował dla Eurocash. Jest to wielkie przedsiębiorstwo. Miało wiele sukcesów na polskim rynku.",
            "[0: [2], [7], [9]]",
            alternative_expected_coref_chains="[0: [0], [4], 1: [2], [7], [9]]",
            excluded_nlps=["core_news_md"],
        )

    def test_common_noun_coreference(self):
        self.compare_annotations(
            "Widziałem dużego psa. Pies polował na kota. Swoim ogonkiem machał.",
            "[0: [2], [4], [9], [11]]",
        )

    def test_entity_coreference(self):
        self.compare_annotations(
            "Był Piotr, który wszystko wiedział. Miłym człowiekiem był.",
            "[0: [1], [8], [9]]",
            excluded_nlps=["core_news_md"],
        )

    def test_reflexive_simple(self):
        self.compare_annotations("Gepard polował na siebie", "[0: [0], [3]]")

    def test_reflexive_coordination(self):
        self.compare_annotations(
            "Gepard i leopard polowały na siebie", "[0: [0, 2], [5]]"
        )

    def test_reflexive_excluded_mix_of_coordination_and_single_member_1(self):
        self.compare_annotations(
            "Piotr i Kasia weszli. Widzieli go.", "[0: [0, 2], [5]]"
        )

    def test_reflexive_preceding_verb(self):
        self.compare_annotations(
            "Chciał, żeby jego syn lepiej sobie poradzał", "[0: [0], [3], 1: [4], [6]]"
        )

    def test_cataphora_simple_verb(self):
        self.compare_annotations(
            "Dopóki był zmęczony, Piotr został w domu.", "[0: [1], [4]]"
        )

    def test_cataphora_simple_pronoun(self):
        self.compare_annotations(
            "Pomimo, że on zmęczony był, Piotr poszedł do domu.", "[0: [3], [7]]"
        )

    def test_documentation_example(self):
        self.compare_annotations(
            "Ponieważ bardzo zajęty był swoją pracą, Janek miał jej dość. Postanowili z jego żoną, że potrzebują wakacji. Pojechali do Hiszpanii, bo bardzo im się ten kraj podobał.",
            "[0: [3], [4], [7], [12], [14], 1: [5], [9], 2: [12, 15], [18], [21], [27], 3: [23], [30]]",
            excluded_nlps=["core_news_md"],
        )

    def test_double_possessive_pronoun_within_conjunctive_coordination(self):
        self.compare_annotations(
            "Widzieli Piotra, jego żonę i jego dzieci.", "[0: [1], [3], [6]]"
        )

    def test_double_possessive_pronoun_within_comitative_coordination(self):
        self.compare_annotations(
            "Widzieli Piotra z jego żoną i jego dziećmi.", "[0: [1], [3], [6]]"
        )
