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
from coreferee.test_utils import get_nlps

class FrenchSmokeTest(unittest.TestCase):

    def setUp(self):

        self.nlps = get_nlps('fr')

    def all_nlps(self, func):
        for nlp in self.nlps:
            func(nlp)

    def compare_annotations(self, doc_text, expected_coref_chains, *, excluded_nlps=[],
        alternative_expected_coref_chains=None):

        def func(nlp):
            sent = 'La femme se leva et regarda Dominique. Elle se tourna et le regarda'
            if nlp.meta['name'] in excluded_nlps:
                return

            doc = nlp(doc_text)
            chains_representation = str(doc._.coref_chains)
            if alternative_expected_coref_chains is None:
                self.assertEqual(expected_coref_chains,
                    chains_representation, nlp.meta['name'])
            else:
                self.assertTrue(expected_coref_chains == chains_representation or
                    alternative_expected_coref_chains == chains_representation, nlp.meta['name'])

        self.all_nlps(func)

    def test_simple(self):
        self.compare_annotations('J\'ai vu un chien et il chassait un chat', '[0: [4], [6]]')

    def test_simple_plural(self):
        self.compare_annotations('J\'ai vu des chiens et ils chassaient des chats', '[0: [4], [6]]')

    def test_simple_conjunction_same_word(self):
        self.compare_annotations(
            'J\'ai vu un chien et un chien et ils chassaient des chats', '[0: [4, 7], [9]]')

    def test_simple_conjunction_different_words(self):
        self.compare_annotations(
            'J\'ai vu un chien et un cheval et ils chassaient un chat', '[0: [4, 7], [9]]',
            excluded_nlps=['core_news_sm'])

    def test_independent_propositions_different_pronouns(self):
        self.compare_annotations(
            'J\'ai vu Jacques et Julie ; elle et lui chassaient un chat', '[0: [3], [9], 1: [5], [7]]',
            excluded_nlps=['core_news_sm'])
            
    def test_conjunction_involving_pronoun(self):
        self.compare_annotations(
            'Je voyais Jacques et Julie. Elle et Richard chassaient un chat', '[0: [4], [6]]')

    def test_different_sentence(self):
        self.compare_annotations(
            'Je voyais Jacques. Il chassait un chat.', '[0: [2], [4]]',
            excluded_nlps="core_news_md")

    def test_proper_noun_coreference(self):
        self.compare_annotations(
            'Je voyais Jean. Jean chassait un chat.', '[0: [2], [4]]')

    def test_proper_noun_coreference_multiword(self):
        self.compare_annotations(
            'Je voyais Jacques Martin. Jacques Martin chassait un chat.', '[0: [2], [5]]')

    def test_proper_noun_coreference_multiword_only_second_repeated(self):
        self.compare_annotations(
            'Je voyais Jacques Martin. Martin chassait un chat', '[0: [2], [5]]')

    def test_proper_noun_coreference_multiword_only_first_repeated(self):
        self.compare_annotations(
            'Je voyais Jacques Martin. Jacques chassait un chat.', '[]')

    def test_proper_noun_coreference_with_gender_difference(self):
        self.compare_annotations(
            'Charles aime La France. Le pays a Paris pour capitale. On peut y voir la Tour Eiffel',
            '[0: [3], [6], [14]]',
            excluded_nlps='core_news_sm',#,
            alternative_expected_coref_chains='[0: [3], [6], 1: [8], [14]]')
            
    def test_common_noun_coreference(self):
        self.compare_annotations(
            'Je voyais un gros chien. Le chien chassait un chat. Il remuait sa queue',
            '[0: [4], [7], [12], [14]]')

    def test_entity_coreference(self):
        self.compare_annotations(
            'Je voyais Jeanne Dupont. Tout le monde aimait cette femme aimable.',
            '[0: [2], [10]]', excluded_nlps='core_news_sm')
            
    def test_reflexive_simple(self):
        self.compare_annotations(
            'La panthère se chassait',
            '[0: [1], [2]]')

    def test_reflexive_doubled(self):
        self.compare_annotations(
            'La panthère se chassait elle-même',
            '[0: [1], [2], [4]]',
            excluded_nlps='core_news_sm')

    def test_reflexive_coordination(self):
        self.compare_annotations(
            'La panthère et le léopard se chassaient',
            '[0: [1, 4], [5]]',
            excluded_nlps=['core_news_md','core_news_sm'])

    def test_reflexive_excluded_mix_of_coordination_and_single_member_1(self):
        self.compare_annotations(
            'Jacques et Julie entrèrent. Ils le virent.',
            '[0: [0, 2], [5]]')
            
    def test_reflexive_excluded_mix_of_coordination_and_single_member_2(self):
        self.compare_annotations(
            'Jacques et Julie entrèrent. Ils les virent.',
            '[0: [0, 2], [5]]')


    def test_reflexive_anaphor_precedes_referent(self):
        self.compare_annotations(
            'On discuta de soi-même et Jacques entra.',
            '[]')

    def test_cataphora_simple(self):
        self.compare_annotations(
            'Bien qu\'il était enervé, Jacques rentra dans le métro',
            '[0: [2], [6]]')

    def test_cataphora_with_coordination(self):
        self.compare_annotations(
            'Alors qu\'ils partaient, l\'homme et la femme étaient tristes',
            '[0: [2], [6, 9]]')


    def test_possessive_pronoun_within_threeway_coordination(self):
        self.compare_annotations(
            'Nous vîment Jacques, ses amis et son chien.',
            '[0: [2], [4], [7]]')

    def test_crossed_demonstrative_anaphors(self):
        self.compare_annotations(
            'J\'admire les colibris et les pigeons. Ceux-ci sont plus gros que ceux-là',
            '[0: [3], [15], 1: [6], [8]]', excluded_nlps='core_news_md'
            )
        
    def test_proadverb_location(self):
        self.compare_annotations(
            'Claire a acheté une nouvelle maison. C\'est là qu\'on ira manger demain avec elle et son mari.',
            '[0: [0], [16], [18], 1: [5], [9]]')
            
    def test_reflexive_noun(self):
        self.compare_annotations(
            'Il est passioné par la rotation de la Terre sur elle-même',
            '[0: [8], [10]]', excluded_nlps='core_news_md'
            )

    def test_relative_clause(self):
        self.compare_annotations(
            "L'homme et la femme qu'ils voyaient, sont rentrés chez eux.",
            '[0: [1, 4], [12]]',
            alternative_expected_coref_chains='[0: [6], [12]]'
            )
        
    def test_masc_over_fem_coordination(self):
        self.compare_annotations(
            "Les australiennes admirent la giraffe et l'hippopotame. Elles boient beaucoup.",
            '[0: [1], [9]]', excluded_nlps='core_news_sm',
            )
            
    def test_documentation_example_1(self):
        self.compare_annotations(
            'Même si elle était très occupée par son travail, Julie en avait marre. Alors, elle et son mari décidèrent qu\'ils avaient besoin de vacances. Ils allèrent en Espagne car ils adoraient le pays',
            '[0: [2], [7], [10], [17], [19], 1: [8], [11], 2: [17, 20], [23], [29], [34], 3: [32], [37]]',
            excluded_nlps = ['core_news_sm']
        )
     
    def test_documentation_example_2(self):
        self.compare_annotations(
            'La femme se leva et regarda Dominique. Elle se tourna et la salua',
            '[0: [1], [2], [12], 1: [6], [8], [9]]',
            excluded_nlps=['core_news_md', 'core_news_sm'],
            alternative_expected_coref_chains='[0: [1], [2], [8], [9], 1: [6] [12]')

    def test_documentation_example_3(self):
        self.compare_annotations(
            'La femme se leva et regarda Dominique. Elle se tourna et le regarda',
            '[0: [1], [2], [8], [9], 1: [6], [12]]',
            alternative_expected_coref_chains='[0: [1], [2], 1: [6], [8], [9]]',
            excluded_nlps=['core_news_sm'])

    def test_documentation_example_4(self):
        self.compare_annotations(
            'Marc et Léa étaient en Espagne. Ils adorèrent le pays et prévoient d\'y retourner l\'an prochain avec leurs parents.',
            '[0: [0, 2], [7], [20], 1: [5], [10], [14]]',
            excluded_nlps=['core_news_md','core_news_sm']
        )
