# Copyright (C) 2021 Valentin-Gabriel Soumah, 2021 msg systems ag, 
# 2021-2022 ExplosionAI GmbH

from typing import List, Set, Tuple, Optional, cast
from spacy.tokens import Token
from ...rules import RulesAnalyzer
from ...data_model import Mention
import sys
import re


class LanguageSpecificRulesAnalyzer(RulesAnalyzer):

    maximum_anaphora_sentence_referential_distance = 5

    maximum_coreferring_nouns_sentence_referential_distance = 3

    random_word = "albatros"

    dependent_sibling_deps = ("conj",)

    conjunction_deps = ("cd", "cc", "punct")

    adverbial_clause_deps = ("advcl", "advmod", "dep")

    or_lemmas = ("ou", "soit")

    entity_noun_dictionary = {
        "PER": [
            "personne",
            "homme",
            "femme",
            "garçon",
            "fille",
            "individu",
            "gars",
            "dame",
            "demoiselle",
            "garçonnet",
            "fillette",
            "monsieur",
            "madame",
            "mec",
            "meuf",
            "nana",
            "enfant",
            "père",
            "mère",
            "fils",
            "frère",
            "soeur",
            "oncle",
            "tante",
            "neveu",
            "nièce",
            "cousin",
            "ami",
            "amie",
            "mari",
            "époux",
            "épouse",
        ],
        "LOC": [
            "lieu",
            "endroit",
            "terrain",
            "secteur",
            "ville",
            "village",
            "zone",
            "site",
            "pays",
            "région",
            "département",
            "commune",
            "quartier",
            "arrondissement",
            "hammeau",
            "continent",
        ],
        "ORG": [
            "entreprise",
            "société",
            "organisation",
            "association",
            "fédération",
            "compagnie",
            "organisme",
            "établissement",
            "institution",
            "communauté",
            "groupe",
            "groupement",
        ],
    }

    quote_tuples = [
        ('"', '"'),
        ("«", "»"),
        ("‹", "›"),
        ("‘", "’"),
        ("“", "”"),
    ]

    person_titles = {
        "m.",
        "mm.",
        "monsieur",
        "messieurs",
        "mgr",
        "monseigneur",
        "président",
        "mme",
        "mmes",
        "madame",
        "mesdames",
        "mlle",
        "mlles",
        "mademoiselle",
        "mesdemoiselles",
        "vve",
        "veuve",
        "présidente",
        "docteur",
        "dr",
        "docteurs",
        "drs",
        "professeur",
        "pr",
        "professeurs",
        "prs" "maitre",
        "maître",
        "me",
        "ministre",
    }

    term_operator_pos = ("DET", "ADJ")

    term_operator_dep = ("det", "amod", "nmod", "nummod")

    clause_root_pos = ("VERB", "AUX")

    disjointed_dep = ("dislocated", "vocative", "parataxis", "discourse")

    french_word = re.compile("[\\-\\w][\\-\\w'&\\.]*$")

    def get_dependent_siblings(self, token: Token) -> List[Token]:
        def add_siblings_recursively(
            recursed_token: Token, visited_set: set
        ) -> Tuple[Set[Token], bool]:
            visited_set.add(recursed_token)
            siblings_set = set()
            coordinator = False
            if recursed_token.lemma_ in self.or_lemmas:
                token._.coref_chains.temp_has_or_coordination = True
            if recursed_token.dep_ in self.dependent_sibling_deps:
                siblings_set.add(recursed_token)
            for child in (
                child
                for child in recursed_token.children
                if child not in visited_set
                and (
                    child.dep_ in self.dependent_sibling_deps
                    or child.dep_ in self.conjunction_deps
                )
            ):
                if child.dep_ == "cc":
                    coordinator = True
                child_siblings_set, returned_coordinator = add_siblings_recursively(
                    child, visited_set
                )
                coordinator = coordinator or returned_coordinator
                siblings_set |= child_siblings_set

            return siblings_set, coordinator

        if (
            token.dep_ not in self.conjunction_deps
            and token.dep_ not in self.dependent_sibling_deps
        ):
            siblings_set, coordinator = add_siblings_recursively(token, set())
            if coordinator:
                return sorted(siblings_set)  # type:ignore[type-var]
        return []

    def is_independent_noun(self, token: Token) -> bool:
        if not self.french_word.match(token.text):
            return False
        if token.pos_ == "PROPN" and re.match("[^A-ZÂÊÎÔÛÄËÏÖÜÀÆÇÉÈŒÙ]", token.lemma_):
            return False
        if (
            token.lemma_ in {"un", "certains", "certain"}
            or self.has_morph(token, "NumType", "Card")
        ) and (
            any(
                child
                for child in token.head.children
                if (child.dep_ == "case" or child.lemma_ == "de")
                and token.i < child.i < token.head.i
            )
            or any(
                child
                for child in token.children
                if child.pos_ == "NOUN" and child.dep_ == "nmod"
            )
        ):
            # Une des filles, certains des garçons...
            pass
        elif self.is_quelqun_head(token):
            pass
        elif (
            token.pos_ not in self.noun_pos + ("ADJ", "PRON")
            or token.dep_ in ("fixed", "flat:name", "flat:foreign", "amod")
            or (token.pos_ in ("ADJ", "PRON") and not self.has_det(token))
        ):
            return False
        elif (
            token.lemma_ == "dernier"
            and any(
                self.has_morph(child, "PronType", "Dem") for child in token.children
            )
            and token.dep_ not in ("amod", "appos")
        ):
            return False
        if (
            token.i > 0
            and token.ent_type_ != ""
            and token.doc[token.i - 1].ent_type_ == token.ent_type_
            and token.doc[token.i - 1] not in token.subtree
        ):
            return False

        if (
            not self.has_det(token)
            and token.lemma_ in self.blacklisted_nouns  # type:ignore[attr-defined]
        ):
            return False
        return not self.is_token_in_one_of_phrases(
            token, self.blacklisted_phrases  # type:ignore[attr-defined]
        )

    def is_potential_anaphor(self, token: Token) -> bool:
        if not self.french_word.match(token.text):
            return False
        # Ce dernier, cette dernière...
        if (
            token.lemma_ == "dernier"
            and any(
                self.has_morph(child, "PronType", "Dem") for child in token.children
            )
            and token.dep_ not in ("amod", "appos")
        ):
            return True
        if self.is_emphatic_reflexive_anaphor(token):
            return True
        if token.lemma_ in {"celui", "celle"}:
            return True
        if token.lower_ in {"-elle"}:
            return True
        if (
            token.lower_ == "-il"
            and token.i > 0
            and token.doc[token.i - 1].lemma_ != "avoir"
            and token.dep_ != "expl:subj"
        ):
            return True
        if (
            token.pos_ == "DET" 
            and token.dep_ == "obj"
            and token.i < len(token.doc) - 1
            and token.head.i == token.i + 1
        ):
        # Covers cases of clitic pronouns wrongly tagged as DET
            return True
        if not (
            (
                token.pos_ == "PRON"
                and (
                    self.has_morph(token, "Person", "3")
                    or self.has_morph(token, "PronType", "Dem")
                )
            )
            or (token.pos_ == "ADV" and token.lemma_ in {"ici", "là"})
            or (token.pos_ == "DET" and self.has_morph(token, "Poss", "Yes"))
        ):
            return False
        if (
            token.pos_ == "DET"
            and self.has_morph(token, "Poss", "Yes")
            and token.lemma_ in {"mon", "ton", "notre", "votre"}
        ):
            return False
        # When anaphoric , the demonstrative refers almost always to a whole proposition and not a noun phrase
        if token.lemma_ in {"ce", "ça", "cela", "-ce"}:
            return False

        if token.lemma_ == "on":
            return False
        # Il y a...
        if token.text == "y" and token.dep_ == "fixed":
            return False
        if any(
            child
            for child in token.children
            if child.dep_ == "fixed" and child.lemma_ == "y"
        ):
            return False

        try:
            if (
                token.lemma_ == "là"
                and token.nbor(1).lemma_ == "bas"
                or (token.nbor(1).lemma_ == "-" and token.nbor(2).lemma_ == "bas")
            ):
                # Typically deictic
                return False
        except IndexError:
            pass
        if token.lemma_ in ("-", "ci", "-ci", "-là"):
            return False
        # Avalent Il. In case some are not marked as expletive
        inclusive_head_children = [token.head] + list(token.head.children)
        if (
            token.dep_ != self.root_dep
            and token.head.pos_ in ("AUX", "VERB")
            and any(
                [
                    1
                    for child in inclusive_head_children
                    if child.lemma_ in self.avalent_verbs  # type:ignore[attr-defined]
                ]
            )
        ):
            return False

        # impersonal constructions
        if (
            token.dep_ in {"expl:comp", "expl:pass", "expl:subj"}
            and token.lemma_ not in {"en"}
            and not self.has_morph(token, "Reflex", "Yes")
        ):
            return False

        # Il fait froid/chaud/soleil/beau
        if token.head.text.lower() == "fait" or token.head.lemma_ == "faire":
            weather_words = {
                "beau",
                "mauvais",
                "gris",
                "chaud",
                "froid",
                "doux",
                "frais",
                "nuageux",
                "orageux",
                "frisquet",
            }
            objects = [
                child
                for child in token.head.children
                if child.dep_
                in {"amod", "obj", "xcomp", "ccomp", "dep", "det", "cop", "fixed"}
            ]
            for obj in objects:
                if obj.lemma_ in weather_words:
                    return False

        if self.has_morph(token, "NumType", "Card"):
            return False

        return True

    def is_emphatic_reflexive_anaphor(self, token: Token) -> bool:
        if token.lemma_ in {"lui-même", "elle-même", "soi-même"}:
            return True
        try:
            if (
                token.nbor(1).lemma_ == "-" and token.nbor(2).lemma_ == "même"
            ) and token.lemma_.lower() in {"lui", "elle", "elles", "eux", "soi"}:
                return True
        except IndexError:
            pass
        return False

    def is_quelqun_head(self, token: Token) -> bool:
        # Special case that is analyzed differently in all the models (due to incorrect tokenisation)
        if (
            token.lemma_ == "un"
            and token.i > 0
            and token.nbor(-1).lower_ in ("quelqu'", "quelqu", "quelque")
        ):
            return True
        return False

    def has_det(self, token: Token) -> bool:
        return any(det for det in token.children if det.dep_ == "det")

    def get_gender_number_info(
        self, token: Token, directly=False, det_infos=False
    ) -> Tuple[bool, bool, bool, bool]:
        masc = fem = sing = plur = False
        if self.is_quelqun_head(token):
            sing = masc = fem = True
        elif self.has_morph(token, "Poss", "Yes") and not det_infos:
            if self.is_potential_anaphor(token):
                # the plural morphs of poss determiner don't mark the owner but the owned
                if token.lemma_ == "leur":
                    plur = True
                if token.lemma_ == "son":
                    sing = True
                masc = fem = True
        else:
            if self.has_morph(token, "Number", "Sing"):
                sing = True
            if self.has_morph(token, "Number", "Plur"):
                plur = True
            if self.has_morph(token, "Gender", "Masc"):
                masc = True
            if self.has_morph(token, "Gender", "Fem"):
                fem = True

            if token.lemma_ in {"ici", "là", "y", "en"}:
                masc = fem = sing = plur = True

            elif self.is_potential_anaphor(token):
                # object pronouns are not well recognized by the  models
                if token.lower_.startswith("lui"):
                    masc = True
                    sing = True
                elif token.lower_.startswith("eux"):
                    masc = True
                    plur = True
                elif token.lower_.startswith("elles"):
                    fem = True
                    plur = True
                elif token.lower_.startswith("elle"):
                    fem = True
                    sing = True
                elif token.lower_.startswith("soi"):
                    masc = fem = sing = plur = True

                if self.has_morph(token, "Reflex", "Yes"):
                    # se
                    if token.head.pos_ in self.clause_root_pos:
                        sing = self.has_morph(token.head, "Number", "Sing")
                        plur = self.has_morph(token.head, "Number", "Plur")
                    masc = fem = True

            elif token.pos_ == "PROPN":

                if token.lemma_ in self.male_names:  # type:ignore[attr-defined]
                    masc = True
                if token.lemma_ in self.female_names:  # type:ignore[attr-defined]
                    fem = True
                if (
                    token.lemma_
                    not in self.male_names  # type:ignore[attr-defined]
                    + self.female_names  # type:ignore[attr-defined]
                ):
                    masc = fem = True
                if not plur:
                    # proper nouns without plur mark are typically singular
                    sing = True
                if not directly and not self.has_det(token):
                    masc = fem = True

        if token.pos_ == "PRON" and token.lower_ == "le" and plur:
            # Je les vois
            masc = fem = True
        # get grammatical info from det
        if token.pos_ in self.noun_pos + ("ADJ",) and not det_infos:
            for det in token.children:
                # prevent recurs for single det phrase
                if det == token:
                    break
                if det.dep_ != "det":
                    continue
                (
                    det_masc,
                    det_fem,
                    det_sing,
                    det_plur,
                ) = self.get_gender_number_info(det, directly=directly, det_infos=True)
                # If determiner has a decisive information it trumps that of noun
                # " Especially in case of epicene nouns : e.g "la ministre"
                if any([det_sing, det_plur]):
                    sing, plur = det_sing, det_plur
                # or invariable nouns : le bras / les bras
                if any([det_fem, det_masc]):
                    fem, masc = det_fem, det_masc
                break

        if not det_infos:
            if not any([sing, plur]):
                sing = plur = True
            if not any([fem, masc]):
                fem = masc = True
        return masc, fem, sing, plur

    def refers_to_person(self, token) -> bool:

        if (
            token.ent_type_ == "PER"
            or self.is_quelqun_head(token)
            or token.lemma_.lower()
            in self.entity_noun_dictionary["PER"]
            + self.person_roles  # type:ignore[attr-defined]
        ):
            return True
        if (
            token.pos_ == self.propn_pos
            and token.lemma_
            in self.male_names + self.female_names  # type:ignore[attr-defined]
            and (
                token.ent_type_ not in ["LOC", "ORG"]
                or token.lemma_
                in ["Caroline", "Virginie", "Salvador", "Maurice", "Washington"]
            )
        ):
            return True

        if token.dep_ in ("nsubj", "nsubj:pass"):
            verb_lemma = token.head.lemma_
            if verb_lemma[-1] == "e" and verb_lemma[-2] != "r":
                # first group verbs that are not lemmatised correctly
                verb_lemma = verb_lemma + "r"
            if (
                verb_lemma
                in self.verbs_with_personal_subject  # type:ignore[attr-defined]
            ):
                return True
        return False

    def is_potential_anaphoric_pair(
        self, referred: Mention, referring: Token, directly: bool
    ) -> int:

        doc = referring.doc
        referred_root = doc[referred.root_index]
        uncertain = False

        if self.is_quelqun_head(referred_root) and referred.root_index > referring.i:
            # qqn can't be cataphoric
            return 0
        if (
            self.has_morph(referring, "Poss", "Yes")
            and referring.head == referred_root
            and referred_root.lemma_ != "personne"
        ):
            # possessive can't be determiner of its own reference
            # * mon moi-même.
            return 0
        (
            referring_masc,
            referring_fem,
            referring_sing,
            referring_plur,
        ) = self.get_gender_number_info(referring, directly=directly)
        # e.g. 'les hommes et les femmes' ... 'ils': 'ils' cannot refer only to
        # 'les hommes' or 'les femmes'
        if (
            len(referred.token_indexes) == 1
            and referring_plur
            and not referring_sing
            and self.is_involved_in_non_or_conjunction(referred_root)
            and not (
                len(referred_root._.coref_chains.temp_dependent_siblings) > 0
                and referring.i > referred.root_index
                and referring.i
                < referred_root._.coref_chains.temp_dependent_siblings[-1].i
            )
            and referring.lemma_ not in ("dernier", "celui", "celui-ci", "celui-là")
        ):
            return 0

        referred_masc = referred_fem = referred_sing = referred_plur = False

        # e.g. 'l'homme et la femme... 'il' : 'il' cannot refer to both
        if len(referred.token_indexes) > 1 and self.is_involved_in_non_or_conjunction(
            referred_root
        ):
            referred_plur = True
            referred_sing = False
            if not referring_plur:
                return 0

        for working_token in (doc[index] for index in referred.token_indexes):
            (
                working_masc,
                working_fem,
                working_sing,
                working_plur,
            ) = self.get_gender_number_info(working_token, directly=directly)
            referred_masc = referred_masc or working_masc
            referred_fem = referred_fem or working_fem
            referred_sing = referred_sing or working_sing
            referred_plur = referred_plur or working_plur

            if (
                referred_masc
                and not referred_fem
                and not referring_masc
                and referring_fem
            ):
                # "Le Masculin l'emporte" rule :
                # If there is any masc in the dependent referred, the referring has to be masc only
                return 0

        if not ((referred_masc and referring_masc) or (referred_fem and referring_fem)):
            return 0

        if not (
            (referred_plur and referring_plur) or (referred_sing and referring_sing)
        ):
            return 0

        #'ici , là... cannot refer to person. only loc and  possibly orgs
        # y needs more conditions
        if self.is_potential_anaphor(referring) and referring.lemma_ in (
            "ici",
            "là",
            "y",
        ):
            if not self.is_independent_noun(
                referred_root
            ) and referred_root.lemma_ not in ["ici", "là", "y"]:
                return 0
            if self.refers_to_person(referred_root):
                return 0
            if referred_root.ent_type_ == "ORG" and referring.lemma_ != "y":
                uncertain = True
            referred_ent_type = self.reverse_entity_noun_dictionary.get(referred_root)
            if referred_ent_type in ("PER", "ORG"):
                uncertain = True

        if directly:
            # possessive det can't be referred to directly
            if self.has_morph(referred_root, "Poss") and referred_root.pos_ == "DET":
                return False
            if self.is_potential_anaphor(referring) > 0:
                try:
                    if (
                        referring.lemma_ == "celui-ci"
                        or referring.lemma_.lower() == "dernier"
                        or (
                            referring.lemma_.lower() == "celui"
                            and (
                                referring.nbor(1).lemma_.lower() in ("-ci", "ci")
                                or (
                                    referring.nbor(1).text == "-"
                                    and referring.nbor(2).lemma_.lower() == "ci"
                                )
                            )
                        )
                    ):
                        #'celui-ci' and 'ce dernier' can only refer to last grammatically compatible noun phrase
                        if referring.i == 0:
                            return 0
                        for previous_token_index in range(referring.i - 1, 0, -1):
                            previous_token = doc[previous_token_index]
                            if self.is_independent_noun(
                                previous_token
                            ) and self.is_potential_anaphoric_pair(
                                Mention(previous_token), referring, directly=False
                            ):
                                if previous_token_index != referred.root_index:
                                    if previous_token.dep_ in ("nmod", "appos"):
                                        continue
                                    # Except if noun phrase is modifier of other noun phrase
                                    # eg: "Le président du pays... ce dernier" can refer to both nouns
                                    return 0
                                break

                    if (
                        referring.lemma_ == "celui"
                        and len(doc) >= referring.i + 1
                        and referring.nbor(1).lemma_.lower() in ("-là", "là")
                    ):
                        #'celui-là' refers to second to last noun phrase or before (but not too far)
                        noun_phrase_count = 0
                        if referring.i == 0:
                            return 0
                        for previous_token_index in range(referring.i - 1, 0, -1):

                            if (
                                self.is_independent_noun(doc[previous_token_index])
                                and previous_token_index in referred.token_indexes
                            ):
                                if noun_phrase_count < 1:
                                    return 0
                            elif self.is_independent_noun(doc[previous_token_index]):
                                noun_phrase_count += 1
                            if noun_phrase_count > 2:
                                return 0
                except IndexError:
                    # doc shorter than the compared index
                    pass
                if referring.lemma_ == "en":
                    # requires list of mass/countable nouns to be implemented
                    if not referred_plur and (self.refers_to_person(referred_root)):
                        uncertain = True

            if (
                referring.pos_ == "PRON"
                and self.has_morph(referring, "Person", "3")
                and self.has_morph(referring, "Number")
                and not self.refers_to_person(referred_root)
            ):
                # Some semantic restrictions on named entities / pronoun pair
                if (
                    referred_root.ent_type_ == "ORG"
                    and referred_root.pos_ in self.propn_pos
                    and not self.has_det(referred_root)
                    and not any(
                        prep for prep in referred_root.children if prep.dep_ == "case"
                    )
                ):
                    # "Twitter ... Il " is not possible
                    return False
                if (
                    referred_root.ent_type_ in {"LOC", "MISC"}
                    and referred_root.pos_ in self.propn_pos
                    and not self.has_det(referred_root)
                    and not any(
                        prep for prep in referred_root.children if prep.dep_ == "case"
                    )
                ):
                    # "Paris... elle" is possible but unlikely
                    # Except for cases when the toponym has a determiner, such as most country name
                    # "La France...elle" is ok. Same for cities with det : "Le Havre... il"
                    uncertain = True

            if (
                self.is_potential_reflexive_pair(referred, referring)
                and self.is_reflexive_anaphor(referring) == 0
                and not self.has_morph(referred_root, "Poss", "Yes")
                and referred_root.dep_ != "obl:mod"
            ):
                # * Les hommes le voyaient. "le" can't refer to "hommes"
                return 0

            if self.is_potential_reflexive_pair(referred, referring) == 0 and (
                self.is_reflexive_anaphor(referring) == 2
            ):
                # * Les hommes étaient sûrs qu'ils se trompaient. "se" can't directly refer to "hommes"
                return 0

        if self.refers_to_person(referring) and not self.refers_to_person(
            referred_root
        ):
            # Le Luxembourg... Il mange ... -> impossible
            if referred_root.ent_type_ in {"ORG", "LOC", "MISC"}:
                return False
            # Le Balcon... il mange... -> impossible but some other nouns are dubious
            if referred_root.pos_ == "NOUN":
                uncertain = True

        referring_governing_sibling = referring
        if referring._.coref_chains.temp_governing_sibling is not None:
            referring_governing_sibling = (
                referring._.coref_chains.temp_governing_sibling
            )
        if (
            referring_governing_sibling.dep_ in ("nsubj:pass", "nsubj")
            and referring_governing_sibling.head.lemma_
            in self.verbs_with_personal_subject  # type:ignore[attr-defined]
        ):
            for working_token in (doc[index] for index in referred.token_indexes):
                if self.refers_to_person(working_token):
                    return 2
            if referred_root.pos == "NOUN":
                uncertain = True

        return 1 if uncertain else 2

    def has_operator_child_with_any_morph(self, token: Token, morphs: dict):
        for child in (
            child
            for child in token.children
            if child.pos_ in self.term_operator_pos + ("ADP",)
        ):
            for morph in morphs:
                if self.has_morph(child, morph, morphs.get(morph)):
                    return True
        return False

    def is_potentially_indefinite(self, token: Token) -> bool:
        return self.has_operator_child_with_any_morph(
            token, {"Definite": "Ind"}
        ) or self.is_quelqun_head(token)

    def is_potentially_definite(self, token: Token) -> bool:

        return self.has_operator_child_with_any_morph(
            token, {"Definite": "Def", "PronType": "Dem"}
        )

    def is_reflexive_anaphor(self, token: Token) -> int:
        if (
            token.lemma_ == "personne"
            and len(
                [
                    det
                    for det in token.children
                    if det.pos_ == "DET"
                    and self.has_morph(det, "Poss", "Yes")
                    and self.has_morph(det, "Person", "3")
                ]
            )
            > 0
        ):
            # sa personne...
            return 2
        if self.is_emphatic_reflexive_anaphor(token):
            return 2
        if self.has_morph(token, "Reflex", "Yes"):
            if self.has_morph(token, "Person", "3"):
                return 2

        return 0

    @staticmethod
    def get_ancestor_spanning_any_preposition(token: Token) -> Optional[Token]:
        if token.dep_ == "ROOT":
            return None
        head = token.head
        return head

    def is_potential_reflexive_pair(self, referred: Mention, referring: Token) -> bool:
        if (
            referring.pos_ != "PRON"
            and not self.is_emphatic_reflexive_anaphor(referring)
            and not (
                referring.pos_ == "DET"
                and referring.dep_ == "obj"
                and referring.lemma_ == "le"
            )
            and referring.lemma_ != "personne"
        ):
            return False

        if referring.dep_ in self.disjointed_dep:
            return False

        referred_root = referring.doc[referred.root_index]

        if referred_root._.coref_chains.temp_governing_sibling is not None:
            referred_root = referred_root._.coref_chains.temp_governing_sibling

        if referring._.coref_chains.temp_governing_sibling is not None:
            referring = referring._.coref_chains.temp_governing_sibling

        if referred_root.dep_ in ("nsubj", "nsubj:pass") and not any(
            selon
            for selon in referring.children
            if selon.lemma_ == "selon" and selon.dep_ == "case"
        ):

            for referring_ancestor in referring.ancestors:
                # Loop up through the verb ancestors of the pronoun
                if referring_ancestor.dep_ in self.disjointed_dep:
                    return False
                if referred_root in referring_ancestor.children:
                    return True
                # Relative clauses
                if (
                    referring_ancestor.pos_ in ("VERB", "AUX")
                    and referring_ancestor.dep_ in ("acl:relcl", "acl")
                    and (
                        referring_ancestor.head == referred_root
                        or referring_ancestor.head.i in referred.token_indexes
                    )
                ):
                    return True

                # The ancestor has its own subject, so stop here
                subjects = [
                    t
                    for t in referring_ancestor.children
                    if t.dep_ in ("nsubj", "nsubj:pass")
                ]
                if any(subjects) and referred_root not in subjects:
                    return False
            return False

        if referring.i < referred_root.i:
            return False

        referring_ancestor = cast(
            Token, self.get_ancestor_spanning_any_preposition(referring)
        )
        referred_ancestor = referred_root.head
        return referring_ancestor is not None and (
            referring_ancestor == referred_ancestor
            or referring_ancestor.i in referred.token_indexes
        )

    # Methods from the parent class that need to be overridden because
    # some cases are not suitable for the french parse tree
    def is_potential_cataphoric_pair(self, referred: Mention, referring: Token) -> bool:
        """Checks whether *referring* can refer cataphorically to *referred*, i.e.
        where *referring* precedes *referred* in the text. That *referring* precedes
        *referred* is not itself checked by the method.

        Overrides the method of the parent class which is not suitable for be clause in french
        """

        doc = referring.doc
        referred_root = doc[referred.root_index]

        if referred_root.sent != referring.sent:
            return False
        if self.is_potential_anaphor(referred_root):
            return False

        referred_verb_ancestors = []
        # Find the ancestors of the referent that are verbs, stopping anywhere where there
        # is conjunction between verbs
        for ancestor in referred_root.ancestors:
            if ancestor.pos_ in self.clause_root_pos or any(
                child for child in ancestor.children if child.dep_ == "cop"
            ):
                referred_verb_ancestors.append(ancestor)
            if ancestor.dep_ in self.dependent_sibling_deps:
                break

        # Loop through the ancestors of the referring pronoun that are verbs,  that are not
        # within the first list and that have an adverbial clause dependency label
        referring_inclusive_ancestors = [referring]
        referring_inclusive_ancestors.extend(referring.ancestors)
        if (
            len(
                [
                    1
                    for ancestor in referring_inclusive_ancestors
                    if ancestor.dep_ in self.adverbial_clause_deps
                ]
            )
            == 0
        ):
            return False
        for referring_verb_ancestor in (
            t
            for t in referring_inclusive_ancestors
            if t not in referred_verb_ancestors
            and t.dep_ in self.adverbial_clause_deps
            and t.pos_ in self.clause_root_pos + self.noun_pos + ("ADJ",)
        ):
            # If one of the elements of the second list has one of the elements of the first list
            # within its ancestors, we have subordination and cataphora is permissible
            if (
                len(
                    [
                        t
                        for t in referring_verb_ancestor.ancestors
                        if t in referred_verb_ancestors
                    ]
                )
                > 0
            ):
                return True
        return False

    def get_propn_subtree(self, token: Token) -> list:
        """Returns a list containing each member M of the subtree of *token* that are proper nouns
        and where all the tokens between M and *token* are themselves proper nouns. If *token*
        is itself not a proper noun or if the head of *token* is a proper noun, an empty list
        is returned.
        """
        """"Has to be edited for french as the titles are parsed as heads of the propn 
        (and are those titles also included in named entities)
        """

        def is_propn_part(token: Token) -> bool:
            if (
                token.lemma_.lower() not in self.person_titles
                and token.text[0].upper() != token.text[0]
                and re.search("\\W", token.text)
            ):
                return False
            return token.pos_ in self.propn_pos or (
                token.lemma_.lower() in self.person_titles
                and token.pos_ in self.noun_pos
            )

        if not is_propn_part(token):
            return []
        if (
            token.dep_ != self.root_dep
            and token.dep_ not in self.dependent_sibling_deps
            and is_propn_part(token.head)
        ):
            return []
        subtree = list(token.subtree)
        before_start_index = -1
        after_end_index = sys.maxsize
        for subtoken in subtree:
            if (
                not is_propn_part(subtoken)
                and subtoken.i < token.i
                and before_start_index < subtoken.i
            ):
                before_start_index = subtoken.i
            elif (
                not is_propn_part(subtoken)
                and subtoken.i > token.i
                and after_end_index > subtoken.i
            ):
                after_end_index = subtoken.i
        return [
            t for t in subtree if t.i > before_start_index and t.i < after_end_index
        ]

    def is_potentially_referring_back_noun(self, token: Token) -> bool:

        if (
            self.is_potentially_definite(token)
            and len(
                [
                    1
                    for c in token.children
                    if c.pos_ not in self.term_operator_pos
                    and c.dep_ not in self.conjunction_deps
                    and c.dep_ not in self.dependent_sibling_deps
                    and c.dep_ not in self.term_operator_dep
                ]
            )
            == 0
        ):
            return True

        return (
            token._.coref_chains.temp_governing_sibling is not None
            and len(
                [
                    1
                    for c in token.children
                    if c.dep_ not in self.conjunction_deps
                    and c.dep_ not in self.dependent_sibling_deps
                ]
            )
            == 0
            and self.is_potentially_referring_back_noun(
                token._.coref_chains.temp_governing_sibling
            )
        )

    def get_noun_core_lemma(self, token):
        prefix = re.compile("^((vice)|(^ex)|(^co))-")
        return prefix.sub("", token.lemma_).lower()

    def is_grammatically_compatible_noun_pair(self, referred: Token, referring: Token):
        (
            referred_masc,
            referred_fem,
            referred_sing,
            referred_plur,
        ) = self.get_gender_number_info(referred, directly=True)
        (
            referring_masc,
            referring_fem,
            referring_sing,
            referring_plur,
        ) = self.get_gender_number_info(referring, directly=True)

        if not (
            (referred_plur and referring_plur) or (referred_sing and referring_sing)
        ) and not (
            referred.ent_type_ == "LOC"
            and referred.lemma_.upper()
            in self.plural_toponyms  # type:ignore[attr-defined]
        ):
            # two nouns with different numbers can't corefer. This is true for substantives and propn alike
            return False

        if (
            referred.ent_type_ == "PER"
            or self.get_noun_core_lemma(referred) in self.person_titles
        ) and not (
            referring.pos_ == "NOUN"
            and self.get_noun_core_lemma(referring)
            in self.mixed_gender_person_roles  # type:ignore[attr-defined]
        ):
            # Gender compatibility is only ensured for person and their roles
            # And only when the role does not allow mixed gender
            # "Sophie... l'auteur du livre'" is possible
            # "Sophie... l'instituteur'" is impossible
            if not (
                (referred_masc and referring_masc) or (referred_fem and referring_fem)
            ):
                return False
        if (
            self.has_morph(referring, "Gender", "Masc")
            and referring_fem
            and not referred_fem
        ):
            # when fem gender is enforced by det
            # eg : la juge
            return False
        return True

    def is_potential_coreferring_pair_with_substantive(
        self, referred: Token, referring: Token
    ) -> bool:
        """
        Returns True if  pragmatical rules of the language
        allow the two nouns to corefer
        """
        # Nouns can't corefer in same predication
        verb_referred_ancestors = [
            t
            for t in referred.ancestors
            if t.dep_ == "ROOT" or t.pos_ in self.clause_root_pos
        ]
        verb_referring_ancestors = [
            t
            for t in referring.ancestors
            if t.dep_ == "ROOT" or t.pos_ in self.clause_root_pos
        ]
        referred_verb_parent = (
            verb_referred_ancestors[0] if verb_referred_ancestors else referred
        )
        referring_verb_parent = (
            verb_referring_ancestors[0] if verb_referring_ancestors else referring
        )
        # Covers cases of unrecognised appos
        if referred_verb_parent == referring_verb_parent and referring.dep_ != "xcomp":
            return False

        for appos_token in [referred, referring]:
            # Prevents any non Propn from appos chain from connecting to other nouns
            # That way we ensure that only the propn will be linked to the bigger chains
            # E.g : "Justin Trudeau.... Le Président, Donald Trump".
            # We don't want "president" to be able to be linked to "Justin"
            appos_children = [c for c in appos_token.children if c.dep_ == "appos"]
            if (
                any(
                    1
                    for propn in appos_children
                    if propn.pos_ == "PROPN" or propn.ent_type_
                )
                and appos_token.pos_ != "PROPN"
                and not appos_token.ent_type_
            ):
                return False

            if (
                appos_token.pos_ != "PROPN"
                and appos_token.dep_ == "appos"
                and not appos_token.ent_type_
                and (appos_token.head.pos_ == "PROPN" or appos_token.ent_type_)
            ):
                return False
        return True

    def language_dependent_is_coreferring_noun_pair(
        self, referred: Token, referring: Token
    ) -> bool:
        """
        Return True if language rules make it necessary
        for the two noun phrases to corever
        """
        # Apposition chains
        if referred == referring.head and referring.dep_ == "appos":
            return True
        if (
            referred == referring.head.head
            and referring.dep_ == "appos"
            and referring.head.dep_ == "appos"
        ):
            return True

        # Cases of apposition wrongly tagged as conj
        if (
            referred == referring.head
            and referring.dep_ == "conj"
            and self.is_involved_in_non_or_conjunction(referring)
            and referred.dep_ in ("nsubj", "nsubj:pass")
            and referred.head.pos_ in ("VERB", "AUX")
        ):

            *_, referred_sing, referred_plur = self.get_gender_number_info(referred)
            if (
                referred_sing
                and not referred_plur
                and self.has_morph(referred.head, "Number", "Sing")
            ):
                return True
        # Other cases of apposition
        if referring not in referred._.coref_chains.temp_dependent_siblings:
            referred_right_in_subtree = list(referred.subtree)[-1]
            referring_left_in_subtree = list(referring.subtree)[0]
            if (
                referring_left_in_subtree.i - referred_right_in_subtree.i == 2
                and referred.doc[referred_right_in_subtree.i + 1].text == ","
            ):
                return True
        # Copular structures
        if (
            referring == referred.head
            and any(
                cop
                for cop in referring.children
                if cop.dep_ == "cop" and cop.lemma_ == "être"
            )
            and not any(prep for prep in referring.children if prep.tag_ == "ADP")
        ):
            return True

        # state verbs
        stative_verbs = ["devenir", "rester", "demeurer"]
        if (
            referred.dep_ in ("nsubj", "nsubj:pass")
            and referring.dep_ == "obj"
            and referring.head == referred.head
            and referring.head.lemma_ in stative_verbs
        ):
            return True
        return False

    def is_potential_coreferring_noun_pair(
        self, referred: Token, referring: Token
    ) -> bool:
        """Returns *True* if *referred* and *referring* are potentially coreferring nouns.
        The method presumes that *is_independent_noun(token)* has
        already returned *True* for both *referred* and *referring* and that
        *referred* precedes *referring* within the document.
        """
        if len(referred.text) == 1 and len(referring.text) == 1:
            return False  # get rid of copyright signs etc.

        if (referred.pos_ not in self.noun_pos and not self.has_det(referred)) or (
            referring.pos_ not in self.noun_pos and not self.has_det(referring)
        ):
            return False
        grammatically_compatible = self.is_grammatically_compatible_noun_pair(
            referred, referring
        )
        # Needs to be here as it covers cases of incorrect parsing
        if (
            self.language_dependent_is_coreferring_noun_pair(referred, referring)
            and grammatically_compatible
        ):
            return True

        if referring in referred._.coref_chains.temp_dependent_siblings:
            return False

        if (
            referring._.coref_chains.temp_governing_sibling is not None
            and referring._.coref_chains.temp_governing_sibling
            == referred._.coref_chains.temp_governing_sibling
        ):
            return False

        # If *referred* and *referring* are names that potentially consist of several words,
        # the text of *referring* must correspond to the end of the text of *referred*
        # e.g. 'Richard Paul Hudson' -> 'Hudson'
        referred_propn_subtree = self.get_propn_subtree(referred)
        if referring in referred_propn_subtree:
            return False
        if len(referred_propn_subtree) > 0:
            referring_propn_subtree = self.get_propn_subtree(referring)
            if len(referring_propn_subtree) > 0 and " ".join(
                t.text for t in referred_propn_subtree
            ).endswith(" ".join(t.text for t in referring_propn_subtree)):
                return True
            if len(referring_propn_subtree) > 0 and " ".join(
                t.lemma_.lower() for t in referred_propn_subtree
            ).endswith(" ".join(t.lemma_.lower() for t in referring_propn_subtree)):
                return True

        if not self.is_potential_coreferring_pair_with_substantive(referred, referring):
            return False
        # e.g. 'Peugeot' -> 'l'entreprise'
        new_reverse_entity_noun_dictionary = {
            noun: "PER" for noun in self.person_roles  # type:ignore[attr-defined]
        } | self.reverse_entity_noun_dictionary

        if (
            self.get_noun_core_lemma(referring) in new_reverse_entity_noun_dictionary
            and self.is_potentially_definite(referring)
            and (
                (
                    referred.ent_type_
                    == new_reverse_entity_noun_dictionary[
                        self.get_noun_core_lemma(referring)
                    ]
                )
                or (
                    new_reverse_entity_noun_dictionary[
                        self.get_noun_core_lemma(referring)
                    ]
                    == "PER"
                    and referred.ent_type_
                    and self.refers_to_person(referred)
                )
            )
            and grammatically_compatible
            and not (referring.ent_type_ != "" and referring.pos_ != "PROPN")
        ):
            return True

        if not self.is_potentially_referring_back_noun(referring):
            return False
        if not self.is_potentially_introducing_noun(
            referred
        ) and not self.is_potentially_referring_back_noun(referred):
            return False
        if self.get_noun_core_lemma(referred) == self.get_noun_core_lemma(
            referring
        ) and referred.morph.get(self.number_morph_key) == referring.morph.get(
            self.number_morph_key
        ):
            return True
        return False
