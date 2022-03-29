import unittest
import os
import spacy
import coreferee
from coreferee.errors import *

class ErrorsTest(unittest.TestCase):

    def test_model_not_supported(self):
        with self.assertRaises(ModelNotSupportedError) as context:
            nlp = spacy.load('de_dep_news_trf')
            nlp.add_pipe('coreferee')
