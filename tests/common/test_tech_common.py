import unittest
import os
from multiprocessing import Process, Manager, Queue as m_Queue
from queue import Queue
from threading import Thread
import spacy
from spacy.tokens import Doc
from thinc.util import prefer_gpu, require_cpu
from coreferee.test_utils import get_nlps

NUMBER_OF_THREADS = 50
NUMBER_OF_PROCESSES = 2

class Worker:

    def listen(self, input_queue):
        while True:
            output_queue, doc = input_queue.get()
            first = str(doc._.coref_chains)
            second = str(doc[0]._.coref_chains)
            third = str(doc[2]._.coref_chains)
            fourth = str(doc[3]._.coref_chains)
            returned_number = doc[-1].text
            output_queue.put((first, second, third, fourth, returned_number))


class CommonUtilsTest(unittest.TestCase):

    def setUp(self):
        nlps = get_nlps('en')
        for nlp in (nlp for nlp in nlps if nlp.meta['name'] == 'core_web_sm'):
            self.sm_nlp = nlp

    def test_serialization_with_scoring(self):
        doc = self.sm_nlp('Peter told Paul he was dissatisfied.')
        self.assertEqual('[0: [0], [3]]', str(doc._.coref_chains))
        self.assertEqual('[0: [0], [3]]', str(doc[0]._.coref_chains))
        self.assertEqual('[]', str(doc[2]._.coref_chains))
        self.assertEqual('[0: [0], [3]]', str(doc[3]._.coref_chains))
        b = doc.to_bytes()
        doc = None
        doc2 = Doc(self.sm_nlp.vocab).from_bytes(b)
        self.assertEqual('[0: [0], [3]]', str(doc2._.coref_chains))
        self.assertEqual('[0: [0], [3]]', str(doc2[0]._.coref_chains))
        self.assertEqual('[]', str(doc2[2]._.coref_chains))
        self.assertEqual('[0: [0], [3]]', str(doc2[3]._.coref_chains))
        self.assertEqual('0: Peter(0), he(3)', doc2[3]._.coref_chains.pretty_representation)
        self.assertEqual(0, doc2._.coref_chains[0].most_specific_mention_index)
        self.assertEqual([doc2[0]], doc2._.coref_chains.resolve(doc2[3]))

    def test_serialization_gpu_to_cpu(self):
        prefer_gpu()
        doc = self.sm_nlp('Peter told Paul he was dissatisfied.')
        self.assertEqual('[0: [0], [3]]', str(doc._.coref_chains))
        self.assertEqual('[0: [0], [3]]', str(doc[0]._.coref_chains))
        self.assertEqual('[]', str(doc[2]._.coref_chains))
        self.assertEqual('[0: [0], [3]]', str(doc[3]._.coref_chains))
        b = doc.to_bytes()
        doc = None
        require_cpu()
        doc2 = Doc(self.sm_nlp.vocab).from_bytes(b)
        self.assertEqual('[0: [0], [3]]', str(doc2._.coref_chains))
        self.assertEqual('[0: [0], [3]]', str(doc2[0]._.coref_chains))
        self.assertEqual('[]', str(doc2[2]._.coref_chains))
        self.assertEqual('[0: [0], [3]]', str(doc2[3]._.coref_chains))
        self.assertEqual('0: Peter(0), he(3)', doc2[3]._.coref_chains.pretty_representation)
        self.assertEqual(0, doc2._.coref_chains[0].most_specific_mention_index)
        self.assertEqual([doc2[0]], doc2._.coref_chains.resolve(doc2[3]))

    def test_serialization_cpu_to_gpu(self):
        require_cpu()
        doc = self.sm_nlp('Peter told Paul he was dissatisfied.')
        self.assertEqual('[0: [0], [3]]', str(doc._.coref_chains))
        self.assertEqual('[0: [0], [3]]', str(doc[0]._.coref_chains))
        self.assertEqual('[]', str(doc[2]._.coref_chains))
        self.assertEqual('[0: [0], [3]]', str(doc[3]._.coref_chains))
        b = doc.to_bytes()
        doc = None
        prefer_gpu()
        doc2 = Doc(self.sm_nlp.vocab).from_bytes(b)
        self.assertEqual('[0: [0], [3]]', str(doc2._.coref_chains))
        self.assertEqual('[0: [0], [3]]', str(doc2[0]._.coref_chains))
        self.assertEqual('[]', str(doc2[2]._.coref_chains))
        self.assertEqual('[0: [0], [3]]', str(doc2[3]._.coref_chains))
        self.assertEqual('0: Peter(0), he(3)', doc2[3]._.coref_chains.pretty_representation)
        self.assertEqual(0, doc2._.coref_chains[0].most_specific_mention_index)
        self.assertEqual([doc2[0]], doc2._.coref_chains.resolve(doc2[3]))
        require_cpu()

    def test_serialization_without_scoring(self):
        doc = self.sm_nlp('Peter said he was dissatisfied.')
        self.assertEqual('[0: [0], [2]]', str(doc._.coref_chains))
        self.assertEqual('[0: [0], [2]]', str(doc[0]._.coref_chains))
        self.assertEqual('[]', str(doc[1]._.coref_chains))
        self.assertEqual('[0: [0], [2]]', str(doc[2]._.coref_chains))
        b = doc.to_bytes()
        doc = None
        doc2 = Doc(self.sm_nlp.vocab).from_bytes(b)
        self.assertEqual('[0: [0], [2]]', str(doc2._.coref_chains))
        self.assertEqual('[0: [0], [2]]', str(doc2[0]._.coref_chains))
        self.assertEqual('[]', str(doc2[1]._.coref_chains))
        self.assertEqual('[0: [0], [2]]', str(doc2[2]._.coref_chains))
        self.assertEqual('0: Peter(0), he(2)', doc2[2]._.coref_chains.pretty_representation)
        self.assertEqual(0, doc2._.coref_chains[0].most_specific_mention_index)
        self.assertEqual([doc2[0]], doc2._.coref_chains.resolve(doc2[2]))

    def test_processing_in_pipe_1_cpu(self):
        doc_texts = (['Peter told Paul he was dissatisfied.', 'Peter said he was dissatisfied'])
        docs = list(self.sm_nlp.pipe(doc_texts))
        self.assertEqual('[0: [0], [3]]', str(docs[0]._.coref_chains))
        self.assertEqual('[0: [0], [3]]', str(docs[0][0]._.coref_chains))
        self.assertEqual('[]', str(docs[0][2]._.coref_chains))
        self.assertEqual('[0: [0], [3]]', str(docs[0][3]._.coref_chains))
        self.assertEqual('[0: [0], [2]]', str(docs[1]._.coref_chains))
        self.assertEqual('[0: [0], [2]]', str(docs[1][0]._.coref_chains))
        self.assertEqual('[]', str(docs[1][1]._.coref_chains))
        self.assertEqual('[0: [0], [2]]', str(docs[1][2]._.coref_chains))

    def test_processing_in_pipe_2_cpu(self):
        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe('coreferee')
        doc_texts = (['Peter told Paul he was dissatisfied.', 'Peter said he was dissatisfied'])
        docs = list(nlp.pipe(doc_texts, n_process=2))
        self.assertEqual('[0: [0], [3]]', str(docs[0]._.coref_chains))
        self.assertEqual('[0: [0], [3]]', str(docs[0][0]._.coref_chains))
        self.assertEqual('[]', str(docs[0][2]._.coref_chains))
        self.assertEqual('[0: [0], [3]]', str(docs[0][3]._.coref_chains))
        self.assertEqual('[0: [0], [2]]', str(docs[1]._.coref_chains))
        self.assertEqual('[0: [0], [2]]', str(docs[1][0]._.coref_chains))
        self.assertEqual('[]', str(docs[1][1]._.coref_chains))
        self.assertEqual('[0: [0], [2]]', str(docs[1][2]._.coref_chains))

    def test_use_in_multithreading_context(self):

        def parse(text, queue):
            queue.put(self.sm_nlp(text))

        queue = Queue()
        for i in range(NUMBER_OF_THREADS):
            text = ' '.join(('Peter told Paul he was dissatisfied', str(i)))
            t = Thread(target=parse,
                       args=(text, queue))
            t.start()
        returned_numbers = set()
        for i in range(NUMBER_OF_THREADS):
            doc = queue.get(True, 60)
            self.assertEqual('[0: [0], [3]]', str(doc._.coref_chains))
            self.assertEqual('[0: [0], [3]]', str(doc[0]._.coref_chains))
            self.assertEqual('[]', str(doc[2]._.coref_chains))
            self.assertEqual('[0: [0], [3]]', str(doc[3]._.coref_chains))
            returned_numbers.add(int(doc[-1].text))
        self.assertEqual(NUMBER_OF_THREADS, len(returned_numbers))

    def test_use_of_data_model_in_multiprocessing_context(self):

        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe('coreferee')
        reference_worker = Worker()
        manager = Manager()
        input_queues = [m_Queue() for i in range(NUMBER_OF_PROCESSES)]
        output_queue = manager.Queue()
        workers = []
        for counter in range(0, NUMBER_OF_PROCESSES):
            worker = Process(
                target=reference_worker.listen, args=(input_queues[counter],),
                daemon=True)
            worker.start()
            workers.append(worker)
        for counter in range(0, NUMBER_OF_PROCESSES):
            doc = nlp(' '.join(('Peter told Paul he was dissatisfied', str(counter))))
            input_queues[counter].put((output_queue, doc))
        returned_numbers = set()
        for counter in range(NUMBER_OF_PROCESSES):
            (first, second, third, fourth, returned_number) = output_queue.get(True, 60)
            self.assertEqual('[0: [0], [3]]', first)
            self.assertEqual('[0: [0], [3]]', second)
            self.assertEqual('[]', third)
            self.assertEqual('[0: [0], [3]]', fourth)
            returned_numbers.add(returned_number)
        self.assertEqual(NUMBER_OF_PROCESSES, len(returned_numbers))
        for worker in workers:
            worker.terminate()
