# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:57:50 2018

@author: FactRank authors (Ivo Merchiers) with modifications by Carlos Ortega
"""

from pattern.es import parsetree, lemma, singularize

postags = [u'CC', u'CD', u'DT', u'EX', u'FW', u'IN', u'JJ', u'JJR', u'JJS', u'LS', u'MD', u'NN', u'NNS', u'NNP', u'NNPS', u'PDT', u'POS', u'PRP', u'PRP$',
           u'RB', u'RBR', u'RBS', u'RP', u'SYM', u'TO', u'UH', u'VB', u'VBZ', u'VBP', u'VBD', u'VBN', u'VBG', u'WDT', u'WP', u'WP$', u'WRB', u'.', u',', u':', u'(', u')']

class Extractor:

    """
    Extractor class enables retrieving all nlp features from `sentence`.
    """

    def __init__(self, sentence):
        self.sentence = sentence
        tree = parsetree(sentence, lemmata=True)
        # tree is actually a sentence in term of pattern definitions
        self.tree = tree[0]
        self.pcounts = self.countPosTags()
        #self.ecounts = self.countEntities()

    def __getitem__(self, k):
        if k == "LEMMA_SENT":
            return self.lemmataSentence()
        elif k == "LEMMA_POS":
            return self.lemmataPosTagSentence()
        elif k == "POS_SENT":
            return self.posTagSentence()
        elif k == "LENGTH":
            return self.length()

    def change(self, sentence):
        self.sentence = sentence
        tree = parsetree(sentence, lemmata=True)
        self.tree = tree[0]

    def length(self):
        return self.tree.stop

    def parsedSentence(self):
        base = self.tree.words
        return [str(b) for b in base]

    def lemmataSentence(self):
        base = self.tree.lemmata
        base = ' '.join(base)
        return base

    def posTagSentence(self):
        wordlist = self.tree.words
        pos = map(lambda x: x.type, wordlist)
        pos_sent = ' '.join(pos)
        return pos_sent

    def lemmataPosTagSentence(self):
        wordlist = self.tree.words
        base = list(self.tree.lemmata)
        pos = list(map(lambda x: x.type, wordlist))
        lem_pos_sent = ' '.join([ lem + "_" + pos for lem, pos in zip(base,pos)])
        return lem_pos_sent

    def countPosTags(self):
        wordlist = self.tree.words
        pos = list(map(lambda x: x.type, wordlist))
        counts = {}
        for i in range(0, len(postags)):
            counts[postags[i]] = sum(map(lambda x: x == postags[i], pos))
        return counts
