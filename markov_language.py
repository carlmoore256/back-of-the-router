import random
from nltk.corpus import stopwords
from coco_utils import generate_assets
from language_processing import word_tokenize, end_sentence
from nltk.tokenize.sonority_sequencing import SyllableTokenizer

# https://www.agiliq.com/blog/2009/06/generating-pseudo-random-text-with-markov-chains-u/

class Markov(object):
	
	def __init__(self):
		self.stop_words = set(stopwords.words("english"))
		self.SSP = SyllableTokenizer()
		
	def set_corpus(self, corpus):
		self.cache = {}
		corpus = [word.lower() for word in corpus if word.isalpha()]
		corpus = [w for w in corpus if w != " "]
		self.words = corpus
		self.word_size = len(self.words)
		self.database()
		
	def triples(self):
		""" Generates triples from the given data string. So if our string were
				"What a lovely day", we'd generate (What, a, lovely) and then
				(a, lovely, day).
		"""
		
		if len(self.words) < 3:
			return
		
		for i in range(len(self.words) - 2):
			yield (self.words[i], self.words[i+1], self.words[i+2])
			
	def database(self):
		for w1, w2, w3 in self.triples():
			key = (w1, w2)
			if key in self.cache:
				self.cache[key].append(w3)
			else:
				self.cache[key] = [w3]

	def generate(self, seed: int, size: int) -> str:
		seed_word, next_word = self.words[seed], self.words[seed+1]
		w1, w2 = seed_word, next_word
		gen_words = []
		gen_words.append(w1)
		attempts = 0
		while len(gen_words) < size:
			try:
				# not really trying to fully understand it cause Im lazy, sometime it
				# doesn't work
				w1, w2 = w2, random.choice(self.cache[(w1, w2)])
				# if w1 != gen_words[-1]:
				gen_words.append(w1)
				attempts -= 1
			except Exception as e:
				attempts += 1
				if attempts > 10:
					break
				continue
		
		# gen_words.append(w2)
		if attempts > 10: # try generating with a new seed
			gen_words = self.generate(random.randint(0, self.word_size-3), size)
		else:
			# if gen_words[-1] == gen_words[-2]:
			# 	gen_words.pop(-1)
			gen_words[0] = gen_words[0].capitalize()
		return gen_words

	# input words and use markov chain to generate a sentence
	def generate_sentence(self, params: dict, corpus: list) -> str:
		self.set_corpus(corpus)
		size = min(params['length'], self.word_size-3)
		seed = random.randint(0, self.word_size-3)
		gen_words = self.generate(seed, size)
		return end_sentence(gen_words)

	# input syllables and use markov chain to generate a word
	def generate_word(self, params: dict, corpus: list) -> str:
		corpus = " ".join(corpus)
		syl_tokens = self.SSP.tokenize(corpus)
		self.set_corpus(syl_tokens)
		size = min(params['length'], self.word_size-3)
		seed = random.randint(0, self.word_size-3)
		syllables = self.generate(seed, size)
		return ''.join(syllables)