import random
import numpy as np
from utils import sort_dict, load_object, check_if_file
from coco_utils import model_path, get_vocab_info
from nltk.tokenize import sent_tokenize, word_tokenize, sonority_sequencing
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize.sonority_sequencing import SyllableTokenizer
from nltk import pos_tag
from config import DATASET_CONFIG
import random

if check_if_file(model_path("vocab_info")):
  VOCAB_INFO = load_object(model_path("vocab_info"))
  try:
    TAGGED_VOCAB = pos_tag(VOCAB_INFO["vocabulary"])
  except Exception as e:
    print(e)
    pass
else:
  VOCAB_INFO = None
  TAGGED_VOCAB = None

SSP = SyllableTokenizer()

def syllable_tokenize(corpus):
  return SSP.tokenize(corpus)

# this is stupid, need to figure out this circular import issue...
def check_get_vocab_info():
    if VOCAB_INFO is None:
      VOCAB_INFO = get_vocab_info()
      TAGGED_VOCAB = pos_tag(VOCAB_INFO["vocabulary"])
    return VOCAB_INFO
    

def generate_name(attributes: dict) -> str:
  word_len = random.randint(5, 20)
  sorted_attrs = dict(sorted(attributes.items(), key=lambda item: item[1], reverse=True))
  name = ""
  attr_idx = 0
  while len(name) < word_len:
      key = list(sorted_attrs.keys())[attr_idx]
      slice_len = int((sorted_attrs[key] * word_len) ** 2)
      if slice_len == 0:
        slice_len = 1
      name += key[:slice_len]
      attr_idx += 1
  return name


def zipf_description(metadata, sentence_len=10, plotDist=False):
    descriptions = metadata.copy().pop("text_metadata")['descriptions']

    all_words = []
    for d in descriptions:
        words = []
        for word in d.split():
            word = word.lower().replace(".", "")
            words.append(word)
            if "." in word:
                words.append("!")
        all_words += words

    zipf_chart = {k: 0 for k in list(set(all_words))}

    for word in all_words:
        zipf_chart[word] += 1
    zipf_chart = dict(sorted(zipf_chart.items(), key=lambda item: item[1], reverse=True))

    # generate distribution to pull from
    dist = np.random.exponential(scale=1, size=sentence_len)
    dist = ((dist - np.min(dist)) / (np.max(dist) - np.min(dist)) * (len(zipf_chart.keys())-1)).astype(int)

    if plotDist:
        plt.title("zipz chart word occurences")
        plt.hist(list(zipf_chart.values()), 50, density=True)
        plt.show()
        plt.title("exponential distribution")
        count, bins, ignored = plt.hist(dist, 50, density = True)
        plt.show()

    sorted_zipf = list(zipf_chart.keys())    
    sentence = [sorted_zipf[idx] for idx in dist]
    str_sentence = ''
    sentence_start = True
    for word in sentence:
        if sentence_start:
            str_sentence += f"{word[0].upper()}{word[1:]}" + " "
            sentence_start = False
        else:
            if word == ".":
                sentence_start = True
            str_sentence += word + " "
    return sentence, str_sentence


def install_nltk_packages():
    nltk.download('punkt')
    nltk.download("stopwords")
    nltk.download('averaged_perceptron_tagger')
    nltk.download('tagsets')

def combine_strings(stringlist):
  string = ""
  for s in stringlist:
    string += s + " "
  return string

def tokenize_sentence(sentence):
  # sentence_tokens = sent_tokenize(sentence)
  word_tokens = word_tokenize(sentence)
  return word_tokens

def tokenize_descriptions(descriptions):
  all_words = combine_strings(descriptions)
  return tokenize_sentence(all_words)

def get_vocabulary(words):
  return list(set(words))

# get the corpus of a list of sentences
def get_corpus(descriptions):
  words = tokenize_descriptions(descriptions)
  return get_vocabulary(words)

def end_sentence(tokenized: list) -> str:
  stop_words = set(stopwords.words("english"))
  last_word = tokenized[-1]

  while last_word in stop_words:
    last_word = tokenized.pop(-1)
    if len(tokenized) < 1:
      return ""

  sentence_end = tokenized.pop(-1) + "."
  tokenized.append(sentence_end)
  return " ".join(tokenized)

def filter_stopwords(words):
  stop_words = set(stopwords.words("english"))
  filtered = [word for word in words if word.casefold() not in stop_words]
  return filtered

def tag_words(words):
  tagged = pos_tag(words)
  return tagged

def get_all_possible_tags(all_words):
  tagged = tag_words(all_words)
  tags = list(set([t[1] for t in tagged]))
  # tags = list(set(tag_words(t)[1] for t in all_words))
  tag_to_id = {tag: i for i, tag in enumerate(sorted(tags))}
  id_to_tag = {i: tag for i, tag in enumerate(sorted(tags))}
  return tags, tag_to_id, id_to_tag

def sentence_to_grammar_ids(sentence):
  words = tokenize_sentence(sentence)
  return words_to_grammar_ids(words)

def words_to_grammar_ids(words):
  tagged = tag_words(words)
  return [check_get_vocab_info()["tag_to_id"][t[1]] for t in tagged]

def sentence_to_ids(sentence):
  words = tokenize_sentence(sentence)
  return words_to_ids(words)

def words_to_ids(words):
  return [check_get_vocab_info()["word_to_id"][w] for w in words]

def ids_to_words(ids):
  return [check_get_vocab_info()["id_to_word"][id] for w in ids]

def get_all_words_for_grammar_id(grammar_id):
  matching = []
  tag = check_get_vocab_info()["id_to_tag"][grammar_id]
  for t in TAGGED_VOCAB:
    if t[1] == tag:
      matching.append(t[0])
  return matching
  # return [VOCAB_INFO["tag_to_id"](t[1]) for t in TAGGED_VOCAB if t[1] == int(grammar_id)]

def grammar_id_to_random_word(grammar_id, corpus=None):
  matching_words = get_all_words_for_grammar_id(grammar_id)
  if corpus is not None: # constrain matching vocab to preset corpus
    matching_words = [w for w in matching_words if w in corpus]
  if len(matching_words) > 0:
    return random.choice(matching_words)
  else:
    return None
  
def grammar_ids_to_random_words(grammar_ids, numerical=True):
  words = []
  for id in grammar_ids:
    words.append(grammar_id_to_random_word(id))
  return words

def words_to_sentence(words):
  sentence = ""
  for i, w in enumerate(words):
    if i == len(words) - 1:
      sentence += w + "."
    else:
      sentence += w + " "
  return sentence

def letters_to_ids(letters):
  return [ord(letter) - 96 + 31 for letter in letters]

def ids_to_letters(ids):
  return [chr(id+96 - 31) for id in ids]


