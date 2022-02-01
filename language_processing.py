import random
import numpy as np
from utils import sort_dict

def generate_name(metadata):
  word_len = random.randint(5, 20)
  sorted_attrs = dict(sorted(metadata.items(), key=lambda item: item[1], reverse=True))
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
