from os import SEEK_END
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from utils import load_json, load_object
from coco_utils import model_path
from language_processing import words_to_sentence, grammar_id_to_random_word, sentence_to_ids, tokenize_sentence, combine_strings
from language_processing import tokenize_descriptions, words_to_grammar_ids, sentence_to_grammar_ids, get_corpus, get_vocabulary
import random
import wandb

from dataset import Dataset
from config import LSTM_CONFIG

VOCAB_INFO = load_object(model_path("vocab_info"))

EMBEDDING_DIM = 12
HIDDEN_DIM = 12

# from pytorch lstm example
class LSTMTagger(nn.Module):

    def __init__(self, config, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        
        self.hidden_dim = config["hidden_dim"]
        self.word_embeddings = nn.Embedding(vocab_size, config["embedding_dim"])
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(config["embedding_dim"], config["hidden_dim"])
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(config["hidden_dim"], tagset_size)

    def generate_sentence(self, params, corpus):
        return self.iterative_sentence(params['seed'], params['iters'], corpus)

    def iterative_sentence(self, seed : str, iters : int = 10, corpus = None):
        x = torch.as_tensor(sentence_to_grammar_ids(seed))  
        words = []
        with torch.no_grad():
            for _ in range(iters):
                try:
                    y_ = self(x)
                    grammar_ids = torch.argmax(y_, dim=1).detach().numpy()
                    batch_words = []
                    for id in grammar_ids:
                        word = grammar_id_to_random_word(id, corpus)
                        if word is None: # in this case use any word
                            word = grammar_id_to_random_word(id)
                        batch_words.append(word)
                    if len(batch_words) > 0:
                        words += batch_words
                        # x = torch.as_tensor(words_to_grammar_ids( batch_words ))
                        x = torch.as_tensor(words_to_grammar_ids( words[-random.randint(1, len(words)):] ), dtype=int)
                except Exception as e:
                    print(e)
                    continue
            sentence = words_to_sentence(words)
        return sentence

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def generate_description_lstm(metadata, iters=3, model_path=None, model_config=None):
    corpus = get_corpus(metadata['text_metadata']['descriptions'])

    if model_config is None: # use default settings
        model_config = LSTM_CONFIG
    if model_path is None:
        model_path = model_config["checkpoint_path"]
    
    model = LSTMTagger(model_config, len(VOCAB_INFO["vocabulary"]), 
        len(VOCAB_INFO["all_tags"]))
    model.load_state_dict(torch.load(model_path))
    seed = random.choice(metadata['text_metadata']['descriptions'])
    seed = tokenize_sentence(seed)
    seed = seed[:random.randint(0, int(len(seed)*0.5))]
    seed = combine_strings(seed)

    description = model.iterative_sentence(seed, iters=iters, corpus=corpus)
    return description

def train_language_model(model, dataset, config):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
    losses = []

    for epoch in range(config["epochs"]):
        avg_loss = 0
        for i, coco_example in enumerate(dataset):
            model.zero_grad()
            caption = coco_example.get_caption()
            try:
                # x input is word ids
                x = torch.as_tensor(sentence_to_ids(caption))
                # y predicted output is the grammar ids
                y = torch.as_tensor(sentence_to_grammar_ids(caption))
            except Exception as e:
                # dataset doesn't seem to find all grammar ids, figure this out later
                print(e)
                continue
            y_ = model(x)
            loss = criterion(y_, y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if i % config["log_freq"] == 0:
                wandb.log({
                    "loss" : loss.item() })
                print(f'step: {i}/{len(dataset)} loss: {loss.item()}')
                 
        avg_loss = avg_loss/len(dataset)

        example_corpus = []
        for _ in range(random.randint(5,10)):
            words = tokenize_sentence(dataset.get_coco_example().get_caption())
            example_corpus += words
        example_corpus = get_vocabulary(example_corpus)
        # example_corpus = list(set([dataset.get_coco_example().get_caption() for _ in range(random.randint(5,10))]))

        seed_words = dataset.get_coco_example().get_caption()
        sentence_loop = random.randint(2, 4)
        gen_sentence = model.iterative_sentence(seed_words, sentence_loop, example_corpus)
        print(f'\n=> Epoch: {epoch} Average Loss: {avg_loss} sentence: {gen_sentence}\n')

        wandb.log({
            "test_sentence" : gen_sentence,
            "avg_loss" : avg_loss
        })
        if epoch > 0 and avg_loss < max(losses):
            print(f'[!] saving checkpoint to {config["checkpoint_path"]}')
            torch.save(model.state_dict(), config["checkpoint_path"])
        losses.append(avg_loss)
    torch.save(model.state_dict(), config["save_path"])

if __name__ == "__main__":
    # train the model
    wandb.init(project="botr-language-model", entity="carl_m")
    wandb.config = LSTM_CONFIG
    dataset = Dataset()
    model = LSTMTagger(
        wandb.config, 
        len(VOCAB_INFO["vocabulary"]), 
        len(VOCAB_INFO["all_tags"]))
    train_language_model(model, dataset, wandb.config)