import wandb
from language_model import LSTMTagger
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from language_processing import VOCAB_INFO, sentence_to_ids, sentence_to_grammar_ids, tokenize_descriptions, get_vocabulary, tokenize_sentence
from utils import load_json, load_object
from coco_utils import model_path
from dataset import Dataset
from config import LSTM_LANGUAGE_CONFIG_PATH, LSTM_LANGUAGE_CONFIG_PATH
import random

LSTM_CONFIG = load_json(LSTM_LANGUAGE_CONFIG_PATH)
VOCAB_INFO = load_object(model_path("vocab_info"))

EMBEDDING_DIM = 12
HIDDEN_DIM = 12

# def test_language_model(mode, dataset, config):

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
                    "step" : i,
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
    wandb.init(project="botr-language-model", entity="carl_m")
    wandb.config = LSTM_CONFIG

    dataset = Dataset()

    model = LSTMTagger(
        wandb.config, 
        len(VOCAB_INFO["vocabulary"]), 
        len(VOCAB_INFO["all_tags"]))

    train_language_model(model, dataset, wandb.config)