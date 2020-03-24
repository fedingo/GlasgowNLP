from .utils import tokenize
import json
import numpy as np

class Vocabulary:
    
    def __init__(self, sentences, vocab_size = -1, tokenize_fn=None, load_path=None):
        
        self.itos = ['<pad>', '<mask>', '<unk>']
        self.stoi = {'<pad>' :0,
                     '<mask>':1,
                     '<unk>' :2 }
        
        self.next_id = 3
        
        self.tokenize = tokenize_fn
        self.vocab_size = vocab_size
        
        if self.tokenize == None:
            self.tokenize = tokenize
            
        if load_path is None:
            self.__create(sentences)
        else:
            self.load_vocab(load_path)
        
    def __create(self, sentences):
        word_list = []
        for s in sentences:
            word_list.extend(self.tokenize(s))
            
        ordered_word_set = self.__build_ordered_word_set(word_list)
        
        for word in ordered_word_set[:self.vocab_size]:
            self.__add_word(word)
        
    def __add_word(self, word):
        assert len(self.itos) == self.next_id, "id missconfiguration"
        
        self.itos.append(word)
        self.stoi[word] = self.next_id
        self.next_id += 1
        
    def __build_ordered_word_set(self, word_list):
        
        word_set = list(set(word_list))
        word_freqs = {}
        
        for word in word_set:
            word_freqs[word] = 0
        
        # Count word frequencies
        for word in word_list:
            word_freqs[word] += 1
            
        # Sorting based on the frequencies of the words
        word_set.sort(key=lambda w: word_freqs[w], reverse=True)
        return word_set
        
        
    def save_vocab(self, path):
        with open(path, 'w') as f:
            json.dump(self.itos, f)
            
    def load_vocab(self, path):
        with open(path, 'r') as f:
            self.itos = json.load(f)
            
        self.stoi = {}
        for idx, word in enumerate(self.itos):
            self.stoi[word] = idx
            
        self.next_id = len(self.itos)
        
    def encode(self, sentence):
        tokens = []
        words = self.tokenize(sentence)
        for word in words:
            tokens.append(self.stoi[word] if word in self.itos
                          else self.stoi['<unk>'])

        return np.array(tokens).astype(np.int64)

    def decode(self, tokens):
        sentence = ""
        for idx in tokens.squeeze():
            sentence += self.itos[idx]
            sentence += " "
        return sentence