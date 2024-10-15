from transformers import AutoTokenizer
from collections import defaultdict
class BPE :

    def __init__(self, corpus, size):
        self.corpus = corpus
        self.size = size
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.word_freqs = self.compute_word_freqs(corpus)
        self.splits = {word: [c for c in word] for word in self.word_freqs.keys()}

    def initialize_bpe(self):
        alphabet = []
        for word in self.word_freqs.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)
        alphabet.sort()
        vocab = ["<|endoftext|>"] + alphabet.copy()
        return alphabet,vocab

    def compute_word_freqs(self,corpus):
        word_freqs = defaultdict(int)
        # We loop through to corups to calculate word frequencies
        for text in corpus:
            words_with_offsets = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                word_freqs[word] = word_freqs[word] + 1
        return word_freqs

    # Define a function to compute the frequency of each pair of characters
    def compute_pair_freqs(self):
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] = pair_freqs[pair] + freq
        return pair_freqs


    def merge_pair(self,a, b):
        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2:]
                else:
                    i += 1
            self.splits[word] = split

        return self.splits

    def compute_bpe_vocab(self):
        alphabet,vocab = self.initialize_bpe()
        while len(vocab) < self.size:
            pair_freqs = self.compute_pair_freqs()
            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)
            a, b = best_pair
            self.splits = self.merge_pair(a, b)
            vocab.append(a + b)
            if len(vocab) % 500 == 0:
                print(f'Current vocab size: {len(vocab)}')
        return vocab

