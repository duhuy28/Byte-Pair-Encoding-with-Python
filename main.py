from BPE import BPE
from datasets import load_dataset
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",split='train')['text']
corpus = [entry for entry in dataset if entry.strip()]
MyBPE = BPE(corpus,size=20000)
vocab =MyBPE.compute_bpe_vocab()
print(vocab[-50:])
with open('vocab.txt', 'w', encoding='utf-8') as f:
    f.write(','.join(vocab))

