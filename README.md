# Byte-Pair-Encoding
# BPE Vocabulary Builder

This project implements a Byte Pair Encoding (BPE) algorithm to build a vocabulary from a given corpus. The BPE algorithm is commonly used in natural language processing tasks to tokenize text data.

## Features

- Load a dataset using the `datasets` library.
- Compute word frequencies using a pre-trained tokenizer from the `transformers` library.
- Implement the BPE algorithm to merge character pairs and build a vocabulary.
- Save the generated vocabulary to a file.

## Installation

To install the required dependencies, run:

```bash
pip install transformers datasets