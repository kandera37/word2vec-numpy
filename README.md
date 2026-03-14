# word2vec-numpy

A small educational project that implements the core training logic of word2vec in pure NumPy.

## What this project does

This project builds a simple skip-gram pipeline with negative sampling:
- reads and tokenizes text
- builds a vocabulary
- generates skip-gram training pairs
- initializes word embeddings
- samples negative words
- computes scores and loss for training examples

## Tech stack

- Python
- NumPy

## Project structure

- `data/sample.txt` — sample text corpus
- `src/data_utils.py` — text preprocessing and training pair generation
- `src/model.py` — embeddings, negative sampling, scores, and loss
- `src/main.py` — simple pipeline runner

## How to run

```bash
python3 src/main.py
```

## Status

Work in progress.
Current version includes preprocessing, embedding initialization, negative sampling, and loss computation.