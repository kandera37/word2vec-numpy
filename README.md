# word2vec-numpy

A NumPy-only implementation of the core training loop of **word2vec** using the **skip-gram with negative sampling** formulation.

## Project goal

The goal of this project is to implement the core optimization procedure of word2vec in pure NumPy, without using PyTorch, TensorFlow, or other machine-learning frameworks.

The implementation includes:

- text preprocessing
- vocabulary construction
- skip-gram pair generation
- embedding initialization
- negative sampling
- forward pass with dot-product scores
- loss computation
- parameter updates
- a simple multi-epoch training loop

## Model variant

This project uses **skip-gram with negative sampling**.

For each training example:

- the **center word** is used as input
- the **context word** is treated as a positive target
- randomly sampled words are used as negative targets

The model learns word embeddings by increasing scores for positive pairs and decreasing scores for negative pairs.

## Dataset

The current implementation uses a small sample text corpus stored in:

- `data/sample.txt`

This keeps the project simple and focused on the mechanics of the training loop itself.

## Project structure

- `data/sample.txt` — sample text corpus
- `src/data_utils.py` — text preprocessing and skip-gram pair generation
- `src/model.py` — embeddings, negative sampling, scores, loss, and training loop
- `src/main.py` — end-to-end demo run of the training pipeline

## How to run

```bash
python3 src/main.py
```

## Current implementation

The pipeline currently performs the following steps:

1. Read and tokenize text 
2. Build a vocabulary 
3. Encode tokens as integer ids 
4. Generate skip-gram training pairs 
5. Initialize input and output embeddings 
6. Compute positive and negative scores 
7. Compute skip-gram loss with negative sampling 
8. Apply one-step gradient-based updates 
9. Run training across multiple epochs 
10. Track average loss across epochs

## Example behavior

The script prints:

- tokenized text 
- vocabulary mapping 
- example skip-gram pairs 
- example positive and negative scores 
- loss before and after one update step 
- average training loss across epochs

A decreasing average loss across epochs indicates that the update rule moves the embeddings in the expected direction.

## Limitations

This is a compact educational implementation intended to demonstrate the core training mechanics of word2vec.

Current limitations:

- it uses a very small toy corpus 
- negative sampling is uniform rather than frequency-based 
- no batching is used 
- no subsampling of frequent words is implemented 
- no evaluation on downstream similarity tasks is included

## Possible improvements

Possible next steps include:

- using a larger text corpus 
- implementing frequency-based negative sampling 
- adding subsampling for frequent words 
- supporting CBOW as an alternative formulation 
- adding nearest-neighbor inspection for learned embeddings 
- vectorizing more of the training loop for efficiency