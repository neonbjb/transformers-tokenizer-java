# Java NLP Tokenize (for Transformers)
This is a Java string tokenizer for natural language processing machine learning models.
Specifically, it was written to output token sequences that are compatible with the
sequences produced by the Transformers library from huggingface, a popular NLP library
written in Python.

## Why?
There are a few good reasons you might want to use Java for string tokenization:
1)  Deploying a NLP model on a mobile device (e.g. Android)
1)  Doing tokenization outside of a typical ML pipeline (for example, maybe you want to just use tensorflow-serving).
1)  Batching ease-of-use (Python doesn't have true multithreading)

## Current Status
Currently, only the GPT-2 tokenizer is implemented.

I originally intended to implement several compatible tokenizers (ex: BERT, XLNet, etc)
from the Transformers
library. However, after working on several of the model types paired with these tokenizers,
I found out that most of the models could not be exported due to either bugs in the Transformers
library or in Tensorflow itself.

That means that Java tokenizers for these alternative model types are effectively useless, since
there is no way to easily deploy them outside of a Python environment - and if you are working
in Python already, you may as well use Python to tokenize.

## Compatibility
This tokenizer guarantees its compatibility via a unit test that loads a set of data that has
been pre-tokenized by the Python transformers library. The strings from this data are then
fed through the Java library and the results are compared. I believe this provides a reasonable
guarantee that this library generates the same output as the transformers version.