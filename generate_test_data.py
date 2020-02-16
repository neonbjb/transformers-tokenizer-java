input_file = 'C:/Users/jbetk/Documents/data/ml/sentiment_analysis/amazon/Digital_Music_5.json'

import transformers
import orjson
import random

if __name__ == '__main__':
    tok = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = "<|endoftext|>"
    with open(input_file, encoding="utf-8") as file:
        line = file.readline()
        toks = {}
        encs = {}
        till_done = 5000
        while till_done > 0:
            entry = orjson.loads(line)
            sentence = entry['reviewText']
            wordparts = sentence.split(' ')
            for i in range(random.randint(0, 3)):
                wordparts.insert((i+1)*2, tok.eos_token)
            sentence = ' '.join(wordparts)

            toks.update({sentence: tok.tokenize(sentence, add_prefix_space=True)})
            encs.update({sentence: tok.encode_plus(sentence, max_length=128, pad_to_max_length=True, return_token_type_ids=True, return_attention_mask=True, return_overflowing_tokens=True, add_prefix_space=True)})

            till_done -= 1
            line = file.readline()

        output_tok = open("gpt2_tokenize_test.json", "wb")
        output_tok.write(orjson.dumps(toks))
        output_tok.close()

        output_enc = open("gpt2_encode_test.json", "wb")
        output_enc.write(orjson.dumps(encs))
        output_enc.close()
