import os
import random
import string
import re
import numpy as np
from tqdm import tqdm

UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"

def prepare_tag_data(filename):
    eng_lines = []
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip().replace('<s>','').replace('</t>','').split('</s> <t>')
            line = line[0].strip()
            line = re.sub('\s+',' ',line)
            eng_lines.append(line)
    return eng_lines

def tuple_data(words,labels):
    data = [(i,j) for i, j in zip(words,labels)]
    return data

def get_vocabs(datasets):
    """Building vocabs present in datasets"""
    print("Building Vocab ....")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("-done vocab words {s}--- vocab_tags {d}".format(s=len(vocab_words),d=len(vocab_tags)))
    return vocab_words, vocab_tags


def get_char_vocab(dataset):
    """Build char vocab present in datasets"""
    vocab_char = set()
    for words, _ in dataset:
        for word in words:
            vocab_char.update(word)
    return vocab_char

def get_glove_vocab(path):
    """Building vocabs present in glove"""
    print("Building Vocab..glove...")
    vocab = set()
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("done. {} tokens in glove".format(len(vocab)))
    return vocab


def final_vocab(vocab_words,vocab_glove,*args):
    """Combine word vocabs >> glove_vocabs+dataset_vocabs"""
    vocab = vocab_words & vocab_glove
    for word in args:
        vocab.add(word)
    return vocab

def build_vocab(vocab,filename):
    "Write all vocabs line by line in a file"
    with open(filename,'w',encoding='utf-8') as f:
        for i, word in enumerate(vocab):
            if i != len(vocab)-1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("written {s} tokens in {d}".format(s=len(vocab), d=filename))


def load_vocab(filename):
    """Assign id to each word in vocab
        returns dictionary"""
    d = {}
    with open(filename,encoding='utf-8') as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx
    return d





def export_glove_vectors(vocab, glove_filename, filename, dim):
    """Bulding compressed file of vectors of words
    that are present in dataset"""
    embeddings = np.zeros([len(vocab),dim],dtype='float32')
    with open(glove_filename,'r',encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            if word in vocab:
                embedding = [float(x) for x in line[1:]]
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)
    np.savez_compressed(filename, embeddings=embeddings)

def get_glove_vectors(filename):
    """Loads the saved numpy file (Embeddings)"""
    with np.load(filename) as data:
        return data['embeddings']
    


def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        UNK = "$UNK$"
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f


class CoNLLDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, data, processing_word=None, processing_tag=None,
                 max_iter=32):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.data = data
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = 32
        self.length = None
        
    def __iter__(self):
#         words, tags = [], []
        for d in self.data:
            words, tags = [], []
            
#             if self.max_iter == niter:
#                 yield words, tags
#                 words, tags = [], []
#             else:
#                 niter+=1
            word, tag = d[0],d[-1]
#             print(word),print(tag)
            for word_, tag_ in zip(word,tag):
                if self.processing_word is not None:
                    word_ = self.processing_word(word_)
                if self.processing_tag is not None:
                    tag_ = self.processing_tag(tag_)
                words += [word_]
                tags += [tag_]
            yield words, tags

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1
        return self.length



def generate_data(lines, max_sents_per_example=6, n_examples=1000,punct=None):
    """
        Generates training data for deepsegment from list of sentences.
        Parameters:
        lines (list): Base sentences for data generation.
        max_sents_per_example (int): Maximum number of sentences to be combined to form a single paragraph.
        
        n_examples (int): Number of training examples to be generated.
        
        Returns:
        list, list: Training data and corresponding labels in BIOU format.
    """
    x, y = [], []
    
    for current_i in tqdm(range(n_examples)):
        x.append([])
        y.append([])

        chosen_lines = []
        for _ in range(random.randint(1, max_sents_per_example)):
            chosen_lines.append(random.choice(lines))
        
        chosen_lines = [bad_sentence_generator(line, remove_punctuation=random.randint(0, 3),use_punct=punct) for line in chosen_lines]
        
        for line in chosen_lines:
            words = line.strip().split()
            for word_i, word in enumerate(words):
                x[-1].append(word)
                label = 'O'
                if word_i == 0:
                    label = 'B-sent'
                y[-1].append(label)
    
    return x, y

punct = ['And','Or','So','After','Once','Since','So','After that','Though',',',':',';','now','now then',""]

def bad_sentence_generator(sent, remove_punctuation = None,use_punct=None):
    """
        Returns sentence with completely/ partially removed punctuation.
        Parameters:
        sent (str): Sentence on which the punctuation removal operation is performed.
        
        remove_punctuation (int): removing punctuation completely if remove_punctuation ==0 or ==1, removing punctuation till a randomly selected point if remove_punctuation ==2
        Returns:
        str: Sentence with modified punctuation
    """

    if not remove_punctuation:
        remove_punctuation = random.randint(0, 3)

    break_point = random.randint(1, len(sent)-2)
    lower_case = random.randint(0, 2)

    if remove_punctuation <= 1:
        # removing punctuation completely if remove_punctuation ==0 or ==1
        sent = re.sub('['+string.punctuation+']', '', sent)
        if punct and remove_punctuation == 0:
            sent = f"{random.choice(punct)}"+" "+sent
    
    elif remove_punctuation == 2:
        # removing punctuation till a randomly selected point if remove_punctuation ==2
        if random.randint(0,1) == 0:
            sent = re.sub('['+string.punctuation+']', '', sent[:break_point]) + sent[break_point:]
        # removing punctuation after a randomly selected point if remove_punctuation ==2        
        else:
            sent = sent[:break_point] + re.sub('['+string.punctuation+']', '', sent[break_point:])    
    
    if lower_case <= 1:
        # lower casing sentence 
        sent = sent.lower()
    
    return sent


#minibatches of data
def minibatches(data,batch_size):
    x_batch, y_batch = [], []
    for (x,y) in data:
        if len(x_batch) == batch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]
        
    if len(x_batch) != 0:
        yield x_batch, y_batch
        
def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    if NONE in tags:
        default = tags[NONE]
    else:
        default = None
        
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
#         print(tok)
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length

def data_builder(config):
    """build all the data files vocab, char, tag, embeddings"""
    data = prepare_tag_data(config.data_path)
    train_length = int(0.7*len(data))
    test_length = len(data)-train_length
    
    x, y = generate_data(lines=data[:train_length],max_sents_per_example=6,n_examples=56000,punct=config.punct)
    x_, y_ = generate_data(lines=data[:test_length],max_sents_per_example=6,n_examples=24000,punct=config.punct)

    config.train = tuple_data(words=x,labels=y)
    config.test = tuple_data(words=x,labels=y)
    
    process_word = get_processing_word(lowercase=True)
    
     # Generators
    test  = CoNLLDataset(data=config.train, processing_word=process_word)
    train = CoNLLDataset(data=config.test, processing_word=process_word)
    
    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, test])
    vocab_glove = get_glove_vocab(config.glove_path)

    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    # Save vocab
    build_vocab(vocab, config.vocab_filename)
    build_vocab(vocab_tags, config.tag_filename)

    # Trim GloVe Vectors
    vocab = load_vocab(config.vocab_filename)
    export_glove_vectors(vocab, config.glove_path,
                                config.vocab_embedding_filename, config.word_dim)

    # Build and save char vocab
    train = CoNLLDataset(data=train)
    vocab_chars = get_char_vocab(train)
    build_vocab(vocab_chars, config.char_filename)




    