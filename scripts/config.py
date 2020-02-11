import os
from data_utils import load_vocab,get_glove_vectors,get_processing_word


class config():
    """Initiates all HYPERPARAMETERS
       Generates and load vocab"""
    def __init__(self,
                 load=True,
                 nepochs=15,
                 batch_size=32,
                 epochs_no_improvement=4,
                 hidden_dim=256,
                 char_hidden_dim=100,
                 word_dim=100,
                 char_dim=50,
                 dropout = None
                 use_chars=True,
                 use_crf=True
                 ):
        self.load = load
        self.nepochs = nepochs
        self.epochs_no_improvement = epochs_no_improvement
        self.dropout = dropout
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.char_hidden_dim = char_hidden_dim
        
        self.word_dim = word_dim
        self.char_dim = char_dim
        
        self.data_path = "data"
        self.model_path = "model"
        self.glove_path = "glove/glove.6B.{}d.txt".format(self.word_dim)
        
        self.vocab_filename = "data/vocabs.txt"
        self.tag_filename = "data/tags.txt"
        self.char_filename = "data/chars.txt"
        
        self.vocab_embedding_filename = "data/vocab_embeddings.npz"
        self.tag_embedding_filename = "data/tag_embeddings.npz"
        self.use_chars = use_chars
        self.use_crf=use_crf
        self.use_pretrained = True
        
    def load(self):
        """creates vocab files and processing functions
        and embeddings"""
        #id dictinories
        #id -- word
        self.word_2_id = load_vocab(self.vocab_filename) 
        self.tag_2_id = load_vocab(self.tag_filename)
        self.char_2_id = (load_vocab(self.char_filename)if self.use_chars else None)
        self.nchars = (len(self.char_2_id)if self.use_chars else None)
        
        
        #load number of unique words present
        self.nwords = len(self.word_2_id)
        self.ntags = len(self,tag_2_id)
        
        #load embeddings 
        self.word_embeddings = (get_glove_vectors(filename=self.vocab_embedding_filename)
                                if self.use_pretrained else None)
        self.tag_embeddings  = get_glove_vectors(filename=self.tag_embedding_filename)
        
        #processing function that will map word str to id 
        self.process_words = get_processing_word(vocab_words=self.word_2_id, vocab_chars=self.char_2_id,chars=self.use_chars)
        self.process_tags = get_processing_word(vocab_words=self.word_2_id, vocab_chars=self.char_2_id,chars=False)
        
        
        
            
            
        
        