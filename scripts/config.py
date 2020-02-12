from .data_utils import load_vocab,get_glove_vectors,get_processing_word
import os

class config():
    """Initiates all HYPERPARAMETERS
       Generates and load vocab"""
    def __init__(self,
                 load=False,
                 nepochs=15,
                 batch_size=32,
                 epochs_no_improvement=4,
                 hidden_dim=256,
                 char_hidden_dim=100,
                 word_dim=100,
                 char_dim=50,
                 data_filename=None,
                 dropout = None,
                 use_chars=True,
                 use_crf=True
                 ):
        
        config.nepochs = nepochs
        config.epochs_no_improvement = epochs_no_improvement
        config.dropout = dropout
        config.batch_size = batch_size
        config.hidden_dim = hidden_dim
        config.char_hidden_dim = char_hidden_dim
        
        config.word_dim = word_dim
        config.char_dim = char_dim
        
        config.data_filename = data_filename
        config.data_path = os.path.join(os.getcwd(),"data/{}".format(self.data_filename))
        config.model_path = os.path.join(os.getcwd(),"model/")
        config.glove_path = os.path.join(os.getcwd(),"glove/glove.6B.{}d.txt".format(self.word_dim))
        
        config.vocab_filename = os.path.join(os.getcwd(),"data/vocabs.txt")
        config.tag_filename = os.path.join(os.getcwd(),"data/tags.txt")
        config.char_filename = os.path.join(os.getcwd(),"data/chars.txt")
        
        config.vocab_embedding_filename = os.path.join(os.getcwd(),"data/vocab_embeddings.npz")
        config.tag_embedding_filename = os.path.join(os.getcwd(),"data/tag_embeddings.npz")
        config.use_chars = use_chars
        config.use_crf=use_crf
        config.use_pretrained = True
        
        config.punct = ['and','or','so','after','once','since','so','after that','though',',',':',';','now','now then',""]
        
        config.train = None
        config.test = None

        config.lr = 0.001
        config.decay_lr = 0.9
        config.lr_method = "Adam"
        config.clip = -1 # clip not performed if -1, use values > 0

        config.use_pretrained=True

        if load:
            self.loads()
        
    def loads(self):
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
        self.ntags = len(self.tag_2_id)
        
        #load embeddings 
        self.word_embeddings = (get_glove_vectors(filename=self.vocab_embedding_filename)
                                if self.use_pretrained else None)
        # self.tag_embeddings  = get_glove_vectors(filename=self.tag_embedding_filename)
        
        #processing function that will map word str to id 
        self.process_words = get_processing_word(vocab_words=self.word_2_id, vocab_chars=self.char_2_id,chars=self.use_chars)
        self.process_tags = get_processing_word(vocab_words=self.word_2_id, vocab_chars=self.char_2_id,chars=False)
        
        
        
            
            
        
        