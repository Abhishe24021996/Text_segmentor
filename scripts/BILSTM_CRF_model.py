import numpy as np
import tensorflow as tf
from .base_model import BaseModel
from .data_utils import minibatches, get_chunks


class BILSTM_CRF(BaseModel):
    def __init__(self,config):
        super(BILSTM_CRF,self).__init__(config)
        
    def add_placeholder(self):
        """initiate placeholders"""
        #intitate placeholders
        self.word_ids = tf.placeholder(tf.int32, shape=[None,None], name="word_id_placeholder")
        self.sequence_length = tf.placeholder(tf.int32, shape=[None,], name="sequence_length_placholder")
        self.label = tf.placeholder(tf.int32,shape=[None,None], name="labels")
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.char_id = tf.placeholder(dtype=tf.int32, shape=[None,None,None], name="char_id")
        self.word_lengths = tf.placeholder(dtype=tf.int32, shape=[None,None], name="word_length_placeholder")

    def get_feed_dict(words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary
        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob
        Returns:
            dict {placeholder: value}
        """
        #perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_length: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_id] = char_ids
            feed[self.word_lengths] = word_lengths
        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.label] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        
    def word_embedding_fn(self):
        with tf.variable_scope("word_embedding"):
            if self.config.word_embeddings is None:
                word_embeddings_ = tf.get_variable(name="word_embeddings_",
                                                  dtype=tf.float32,
                                                  shape=[self.config.nwords,self.config.word_dim])
            else:
                word_embeddings_ = tf.Variable(self.config.word_embeddings,
                                               name="word_embeddings_",
                                               dtype=tf.float32,
                                               shape=[self.config.nwords,self.config.word_dim],
                                               trainable=True
                                               ) 
            self.word_embeddings = tf.nn.embedding_lookup(word_embeddings_,self.word_ids,name="word_embeddings")
        
        with tf.variable_scope("char"):
            print('using char embedding')
            if self.config.use_chars:
                char_embeddings_ = tf.Variable(tf.random_uniform([self.config.nchars,self.config.char_dim],-1.0,1.0),
                                                name="_char_embeddings",
                                                dtype=tf.float32)
                            #shape=[nchars, char_hidden_size])
                char_embeddings = tf.nn.embedding_lookup(char_embeddings_,
                                                        self.char_id,
                                                        name="char_embeddings")
                #including time dimesion
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                                            shape=[s[0]*s[1],s[-2],50])
                word_length = tf.reshape(self.word_lengths,shape=[s[0]*s[1]])
                #lstm bidir over chars
                fw_cell = tf.contrib.rnn.LSTMCell(self.config.char_hidden_dim, state_is_tuple=True)
                bw_cell = tf.contrib.rnn.LSTMCell(self.config.char_hidden_dim, state_is_tuple=True)
                
                output_ = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell, char_embeddings,
                                                         sequence_length = word_length, dtype=tf.float32)
                #read tthe output_
                _, ((_,fw_output),(_,bw_output)) = output_
                output = tf.concat([fw_output,bw_output],axis=-1)
                #shape [bs, max_sequnce_length, char_hidden_size]
                output = tf.reshape(output,shape=[s[0],s[1],2*self.config.char_hidden_dim])
                self.word_embeddings = tf.concat([self.word_embeddings, output],axis=-1)
        if self.config.dropout:
            self.word_embeddings = tf.nn.dropout(self.word_embeddings, self.config.dropout)
    
    def logits_op(self):
        with tf.variable_scope("bidirectional_lstm"):
            for_cell = tf.contrib.rnn.LSTMCell(self.config.hidden_dim)
            bac_cell = tf.contrib.rnn.LSTMCell(self.config.hidden_dim)
            (for_out, bac_out), _ = tf.nn.bidirectional_dynamic_rnn(for_cell, bac_cell,
                                                                    self.word_embeddings,
                                                                    sequence_length=self.sequence_length,
                                                                    dtype=tf.float32)
            #shape >> [bs, sequence_length, 2EMD]
            output = tf.concat([for_out, bac_out], axis=-1)
    
        #         output = tf.nn.dropout(output,)

        with tf.variable_scope("w_b"):
            W = tf.get_variable(name="W",dtype=tf.float32,
                                shape = [2*self.config.hidden_dim,self.config.ntags])
            b = tf.get_variable(name="b", dtype=tf.float32,
                                shape = [self.config.ntags],
                                initializer = tf.zeros_initializer())
    
    
            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1,2*self.config.hidden_dim])
    
            pred = tf.matmul(output,W) + b
            self.logits = tf.reshape(pred,[-1,nsteps,self.config.ntags])
            
    def pred_op(self):
        if not self.config.use_crf:
            self.label_pred = tf.cast(tf.argmax(self.logits, axis=-1),tf.int32)
        
    def loss_op(self):
        if self.config.use_crf:
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                                            self.logits, self.label, self.sequence_length)
            self.trans_params = trans_params
            self.loss =tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.label)
            mask = tf.sequence_mask(self.sequence_length)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        self.training_op = optimizer.minimize(self.loss)

        
    def build(self):
        # NER specific functions
        self.add_placeholder()
        self.word_embedding_fn()
        self.logits_op()
        self.pred_op()
        self.loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths


    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, self.config.batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

# =============================================================================
#             # tensorboard
#             if i % 10 == 0:
#                 self.file_writer.add_summary(summary, epoch*nbatches + i)
# =============================================================================

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        print(msg)

        return metrics["f1"]


    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100*acc, "f1": 100*f1}


    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds
    


