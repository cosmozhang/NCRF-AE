'''
Cosmo Zhang @ Purdue
neurocrf_autodecoder_EM.py
'''

import theano
import theano.tensor as T
import numpy as np
from neuromodules import *
from utils import *
import nnoptimizer
import os
import cPickle
# theano.config.compute_test_value = 'raise'

class CRF_Auto_Encoder(object):
    def __init__(self, rng, embeddings, char_embeddings, hiddensize, char_hiddensize, embedding_dim, char_embedding_dim, window_size, num_tags, dic_size, dropout_rate = 0.7):
        self.rng = rng
        self.inputX = T.imatrix('inputX') # a sentence, shape (T * window_size)
        self.inputX_chars = T.itensor3('inputX_chars') # a sentence, shape (T * max numbe of chars in a word)
        self.inputY = T.ivector('inputY') # tags of a sentence
        self.is_train = T.iscalar('is_train')

        self.new_theta = T.fmatrix('new_theta')

        self.dropout_rate = dropout_rate
        self.nhidden = hiddensize
        self.char_nhidden = char_hiddensize # for now set the number of hidden units the same
        self.embedding_dim = embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.window_size = window_size
        self.n_classes = num_tags
        self.dic_size = dic_size

        # for testing in compling
        self.inputX.tag.test_value = np.ones((10, window_size)).astype(np.int32)
        self.inputX_chars.tag.test_value = np.ones((10, window_size, 8)).astype(np.int32)
        self.inputY.tag.test_value = np.ones(10).astype(np.int32)

        self.Embeddings = theano.shared(value = embeddings, name = "Embeddings", borrow = True)
        self.Char_Embeddings = theano.shared(value = char_embeddings, name = "Char_Embeddings", borrow = True)

        # word embeddings
        self.inputW = self.Embeddings[self.inputX]

        # char embeddings
        self.inputC = self.Char_Embeddings[self.inputX_chars].dimshuffle([2, 0, 1, 3])

        self.params = [self.Embeddings, self.Char_Embeddings]

    def forward_z(self, unary_potentials, interaction_potentials, viterbi = False):

        def inner_function(unary, alpha_tm1_max, alpha_tm1_min, alpha_tm1, interaction):
            # interaction is a (classes by classes) matrix
            # unary is a (classes) vector
            alpha_tm1_max = alpha_tm1_max.dimshuffle(0, 'x')
            alpha_tm1_min = alpha_tm1_min.dimshuffle(0, 'x')
            alpha_tm1 = alpha_tm1.dimshuffle(0, 'x')
            unary = unary.dimshuffle('x', 0)

            out1 = T.max( alpha_tm1_max + unary + interaction, axis = 0)
            out2 = T.min( alpha_tm1_min + unary + interaction, axis = 0)
            out3 = theano_logsumexp( alpha_tm1 + unary + interaction, axis = 0)
            out_argmax = T.argmax( alpha_tm1 + unary + interaction, axis = 0)

            return [out1, out2, out3, out_argmax]

        assert unary_potentials.ndim == 2 #timesteps, classes
        assert interaction_potentials.ndim == 2 #classes+2, classes+2

        initial = unary_potentials[0]

        [alpha_max, alpha_min, alpha, argmax_preds], _ = theano.scan(fn = inner_function,
                                sequences = [unary_potentials[1:]],
                                outputs_info = [initial, initial, initial, None],
                                non_sequences = interaction_potentials)

        def return_seq(trace_at_t, label_idx):

            return trace_at_t[label_idx]

        bestseq, _ = theano.scan(fn = return_seq,
                                sequences = argmax_preds[::-1],
                                outputs_info = T.argmax(alpha[-1]))

        pred_seq = T.concatenate([bestseq[::-1], [T.argmax(alpha[-1])]], axis = 0)

        adplr = T.exp(T.max(alpha_max[-1], axis = 0) - theano_logsumexp(alpha[-1], axis=0)) - T.exp(T.min(alpha_min[-1], axis = 0) - theano_logsumexp(alpha[-1], axis=0))

        if viterbi:
            return T.max(alpha_max[-1], axis = 0), pred_seq, adplr
        else:
            return theano_logsumexp(alpha[-1], axis=0), alpha[:-1, 0:-2], adplr

    def backward_z(self, unary_potentials, interaction_potentials):

        def inner_function(unary, beta_tm1_max, beta_tm1, interaction):
            # interaction is a (classes by classes) matrix
            # unary is a (classes) vector
            beta_tm1_max = beta_tm1_max.dimshuffle(0, 'x')
            beta_tm1 = beta_tm1.dimshuffle(0, 'x')
            unary = unary.dimshuffle('x', 0)

            out1 = T.max( beta_tm1_max + unary + interaction, axis = 0)
            out3 = theano_logsumexp( beta_tm1 + unary + interaction, axis = 0)
            out_argmax = T.argmax( beta_tm1 + unary + interaction, axis = 0)

            return [out1, out3, out_argmax]

        assert unary_potentials.ndim == 2 #timesteps, classes
        assert interaction_potentials.ndim == 2 #classes+2, classes+2

        initial = unary_potentials[-1]

        [beta_max, beta, argmax_preds], _ = theano.scan(fn = inner_function,
                                sequences = [unary_potentials[::-1][1:]],
                                outputs_info = [initial, initial, None],
                                non_sequences = interaction_potentials)

        return theano_logsumexp(beta[-1], axis=0), beta[::-1][1:, 0:-2]

    def trainsition(self):

        a = self.rng.uniform(-1, 1, size = (self.n_classes + 2, self.n_classes + 2)).astype(np.float32)
        trans = theano.shared(value = a, name = 'trainsition')

        self.params = self.params + [trans]
        return trans

    def simple_decoder(self):

        # parameters of simple look_up decoder
        temp = self.rng.uniform(-1, 1, size = (self.n_classes, self.dic_size)).astype(np.float32)
        self.MAT = theano.shared(value = temp, name = "MAT", borrow = True)

        mat = T.nnet.softmax(self.MAT).T
        xinds = self.inputX[:, self.window_size/2]
        ret = T.log(mat[xinds])

        return ret, mat

    def compile(self):

        ### Encoder part

        ## char RNN layer, use bidirectional LSTM
        self.charLayer_f = LstmLayer(self.rng, self.inputC, self.char_embedding_dim, self.char_nhidden, activation = T.nnet.relu, retrun_final_only = True, batch_mode = True)
        self.charLayer_b = LstmLayer(self.rng, self.inputC[::-1], self.char_embedding_dim, self.char_nhidden, activation = T.nnet.relu, retrun_final_only = True, batch_mode = True)

        l_inputWC = T.concatenate([self.inputW, self.charLayer_f.output, self.charLayer_b.output], axis = 2).reshape((self.inputX.shape[0], (self.embedding_dim+2*self.char_nhidden)*self.window_size))

        ## Adding dropout layer
        if self.dropout_rate:
            self.dropout_layer = DropoutLayer(self.rng, l_inputWC, dropout_rate=self.dropout_rate)
            input_train = self.dropout_layer.output
            input_test = (1 - self.dropout_rate) * l_inputWC
            l_inputWC = T.switch(T.neq(self.is_train, T.constant(0)), input_train, input_test)

        ## MLP
        self.non_linearLayer = HiddenLayer(self.rng, l_inputWC, (self.embedding_dim+2*self.char_nhidden)*self.window_size, self.nhidden, activation = T.nnet.relu)
        self.finalLayer = HiddenLayer(self.rng, self.non_linearLayer.output, self.nhidden, self.n_classes, activation = None)

        up_input = self.finalLayer.output # shape (n_timesteps * n_classes)

        ## transition
        ip_input = self.trainsition() # shape ((n_classes+2) * (n_classes+2))

        ### decoder part

        ## simple decoder
        down_input, mat = self.simple_decoder() # shape (n_timesteps * n_classes)

        ### compute score

        ## score from encoder + decoder
        u_score_input = down_input + up_input

        # params for MLP as encoder
        self.params = self.params + self.non_linearLayer.params + self.finalLayer.params + self.charLayer_f.params + self.charLayer_b.params

        small = -10000
        b_s = np.array([[small] * self.n_classes + [0, small]]).astype(np.float32)
        e_s = np.array([[small] * self.n_classes + [small, 0]]).astype(np.float32)

        # the numerater
        tempn = T.concatenate([u_score_input, small * T.ones((self.inputX.shape[0], 2))], axis=1)
        score_n = T.concatenate([b_s, tempn, e_s], axis=0)

        # the partition function
        tempd = T.concatenate([up_input, small * T.ones((self.inputX.shape[0], 2))], axis=1)
        score_z = T.concatenate([b_s, tempd, e_s], axis=0)

        # Score from transitions
        b_id = theano.shared(value=np.array([self.n_classes], dtype=np.int32))
        e_id = theano.shared(value=np.array([self.n_classes + 1], dtype=np.int32))
        padded_tags_ids = T.concatenate([b_id, self.inputY, e_id], axis=0)

        # encoder + decoder
        left_supervise = u_score_input[T.arange(self.inputY.shape[0]), self.inputY].sum() + ip_input[padded_tags_ids[T.arange(self.inputY.shape[0] + 1)], padded_tags_ids[T.arange(self.inputY.shape[0] + 1) + 1]].sum()

        # the numerator for un_labeled
        left_unsupervise, alpha, alr = self.forward_z(score_n, ip_input)

        _, beta = self.backward_z(score_n, ip_input)

        # the partition function
        log_partition, _, _ = self.forward_z(score_z, ip_input)

        ret_v, prediction, _ = self.forward_z(score_n, ip_input, viterbi = True)

        cost_unsupervise = - (left_unsupervise - log_partition)

        # combine two loss
        cost_supervise = - (left_supervise - log_partition) + cost_unsupervise

        updates_supervise = nnoptimizer.Adadelta(cost=cost_supervise, params = self.params)

        ## using Adadelta to optimize the encoder
        updates_unsupervise = nnoptimizer.Adadelta(cost=cost_unsupervise, params = self.params)

        self.train_xy_func = theano.function(inputs = [self.inputX, self.inputX_chars, self.inputY], outputs = [cost_supervise, prediction], updates = updates_supervise, on_unused_input = 'ignore', givens=({self.is_train: np.asarray(1).astype(np.int32)} if self.dropout_rate else {self.is_train: np.asarray(0).astype(np.int32)}))

        self.train_x_func = theano.function(inputs = [self.inputX, self.inputX_chars], outputs = [cost_unsupervise, prediction], updates = updates_unsupervise, on_unused_input = 'ignore', givens=({self.is_train: np.asarray(1).astype(np.int32)} if self.dropout_rate else {self.is_train: np.asarray(0).astype(np.int32)}))

        self.infer_func = theano.function(inputs = [self.inputX, self.inputX_chars], outputs = prediction, on_unused_input = 'ignore', givens=({self.is_train: np.asarray(0).astype(np.int32)} if self.dropout_rate else {self.is_train: np.asarray(1).astype(np.int32)}))

        self.forward_backward_func = theano.function(inputs = [self.inputX, self.inputX_chars], outputs = [alpha, beta, log_partition], on_unused_input = 'ignore', givens=({self.is_train: np.asarray(1).astype(np.int32)}))

        self.decoder_update_func = theano.function(inputs = [self.new_theta], updates = {self.MAT: self.new_theta})

        print "Model compiled!"

    def save(self, folder, model_name):
        pvals = [p.get_value() for p in self.params]
        # print folder, model_name
        f = open(os.path.join(folder, model_name), 'wb')
        cPickle.dump(pvals, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

def main():
    # For debug purpose

    randomseed = 12345
    rng = np.random.RandomState(randomseed)
    embeddings = 0.2 * rng.uniform(-1.0, 1.0, (200+1, 20)).astype(np.float32)
    char_embeddings = 0.2 * rng.uniform(-1.0, 1.0, (10, 5)).astype(np.float32)
    crfdecoder = CRF_Auto_Encoder(rng, embeddings, char_embeddings, 10, 20, 20, 5, 3, 10, 201)
    crfdecoder.compile()

    # create simulated dataset
    Trainset, Testset = [], []
    for i in range(20):
        n_timesteps = rng.random_integers(1, 10).astype(np.int32)

        trainX_chars = rng.randint(0, 10, size = (n_timesteps, 3, 8)).astype(np.int32)
        trainX = rng.randint(0, 201, size=(n_timesteps, )).astype(np.int32)
        trainX_win = contextwin(trainX, 3)
        trainY = rng.randint(0, 10, size=(n_timesteps, )).astype(np.int32)
        Trainset.append((trainX_chars, trainX, trainX_win, trainY))

    # training and testing
    for i in range(20):
        # labeled data
        trainX_chars, trainX, trainX_win, trainY = Trainset[i]

        cost_value, predicted_value = crfdecoder.train_xy_func(trainX_win, trainX_chars, trainY, 0.002)

    new_theta_table = np.zeros((crfdecoder.n_classes, crfdecoder.dic_size))

    for i in range(20):
        # labeled data
        trainX_chars, trainX, trainX_win, trainY = Trainset[i]

        # train encoder
        cost_value, predicted_value = crfdecoder.train_x_func(trainX_win, trainX_chars, 0.002)

        # fix encoder and get global optimization of decoder based on EM algorithm
        alpha_table, beta_table, Z = crfdecoder.forward_backward_func(trainX_win, trainX_chars)
        for t in xrange(trainX.shape[0]):
            expected_count = alpha_table[t] * beta_table[t] / Z
            v_id = trainX[t]
            new_theta_table[:,v_id] += expected_count

    new_theta_table = (np.exp(new_theta_table) / np.sum(np.exp(new_theta_table), axis=1)[:, np.newaxis]).astype(np.float32)

    print crfdecoder.MAT.get_value().shape, new_theta_table.shape

    crfdecoder.decoder_update_func(new_theta_table)


    for i in range(20):

        trainX_chars, trainX, trainX_win, trainY = Trainset[i]

        predicted_value = crfdecoder.infer_func(trainX_win, trainX_chars)
        print trainY, predicted_value[1:-1]

if __name__ == '__main__':
    main()
