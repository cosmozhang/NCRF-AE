import numpy as np
import time
import sys
import subprocess
import os
import random
from NCRFAE.neurocrf_autoencoder_EM import *
import gzip
import cPickle
from NCRFAE.utils import *
import argparse

def run(parsed_args):

    lfilename = parsed_args.labeled_set
    per_labeled = parsed_args.percent_labeled
    per_unlabeled = parsed_args.percent_unlabeled
    em_weight_unlabeled = parsed_args.em_weight_unlabeled

    lg_head = lfilename.split('/')[1]


    folder = lg_head + '_semi_EM' + '_results'
    if not os.path.exists(folder):
        os.mkdir(folder)

    # load the dataset
    f = gzip.open(lfilename,'rb')
    train_set, valid_set, test_set, dic = cPickle.load(f)
    print lfilename + " loaded."

    idx2label = dict((k, v) for v, k in dic['labels2idx'].iteritems())
    idx2word = dict((k, v) for v, k in dic['words2idx'].iteritems())
    idx2char = dict((k, v) for v, k in dic['chars2idx'].iteritems())

    idx2vec = dic['idx2vec']

    train_lex, train_char, train_y = train_set
    valid_lex, valid_char, valid_y = valid_set
    test_lex,  test_char,  test_y = test_set

    vocsize = len(idx2vec)

    charsize = len(idx2char)

    # number of classes
    n_classes = len(idx2label)
    # print n_classes

    char_embeddingdim = parsed_args.char_emb_dim
    embeddingdim = parsed_args.emb_dimension
    hiddensize = parsed_args.hiddensize
    char_hiddensize = parsed_args.char_hiddensize
    randomseed = parsed_args.seed
    windowsize = parsed_args.context_win
    dropout_rate = parsed_args.dropout_rate

    # initialize a random number generator
    rng = np.random.RandomState(randomseed)

    # word embeddings
    if parsed_args.random_emb:
        # add one for PADDING at the end or beginning(the dummy word) ; word vectors are parameters as well
        embeddings = 0.2 * rng.uniform(-1.0, 1.0, (vocsize+1, embeddingdim)).astype(np.float32)
    else:
        # using Mikolov's embeddings
        embeddings = np.zeros((vocsize+1, embeddingdim), dtype=np.float32)
        for idx, value in idx2vec.iteritems():
            embeddings[idx] = value

    # char embeddings
    char_embeddings = 0.2 * rng.uniform(-1.0, 1.0, (charsize+1, char_embeddingdim)).astype(np.float32)

    # instanciate the model
    classifier = CRF_Auto_Encoder(rng, embeddings, char_embeddings, hiddensize, char_hiddensize, embeddingdim, char_embeddingdim, windowsize, n_classes, vocsize+1, dropout_rate = dropout_rate)
    classifier.compile()

    # semi-supervised learning starting from here

    training_idxs = np.arange(len(train_lex))

    # train with early stopping on validation set
    best_res = -np.inf  # infinity

    #divide the training set into labeled data and unlabeled data
    n_threshold_labeled = len(train_lex)/100*per_labeled
    n_threshold_unlabeled = n_threshold_labeled + len(train_lex)/100*per_unlabeled

    #initialize parameters of decoder by using labeled dataset

    temp_theta_table = np.zeros((classifier.n_classes, classifier.dic_size))
    for idx, i in enumerate(training_idxs):
        if i < n_threshold_labeled:
            for x, y in zip(train_lex[i], train_y[i]): # x, y are indices of word, label
                temp_theta_table[y, x] += 1
    temp_theta_table = npsoftmax(temp_theta_table)
    classifier.decoder_update_func(temp_theta_table)

    for e in xrange(parsed_args.nepochs):
        # shuffle
        rng.shuffle(training_idxs)
        current_epoch = e

        # training the encoder
        tic = time.time()
        for idx, i in enumerate(training_idxs):
            trainx = contextwin(train_lex[i], windowsize)
            trainx_char = contextwin_char(train_char[i], windowsize)
            trainy = train_y[i]
            if i < n_threshold_labeled:
                cost_value, predicted_value = classifier.train_xy_func(trainx, trainx_char, trainy)
            elif i >= n_threshold_labeled and i < n_threshold_unlabeled:
                cost_value, predicted_value = classifier.train_x_func(trainx, trainx_char)
            else:
                continue

            if parsed_args.verbose:
                print '[Semi-supervised learning] per %2.2f%% epoch %i >> %2.2f%%' % (1*per_labeled, e, (idx+1)*100./len(train_lex)), 'completed in %.2f (sec) <<\r' % (time.time()-tic),
                sys.stdout.flush()

        new_theta_table = np.zeros((classifier.n_classes, classifier.dic_size))
        # directly optimize the decoder
        for idx, i in enumerate(training_idxs):
            if i < n_threshold_labeled:
                for x, y in zip(train_lex[i], train_y[i]): # x, y are indices of word, label
                    new_theta_table[y, x] += 1
            elif i >= n_threshold_labeled and i < n_threshold_unlabeled:
                trainx = contextwin(train_lex[i], windowsize)
                trainx_char = contextwin_char(train_char[i], windowsize)
                alpha_table, beta_table, Z = classifier.forward_backward_func(trainx, trainx_char)
                for t in xrange(train_lex[i].shape[0]):
                    expected_count = alpha_table[t] * beta_table[t] / Z * em_weight_unlabeled
                    v_id = train_lex[i][t]
                    new_theta_table[:,v_id] += expected_count
            else:
                continue
        new_theta_table = npsoftmax(new_theta_table)

        classifier.decoder_update_func(new_theta_table)

        # evaluation // back into the real world : id -> words

        # validation
        tic = time.time()
        predictions_valid = []
        for i in xrange(len(valid_lex)):
            validx = contextwin(valid_lex[i], windowsize)
            validx_char = contextwin_char(valid_char[i], windowsize)
            temp = classifier.infer_func(validx, validx_char).astype(np.int32)
            validpred = temp[1:-1]
            predictions_valid.append(map(lambda u: idx2label[u], validpred))

            if parsed_args.verbose:
                print '[Testing on validation set] per %2.2f%% epoch %i >> %2.2f%%' % (1*per_labeled, e, (i+1)*100./len(valid_lex)), 'completed in %.2f (sec) <<\r' % (time.time()-tic),
                sys.stdout.flush()


        groundtruth_valid = [map(lambda u: idx2label[u], y) for y in valid_y]
        words_valid = [map(lambda u: idx2word[u], w) for w in valid_lex]

        # compute the accuracy using pos

        res_valid = poseval(predictions_valid, groundtruth_valid, words_valid, folder + '/' + str(per_labeled) + '_current.valid.txt')

        if res_valid['wordacc'] > best_res:

            # testing
            tic = time.time()
            predictions_test = []
            for i in xrange(len(test_lex)):
                testx = contextwin(test_lex[i], windowsize)
                testx_char = contextwin_char(test_char[i], windowsize)
                temp = classifier.infer_func(testx, testx_char).astype(np.int32) # a list of integers
                testpred = temp[1:-1]
                predictions_test.append(map(lambda u: idx2label[u], testpred))

                if parsed_args.verbose:
                    print '[Testing on testing set] per %2.2f%% epoch %i >> %2.2f%%' % (1*per_labeled, e, (i+1)*100./len(test_lex)), 'completed in %.2f (sec) <<\r' % (time.time()-tic),
                    sys.stdout.flush()

            groundtruth_test = [map(lambda u: idx2label[u], y) for y in test_y]
            words_test = [map(lambda u: idx2word[u], w) for w in test_lex]

            res_test = poseval(predictions_test, groundtruth_test, words_test, folder + '/' + str(per_labeled) + '_current.test.txt')

            classifier.save(folder, '_' + str(current_epoch) + '.model')
            best_res = res_valid['wordacc']
            if parsed_args.verbose:
                print 'NEW BEST: epoch', e, 'valid acc', res_valid['wordacc'], 'best test acc', res_test['wordacc'], ' '*20
                print ''
            vsacc, vwacc = res_valid['sentenceacc'], res_valid['wordacc']
            tsacc, twacc = res_test['sentenceacc'],  res_test['wordacc']
            best_epoch = e
            subprocess.call(['mv', folder + '/' + str(per_labeled) + '_current.valid.txt', folder + '/' + str(per_labeled) + '_best.valid.txt'])

    print("semi-supervised")
    subprocess.call(['mv', folder + '/' + str(per_labeled) + '_current.test.txt', folder + '/' + str(per_labeled) + '_best.test.txt'])
    print 'BEST RESULT: epoch', best_epoch, 'with the model', folder, 'with percent of labeled data', per_labeled, 'percent of un-labeled data', per_unlabeled
    print 'valid word accuracy', vwacc, 'best test word accuracy', twacc

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--labeled_set", type=str, help="The labeled dataset")
    argparser.add_argument("--percent_labeled", type=int, default=50, help="Percentage of labeled data")
    argparser.add_argument("--percent_unlabeled", type=int, default=50, help="Percentage of unlabeled data")
    argparser.add_argument("--em_weight_unlabeled", type=float, default=1.00, help="weight for unlabled data in EM")
    argparser.add_argument("--verbose", type=bool, default=False, help="Verbose output")
    argparser.add_argument("--seed", type=int, default=2017, help="Set up the random seed")
    argparser.add_argument("--random_emb", type=bool, default=False, help="Use the randomized word embedding")
    argparser.add_argument("--emb_dimension", type=int, default=200, help="Word embedding dimension")
    argparser.add_argument("--char_emb_dim", type=int, default=15, help="Char embedding dimension")
    argparser.add_argument("--context_win", type=int, default=3, help="Context window size")
    argparser.add_argument("--hiddensize", type=int, default=20, help="Number of nodes in the hidden layer")
    argparser.add_argument("--char_hiddensize", type=int, default=20, help="Number of nodes in the hidden layer for char layer")
    argparser.add_argument("--nepochs", type=int, default=25, help="Maximum number of epochs")
    argparser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate for the dropout layer")

    parsed_args = argparser.parse_args()


    run(parsed_args)
