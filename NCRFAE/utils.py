import random
import sys
import numpy as np
import theano.tensor as T

def npsoftmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return (e_x / div).astype(np.float32)

def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    newbigls = []
    # compress
    for i in range(len(lol[0])):
        eg = [lol[0][i], lol[1][i], lol[2][i]]
        newbigls.append(eg)

    # shuffle
    random.seed(seed)
    random.shuffle(newbigls)

    # decompress
    train_lex, train_char, train_y = [], [], []
    for i in range(len(newbigls)):
        eg = newbigls[i]
        train_lex.append(eg[0])
        train_char.append(eg[1])
        train_y.append(eg[2])

    return (train_lex, train_char, train_y)

def poseval(p, g, w, filename):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS BOS BOS\n'
        for wl, wp, ww in zip(sl, sp, sw):
            out += ww + '\t' + wl + '\t' + wp + '\n'
        out += 'EOS EOS EOS\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()

    nsentence = len(w)
    nword = 0.0
    nwordcorrect = 0.0
    nsentencecorrect = 0.0
    if (len(p) != len(g)):
        # print p
        # print g
        sys.exit("Error, sentence length not equal")
    else:
        # print g, p, w
        for sl, sp, sw in zip(g, p, w):  # sl, sp, sw is in the sentence level
            if (sl == sp):
                nsentencecorrect += 1
            for wl, wp, ww in zip(sl, sp, sw):
                nword += 1
                if (wl == wp):
                    nwordcorrect += 1
    # print w
    # print nsentence, nword, nwordcorrect, nsentencecorrect
    sentenceacc = round(nsentencecorrect*100.0/nsentence, 2)
    wordacc = round(nwordcorrect*100.0/nword, 2)
    return {'sentenceacc': sentenceacc, 'wordacc': wordacc}

def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1 #ensure it is odd number
    assert win >=1 #ensure the window >=1
    l = list(l)

    lpadded = [-1 for i in range(win/2)] + l + [-1 for i in range(win/2)] #padding for each sentence before and after the real sentence
    out = [ lpadded[i:i+win] for i in xrange(len(l)) ] #return a list of lists

    assert len(out) == len(l) #ensure the length are equal
    return np.asarray(out, dtype = np.int32)

def contextwin_char(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1 #ensure it is odd number
    assert win >=1 #ensure the window >=1
    l = list(l)
    # l is a list of lists

    len_chars = len(l[0])

    lpadded = [[-1 for i in range(len_chars)] for j in range(win/2)] + l + [[-1 for i in range(len_chars)] for j in range(win/2)]

    out = [ lpadded[i:i+win] for i in xrange(len(l)) ]

    assert len(out) == len(l) #ensure the length are equal
    return np.asarray(out, dtype = np.int32)


def theano_logsumexp(x, axis=None):
    """
    Compute log(sum(exp(x), axis=axis) in a numerically stable
    fashion.
    Parameters
    ----------
    x : tensor_like
        A Theano tensor (any dimension will do).
    axis : int or symbolic integer scalar, or None
        Axis over which to perform the summation. `None`, the
        default, performs over all axes.
    Returns
    -------
    result : ndarray or scalar
        The result of the log(sum(exp(...))) operation.
    """
    xmax = T.max(x, axis = axis, keepdims = True)
    xmax_ = T.max(x, axis = axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis = axis))
