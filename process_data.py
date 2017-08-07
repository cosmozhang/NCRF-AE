import cPickle
import numpy as np
import sys
from collections import OrderedDict

def format_chars(chars_sent_ls):
    max_leng = max([len(l) for l in chars_sent_ls])
    to_pads = [max_leng - len(l) for l in chars_sent_ls]
    for i, to_pad in enumerate(to_pads):
        if to_pad % 2 == 0:
            chars_sent_ls[i] = [0] * (to_pad / 2) + chars_sent_ls[i] + [0] * (to_pad / 2)
        else:
            chars_sent_ls[i] = [0] * (1 + (to_pad / 2)) + chars_sent_ls[i] + [0] * (to_pad / 2)
    return chars_sent_ls


def load_bin_vec(fname, vocab):
    """
    Loads word vecs from word2vec bin file
    """
    word_vecs = OrderedDict()
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                idx = vocab[word]
                word_vecs[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=200):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs:
            idx = vocab[word]
            word_vecs[idx] = np.random.uniform(-0.25,0.25,k)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_user(s):
    if len(s)>1 and s[0] == "@":
        return True
    else:
        return False

def is_url(s):
    if len(s)>4 and s[:5] == "http:":
        return True
    else:
        return False

def digits(n):
    digit_str = ''
    for i in range(n):
        digit_str = digit_str + 'DIGIT'
    return digit_str

def establishdic(fname, gname, hname, binfile):
    data = open(fname, "rb").readlines() + open(gname, "rb").readlines() + open(hname, "rb").readlines()

    char_dict = OrderedDict()
    vocab_dict = OrderedDict()
    tag_dict = OrderedDict()

    char_count = 0
    vocab_count = 0
    tag_count = 0

    for line in data:
        line = line.replace('\n', '').replace('\r', '')
        line = line.split("\t")
        if line == ['', ''] or line == [''] or line[0].isdigit() != True:
            continue
        vocab = line[1]
        tag = line[3]
        if is_number(vocab):  # check if the term is a number
            vocab = digits(len(vocab))
        if is_url(vocab):
            vocab = "URL"
        if is_user(vocab):
            vocab = "USR"
        if vocab not in vocab_dict:
            vocab_dict[vocab] = vocab_count
            vocab_count += 1
        if tag not in tag_dict:
            tag_dict[tag] = tag_count
            tag_count += 1

        # generate char dictionary
        chars = list(vocab)
        for char in chars:
            if char not in char_dict:
                char_dict[char] = char_count
                char_count += 1


    pos_dictionary = OrderedDict()
    pos_dictionary['words2idx'] = vocab_dict

    pos_dictionary['labels2idx'] = tag_dict

    pos_dictionary['chars2idx'] = char_dict

    wordvec_dict = load_bin_vec(binfile, vocab_dict)
    add_unknown_words(wordvec_dict, vocab_dict)

    pos_dictionary['idx2vec'] = wordvec_dict
    return pos_dictionary

def sepdata(fname, gname, hname, pos_dictionary):
    vocab_dict = pos_dictionary['words2idx']
    tag_dict = pos_dictionary['labels2idx']
    char_dict = pos_dictionary['chars2idx']

    # of all sets
    dataset_words = []
    dataset_labels = []
    dataset_chars = []
    for f in [fname, gname, hname]:
        data = open(f, "rb").readlines()

        # of a whole set
        words_set = []
        tag_labels_set = []
        chars_set = []

        # of a whole sentence
        example_words = []
        example_tag_labels = []
        example_char = []
        count = 0
        for line in data:
            line = line.replace('\n', '').replace('\r', '')
            line = line.split("\t")

            if (not line[0].isdigit()) and (line != ['']):
                continue # this is the heading line

            # this means a example finishes
            if (line == ['', ''] or line == ['']) and (len(example_words) > 0):
                words_set.append(np.array(example_words, dtype = "int32"))
                tag_labels_set.append(np.array(example_tag_labels, dtype = "int32"))
                chars_set.append(np.array(example_char, dtype = "int32"))

                # restart a new example after one finishes
                example_words = []
                example_tag_labels = []
                example_char = []
                count += 1

            else:  # part of an example
                vocab = line[1]
                tag = line[3]
                if is_number(vocab):  # check if the term is a number
                    vocab = digits(len(vocab))
                if is_url(vocab):
                    vocab = "URL"
                if is_user(vocab):
                    vocab = "USR"

                example_words.append(vocab_dict[vocab])
                example_tag_labels.append(tag_dict[tag])
                char_word_list = map(lambda u: char_dict[u], list(vocab))
                example_char.append(char_word_list)
                example_char = format_chars(example_char)
                # for each example do a padding

        dataset_words.append(words_set)
        dataset_labels.append(tag_labels_set)
        dataset_chars.append(chars_set)

    train_pos= [dataset_words[0], dataset_chars[0], dataset_labels[0]]

    valid_pos = [dataset_words[1], dataset_chars[1], dataset_labels[1]]

    test_pos = [dataset_words[2], dataset_chars[2], dataset_labels[2]]

    assert len(dataset_words[0]+dataset_words[1]+dataset_words[2]) == len(train_pos[0]) + len(valid_pos[0]) + len(test_pos[0])

    return train_pos, valid_pos, test_pos


def main():
    if len(sys.argv) != 6:
        sys.exit("file paths not specified")

    binfile = sys.argv[1]
    fname = sys.argv[2] # train file
    gname = sys.argv[3] # validation file
    hname = sys.argv[4] # test file
    outfilename = sys.argv[5]

    pos_dictionary = establishdic(fname, gname, hname, binfile)

    train_pos, valid_pos, test_pos = sepdata(fname, gname, hname, pos_dictionary)

    print "train pos examples", len(train_pos[0])
    print "valid pos examples", len(valid_pos[0])
    print "test pos examples", len(test_pos[0])

    with open(outfilename + ".pkl", "wb") as f:
        cPickle.dump([train_pos, valid_pos, test_pos, pos_dictionary], f)
    print "data %s is generated." % (outfilename + ".pkl")

if __name__ == '__main__':
    main()
