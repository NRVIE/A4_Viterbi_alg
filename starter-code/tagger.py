# The tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)
import os
import sys
import numpy as np


def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")
    ##########################
    init_prob, trans_prob, em_prob, tag_list = training(training_list)

    # Read test_file
    f = open(test_file)
    lines = f.readlines()

    ##########################


def split_sen(text: list):
    """Return a nested list of sentence.
        ex: [['Hello', 'world', '!'],
             ['Hi', ',', 'James', '.']]
    """
    quote = False
    prev_word = None
    curr_word = None
    result = []
    for line in text:
        curr_word = line
        curr_sen = []
        if prev_word is None:
            ...
        elif (prev_word == '.' or prev_word == '!') and quote is False:
            ...
        elif ...:
            ...


def training(training_list):
    """Getting all the probabilities we need for HMM."""
    # Dictionary for counting the occurrence (corresponding value) of a given tag (key) when the tag at the beginning of a sentence
    begin_count = {}
    # Nested dictionary: the observed tag is a key for the dictionary,
    # and its correspond value is another dictionary with
    # previous tag of it (key in nested dictionary) and correspond value is the times this two tags occurred in the same position.
    prev_count = {}
    # Nested dictionary: the tag is a key for the dictionary,
    # and its correspond value is another dictionary with
    # the specific word of this tag (key) and correspond value is the times this two word occurred.
    word_count = {}
    # Dictionary for counting the occurrence (corresponding value) of a given tag (key)
    tag_count = {}
    total_sentence = 0
    tag_list = []  # Stores all possible hidden states

    # Read the training date by using training_list.
    f = open(training_list)
    training_text = f.readlines()

    # Counting for begin_count, prev_count, tag_count, and word_count
    prev_word = None
    prev_tag = None
    for line in training_text:
        new_line = line.strip('\n')
        new_line = new_line.split(' : ')
        curr_word = new_line[0]
        curr_tag = new_line[1]
        # Add curr_tag to tag_list if curr_tag not in tag_list
        if curr_tag not in tag_list:
            tag_list.append(curr_tag)
        if prev_word is None or prev_word == '.':
            # curr_word is a word at the beginning of a sentence,
            # then add the count of curr_tag to begin_count
            if curr_tag not in begin_count:
                begin_count[curr_tag] = 1
            else:
                begin_count[curr_tag] += 1

        else:  # not the beginning of the sentence
            # Count prev_count
            if curr_tag not in prev_count:
                prev_count[curr_tag] = dict()
                prev_count[curr_tag][prev_tag] = 1
            else:
                if prev_tag not in prev_count[curr_tag]:
                    prev_count[curr_tag][prev_tag] = 1
                else:
                    prev_count[curr_tag][prev_tag] += 1

        # Count tag
        if curr_tag not in tag_count:
            tag_count[curr_tag] = 1
        else:
            tag_count[curr_tag] += 1

        # Count word
        if curr_tag not in word_count:
            word_count[curr_tag] = dict()
            word_count[curr_tag][curr_word] = 1
        else:
            if curr_word not in word_count[curr_tag]:
                word_count[curr_tag][curr_word] = 1
            else:
                word_count[curr_tag][curr_word] += 1

        # Set prev_word to curr_word
        prev_word = curr_word
        # Set prev_tag to curr_tag
        prev_tag = curr_tag
        # Check whether is a sentence end
        if curr_word == '.':
            total_sentence += 1

    # Calculating the probabilities for HMM.
    init_prob = dict()  # Same structure as begin_count
    trans_prob = dict()  # Same structure as prev_count
    em_prob = dict()  # Same structure as word_count

    for key in begin_count:
        init_prob[key] = begin_count[key] / total_sentence

    for key in prev_count:
        trans_prob[key] = dict()
        for prev_tag in prev_count[key]:
            trans_prob[key][prev_tag] = prev_count[key][prev_tag] / tag_count[prev_tag]

    for key in word_count:
        em_prob[key] = dict()
        for word in word_count[key]:
            em_prob[key][word] = word_count[key][word] / tag_count[key]

    f.close()
    return init_prob, trans_prob, em_prob, tag_list


def viterbi_alg(tag_list: list, init_prob: dict, trans_prob: dict, em_prob: dict, sentence: list):
    """Applying viterbi algorithm for finding the most likely tag sequence."""
    prob = np.zeros((len(sentence), len(tag_list)))
    prev = np.zeros((len(sentence), len(tag_list)))

    for i in range(len(tag_list)):
        if tag_list[i] in init_prob and sentence[0] in em_prob[tag_list[i]]:
            prob[0, i] = init_prob[tag_list[i]] * em_prob[tag_list[i]][sentence]
        else:
            # if we never observed a tag at the beginning of a sentence or a word,
            # then we assume the probability of it is 0
            prob[0, i] = 0
        prev[0, i] = None

    for t in range(1, len(sentence)):
        for i in range(len(tag_list)):
            if sentence[t] in em_prob[tag_list[i]]:
                curr_em = em_prob[tag_list[i]][sentence[t]]
            else:
                # if we never observed a word with that tag, then set the probability to 0
                curr_em = 0
            prob[t, i], prev[t, i] = find_max(tag_list, list(prob[t - 1]), trans_prob, curr_em, i)

    return prob, prev


def find_max(tag_list: list, prob: list, trans_prob: dict, curr_em: float, curr_i: int):
    """Helper function for viterbi_alg."""
    max_prob = 0
    max_x = None
    for x in range(len(tag_list)):
        if tag_list[x] in trans_prob[tag_list[curr_i]]:
            curr_trans = trans_prob[tag_list[curr_i]][tag_list[x]]
        else:
            curr_trans = 0
        curr_prob = prob[x] * curr_trans * curr_em

        # max detection
        if curr_prob > max_prob:
            max_prob = curr_prob
            max_x = x

    return max_prob, max_x



if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[parameters.index("-d") + 1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t") + 1]
    output_file = parameters[parameters.index("-o") + 1]
    # print("Training files: " + str(training_list))
    # print("Test file: " + test_file)
    # print("Output file: " + output_file)

    # Start the training and tagging operation.
    tag(training_list, test_file, output_file)
