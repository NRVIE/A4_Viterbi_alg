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
    f1 = open(test_file)
    lines = f1.readlines()
    split_text = split_sen(lines)
    # Load output_file
    f2 = open(output_file, "a")

    for text in split_text:
        prob, prev = viterbi_alg(init_prob, trans_prob, em_prob, tag_list, text)
        text_with_tag = read_prev(tag_list, prob, prev, text)
        # Write text_with_tag to output_file
        for output in text_with_tag:
            f2.write(output + '\n')
    ##########################


def split_sen(text: list):
    """Return a nested list of sentence.
        ex: [['Hello', 'world', '!'],
             ['Hi', ',', 'James', '.']]
    """
    quote = False
    prev_word = None
    result = []
    curr_sen = []
    for line in text:
        curr_line = line.strip('\n')
        curr_word = curr_line.split(' : ')[0]

        curr_sen.append(curr_line)
        # Check if curr_word is the last word in a sentence
        if curr_word == '.' or curr_word == '!' or curr_word == '?':
            if not quote:
                # The end of sentence
                result.append(curr_sen)
                curr_sen = []
        elif curr_word == '"':
            if quote:
                # Change the status of quote
                quote = False
                if prev_word == '.' or prev_word == '!' or prev_word == '?':
                    # The end of sentence
                    result.append(curr_sen)
                    curr_sen = []
            else:
                # Change the status of quote
                quote = True
        # Update prev_word
        prev_word = curr_word

    if len(curr_sen) != 0:  # The last sentence doesn't end correctly.
        result.append(curr_sen)
    return result


def training(training_list: list):
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
    tag_list = []  # Stores all possible hidden states

    # Read the training date by using training_list.
    split_text = []
    for file in training_list:
        f = open(file)
        training_text = f.readlines()
        split_text = split_text + split_sen(training_text)
        f.close()
    total_sentence = len(split_text)

    # Counting for begin_count, prev_count, tag_count, and word_count
    prev_tag = None
    for sentence in split_text:
        for i in range(len(sentence)):
            new_line = sentence[i].split(' : ')
            curr_word = new_line[0]
            curr_tag = new_line[1]

            # Add curr_tag to tag_list if curr_tag not in tag_list
            if curr_tag not in tag_list:
                tag_list.append(curr_tag)

            if i == 0:  # The beginning of a sentence
                # Add curr_tag to begin_count
                if curr_tag not in begin_count:
                    begin_count[curr_tag] = 1
                else:
                    begin_count[curr_tag] += 1
            else:
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

            # Set prev_tag to curr_tag
            prev_tag = curr_tag

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

    return init_prob, trans_prob, em_prob, tag_list


def viterbi_alg(init_prob: dict, trans_prob: dict, em_prob: dict, tag_list: list, sentence: list):
    """Applying viterbi algorithm for finding the most likely tag sequence."""
    prob = np.zeros((len(sentence), len(tag_list)))
    prev = np.zeros((len(sentence), len(tag_list)))

    for i in range(len(tag_list)):
        if tag_list[i] in init_prob and sentence[0] in em_prob[tag_list[i]]:
            prob[0, i] = init_prob[tag_list[i]] * em_prob[tag_list[i]][sentence[0]]
        else:
            # if we never observed a tag at the beginning of a sentence or a word,
            # then we assume the probability of it is 0
            prob[0, i] = 0
        prev[0, i] = None

    for t in range(1, len(sentence)):
        sum_prob = 0
        for i in range(len(tag_list)):
            if sentence[t] in em_prob[tag_list[i]]:
                curr_em = em_prob[tag_list[i]][sentence[t]]
            else:
                # if we never observed a word with that tag, then set the probability to 0
                curr_em = 0
            prob[t, i], prev[t, i] = find_max(tag_list, list(prob[t - 1]), trans_prob, curr_em, i)
            sum_prob += prob[t, i]
        # Normalize
        for i in range(len(tag_list)):
            prob[t, i] = prob[t, i]/sum_prob

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


def read_prev(tag_list, prob, prev, sentence: list):
    """Return the output of the most likely tag sequence with a given sentence"""
    # Find the highest probability at the last word of the sentence
    last_max = 0
    last_max_ind = None
    tag_result = []
    for i in range(prob.shape[1]):
        if prob[prob.shape[0] - 1, i] > last_max:
            last_max = prob[prob.shape[0] - 1][i]
            last_max_ind = i
    # Backward traverse the most likely path
    for i in range(prev.shape[0] - 1, -1, -1):
        tag_result.append(sentence[i] + ' : ' + tag_list[last_max_ind])
        if i != 0:
            last_max_ind = int(prev[i, last_max_ind])
    tag_result.reverse()

    return tag_result


if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[parameters.index("-d") + 1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t") + 1]
    output_file = parameters[parameters.index("-o") + 1]
    print("Training files: " + str(training_list))
    print("Test file: " + test_file)
    print("Output file: " + output_file)

    # Start the training and tagging operation.
    tag(training_list, test_file, output_file)

    # # Test
    # f = open('data/test1.txt')
    # lines = f.readlines()
    # text = split_sen(lines)
    # init_prob, trans_prob, em_prob, tag_list = training('data/training1.txt')
    # prob, prev = viterbi_alg(init_prob, trans_prob, em_prob, tag_list, text[0])
    # result = read_prev(tag_list, prob, prev, text[0])
