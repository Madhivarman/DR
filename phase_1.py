""" program description """
# tokenize the word
# remove the noisy words
# remove regex expression
# remove unwanted words
# abbrevate the words

# importing library we need
from __future__ import  division
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import operator
import  nltk
import string

import re

# if and else dictionary
dict_if = ['assuming that','in case that','in case of','on the assumption that','whenever','on the occasion','supposing that','granted that']
dict_else = ['otherwise','orthen','under other conditions','variously','if not','elseways','any other way']

#to see what are all the stopwords
#print(set(stopwords.words('english')))

# calling the main function
def main(input_text):

    #remove stopwords
    stop_words = set(stopwords.words('english'))
    #tokenizing the words
    word_token= word_tokenize(input_text)
    #filtering the sentence
    filtered_sentence = [w for w in word_token if not w in stop_words]
    #list
    filtered_sentence = []

    for w in word_token:
        if w not in stop_words:
            filtered_sentence.append(w)

    #join the list words
    filtered_sentence = " ".join(filtered_sentence)
    print("Filtered Sentence is :");print(filtered_sentence)


    # lemmatizer
    lem = WordNetLemmatizer()
    lem_words = lem.lemmatize(filtered_sentence,"v")
    # stemming
    ps = PorterStemmer()
    stemming_words = word_tokenize(filtered_sentence)

    #list formation
    stemming_collection_words = []

    for w in stemming_words:
        stemming_collection_words.append(ps.stem(w))

    #joining the words
    stemmed_words = " ".join(stemming_collection_words)
    print("The Sentence After Stemmed:");print(stemmed_words)


    #object standardize lookup
    __object_standardize(stemmed_words)

#__object_Standardize function called

def __object_standardize(user_input_object_expand):
    #lookup dictionary
    lookup_dict = {
        'abt': 'about', 'rt': 'Retweet', 'fb': 'Facebook', 'wanna': 'Want to',
        'dm': 'Direct message', 'ttul': 'Talk to you Later', 'tysm': 'Thankyou So much',
        'awsm': 'awesome', 'luv': 'love', 'k': 'Okay', 'msg': 'message','gltra':'ground left thrust resolve angle',
        'agltra' : 'aircraft ground left thrust resolve angle'
    }
    user_text = user_input_object_expand.split()

    after_obj_standardize = []

    for i in user_text:
        if i.lower() in lookup_dict:
            i = lookup_dict[i.lower()]

        after_obj_standardize.append(i)

    obj_standard_word = " ".join(after_obj_standardize)
    print("After Object Standardization:");print(obj_standard_word)

    #extraction of keyword phase done here
    __extraction_keywords(obj_standard_word)

def __pos_tag(text_to_post_tag):
    tokens = word_tokenize(text_to_post_tag)
    print("tokens are:",tokens)
    print("After PostTagging:");print(pos_tag(tokens))

#load stop word function called here
def load_stop_words(stop_word_file):
    stop_words = []
    for line in open(stop_word_file):
        if line.strip()[0:1] != '#':
            for word in line.split(): #incase more then one word is present in the line
                stop_words.append(word)
    return  stop_words

#build_stop_word_regex function called here
def build_stop_word_regex(stop_word_file_path):
    stop_word_list = load_stop_words(stop_word_file_path)
    stop_word_regex_list =[]
    for word in stop_word_list:
        word_regex = r'\b' + word + r'(?![\w-])' #to look for hiphen
        stop_word_regex_list.append(word_regex)

    stop_word_pattern = re.compile('|'.join(stop_word_regex_list),re.IGNORECASE)

    return  stop_word_pattern

#is_number function is called
def is_number(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False

#split_sentences function called
def split_sentences(text):
    sentence_delimiters = re.compile(u'[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s]')
    sentences = sentence_delimiters.split(text)
    return sentences

#separate_words function is called
def separate_words(text,min_word_return_size):
    splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
    words = []
    for single_word in splitter.split(text):
        current_word = single_word.strip().lower()
        #leave number in phrase
        if len(current_word) > min_word_return_size and current_word != '' and not is_number(current_word):
            words.append(current_word)

    return words

def generate_candidate_keywords(sentence_list,stopword_pattern):
    phrase_list = []
    for s in sentence_list:
        tmp = re.sub(stopword_pattern,'|',s.strip())
        phrase = tmp.split()
        for phrase in phrase:
            phrase = phrase.strip().lower()
            if phrase != "":
                phrase_list.append(phrase)

    return  phrase_list

#calculate_word_scores function called
def calculate_word_scores(phraseList):
    word_frequency = {}
    word_degree = {}
    for phrase in phraseList:
        word_list = separate_words(phrase, 0)
        word_list_length = len(word_list)
        word_list_degree = word_list_length - 1
        # if word_list_degree > 3: word_list_degree = 3 #exp.
        for word in word_list:
            word_frequency.setdefault(word, 0)
            word_frequency[word] += 1
            word_degree.setdefault(word, 0)
            word_degree[word] += word_list_degree  # orig.
            # word_degree[word] += 1/(word_list_length*1.0) #exp.
    for item in word_frequency:
        word_degree[item] = word_degree[item] + word_frequency[item]

    # Calculate Word scores = deg(w)/frew(w)
    word_score = {}
    for item in word_frequency:
        word_score.setdefault(item, 0)
        word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)  # orig.
    # word_score[item] = word_frequency[item]/(word_degree[item] * 1.0) #exp.
    return word_score

# keyword scores function called
def generate_candidate_keyword_scores(phrase_list, word_score):
    keyword_candidates = {}
    for phrase in phrase_list:
        keyword_candidates.setdefault(phrase, 0)
        word_list = separate_words(phrase, 0)
        candidate_score = 0
        for word in word_list:
            candidate_score += word_score[word]
        keyword_candidates[phrase] = candidate_score
    return keyword_candidates

#class is called here
class RakeKeyWordExtractor(object):
    def __init__(self,stop_words_path):
        self.stop_words_path = stop_words_path
        self.__stop_words_pattern = build_stop_word_regex(stop_words_path)

    def run(self,text):
        sentence_split = split_sentences(text)

        phrase_list = generate_candidate_keywords(sentence_split, self.__stop_words_pattern )

        word_scores = calculate_word_scores(phrase_list)

        keyword_candidates = generate_candidate_keyword_scores(phrase_list,word_scores)

        sorted_keywords = sorted(keyword_candidates.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sorted_keywords


#main module of the function called
def __extraction_keywords(want_to_extract_words):
    rake = RakeKeyWordExtractor("SmartStoplist.txt")
    keywords = rake.run(want_to_extract_words)
    print("The important Keywords are:")
    print(keywords)

# structure the data is done here
def __structure_the_data(structured_data):
    # changing the usertext according to if..else statement

    statement = structured_data
    removed_lexical_statement = statement.replace(",", " ")

    # splitting the statement according the question

    for i in dict_if:
        removed_lexical_statement = removed_lexical_statement.replace(i, "if")

    for i in dict_else:
        removed_lexical_statement = removed_lexical_statement.replace(i, "else")

    # splitting the statement into two statements
    # into if and else

    with_else = removed_lexical_statement.split("else")

    # tokenize the words which are in with_else

    for i in with_else:
        with_else_tokenize_path(i) # function calling


def with_else_tokenize_path(tokenization_words):
    stop_words = set(stopwords.words('english'))

    #tokenizations
    word_token = word_tokenize(tokenization_words)

    #filtered sentences
    filtered_sentences = [w for w in word_token if not w in stop_words]

    #list
    filtered_sentences = []

    #loop
    for w in word_token:
        if w not in stop_words:
            filtered_sentences.append(w)


    filtered_sentences = " ".join(filtered_sentences)
    print("Filtered Keywords are:")
    print("______________________")
    print(filtered_sentences)





#start of the function
user_variable = raw_input("Input the text :")

if __name__ == '__main__':
    main(user_variable)
    __pos_tag(user_variable)

    # second module work is done here
    __structure_the_data(user_variable)


