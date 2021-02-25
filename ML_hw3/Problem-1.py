"""
Put your NetID here. NetID: zc969 Name:Skye Chen
"""


import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

class Opts:
    def __init__(self, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('train', default=None, help='The path to the training set file.')
        parser.add_argument('--test', default=None, help='The path to the test set file. If used, the model should be in the final state and no further hyper-parameter tuning is allowed.')
        parser.add_argument('-t', '--threshold', type=int, default=26, help='The threshold of the number of times to choose a word.')
        self.__dict__.update(parser.parse_args(argv).__dict__)


def read_data(filename):
    """
    Read the dataset from the file given by name $filename.
    The returned object should be a list of pairs of data, such as
        [
            (True , ['a', 'b', 'c']),
            (False, ['d', 'e', 'f']),
            ...
        ]
    """
    # Fill in your code here.
    f=open(filename,'r')
    data_list = []
    for line in f.readlines():
        #get the label
        label, feature = line.split()[0], line.split()[1:] 
        if int(label) == 0:
            label = False
        else:
            label = True
        data_list.append((label,feature))
    return data_list

def split_train(original_train_data):
    """
    Split the original training set into two sets:
        * the training set, and
        * the validation set.
    """
    # Fill in your code here.
    validation_data = original_train_data[:1000]
    train_data =  original_train_data[1000:]
    return  train_data,validation_data


def create_wordlist(original_train_data, threshold=26):
    """
    Create a word list from the original training set.
    Only get a word if it appears in at least $threshold emails.
    """

    # Fill in your code here.
    word_dic = {}
    word_list = []
    for line in original_train_data:
        line_lst = list(line[1])
        line_lst = list(dict.fromkeys(line_lst))
        for word in line_lst:
            if word in word_dic.keys():
                word_dic[word]+=1
            else:
                word_dic[word]=1
                
    for word in word_dic:
        if word_dic[word]>=threshold:
            word_list.append(word)
    print('length of word list',len(word_list))
    return word_list


class Model:
    @staticmethod
    def count_labels(data):
        """
        Count the number of positive labels and negative labels.
        Returns (a tuple or a numpy array of two elements):
            * negative_count: a non-negative integer, which represents the number of negative labels;
            * positive_count: a non-negative integer, which represents the number of positive labels.
        """

        # Fill in your code here.
        positive_count = 0
        for item in data:
            positive_count += int(item[0])
        negative_count = len(data)-positive_count
        print('count positive_count: ',positive_count)
        print('count negative_count: ',negative_count)
        return (negative_count,positive_count)

    @staticmethod
    def count_words(wordlist, data):
        """
        Count the number of times that each word appears in emails under a given label.
        Returns (a numpy array):
            * word_counts: a numpy array with shape (2, L), where L is the length of $wordlist,
                - word_counts[0, i] represents the number of times that word $wordlist[i] appears in non-spam (negative) emails, and
                - word_counts[1, i] represents the number of times that word $wordlist[i] appears in spam (positive) emails.
        """
        # Fill in your code here.
        ################### Problem 1 #####################  
        word_counts = np.zeros([2,len(wordlist)])
        for i in range(len(wordlist)):
            for email in data:
                content_list = email[1]
                label = int(email[0])
                times = content_list.count(wordlist[i])
                if times != 0:
                    word_counts[label,i] = 1 + int(word_counts[label,i])
        # print("word_counts",word_counts)
        return word_counts
                
        
        
        ################### Bag of words #####################  
#        word_counts = np.zeros([2,len(wordlist)])
#        for i in range(len(wordlist)):
#            for email in data:
#                content_list = email[1]
#                label = int(email[0])
#                times = content_list.count(wordlist[i])
#                
#                if label == 1: #if spam
#                    word_counts[1,i] += times
#                else:
#                    word_counts[0,i] += times
                
        # return word_counts
        

    @staticmethod
    def calculate_probability(label_counts, word_counts):
        """
        Calculate the probabilities, both the prior and likelihood.
        Returns (a pair of numpy array):
            * prior_probs: a numpy array with shape (2, ), only two elements, where
                - prior_probs[0] is the prior probability of negative labels, and
                - prior_probs[1] is the prior probability of positive labels.
            * likelihood_probs: a numpy array with shape (2, L), where L is the length of the word list,
                - likelihood_probs[0, i] represents the likelihood probability of the $i-th word in the word list, given that the email is non-spam (negative), and
                - likelihood_probs[1, i] represents the likelihood probability of the $i-th word in the word list, given that the email is spam (positive).
        """
        # Fill in your code here.
        # Do not forget to add the additional counts.
        # ################### MLE #####################     
        # neg_pri_prob = label_counts[0]/sum(label_counts)
        # pos_pri_prob = 1-neg_pri_prob
        # prior_probs = np.array([neg_pri_prob,pos_pri_prob])
        # likelihood_probs = np.zeros([2,np.shape(word_counts)[1]])
        # for i in range(np.shape(word_counts)[1]):
        #     likelihood_probs[0,i] = word_counts[0,i]/label_counts[0]
        #     likelihood_probs[1,i] = word_counts[1,i]/label_counts[1]
        # print("sum of neg",sum(likelihood_probs[0,:]))
        ################### MAP #####################  
        ## take hallucinated word count = 1   
        neg_pri_prob = (label_counts[0]+1)/(sum(label_counts)+2)
        pos_pri_prob = 1-neg_pri_prob
        prior_probs = np.array([neg_pri_prob,pos_pri_prob])
        
        likelihood_probs = np.zeros([2,np.shape(word_counts)[1]])
        
        for i in range(np.shape(word_counts)[1]):
            likelihood_probs[0,i] = (word_counts[0,i]+1)/(label_counts[0]+2)
            likelihood_probs[1,i] = (word_counts[1,i]+1)/(label_counts[1]+2)
        return prior_probs,likelihood_probs

    def __init__(self, wordlist):
        self.wordlist = wordlist

    def fit(self, data):
        label_counts = self.__class__.count_labels(data)
        word_counts = self.__class__.count_words(self.wordlist, data)

        self.prior_probs, self.likelihood_probs = self.__class__.calculate_probability(label_counts, word_counts)

        # You may do some additional processing of variables here, if you want.
        # Suggestion: You may get the log of probabilities.

    def predict(self, x):
        """
        Predict whether email $x is a spam or not.
        Returns:
            * y: a boolean value indicating whether $x is a spam or not.
        """
        # Fill in your code here.
        prob_non_spam = 1
        prob_spam = 1
        for i in range(len(self.wordlist)):
            word = self.wordlist[i]
            if word in x:
                prob_non_spam = prob_non_spam * self.likelihood_probs[0,i]
                prob_spam = prob_spam * self.likelihood_probs[1,i]
            else:
                prob_non_spam = prob_non_spam * (1-self.likelihood_probs[0,i])
                prob_spam = prob_spam * (1-self.likelihood_probs[1,i])               
        prob_non_spam *= self.prior_probs[0]
        prob_spam *= self.prior_probs[1]

        if prob_non_spam >= prob_spam:
            return False
        else:
            return True
        


def main(argv):
    opts = Opts(argv)

    if opts.test is None:
        original_train_data = read_data(opts.train)
        train_data, val_data = split_train(original_train_data)
    else:
        original_train_data = read_data(opts.train)
        train_data = original_train_data
        val_data = read_data(opts.test)

    # Create the word list.
    wordlist = create_wordlist(original_train_data, opts.threshold)

    model = Model(wordlist)
    model.fit(train_data)

    error_count = sum([y != model.predict(x) for y, x in val_data])
    error_percentage = error_count / len(val_data) * 100
    # Test training data 
    error_count_train = sum([y != model.predict(x) for y, x in train_data])
    error_percentage_train = error_count_train / len(train_data) * 100
    ###########   code for 1b plot  ############
    # N_list = [200,600,1200,2400,4000]
    # Train_error = []
    # Val_error = []
    # for N in N_list:
    #     train_N = train_data[:N]
    #     model.fit(train_N)
    #     error_count = sum([y != model.predict(x) for y, x in val_data])
    #     error_percentage = error_count / len(val_data) * 100
    #     # Test training data 
    #     error_count_train = sum([y != model.predict(x) for y, x in train_N])
    #     error_percentage_train = error_count_train / len(train_N) * 100
    #     Train_error.append(error_percentage_train)
    #     Val_error.append(error_percentage)
    # print("Train_error  ",Train_error)
    # print('Val_error  ',)
    # plt.plot(N_list, Val_error, label='Validation Error')
    # plt.plot(N_list, Train_error,label='Training Error')
    # plt.xlabel("N")
    # plt.ylabel("Error rate")
    # plt.legend(loc='upper right')
    # plt.show()
    
    
    if opts.test is None:
        print("Validation error, # = {:>4d}, % = {:>8.4f}%.".format(error_count, error_percentage))
        print("Training error, # = {:>4d}, % = {:>8.4f}%.".format(error_count_train, error_percentage_train))
    else:
        print("Test error, # = {:>4d}, % = {:>8.4f}%.".format(error_count, error_percentage))


if __name__ == '__main__':
    main(sys.argv[1:])

#######  plot  #####


########## data for Problem 1c
# X = 30 N = 4000 MLE 
# Validation error, # =   48, % =   4.8000%.
# Training error, # =  194, % =   4.8500%.
# X = 28 N = 4000 MLE 
# Validation error, # =   51, % =   5.1000%.
# Training error, # =  196, % =   4.9000%.
# X = 26 N = 4000 MLE 
# Validation error, # =   57, % =   5.7000%.
# Training error, # =  199, % =   4.9750%.
# X = 24 N = 4000 MLE 
# Validation error, # =   60, % =   6.0000%.
# Training error, # =  203, % =   5.0750%.
# X = 22 N = 4000 MLE 
# Validation error, # =   63, % =   6.3000%.
# Training error, # =  207, % =   5.1750%.

###### Uncomment data below to get plot of Problem 1c
# X = [22,24,26,28,30] 
# Val_error = [0.063,0.06,0.057,0.051,0.048]
# Train_error  = [0.05175,0.05075,0.04975,0.049,0.0485]
# plt.plot(X, Val_error, label='Validation Error')
# plt.plot(X, Train_error,label='Training Error')
# plt.xlabel("X(word_threshold)")
# plt.ylabel("Error rate")
# plt.legend(loc='upper right')
# plt.show()


