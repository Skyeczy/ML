"""
Net Id: zc969 Name: Skye Chen
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
    # print('length of word list',len(word_list))
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
    def count_words(wordlist, label_counts, data):
        """
        Count the number of times that each word appears in emails under a given label.
        Returns (a numpy array):
            * negative_word_counts: a numpy array with shape (L, N-non-spam), where L is the length of $wordlist, N-non-spam is the number of non-spam emails, and
                - negative_word_counts[i, j] represents the number of times that word $wordlist[i] appears in the j-th non-spam email; and
            * positive_word_counts: a numpy array with shape (L, N-spam), where L is the length of $wordlist, N-spam is the number of spam emails, and
                - positive_word_counts[i, j] represents the number of times that word $wordlist[i] appears in the j-th spam email.
        """
        # Fill in your code here.
        N_non_spam = label_counts[0]
        N_spam = label_counts[1]
        L = len(wordlist)
        negative_word_counts = np.zeros([L,N_non_spam])
        positive_word_counts = np.zeros([L,N_spam])
        non_spam_data = []
        spam_data = []
        for item in data:
            if int(item[0])==0:
                non_spam_data.append(item)
            else:
                spam_data.append(item)
        
        print('len of non spam', len(non_spam_data))
        print('count non spam', N_non_spam)
        for i in range(L):
            for j in range(N_non_spam):
                email = list(non_spam_data[j][1])
                label = non_spam_data[j][0]
                times = email.count(wordlist[i])
                negative_word_counts[i, j] = times 
                
        for i in range(L):
            for j in range(N_spam):
                email = list(spam_data[j][1])
                label = spam_data[j][0]
                times = email.count(wordlist[i])
                positive_word_counts[i, j] = times 
        # Add to more hallicated email(of spam and non-spam) containing every word to avoid zero count problem
        negative_word_counts = np.concatenate((negative_word_counts, 1*np.ones((len(wordlist),1))), axis=1)
        positive_word_counts = np.concatenate((positive_word_counts, 1*np.ones((len(wordlist),1))), axis=1)
        return negative_word_counts, positive_word_counts
                    

    @staticmethod
    def calculate_probability(label_counts, negative_word_counts, positive_word_counts):
        """
        Calculate the probabilities, both the prior and likelihood.
        Returns (a pair of numpy array):
            * prior_probs: a numpy array with shape (2, ), only two elements, where
                - prior_probs[0] is the prior probability of negative labels, and
                - prior_probs[1] is the prior probability of positive labels;
            * likelihood_mus: a numpy array with shape (2, L), where L is the length of the word list,
                - likelihood_mus[0, i] represents the mean value (mu) of the likelihood probability of the $i-th word in the word list, given that the email is non-spam (negative), and
                - likelihood_mus[1, i] represents the mean value (mu) of the likelihood probability of the $i-th word in the word list, given that the email is spam (positive); and
            * likelihood_sigmas: a numpy array with shape (2, L), where L is the length of the word list,
                - likelihood_sigmas[0, i] represents the deviation value (sigma) of the likelihood probability of the $i-th word in the word list, given that the email is non-spam (negative), and
                - likelihood_sigmas[1, i] represents the deviation value (sigma) of the likelihood probability of the $i-th word in the word list, given that the email is spam (positive).
        """

        # Fill in your code here.
        ################### MAP #####################     
        ### Since previously added two emails, one spam, one non-spam
        ### Here the prior probability need to add 1 to each label
        neg_pri_prob = (label_counts[0]+1)/(sum(label_counts)+2)
        pos_pri_prob = 1-neg_pri_prob
        prior_probs = np.array([neg_pri_prob,pos_pri_prob])
        
        L= np.shape(negative_word_counts)[0]
        N_non_spam = np.shape(negative_word_counts)[1]
        N_spam = np.shape(positive_word_counts)[1]
        
        likelihood_mus = np.zeros([2,L])
        for i in range(L):
            likelihood_mus[0,i] = np.mean(negative_word_counts[i,:])
            likelihood_mus[1,i] = np.mean(positive_word_counts[i,:])
            
        likelihood_sigmas = np.zeros([2,L])
        for i in range(L):
            sigma = 0
            for j in range(N_non_spam):
                sigma += (negative_word_counts[i,j]-likelihood_mus[0,i])**2
            sigma = (sigma/N_non_spam)**0.5
            likelihood_sigmas[0,i] = sigma 

        for i in range(L):
            sigma = 0
            for j in range(N_spam):
                sigma += (positive_word_counts[i,j]-likelihood_mus[1,i])**2
            sigma = (sigma/N_spam)**0.5
            likelihood_sigmas[1,i] = sigma   
        print('pri:',prior_probs)
        print("likelihood_mus: ",likelihood_mus)
        print('likelihood_sigmas:', likelihood_sigmas)
        print('largest neg:',max(likelihood_sigmas[0]))
        return prior_probs,likelihood_mus,likelihood_sigmas

    def __init__(self, wordlist):
        self.wordlist = wordlist

    def fit(self, data):
        label_counts = self.__class__.count_labels(data)
        negative_word_counts, positive_word_counts = self.__class__.count_words(self.wordlist, label_counts, data)

        self.prior_probs, self.likelihood_mus, self.likelihood_sigmas = self.__class__.calculate_probability(label_counts, negative_word_counts, positive_word_counts)

        # You may do some additional processing of variables here, if you want.

    def predict(self, x):
        """
        Predict whether email $x is a spam or not.
        Returns:
            * y: a boolean value indicating whether $x is a spam or not.
        """

        # Fill in your code here.
        # In this problem, take log of the function, so that we only need to calculate the sum of conditional probabilities
        # And we can just compare the sum, therefore here they are added together
        prob_non_spam = np.log(self.prior_probs[0])
        prob_spam = np.log(self.prior_probs[1])
        
        # prob_non_spam = self.prior_probs[0]
        # prob_spam = self.prior_probs[1]
        
        for i in range(len(self.wordlist)):
            word = self.wordlist[i]
            if word in x:
                time = x.count(word)
                non_prob = -0.5*((time-self.likelihood_mus[0,i])/self.likelihood_sigmas[0,i])**2- np.log((2*np.pi*self.likelihood_sigmas[0,i]**2)**0.5)
                prob_non_spam += non_prob
                spam_prob =  -0.5*((time-self.likelihood_mus[1,i])/self.likelihood_sigmas[1,i])**2- np.log((2*np.pi*self.likelihood_sigmas[1,i]**2)**0.5)
                prob_spam += spam_prob

            else:
                # if self.likelihood_sigmas[1,i] == 0:
                #     self.likelihood_sigmas[1,i] = 0.0000001
                # if self.likelihood_sigmas[0,i] == 0:
                #     self.likelihood_sigmas[0,i] = 0.0000001
                    
                non_prob = -0.5*((0-self.likelihood_mus[0,i])/self.likelihood_sigmas[0,i])**2 -np.log((2*np.pi*self.likelihood_sigmas[0,i]**2)**0.5)
                prob_non_spam += non_prob
                spam_prob =  -0.5*((0-self.likelihood_mus[1,i])/self.likelihood_sigmas[1,i])**2 - np.log((2*np.pi*self.likelihood_sigmas[1,i]**2)**0.5)
                prob_spam += spam_prob
        return True if prob_non_spam < prob_spam else False
        # if prob_non_spam >= prob_spam:
        #     return False
        # else:
        #     return True
        
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
    ##### take N = 600 
    best_N = 600
    train_data = train_data[:best_N]
    model.fit(train_data)
    
    ###########   code for 3b plot  ############
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
    
    # Test training data 
    error_count_train = sum([y != model.predict(x) for y, x in train_data])
    error_percentage_train = error_count_train / len(train_data) * 100
    
    
    error_count = sum([y != model.predict(x) for y, x in val_data])
    error_percentage = error_count / len(val_data) * 100

    if opts.test is None:
        print("Validation error, # = {:>4d}, % = {:>8.4f}%.".format(error_count, error_percentage))
        print("Training error, # = {:>4d}, % = {:>8.4f}%.".format(error_count_train, error_percentage_train))

    else:
        print("Test error, # = {:>4d}, % = {:>8.4f}%.".format(error_count, error_percentage))


if __name__ == '__main__':
    main(sys.argv[1:])
