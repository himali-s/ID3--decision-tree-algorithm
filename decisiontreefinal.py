import csv

import sys

import ast

from collections import Counter

import copy

import random

from math import log


#function calculates the  entropy of data set

def entropy(rows):

    log_base_2 = lambda x: log(x) / log(2)

    results = uniquecounts(rows)

    entr = 0.0

    for r in results.keys():

        p = float(results[r]) / len(rows)

        entr = entr - p * log_base_2(p)

    return entr


#function  splits the  dataset based on an attribute

def divideset(rows, column, value):

    split_function = None

    if isinstance(value, int):

        split_function = lambda row: row[column] >= value

    else:

        split_function = lambda row: row[column] == value



    set1 = [row for row in rows if split_function(row)]

    set2 = [row for row in rows if not split_function(row)]

    return (set1, set2)







#function to calculate the variance impurity of data set

def varianceImpurity(rows):

    if len(rows) == 0: return 0

    results = uniquecounts(rows)

    total_samples = len(rows)

    variance_impurity = (results['0'] * results['1']) / (total_samples ** 2)

    return variance_impurity




#count number of values based on class attribute and return a dictionary

def uniquecounts(rows):

    results = {}

    for row in rows:

        # The target variable is the last column

        r = row[len(row) - 1]

        if r not in results: results[r] = 0

        results[r] += 1

    return results


#create class for a node of tree.

#A leaf node will have results as unique counts for each class variable

#A non leaf node will have subsequent branches as true branch and false branch

class decisionnode:

    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):

        
        self.col = col

        self.value = value

        self.results = results

        self.tb = tb

        self.fb = fb





#function used to split data set based on entropy(default) or variance impurity after calculating gain

def buildtree(rows, scoref=entropy):

    if len(rows) == 0: return decisionnode()

    current_score = scoref(rows)



    best_gain = 0.0

    best_criteria = None

    best_sets = None



    #the last column is the target attribute

    column_count = len(rows[0]) - 1

    for col in range(0, column_count):

        #divide data sets based on each attribute and calculate gain based on column. Select the attribute which results in best gain

        global column_values

        column_values = {}

        for row in rows:

            column_values[row[col]] = 1

        for value in column_values.keys():

            (set1, set2) = divideset(rows, col, value)



            # Calculate gain based on entropy(Information gain) or variance impurity based on requirement

            p = float(len(set1)) / len(rows)

            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)

            if gain > best_gain and len(set1) > 0 and len(set2) > 0:  # set must not be empty

                best_gain = gain

                best_criteria = (col, value)

                best_sets = (set1, set2)



    # Create the sub branches

    if best_gain > 0:

        trueBranch = buildtree(best_sets[0])

        falseBranch = buildtree(best_sets[1])

        return decisionnode(col=best_criteria[0], value=best_criteria[1],

                            tb=trueBranch, fb=falseBranch)

    else:

        return decisionnode(results=uniquecounts(rows))





#print tree in required format

def printtree(tree, header_data, indent):

    if tree.results != None:

        for key in tree.results:

            print(str(key))

    else:

        print("")

        print(indent + str(header_data[tree.col]) + ' = ' + str(tree.value) + ' : ', end="")

        printtree(tree.tb, header_data, indent + '  |')



        print(indent + str(header_data[tree.col]) + ' = ' + str(int(tree.value) ^ 1) + ' : ', end="")

        printtree(tree.fb, header_data, indent + '  |')





#function to calculate the accuracy

def tree_accuracy(rows, tree):

    correct_predictions = 0

    for row in rows:

        pred_val = classify(row, tree)

        if row[-1] == pred_val:

            correct_predictions += 1

    accuracy = 100 * correct_predictions / len(rows)

    return accuracy





#function to classify data set based on a learned tree

def classify(observation, tree):

    if tree.results != None:

        for key in tree.results:

            predicted_value = key

        return predicted_value

    else:

        v = observation[tree.col]

        if isinstance(v, int) or isinstance(v, float):

            if v >= tree.value:

                branch = tree.tb

            else:

                branch = tree.fb

        else:

            if v == tree.value:

                branch = tree.tb

            else:

                branch = tree.fb

        predicted_value = classify(observation, branch)

    return predicted_value





#function to count total number of non leaf nodes and label them according to number

def list_nodes(nodes, tree, count):

    if tree.results != None:

        return nodes, count

    count += 1

    nodes[count] = tree

    (nodes, count) = list_nodes(nodes, tree.tb, count)

    (nodes, count) = list_nodes(nodes, tree.fb, count)

    return nodes, count





def count_class_occurence(tree, class_occurence):

    if tree.results != None:

        for key in tree.results:

            class_occurence[key] += tree.results[key]

        return class_occurence



    left_branch_occurence = count_class_occurence(tree.fb, class_occurence)

    right_branch_occurence = count_class_occurence(tree.tb, left_branch_occurence)



    return right_branch_occurence





#replace subtree according to the pruning algorithm

def findAndReplaceSubtree(tree_copy, subtree_to_replace, subtree_to_replace_with):

    if (tree_copy.results != None):

        return tree_copy



    if (tree_copy == subtree_to_replace):

        tree_copy = subtree_to_replace_with

        return tree_copy



    tree_copy.fb = findAndReplaceSubtree(tree_copy.fb, subtree_to_replace, subtree_to_replace_with)

    tree_copy.tb = findAndReplaceSubtree(tree_copy.tb, subtree_to_replace, subtree_to_replace_with)



    return tree_copy



#function to prune tree

def prune_tree(tree, l, k, data):

    tree_best = tree

    best_accuracy = tree_accuracy(data, tree)

    tree_copy = None

    for i in range(1, l):

        m = random.randint(1, k)

        tree_copy = copy.deepcopy(tree)

        for j in range(1, m):

            (nodes, initial_count) = list_nodes({}, tree_copy, 0)

            if (initial_count > 0):

                p = random.randint(1, initial_count)

                # replcae subtree rooted in p

                subtree_p = nodes[p]



                # count examples with class variable as 0 and 1 in the subtree

                class_occurence = {'0': 0, '1': 0}

                count = count_class_occurence(subtree_p, class_occurence)

                # replace subtree with leaf node depending if zero or one count is greater

                if count['0'] > count['1']:

                    count['0'] = count['0'] + count['1']

                    count.pop('1')

                    subtree_p = decisionnode(results=count)

                else:

                    count['1'] = count['0'] + count['1']

                    count.pop('0')

                    subtree_p = decisionnode(results=count)



                tree_copy = findAndReplaceSubtree(tree_copy, nodes[p], subtree_p)



        # calculate accuracy based on pruned tree

        curr_accuracy = tree_accuracy(data, tree_copy)

        if (curr_accuracy > best_accuracy):

            best_accuracy = curr_accuracy

            tree_best = tree_copy

    return tree_best, best_accuracy


def main():

    args = str(sys.argv)

    args = ast.literal_eval(args)

	# ast.literal_eval raises an exception if the input isn't a valid Python datatype, so the code won't be executed if it's not.
	
    if (len(args) < 6):

        print ("Input arguments should be 6. Please refer the Readme file regarding input format.")

    elif (args[3][-4:] != ".csv" or args[4][-4:] != ".csv" or args[5][-4:] != ".csv"):

        print(args[2])

        print ("Your training, validation and test file must be a .csv!")

    else:

        l = int(args[1])

        k = int(args[2])

        training_set = str(args[3])

        validation_set = str(args[4])

        test_set = str(args[5])

        to_print = str(args[6])



        with open(training_set, newline='', encoding='utf_8') as csvfile:

                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

                header_data = next(spamreader)
		
                train_training_data = list(spamreader)



        with open(validation_set, newline='', encoding='utf_8') as csvfile:

            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

            validation_training_data = list(spamreader)



        with open(test_set, newline='', encoding='utf_8') as csvfile:

            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

            test_training_data = list(spamreader)



            using_IG_str = "----- Using Information Gain heuristic -----"

            using_VI_str = "----- Using Variance Impurity heuristic -----"



            l_arr = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

            k_arr = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]



            # build tree using information gain heuristic

            learned_tree_IG = buildtree(train_training_data, scoref=entropy)

            print(using_IG_str)

            if(to_print.lower() == "yes"):

                print("Printing the learned tree : ")

                printtree(learned_tree_IG, header_data, '')



            train_accuracy = tree_accuracy(train_training_data, learned_tree_IG)

            print("Training data accuracy : ", train_accuracy)



            validation_accuracy = tree_accuracy(validation_training_data, learned_tree_IG)

            print("Validation data accuracy : ", validation_accuracy)



            test_accuracy = tree_accuracy(test_training_data, learned_tree_IG)

            print("Test data accuracy : ", test_accuracy)



            (pruned_best_tree_validation, pruned_best_accuracy_validation) = prune_tree(learned_tree_IG, l, k,

                                                                                        validation_training_data)



            print("Validation data accuracy after pruning : ", pruned_best_accuracy_validation)



            (pruned_best_tree_test, pruned_best_accuracy_test) = prune_tree(learned_tree_IG, l, k, test_training_data)

            if (to_print.lower() == "yes"):

                print("Printing the pruned tree using on test data : ")

                printtree(pruned_best_tree_test, header_data, '')



            print("Test data accuracy after pruning : ", pruned_best_accuracy_test)



            #check accuracies of test data with 10 combinations of l and k

            print("Calculating accuracies of test data with 10 combinations of l and k :")

            for l_val, k_val in  zip(l_arr, k_arr):

                (pruned_best_tree_test, pruned_best_accuracy_test) = prune_tree(learned_tree_IG, l_val, k_val,

                                                                                test_training_data)

                print("Test data accuracy after pruning with l = ", l_val," and k = " , k_val,

                      " : ", pruned_best_accuracy_test)





            # build tree using variance impurity heuristic

            learned_tree_VI = buildtree(train_training_data, scoref=varianceImpurity)

            print(using_VI_str)

            if (to_print.lower() == "yes"):

                print("Printing the learned tree : ")

                printtree(learned_tree_VI, header_data, '')



            train_accuracy_VI = tree_accuracy(train_training_data, learned_tree_VI)

            print("Training data accuracy : ", train_accuracy_VI)



            validation_accuracy_VI = tree_accuracy(validation_training_data, learned_tree_VI)

            print("Validation data accuracy : ", validation_accuracy_VI)



            test_accuracy_VI = tree_accuracy(test_training_data, learned_tree_VI)

            print("Test data accuracy : ", test_accuracy_VI)



            (pruned_best_tree_validation_VI, pruned_best_accuracy_validation_VI) = prune_tree(learned_tree_VI, l, k, validation_training_data)

            print("Validation data accuracy after pruning: ", pruned_best_accuracy_validation_VI)



            (pruned_best_tree_test_VI, pruned_best_accuracy_test_VI) = prune_tree(learned_tree_VI, l, k, test_training_data)

            if (to_print.lower() == "yes"):

                print("Printing the pruned tree using on test data : ")

                printtree(pruned_best_tree_test_VI, header_data, '')



            print("Test data accuracy after pruning : ", pruned_best_accuracy_test_VI)



            # check accuracies of test data with 10 combinations of l and k

            print("Calculating accuracies of test data with 10 combinations of l and k :")

            for l_val, k_val in zip(l_arr, k_arr):

                (pruned_best_tree_test_VI, pruned_best_accuracy_test_VI) = prune_tree(learned_tree_VI, l_val, k_val,

                                                                                test_training_data)

                print("Test data accuracy after pruning with l = ", l_val, " and k = ", k_val,

                      " : ", pruned_best_accuracy_test_VI)



            #write results to a file using Information gain heuristic

            with open("Results.txt", "w") as text_file:

                text_file.write("%s\n\n" % str(using_IG_str))



                if (to_print.lower() == "yes"):

                    text_file.write("%s\n\n" % "Printing the learned tree : ")

                   # writeTreetoFile(learned_tree_IG, header_data, '', text_file)



                train_accuracy_str = "Training data accuracy : ", train_accuracy

                text_file.write("%s\n" % str(train_accuracy_str))



                validation_accuracy_str = "Validation data accuracy : ", validation_accuracy

                text_file.write("%s\n" % str(validation_accuracy_str))



                test_accuracy_str = "Test data accuracy : ", test_accuracy

                text_file.write("%s\n" % str(test_accuracy_str))



                if (to_print.lower() == "yes"):

                    text_file.write("%s\n" % "Printing the pruned tree using on test data : ")

                     #writeTreetoFile(pruned_best_tree_test, header_data, '', text_file)



                pruned_best_accuracy_validation_str = "Validation data accuracy after pruning : ", pruned_best_accuracy_validation

                text_file.write("%s\n" % str(pruned_best_accuracy_validation_str))



                pruned_best_accuracy_test_str = "Test data accuracy after pruning : ", pruned_best_accuracy_test

                text_file.write("%s\n" % str(pruned_best_accuracy_test_str))



                # check accuracies of test data with 10 combinations of l and k

                text_file.write("%s\n\n" % "Calculating accuracies of test data with 10 combinations of l and k :")

                for l_val, k_val in zip(l_arr, k_arr):

                    (pruned_best_tree_test, pruned_best_accuracy_test) = prune_tree(learned_tree_IG, l_val, k_val,

                                                                                    test_training_data)

                    test_accuracy_str_i = "Test data accuracy after pruning with l = ", l_val, " and k = ", k_val, " : ", pruned_best_accuracy_test

                    text_file.write("%s\n" %str(test_accuracy_str_i))



                #write results to a file using Variance Impurity heuristic

                text_file.write("%s\n\n\n\n")

                text_file.write("%s\n\n" % str(using_VI_str))

                if (to_print.lower() == "yes"):

                    text_file.write("%s\n\n" % "Printing the learned tree : ")

                   # writeTreetoFile(learned_tree_VI, header_data, '', text_file)



                train_accuracy_str_VI = "Training data accuracy : ", train_accuracy_VI

                text_file.write("%s\n" % str(train_accuracy_str_VI))



                validation_accuracy_str_VI = "Validation data accuracy : ", validation_accuracy_VI

                text_file.write("%s\n" % str(validation_accuracy_str_VI))



                test_accuracy_str_VI = "Test data accuracy : ", test_accuracy_VI

                text_file.write("%s\n" % str(test_accuracy_str_VI))



                if (to_print.lower() == "yes"):

                    text_file.write("%s\n" % "Printing the pruned tree using on test data : ")

                   # writeTreetoFile(pruned_best_tree_test_VI, header_data, '', text_file)



                pruned_best_accuracy_validation_str_VI = "Validation data accuracy after pruning : ", pruned_best_accuracy_validation_VI

                text_file.write("%s\n" % str(pruned_best_accuracy_validation_str_VI))



                pruned_best_accuracy_test_str_VI = "Test data accuracy after pruning : ", pruned_best_accuracy_test_VI

                text_file.write("%s\n" % str(pruned_best_accuracy_test_str_VI))



                # check accuracies of test data with 10 combinations of l and k

                text_file.write("%s\n\n" % "Calculating accuracies of test data with 10 combinations of l and k :")

                for l_val, k_val in zip(l_arr, k_arr):

                    (pruned_best_tree_test_VI, pruned_best_accuracy_test_VI) = prune_tree(learned_tree_VI, l_val, k_val,

                                                                                    test_training_data)

                    test_accuracy_str_VI_i = "Test data accuracy after pruning with l = ", l_val, " and k = ", k_val," : ", pruned_best_accuracy_test_VI

                    text_file.write("%s\n" %str(test_accuracy_str_VI_i))





if __name__ == "__main__":

    main()