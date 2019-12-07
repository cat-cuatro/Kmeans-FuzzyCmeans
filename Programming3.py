""" John Lorenz IV // 12-4-19
    Fuzzy C-Means and K-Means Algorithms
    This file hosts the main, and the methods for calling K-means
    or Fuzzy C-Means. They're similar methods of unsupervised learning
    that focus on clustering like-data so that it can be easily
    differentiated. Data is 2-D simulated from 3-D Gaussians
"""
import numpy as np
import matplotlib.pyplot as plot
import testClass
plot.style.use('ggplot')

def main():
    kClassifier = testClass.kMeans() # hold data for class
    fuzzyClassifier = testClass.fuzzyCMeans() 
    data_file = open("GMM_data_fall2019.txt",'r') 
    data = data_file.readlines()
    graphs = [] # currently unused
    vectors = [] # make the array to store our data


    for i in data:
        vectors.append(i.split()) 

    for i in range(len(vectors)): # convert all array elements into floats
        for j in range(len(vectors[i])):
            vectors[i][j] = float(vectors[i][j])

    userInput = input("How many clusters should be data be run on? (1-70)\n")
    clusters = int(userInput)
    userInput = input("How many times would you like to re-cluster?\n")
    iterations = int(userInput)
    userInput = input("Input 1 to run Fuzzy C-Means, or 0 to run K-Means\n")
    runFuzzyCMeans = int(userInput)

    em_steps = 400 ## Change this parameter for more or less centroid and weight adjustments in fuzzy C means ##
    K_em_steps = 100
    if runFuzzyCMeans == 1:
        print("Now testing Fuzzy C-Means. . .")
        graph = fuzzyClassifier.CMeansClassifier(iterations, clusters, vectors, em_steps)
    else:
        print("Now testing K-Means. . .")
        graph = kClassifier.KMeansClassifier(iterations, clusters, vectors, K_em_steps)





if __name__ == "__main__":
    main()