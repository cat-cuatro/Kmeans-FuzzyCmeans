import numpy as np
import matplotlib.pyplot as plot
import matplotlib.animation as animation
import random
import pickle
from math import sqrt
plot.style.use('ggplot')

class fuzzyCMeans(object):
    centroids = [] # 1-d array of centroids
    assignments = [] # k-d array of assignments per vector
    bestCentroids = []
    bestAssignments = []
    worstCentroids = []
    worstAssignments = []
    worst_wcss = 0
    all_wcss = []
    best_wcss = 10000
    current_wcss = 0
    bests_found = 0
    m_parameter = 2 # fuzzifier parameter
    epsilon = .001   # early stopping parameter
    plotcolors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','b',
                 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 
                 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 
                 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 
                 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 
                 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 
                 'k','b', 'g', 'r', 'c', 'm', 'y', 'k',
                 'b', 'g', 'r', 'c', 'm', 'y', 'k','b', 
                 'g', 'r', 'c', 'm', 'y', 'k'] # supports up to 70 clusters. Brute force color cycle

    def print_wcss(self):
        print("All wcss sums:")
        print(self.all_wcss)
        print("Best wcss sum:")
        print(self.best_wcss)
    
    def assess_wcss(self):
        temp_wcss = 0
        self.all_wcss.append(self.current_wcss)
        if self.current_wcss < self.best_wcss:
            self.bests_found += 1
            print("New best wcss! #: ", self.bests_found)
            self.best_wcss = self.current_wcss
            self.bestCentroids = self.centroids
            self.bestAssignments = self.assignments

    def constRandomWeights(self, vectors, clusters):
        self.assignments = []
        for i in range(len(vectors)):
            nums = random.sample(range(0,1000), clusters)
            nums = np.array(nums)
            sums = nums.sum()
            nums = nums/sums
            self.assignments.append(list(nums))


    def randomizeWeights(self, vectors, clusters): # First step is to randomize the membership weights
        self.assignments = [] # reset this list since we're appending to it!
        temp = 0
        toAssign = []
        current_sum = 0
        cost = 0
        distanceList = [] # this will hold the distance of a point to each centroid so that we can compute efficient weights
        for i in range(len(vectors)):
            for x in range(clusters): # For every vector, and for every centroid
                cost = self.costFunction(vectors[i], self.centroids[x]) ## compute cost separate for code readability
                distanceList.append(cost)
                current_sum += cost
            for x in range(clusters): # unfortunately we'll have to loop again
                temp = distanceList[x]/current_sum # this will yield a % value of overall distance from **all** centroids
                toAssign.append(temp)      # start contructing a list of weights
            # now that we've done the number operations, append to assignments list and reset variables
            self.assignments.append(toAssign)
            current_sum = 0
            toAssign = []
    """ What's nice about this is that we're using the Euclidean distance of a point from 
        all of the clusters as a basis of weight assignments, and since the centroids are
        random, this process is inherently random and will also give us good results which
        will justify the extra computational overhead.
        Additionally, if we added all the weights of any given (x,y) data point, it will add up to 1
        (or 100% of the distance)
    """
    
    def computeCentroids(self, vectors, clusters):
        xComponent = 0
        yComponent = 0
        member_weight = 0
        tempX = 0
        tempY = 0
        tempDenomX = 0
        tempDenomY = 0
        for i in range(clusters): ## for each centroid
            for x in range(len(vectors)): ## for each data point
                member_weight = self.assignments[x][i]**self.m_parameter
                if(member_weight == 0):
                    member_weight = .00001 
                   # prevent dividing by 0
                tempX += (member_weight * vectors[x][0]) # multiply x-component and add to self
                tempY += (member_weight * vectors[x][1]) # multiply y-component and add to self

                tempDenomX += member_weight
                tempDenomY += member_weight
                # each membership weight is in the same order as clusters!
            xComponent = (tempX/tempDenomX) # final component-wise adjustment
            yComponent = (tempY/tempDenomY)
            self.centroids[i] = [xComponent, yComponent] # assign new centroid

            tempX, tempY = 0,0
            tempDenomX, tempDenomY = 0,0
             # reset our variables

    
    def computeMembership(self, vectors, clusters): # The highest cost function. Probably reduceable with clever numpy arrays
        halt = False
        numerator_cost = 0
        denominator_cost = 0
        current_sum = 0
        old_wcss = self.current_wcss #first make a copy of our old wcss
        self.current_wcss = 0 # reset this value, as we're going to re-compute it
        for x in range(len(vectors)): ## for each vector
            for i in range(clusters): ## for each membership weight
                numerator_cost = self.costFunction(vectors[x], self.centroids[i])
                for k in range(clusters):
                    denominator_cost = self.costFunction(vectors[x], self.centroids[k])
                    if(denominator_cost == 0):
                        denominator_cost = .00001 # prevent dividing by 0
                    current_sum += (numerator_cost/denominator_cost)**(2/(self.m_parameter-1))
                if(current_sum ==0):
                    current_sum = .00001 # prevent division by 0
                self.assignments[x][i] = (1/current_sum)
                self.current_wcss += (self.assignments[x][i] * numerator_cost)
                # We want to keep track of our within-cluster-sum-of-squares (wcss)
                current_sum = 0 # reset our variables
        if (old_wcss - self.current_wcss <= self.epsilon):
            halt = True
        return halt

    def costFunction(self, a_point, a_centroid): #euclidean distance
        return sqrt((a_point[0]-a_centroid[0])**2 + (a_point[1]-a_centroid[1])**2)

    def pickCentroids(self, centroids, upperBound, vectors):
        self.centroids = [] ## reset centroids, as we're going to re-assign every centroid anyways!
        indices = [] # make an array of random indices
        indices = random.sample(range(0,upperBound), centroids)
        for i in indices:
            self.centroids.append(vectors[i])


    def displayGraph(self, vectors, computation, iteration):
        vectors = np.array(vectors)
        localCentroids = np.array(self.centroids)
        x,y = vectors.T
        figure = plot.figure()
        ax1 = figure.add_subplot(1,1,1)
        counter = 0
        for numOf in range(len(vectors)):
            x,y = vectors[numOf].T
#            color = self.assignments[numOf]
            greatest = 0
#            color = 1
            for k in range(len(self.centroids)):
                if self.assignments[numOf][k] > greatest:
                    greatest = self.assignments[numOf][k]
                    color = k
            ax1.scatter(x,y, s=10, c=self.plotcolors[color])       
        for i in localCentroids: # now plot the centroid markers
            x,y = i.T
            ax1.scatter(x,y, s=200, c=self.plotcolors[counter], marker='X', edgecolors='k')
            counter = counter+1

#        print("Saving graph")
        plot.title("CMeans_graph"+str(iteration)+"_step"+str(computation)+"\n"+"WCSS: " + str(self.current_wcss))
        plot.savefig("CMeans_graph"+str(iteration)+"_step"+str(computation)+".png")
#        plot.show() #show graph
        ax1.clear()
        return figure

    def seedBestResult(self):
        self.centroids = self.bestCentroids
        self.assignments = self.bestAssignments

    def CMeansClassifier(self, iterations, clusters, vectors, computations):
        for i in range(iterations):
            halt = False
            if(i % 5):
                print(i, " iterations processed..")
            self.pickCentroids(clusters, len(vectors), vectors) # pick initial random centroids
            self.randomizeWeights(vectors, clusters) # euclidean weights
#            self.constRandomWeights(vectors, clusters) #'true' random weights

            for x in range(computations):
                halt = False
                if(x % 5 == 0):
                    print(x, " steps processed..")

                halt = self.computeMembership(vectors, clusters)
                self.computeCentroids(vectors, clusters)
                graph = self.displayGraph(vectors, x, i)
                if halt == True and x >= 25:
                    print("Epsilon reached. Stopping early.")
                    break
            self.assess_wcss()
        self.print_wcss()
        self.seedBestResult()
        graph = self.displayGraph(vectors, 0, "best") ## not sure what to do with the graph yet
        return graph

class kMeans(object):
    centroids = []
    assignments = []
    clusterMeans = []
    bestCentroids = []
    bestAssignments = []
    all_wcss = []
    best_wcss = 10000
    bests_found = 0
    current_wcss = 0
    epsilon = .001
    plotcolors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','b',
                 'g', 'r', 'c', 'm', 'y', 'k','b', 'g', 
                 'r', 'c', 'm', 'y', 'k','b', 'g', 'r', 
                 'c', 'm', 'y', 'k','b', 'g', 'r', 'c', 
                 'm', 'y', 'k','b', 'g', 'r', 'c', 'm', 
                 'y', 'k','b', 'g', 'r', 'c', 'm', 'y', 
                 'k','b', 'g', 'r', 'c', 'm', 'y', 'k',
                 'b', 'g', 'r', 'c', 'm', 'y', 'k','b', 
                 'g', 'r', 'c', 'm', 'y', 'k'] # supports up to 70 clusters. Brute force color cycle

    def printAssignments(self, vectors):
        for i in range(len(vectors)):
            print(vectors[i], " is assigned to centroid ", self.centroids[self.assignments[i]])

    def print_wcss(self):
        print("All wcss sums:")
        print(self.all_wcss)
        print("Best wcss sum:")
        print(self.best_wcss)

    def computeMeans(self, vectors, clusters): 
        ## for each set in each cluster, compute the mean
        halt = False
        sums = []
        lengths = []
        means = []
        tempMeanX = 0
        tempMeanY = 0
        wcss_sum = 0
        for x in range(clusters):
            sums.append([0, 0]) ## there should be the same number of sums as clusters*2
            lengths.append(0)
        for num in range(len(vectors)):
            index = self.assignments[num]
            sums[index][0] += vectors[num][0] # sum all x coordinates
            sums[index][1] += vectors[num][1] # sum all y coordinates
            lengths[index] += 1
        for i in range(clusters):
            tempMeanX = sums[i][0]/lengths[i]
            tempMeanY = sums[i][1]/lengths[i]
            means.append([tempMeanX, tempMeanY])
        self.centroids = means # technically the means are exactly the new centroids

        for num in range(len(vectors)):
            index = self.assignments[num]
            cost = self.costFunction(vectors[num], self.centroids[index])
            wcss_sum += self.costFunction(vectors[num], self.centroids[index]) # now compute within cluster sum of squares

        if wcss_sum < self.best_wcss:
            self.bests_found += 1
            self.bestCentroids = means
            self.bestAssignments = self.assignments
            self.best_wcss = wcss_sum
        self.all_wcss.append(wcss_sum)
        if(self.current_wcss - wcss_sum <= self.epsilon):
            halt = True
            print("Epsilon reached. Early halting")
        self.current_wcss = wcss_sum
        wcss_sum = 0
        return halt

    def seedBestResult(self):
        self.centroids = self.bestCentroids
        self.assignments = self.bestAssignments

    def displayGraph(self, vectors, computation, iteration):
        vectors = np.array(vectors)
        localCentroids = np.array(self.centroids)
        x,y = vectors.T
        figure = plot.figure()
        ax1 = figure.add_subplot(1,1,1)
        counter = 0
        for numOf in range(len(vectors)):
            x,y = vectors[numOf].T
            color = self.assignments[numOf]
            ax1.scatter(x,y, s=10, c=self.plotcolors[color])       
        for i in localCentroids: # now plot the centroid markers
            x,y = i.T
            ax1.scatter(x,y, s=200, c=self.plotcolors[counter], marker='X', edgecolors='k')
            counter = counter+1
#        print("Saving graph")
        plot.title("KMeans_graph"+str(iteration)+"_step"+str(computation)+"\n"+"WCSS: " + str(self.current_wcss))
        plot.savefig("KMeans_graph"+str(iteration)+"_step"+str(computation)+".png")
#        plot.show() #show graph
        ax1.clear()
        return figure

    def pickCentroids(self, centroids, upperBound, vectors):
        self.centroids = [] ## reset centroids, as we're going to re-assign every centroid anyways!
        indices = [] # make an array of random indices
        indices = random.sample(range(0,upperBound), centroids)
        for i in indices:
            self.centroids.append(vectors[i])

    def assignPoints(self, vectors): # assign points to nearest clusters
        self.assignments = [] ## reset assignments, as we're going to re-assign every point anyways!
        assigned_centroid = 0 
        cost = 100 # assign large initial val
        nextCost = 0
        wcss_sum = 0
        for i in range(len(vectors)):
            for j in range(len(self.centroids)):
                nextCost = self.costFunction(vectors[i], self.centroids[j])
                if cost > nextCost: # if current cost is larger, reassign
                    cost = nextCost
                    assigned_centroid = j
            self.assignments.append(assigned_centroid) # assign index of centroid to this vector
            cost = 100

    def costFunction(self, a_point, a_centroid): #euclidean distance
        return sqrt((a_point[0]-a_centroid[0])**2 + (a_point[1]-a_centroid[1])**2)

    def KMeansClassifier(self, iterations, clusters, vectors, em_steps):
        for i in range(iterations):
            halt = False
            if(i % 5 == 0):
                print(i, " iterations processed..")
            self.pickCentroids(clusters, len(vectors), vectors)
            self.assignPoints(vectors)
            for x in range(em_steps):
                halt = self.computeMeans(vectors, clusters)
                self.assignPoints(vectors)
                graph = self.displayGraph(vectors, x, i)
                if(halt == True):
                    break
        #display the best result one last time
        self.seedBestResult() ## load best result into data
        self.print_wcss() ## display all attempts
        print("Graphing the best result . . .")
        graph = self.displayGraph(vectors, 0, "best") 
        return graph