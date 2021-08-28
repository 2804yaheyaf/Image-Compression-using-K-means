Jason Joseph Rebello
Carnegie Mellon University (Jan 2012 - May 2013)
Masters in Electrical & Computer Engineering

K means clustering algorithm with the application to image compression.
This program uses the K means clustering algorithm to group pixels in an image in order to provide image compression

This algorithm will run slower as the number of clusters , size of the image and number of iterations increase.

K indicates the number of clusters that we want to group the pixels into. iterCentroids will be the number of iterations executed in order to find the best centroid and best clustering. iterKMeans is the number of times we wish to run K means with different initial centroids. 
The difference between the two iterations. Imagine a 2D coordinate system for simplicity. There are 4 quadrants and initialize the number of clusters to 4. Now if we have points such that by looking at the data we can see that points belonging to each quadrant form one cluster. Now the initial centroids are chosen at random from the data points. If we get lucky, then we will get 4 initial centroids each belonging to one quadrant. But chances are that we might get 2 initial centroids that are two data points from one quadrant. Hence what iterKMeans aims to do is run the algorithm 'iterKMeans' times in order to get at least one 'initial centroids' that will lie in each of the 4 quadrants. For each iterKMeans, the clustering algorithm will update the new centroids and points belonging to that centroid 'iterCentroids' times.

The image is an RGB image of m x n size. So in order to perform K means clustering I rearrange the data  to form an [(m x n) x 3]  2D matrix. Not R,G,B are the features of the image so effectively we have a (total pixels x total features) matrix.

For each iterKMeans iteration runKMeans is called.

runKMeans : This function first calls the initCentroids function to initialize the centroids ( initCentroids randomly chooses the first K points to act as the centroids after randomly per mutating the rows). For each iteration in runKMeans I call the findClosestCentroids functions to assign the closest centroids for each pixel by calculating the distance between the point and the centroid. The computeNewCentroids is then called. After which the total cost is calculated.

findClosestCentroid : For each point (i.e each row in X), I compute the distance between it and all the centroids. The centroid with the minimum distance is assigned to the point. 

computeNewCentroids :  For each cluster the new centroid position is achieved by calculating the mean of all the points belonging to that cluster.

computeCost : The total cost is calculated by summing the distance of each point to its cluster centre and then summing over all the clusters. Based on the minimum overall cost achieved during each iteration of 'iterKMeans' the pixels assignment to their respective clusters are made.

During each iteration if iterKMeans the image is showed based on the final clustering. As can be seen based on the initial centroids that are selected the final clustered image is different. This is the reason why K means is performed with different initial centroids.

The original and final compressed image is then displayed.

How is compression achieved ?
The original image required 24 bits (8 for red ,8 for green and 8 for blue) for each one of the 128 by 128 pixel locations, resulting in total size of 128 * 128 * 24 = 393, 216 bits. The new representation requires some overhead storage in form of a dictionary of 16 colors, each of which require 24 bits, but the image itself then only requires 4 bits per pixel location (since 16 numbers can be represented by 4 bits). The final number of bits used is therefore 16 * 24 + 128 * 128 * 4 = 65, 920 bits, which corresponds to compressing the original image by about a factor of 6.