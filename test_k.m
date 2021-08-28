clear all;
clc;
close all;

fprintf('K means clustering algorithm used for image compression\n\n');
t = cputime;

%% Read the image
fprintf('Reading image');
I = imread('bird_small.png');
%imshow(I);
I = (double(I))/255;
fprintf('...done\n\n');

%% Declare and Initialize Variabels
fprintf('Initializing variables');
K = 16; % number of clusters
imgSize = size(I); % get size of image
iterCentroids = 10; % number of times K means runs to find the best centroid
iterKMeans = 10; % number of times K means runs with different initial centroids
fprintf('...done\n\n');

%% Get input
fprintf('Formatting input');
X = reshape(I, imgSize(1) * imgSize(2), 3); % resize into (total pixel x features)
fprintf('...done\n\n');


function initialCentroids = initCentroids(X, K)

% This function will be used to initialize K centroids by randomly choosing
% K data points from X to act as the initial centroids.

temp = (randperm(length(X)))'; % randomize rows
initialCentroids = X(temp(1:K,1),:); % select first K rows as initial centroids
end

function idx = findClosestCentroids(X, centroids)

% findClosestCentroids computes the closest centroid for each point based
% on the Euclidean distance between the point and the centroid

% Initialize variables
K = size(centroids, 1); 
idx = zeros(size(X,1), 1); % returns index of closest centroid

for i=1:size(X,1)
    temp = X(i,:);
    [~,idx(i,1)] = min(sum(((bsxfun(@minus,temp,centroids)).^2),2));
end
end

function centroids = computeNewCentroids(X, idx, K)

% computeNewCentroids computes the new centroids of each cluster based on
% the mean value of the all the points belonging to that cluster.

% Initialize variables
[m n] = size(X);
centroids = zeros(K, n);

for i=1:K
    temp = find(idx==i);
    Xtemp = X(temp,:); % Get all points belonging to that cluster
    centroids(i,:) = (sum(Xtemp,1))./length(Xtemp); % Assign new centroid based on mean
end
end
function cost = computeCost(X, idx, centroids, K)

% computeCost is used to compute the final cost of all the points belonging
% to their respective clusters

% Initialize variables
cost = 0;

% Compute cost
for i=1:K
    temp = find(idx==i);
    Xtemp = X(temp,:); % Get all points belonging to respective cluster
    cost = cost + (1/length(Xtemp))*sum((sum(((bsxfun(@minus,centroids(i,:),Xtemp)).^2),2))...
                                                        .^(-1/2));
end
end
% Formula : {sqrt[(q1-p1)^2 + (q2-p2)^2 +...]}/{number of points belonging
% to cluster}
% q = q1 + q2 + ...

function displayImage(I, XCompressed, K)

% Displays best final compressed image.

% Display the original image 
subplot(1, 2, 1);
imagesc(I); 
title('Original');

% Display compressed image side by side
subplot(1, 2, 2);
imagesc(XCompressed)
title(sprintf('Compressed, with %d colors.', K));

function [centroids cost idx] = runKMeans(X, K, iterCentroids)

% This function runs K means the number of times specified by iterKMeans.
% It returns the final centroids and the final cost of that iteration.

% Initialize centroids
fprintf('Initializing centroids');
centroids = initCentroids(X,K);
fprintf('...done\n\n');

end
for num=1:iterCentroids
    
    %fprintf('Starting iteration number %d\n\n',num);
    
    % return index of closest centroid for each point
    idx = findClosestCentroids(X, centroids);
    
    % Compute New centroid of each cluster
    centroids = computeNewCentroids(X, idx, K);
    
end

% Compute cost of the final clusters
cost = computeCost(X, idx, centroids, K);
end
%% Run K Means
for i=1:iterKMeans
    
    fprintf(' ********* Running K means iteration %d ***********\n\n',i);
    [centroids cost idx] = runKMeans(X, K, iterCentroids);
    fprintf('Cost after %d iteration : %f\n\n',i,cost);
    
    if i==1
        bestCentroids = centroids;
        bestCost = cost;
        bestidx = idx;
    elseif (i>1 && cost<bestCost) % stores the best clustering
        bestCentroids = centroids;
        bestCost = cost;
        bestidx = idx;   
    end
    fprintf('Best cost : %f\n\n',bestCost);  

XCompressed = centroids(idx,:);

% Reshape the recovered image into proper dimensions
XCompressed = reshape(XCompressed, imgSize(1), imgSize(2), 3);
imshow(XCompressed); % display final compressed image for each iteration
pause(1);

end

%% Display original and best compressed image
XCompressed = bestCentroids(bestidx,:);

% Reshape the recovered image into proper dimensions
XCompressed = reshape(XCompressed, imgSize(1), imgSize(2), 3);
displayImage(I, XCompressed, K);

fprintf('Program executed in %f seconds or %f minutes\n\n', cputime-t, (cputime-t)/60);