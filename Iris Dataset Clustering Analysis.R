library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
library(fpc)        # for DBScan

#Loading Data and making use of it
data("iris")
iris1<-iris[-5]

#scaling Data into normalization
iris2<-as.data.frame(scale(iris1))

#elbow method
set.seed(123)
wss <- function(k){kmeans(iris1, k)$tot.withinss}
k.values<- 1:10
wss_values <- map_dbl(k.values, wss)
wss_values

#plot wss model
plot(k.values, wss_values, ask=FALSE,
     type="b", pch = 19, frame= FALSE,
     xlab="Number Of Clusters K",
     ylab="Total Within-clusters sum of squares")

#Another Elbow method
# First we make A func to take the Data(Points and number of clusters as wanted)
calculate_BSS_WSS <- function(points, kmax) {
  # Set up parameters so store the Wss as the algorithm go
  wss <- numeric()
  bss <- numeric()
  # Here we make a for loop to fit a cluster model for each K to get the centers and the cluster so Calc the Wss, Bss
  for (k in 1:kmax) {
    kmeans_model <- kmeans(points, k)
    # STORE the data to use them in Sum Method (Faster)
    centroids <- kmeans_model$centers
    pred_clusters <- kmeans_model$cluster
    
    # Calculate WSS
    curr_wss <- sum((points - centroids[pred_clusters,])^2)
    wss <- c(wss, curr_wss)
    
  }
  return(list(wss = wss, bss = bss))
}

# Calculate WSS and BSS for k values from 1 to 10
bss_wss_scores <- calculate_BSS_WSS(iris1, 10)
bss_wss_scores

# Plot the WSS scores for each k
plot(1:10, bss_wss_scores$wss, ask=FALSE,
     type="b", pch = 19, frame= FALSE,
     xlab="Number of clusters (k)",
     ylab="Total Within-clusters sum of squares(WSS score)")


# Print the WSS scores
cat("WSS Scores:", bss_wss_scores$wss, "\n")

#silhouette method
silhouette_score <- function(X, labels) {
  n_samples <- nrow(X)
  cluster_labels <- unique(labels)
  n_clusters <- length(cluster_labels)
  
  # Calculate mean distance between each point and all other points in its cluster
  cluster_distances <- matrix(NA, n_samples, n_clusters)
  for (i in 1:n_samples) {
    for (j in 1:n_clusters) {
      mask <- (labels == cluster_labels[j])
      if (sum(mask) > 0) {
        cluster_distances[i, j] <- sqrt(sum((X[i, ] - colMeans(X[mask, ]))^2) / sum(mask))
      }
    }
  }
  
  # Calculate mean distance between each point and all other points in its nearest neighboring cluster
  nn_distances <- matrix(NA, n_samples, 1)
  for (i in 1:n_samples) {
    cluster_label <- labels[i]
    mask <- (labels != cluster_label)
    nn_distances[i] <- min(apply(X[mask, ], 1, function(x) sqrt(sum((X[i, ] - x)^2))))
  }
  
  # Calculate silhouette width for each point
  silhouette_widths <- matrix(NA, n_samples, 1)
  for (i in 1:n_samples) {
    cluster_label <- labels[i]
    a_i <- cluster_distances[i, which(cluster_labels != cluster_label)]
    b_i <- nn_distances[i]
    silhouette_widths[i] <- (b_i - a_i) / max(a_i, b_i)
  }
  
  # Calculate mean silhouette width for all points
  mean_silhouette_width <- mean(silhouette_widths)
  
  return(mean_silhouette_width)
}
results = c()
for(i in 2:9){
  kmeans.re <- kmeans(iris1, centers = i, nstart = 25)
  l=kmeans.re$cluster
  results[i-1]=silhouette_score(iris1,l)
}
plot(results, ask=FALSE, type="b", pch = 19,
    xlab="Numer of clusters (K-1)",
    ylab="Silhouette scores",
    main="Silhouette Plot")

# Compute k-means clustering with k = 3
set.seed(123)
final <- kmeans(iris1, 3, nstart = 25)
print(final)

#Now We can visualize the results for k = 3
fviz_cluster(final, data = iris1)

# Compute k-means clustering with k = 2
set.seed(123)
final2 <- kmeans(iris1, 2, nstart = 25)
print(final2)

#Now We can visualize the results for k = 2
fviz_cluster(final2, data = iris1)

#Hierarchical
#hclust takes distance matrix of the data
hierarchical=hclust(dist(iris1),method="complete")
plot(hierarchical, ask=FALSE,
     main="Hierarchical Plot")

#Agglomerative
ag=agnes(iris1,method='complete')
ag$ac
plot(ag, ask=FALSE)
p1=as.dendrogram(ag) #to only get dendogram graph
plot(p1, ask=FALSE)

#Divisive
dv=diana(iris1)
dv$dc
plot(dv, ask=FALSE)
p2=as.dendrogram(dv)
plot(p2, ask=FALSE)

#DBScan
db=dbscan(iris1,eps=0.7,MinPts=10)
db
plot(db,iris1)