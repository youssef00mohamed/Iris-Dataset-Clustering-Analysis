# Iris Dataset Clustering Analysis

This project explores clustering analysis on the Iris dataset using various algorithms implemented in R. It includes k-means clustering with both elbow method and silhouette analysis, hierarchical clustering, agglomerative clustering, divisive clustering, and DBScan. Each method is applied to identify natural groupings within the dataset and visualize clustering results.

## Usage

1. **Requirements:**
   - R programming language
   - Required libraries: `tidyverse`, `cluster`, `factoextra`, `fpc`

2. **Instructions:**
   - Load the dataset using `data("iris")`.
   - Run the R script `clustering_analysis.R` to perform clustering using different methods.
   - Review the generated plots to analyze clustering results for various algorithms.

3. **Output:**
   - Plots illustrating clustering results for k-means, hierarchical, agglomerative, divisive clustering, and DBScan.
   - Analysis and visualization of clusters in the Iris dataset.

## Files

- `clustering_analysis.R`: R script containing implementations of various clustering algorithms on the Iris dataset.
- `README.md`: This file providing an overview of the project, instructions, and usage.

## Algorithms Used

- **k-means Clustering:**
  - Elbow method and silhouette analysis to determine optimal number of clusters.
- **Hierarchical Clustering:**
  - Complete linkage method for hierarchical clustering.
- **Agglomerative Clustering:**
  - Agnes function for agglomerative clustering.
- **Divisive Clustering:**
  - Diana function for divisive clustering.
- **DBScan:**
  - DBScan function for density-based clustering.

## Notes

- Each clustering method provides insights into the structure of the Iris dataset.
- The project serves as an educational resource for exploring different clustering techniques in R.

## Contribution

Contributions to enhance this project are welcome. Feel free to fork the repository, make improvements, and submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE).
