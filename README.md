# SCT_ML_02
Customer Segmentation using KMeans Clustering
Project Overview:
This project focuses on customer segmentation for a retail store based on customer purchasing history. The goal is to categorize customers into different segments based on their age, annual income, and spending score (1-100). The segmentation process is performed using KMeans clustering, followed by analysis and visualization of the resulting clusters. We use feature engineering to enhance the data before clustering, including ratios and interactions between key features.

Key Features
Age: Customer's age
Annual Income (k$): Customer's annual income in thousands
Spending Score (1-100): A score that represents how much the customer spends
Gender: Female or Male (encoded as a boolean)
Age Group: Age category (Teenager, Adult, Senior)
Income Group: Income category (Low, Mid, High)
Additional engineered features like income_age_ratio, score_to_income_ratio, etc.



KMeans Clustering: The main clustering task is performed using the KMeans algorithm. The number of clusters (k) is determined using the elbow method, which is visualized in the elbow_plot.png. The silhouette score is also computed for cluster validation and visualized in the silhouette_plot.png.

PCA for Visualization A PCA (Principal Component Analysis) is applied to reduce the dimensions of the data to 2D for easy visualization. The resulting plot is saved as pca_plot.png.

Clustering Results The final cluster assignments are saved to output_segments.csv. Each customer is assigned to a specific cluster, which helps in understanding different segments based on their purchasing behavior.

Outputs

Cluster Summary: The mean values of features per cluster.
Cluster Visualizations: Visual representations of the clusters (PCA plot,t-SNE plot, elbow plot, silhouette score plot).
Output CSV: A CSV file containing the cluster assignment for each customer.

Conclusion
This project helps in understanding the behavior of customers based on their demographics and purchasing behavior. The segmentation can be used for targeted marketing strategies, personalized recommendations, and better customer retention.

