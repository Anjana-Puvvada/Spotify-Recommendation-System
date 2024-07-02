# Music Recommendation System using Spotify Dataset

This project focuses on creating a Music Recommendation System using the Spotify Dataset. By leveraging various data analysis and machine learning techniques, the system provides insightful visualizations and accurate recommendations to enhance the user experience.

## Data Source
The dataset is sourced from the Spotify API, which includes a comprehensive collection of audio features and metadata for different songs.

## Processes

### Data Extraction
- **Spotify API Integration**: Using Spotipy, a Python client for the Spotify Web API, to fetch data and query Spotify’s catalog for songs.
- **Data Collection**: Extracting audio features and metadata for a vast number of songs to build a rich dataset.

### Data Exploration and Cleaning
- **Reading and Loading Data**: Utilizing Pandas to read and load the extracted data into a suitable format for analysis.
- **Handling Missing Values**: Replacing missing values with zeroes to ensure data integrity.
- **Data Transformation**: Transposing and transforming the dataset for time series analysis.

### Analysis and Visualization
- **Music Over Time**: Analyzing how the overall sound of music has evolved from 1921 to 2020 using grouped data by year.
- **Characteristics of Different Genres**: Comparing audio features of different genres to understand their unique differences in sound.

### Clustering Genres with K-Means
- **K-Means Clustering**: Dividing genres into ten clusters based on numerical audio features, providing insights into genre similarities.

### Visualizing Clusters with t-SNE and PCA
- **t-SNE Visualization**: Creating a 2D representation of genre clusters to visualize their distribution.
- **PCA Visualization**: Using Principal Component Analysis to visualize song clusters, aiding in understanding the data's structure.

### Building the Recommender System
- **Model Selection**: Employing LSTM and ARIMA models for accurate time series forecasting.
- **Recommendation Algorithm**: Recommending songs by finding similar data points based on the user's listening history and the extracted audio features.

## Technologies Used
- **Programming Languages**: Python
- **Libraries and Frameworks**: Pandas, Scikit-learn, Spotipy, Plotly, Seaborn, Matplotlib
- **Machine Learning Models**: K-Means Clustering,PCA, t-SNE
- **Visualization Tools**: Plotly, Matplotlib, Seaborn

## Authors
- Anjana Sowmya Puvvada


## Acknowledgements
- Built using Python, Pandas, Scikit-learn, Spotipy, and Plotly.
- Inspired by the need for an efficient music recommendation system using time series data.
