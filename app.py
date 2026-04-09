import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
import io

# Page Configuration
st.set_page_config(page_title="DAVL Lab Portal", layout="wide")

# Title and Sidebar
st.title("Data Analysis and Visualization Lab Portal")
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose an Experiment", 
    ["Home", "Data Cleaning & Stats (Exp 6/9)", "Web Data Table (Exp 7)", "EDA & Distributions (Exp 8)", "Regression Analysis (Exp 11)", "Dimensionality Reduction (Exp 12)", "Clustering (Exp 13)"])

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("winequality-red.csv")
    return df

df = load_data()

if app_mode == "Home":
    st.header("Welcome to the DAVL Portal")
    st.write("This application demonstrates various data analysis concepts implemented across your lab assignments.")
    st.subheader("Dataset Overview: Wine Quality")
    st.dataframe(df.head())
    st.info("Use the sidebar to navigate through different experiment concepts.")

elif app_mode == "Data Cleaning & Stats (Exp 6/9)":
    st.header("Experiment 6 & 9: Data Stats & Preprocessing")
    st.write("Concepts: Computing summary statistics and cleaning data.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Summary Statistics")
        st.write(df.describe())
    
    with col2:
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

elif app_mode == "Web Data Table (Exp 7)":
    st.header("Experiment 7: Flask-style Web Data Table")
    st.write("Concepts: Rendering data tables and summary metrics in a web interface.")
    
    # Filter functionality
    quality_filter = st.slider("Filter by Quality Score", int(df.quality.min()), int(df.quality.max()), (5, 7))
    filtered_df = df[(df.quality >= quality_filter[0]) & (df.quality <= quality_filter[1])]
    
    st.write(f"Showing {len(filtered_df)} records:")
    st.dataframe(filtered_df)

elif app_mode == "EDA & Distributions (Exp 8)":
    st.header("Experiment 8: EDA & Distributions")
    st.write("Concepts: Histograms, Box plots, and Heatmaps.")
    
    feature = st.selectbox("Select Feature for Distribution", df.columns[:-1])
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[feature], kde=True, ax=ax[0], color='skyblue')
    ax[0].set_title(f"Histogram of {feature}")
    
    sns.boxplot(y=df[feature], ax=ax[1], color='lightgreen')
    ax[1].set_title(f"Boxplot of {feature}")
    st.pyplot(fig)
    
    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

elif app_mode == "Regression Analysis (Exp 11)":
    st.header("Experiment 11: Correlation & Regression")
    st.write("Concepts: Relationship between variables using Linear Regression.")
    
    x_axis = st.selectbox("Select Independent Variable (X)", df.columns[:-1], index=10) # Default to Alcohol
    y_axis = "quality"
    
    fig, ax = plt.subplots()
    sns.regplot(x=df[x_axis], y=df[y_axis], ax=ax, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    st.pyplot(fig)
    
    corr_value = df[x_axis].corr(df[y_axis])
    st.success(f"Correlation between {x_axis} and Quality: {corr_value:.2f}")

elif app_mode == "Dimensionality Reduction (Exp 12)":
    st.header("Experiment 12: PCA & LDA")
    st.write("Concepts: Reducing high-dimensional data for visualization.")
    
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    method = st.radio("Select Method", ["PCA", "LDA"])
    
    if method == "PCA":
        model = PCA(n_components=2)
        components = model.fit_transform(X)
    else:
        model = LDA(n_components=2)
        components = model.fit_transform(X, y)
        
    res_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
    res_df['quality'] = y
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=res_df, x='PC1', y='PC2', hue='quality', palette='viridis', ax=ax)
    st.pyplot(fig)

elif app_mode == "Clustering (Exp 13)":
    st.header("Experiment 13: K-Means Clustering")
    st.write("Concepts: Grouping data based on chemical characteristics.")
    
    k = st.slider("Select Number of Clusters (K)", 2, 6, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    
    # Using Alcohol and pH for clustering visualization
    features = ['alcohol', 'pH']
    X_clust = df[features]
    df['cluster'] = kmeans.fit_transform(X_clust).argmax(axis=1)
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='alcohol', y='pH', hue='cluster', palette='Set1', ax=ax)
    st.pyplot(fig)
    st.write(f"Clusters generated based on Alcohol and pH levels.")
