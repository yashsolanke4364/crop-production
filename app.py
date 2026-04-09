import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# --- PAGE CONFIGURATION (Frontend Setup in Python) ---
st.set_page_config(page_title="Agri-Data Intelligence Dashboard", layout="wide")

# --- DATA PROCESSING (Assignment Concepts: Exp 6, 8, 9) ---
@st.cache_data
def load_and_clean_data():
    # Load dataset
    df = pd.read_csv('India Agriculture Crop Production.csv')
    
    # Data Cleaning (Exp 9)
    # Removing rows where essential metrics are missing
    df = df.dropna(subset=['Production', 'Area', 'Yield'])
    
    # Feature Engineering: Simplified Year (Exp 9)
    df['Year_Numeric'] = df['Year'].str.split('-').str[0].astype(int)
    
    return df

df = load_and_clean_data()

# --- SIDEBAR FILTERS (Interactive Elements) ---
st.sidebar.header("Dashboard Filters")
selected_state = st.sidebar.multiselect("Select State(s)", options=df['State'].unique(), default=df['State'].unique()[:5])
filtered_df = df[df['State'].isin(selected_state)]

# --- HEADER SECTION ---
st.title("🌾 India Agriculture Crop Production Analytics")
st.markdown("This dashboard applies concepts from **Experiments 6-13**: Cleaning, Statistical Analysis, Linear Regression, PCA, and Clustering.")

# --- 1. KEY METRICS (Exp 6: Summary Statistics) ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(filtered_df))
col2.metric("Avg Yield", f"{round(filtered_df['Yield'].mean(), 2)} T/H")
col3.metric("Total Production", f"{round(filtered_df['Production'].sum() / 1e6, 2)}M Tonnes")
col4.metric("Max Area", f"{int(filtered_df['Area'].max())} Hectares")

st.divider()

# --- 2. TREND & DISTRIBUTION (Exp 10: Interactive Visualization) ---
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Production by Season")
    fig_season = px.pie(filtered_df, values='Production', names='Season', hole=0.4, 
                        color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_season, use_container_width=True)

with row1_col2:
    st.subheader("Crop Production Trends (Top 10 Crops)")
    top_crops = filtered_df.groupby('Crop')['Production'].sum().nlargest(10).reset_index()
    fig_crop = px.bar(top_crops, x='Crop', y='Production', color='Crop', text_auto='.2s')
    st.plotly_chart(fig_crop, use_container_width=True)

# --- 3. LINEAR REGRESSION (Exp 11: Correlation Analysis) ---
st.divider()
st.subheader("📈 Machine Learning: Linear Regression (Area vs Production)")
reg_col1, reg_col2 = st.columns([1, 3])

# Sample data for regression visualization
sample_size = min(2000, len(filtered_df))
reg_df = filtered_df.sample(sample_size)

X = reg_df[['Area']].values
y = reg_df['Production'].values
model = LinearRegression().fit(X, y)
r_squared = model.score(X, y)

with reg_col1:
    st.write("Regression Summary:")
    st.info(f"**Coefficient:** {model.coef_[0]:.2f}")
    st.info(f"**R² Score:** {r_squared:.4f}")
    st.write("This indicates how strongly the land area determines the total production output.")

with reg_col2:
    fig_reg = px.scatter(reg_df, x='Area', y='Production', trendline="ols", 
                         trendline_color_override="red", opacity=0.5)
    st.plotly_chart(fig_reg, use_container_width=True)

# --- 4. CLUSTERING & PCA (Exp 12 & 13: Unsupervised Learning) ---
st.divider()
st.subheader("🤖 Pattern Recognition: K-Means Clustering & PCA")

# Prepare data for Clustering (Grouping states by performance)
cluster_df = df.groupby('State')[['Area', 'Yield', 'Production']].mean()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_df)

# K-Means (Exp 13)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_df['Cluster'] = kmeans.fit_predict(scaled_data).astype(str)

# PCA for Visualization (Exp 12)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)
cluster_df['PC1'] = pca_components[:, 0]
cluster_df['PC2'] = pca_components[:, 1]

clust_col1, clust_col2 = st.columns(2)

with clust_col1:
    st.write("**K-Means Clustering (Area vs Yield)**")
    fig_cl = px.scatter(cluster_df.reset_index(), x='Area', y='Yield', color='Cluster', 
                        hover_name='State', size='Production')
    st.plotly_chart(fig_cl, use_container_width=True)

with clust_col2:
    st.write("**PCA: Dimensionality Reduction**")
    fig_pca = px.scatter(cluster_df.reset_index(), x='PC1', y='PC2', color='Cluster',
                         hover_name='State', title="States Projected onto 2 Principal Components")
    st.plotly_chart(fig_pca, use_container_width=True)

# --- DATA TABLE (Exp 7: Displaying Records) ---
st.divider()
st.subheader("Raw Data Preview")
st.dataframe(filtered_df.head(100), use_container_width=True)
