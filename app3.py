import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Page configuration
st.set_page_config(layout="wide", page_title="Streaming Platform Analysis")

# Set Matplotlib style
plt.style.use('fivethirtyeight')

# --- Data Loading ---
@st.cache_data
def load_data():
    """
    Loads the single dataset and processes it for platform comparison.
    """
    try:
        # Load the dataset
        df = pd.read_csv('data/MoviesOnStreamingPlatforms_updated.csv')
    except FileNotFoundError as e:
        st.error(f"Error loading file: {e}")
        st.error("Please make sure 'MoviesOnStreamingPlatforms_updated.csv' is in a folder named 'data'.")
        return None, None

    # --- 1. Clean and Prepare Data ---
    
    # Standardize column names (remove spaces, make lowercase)
    df.columns = df.columns.str.strip().str.lower()
    
    # Check for essential columns
    rt_col = 'rotten tomatoes'
    platform_cols = ['netflix', 'hulu', 'prime video', 'disney+']
    if rt_col not in df.columns:
        st.error(f"Critical Error: Column '{rt_col}' not found in the dataset.")
        st.info(f"Available columns are: {list(df.columns)}")
        return None, None
        
    # --- 2. Process Rotten Tomatoes Rating ---
    # The 'rotten tomatoes' column is a string like '95/100'
    df['rt_rating'] = df[rt_col].str.split('/').str[0]
    df['rt_rating'] = pd.to_numeric(df['rt_rating'], errors='coerce')
    
    # Drop rows where we have no rating, year, or age
    df_cleaned = df.dropna(subset=['rt_rating', 'year', 'age']).copy()
    df_cleaned['year'] = df_cleaned['year'].astype(int)
    df_cleaned['age'] = df_cleaned['age'].astype(str) # Ensure 'age' is a string for grouping
    
    # --- 3. "Melt" Data for Platform Comparison ---
    # This is the key step: we unpivot the platform columns
    # From: [Title, RT_Rating, Netflix, Hulu]
    # To:   [Title, RT_Rating, Platform]
    
    id_vars = ['title', 'year', 'age', 'rt_rating']
    # Ensure all id_vars exist, otherwise use a minimal set
    id_vars = [col for col in id_vars if col in df_cleaned.columns]
    
    df_melted = df_cleaned.melt(
        id_vars=id_vars, 
        value_vars=platform_cols, 
        var_name='platform', 
        value_name='on_platform'
    )
    
    # We only care about rows where the movie IS on the platform (on_platform == 1)
    df_platform_data = df_melted[df_melted['on_platform'] == 1].copy()
    
    # Capitalize platform names for display
    df_platform_data['platform'] = df_platform_data['platform'].str.title()

    return df_cleaned, df_platform_data

# --- Load all data ---
st.title("Streaming Platform Content Analysis")
st.markdown("A comparative analysis of **Netflix, Hulu, Prime Video, and Disney+** using Rotten Tomatoes Ratings.")

# Run the heavy loading function
with st.spinner("Loading and processing data..."):
    df_cleaned, df_platform_data = load_data()

if df_cleaned is None:
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", 
    [
        "1. Platform Overview",
        "2. Platform Quality Comparison",
        "3. Top Rated Content by Platform",
        "4. Library Age & Maturity"
    ]
)
st.sidebar.markdown("---")
st.sidebar.markdown("""
This app analyzes the "Movies on Streaming Platforms" dataset to compare the content libraries of the top 4 streaming services.
""")

# ==============================================================================
# --- PAGE 1: Platform Overview ---
# ==============================================================================
if page == "1. Platform Overview":
    st.header("Platform Overview")
    st.markdown("How much content does each platform have in this dataset?")

    # 1. Bar chart of content counts
    st.subheader("Total Titles per Platform")
    platform_counts = df_platform_data['platform'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#ff7f0e', '#1f77b4', '#d62728', '#2ca02c'] # Assign colors to match platform
    ax.bar(platform_counts.index, platform_counts.values, color=colors)
    ax.set_title("Number of Titles per Streaming Service")
    ax.set_ylabel("Number of Titles")
    st.pyplot(fig)
    st.caption("This chart shows the total number of titles listed for each platform in the dataset. Note: 'Prime Video' has a significantly larger library.")

    # 2. Data Summary
    st.subheader("Data Summary")
    col1, col2 = st.columns(2)
    col1.metric("Total Unique Titles in Dataset", f"{len(df_cleaned)}")
    col2.metric("Total Platform Listings Analyzed", f"{len(df_platform_data)}")

    st.subheader("Data Sample (Original Cleaned Data)")
    st.dataframe(df_cleaned[['title', 'year', 'rt_rating', 'age', 'netflix', 'hulu', 'prime video', 'disney+']].head())

# ==============================================================================
# --- PAGE 2: Platform Quality Comparison ---
# ==============================================================================
elif page == "2. Platform Quality Comparison":
    st.header("Which platform has the highest quality content?")
    st.markdown("We can't just use 'Top 10' lists. A true comparison looks at the *entire* library.")

    # 1. Box Plot - The best way to compare distributions
    st.subheader("Distribution of Rotten Tomatoes Ratings")
    st.markdown("A box plot is the best way to see the range (min, max, median, 25th/75th percentiles) of ratings for each platform. It shows the *full picture* of a library's quality.")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    # Create a list of data for each platform to control order and colors
    platforms_in_order = ['Prime Video', 'Netflix', 'Hulu', 'Disney+']
    data_to_plot = [df_platform_data[df_platform_data['platform'] == p]['rt_rating'] for p in platforms_in_order]
    
    bp = ax.boxplot(data_to_plot, patch_artist=True, labels=platforms_in_order)
    
    colors = ['#ff7f0e', '#d62728', '#1f77b4', '#2ca02c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        
    ax.set_title("Rating Distribution by Platform")
    ax.set_ylabel("Rotten Tomatoes Rating (1-100)")
    ax.set_xlabel("Platform")
    st.pyplot(fig)
    
    # 2. Bar Chart of Average Ratings
    st.subheader("Average Rotten Tomatoes Rating")
    st.markdown("While a simple average can be misleading, it gives a good high-level summary.")
    
    avg_ratings = df_platform_data.groupby('platform')['rt_rating'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(avg_ratings.index, avg_ratings.values, color=['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e'])
    ax.set_title("Average Rating by Platform")
    ax.set_ylabel("Average Rotten Tomatoes Rating")
    ax.set_xlabel("Platform")
    ax.set_ylim(bottom=avg_ratings.min() - 5) # Start y-axis lower
    st.pyplot(fig)
    
    st.subheader("Insights")
    st.markdown(f"""
    * **Prime Video** has the largest library, but its box plot is centered lower, showing a very wide range of quality (many low-rated titles).
    * **Disney+** has a smaller library, but its ratings are high and tightly grouped (a high 'quality floor').
    * **Netflix** and **Hulu** are very competitive, with similar median ratings and distributions.
    * The highest average rating belongs to **{avg_ratings.index[0]}** with a score of **{avg_ratings.iloc[0]:.2f}**.
    """)

# ==============================================================================
# --- PAGE 3: Top Rated Content by Platform ---
# ==============================================================================
elif page == "3. Top Rated Content by Platform":
    st.header("Top Rated Content by Platform")
    st.markdown("Discover the best content available on each service, or see the best of the best from all of them.")

    # 1. Platform Selector
    platforms = ['All Platforms (Unique Titles)'] + list(df_platform_data['platform'].unique())
    selected_platform = st.selectbox("Select a platform to see its top-rated content:", platforms)

    # 2. Filter data based on selection
    if selected_platform == 'All Platforms (Unique Titles)':
        st.subheader("Top 20 Rated Titles (All Platforms)")
        # Use the original cleaned df for a unique list
        data_to_show = df_cleaned
        
    else:
        st.subheader(f"Top 20 Rated Titles on {selected_platform}")
        # Filter the melted dataframe for the specific platform
        data_to_show = df_platform_data[df_platform_data['platform'] == selected_platform]

    # 3. Display the top 20
    top_20 = data_to_show[['title', 'year', 'rt_rating', 'age']].sort_values(by='rt_rating', ascending=False).head(20).set_index('title')
    st.dataframe(top_20, use_container_width=True)

# ==============================================================================
# --- PAGE 4: Library Age & Maturity ---
# ==============================================================================
elif page == "4. Library Age & Maturity":
    st.header("Library Age & Maturity Analysis")
    st.markdown("What kind of content does each platform have? Is it new or old? Is it for kids or adults?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Library Age Distribution (KDE Plot)
        st.subheader("Library Age Distribution")
        st.markdown("This shows the distribution of release years for each platform's library. A peak on the right means a 'newer' library.")
        
        fig, ax = plt.subplots(figsize=(10, 7))
        platforms_in_order = ['Prime Video', 'Netflix', 'Hulu', 'Disney+']
        colors = ['#ff7f0e', '#d62728', '#1f77b4', '#2ca02c']
        
        for platform, color in zip(platforms_in_order, colors):
            platform_data = df_platform_data[df_platform_data['platform'] == platform]
            # Use pandas plotting to get a Kernel Density Estimate
            platform_data['year'].plot(kind='kde', ax=ax, label=platform, color=color, linewidth=3)

        ax.set_title("Distribution of Title Release Years")
        ax.set_xlabel("Year of Release")
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_xlim(left=1980, right=2021) # Focus on the modern era
        st.pyplot(fig)

    with col2:
        # 2. Content Maturity (Stacked Bar Plot)
        st.subheader("Content Maturity by Age Rating")
        st.markdown("This shows the *proportion* of each library that falls into different age ratings.")
        
        # Get counts of each age rating per platform
        age_counts = df_platform_data.groupby(['platform', 'age']).size().unstack(fill_value=0)
        
        # We only care about the main ratings
        common_age_ratings = [col for col in ['all', '7+', '13+', '16+', '18+'] if col in age_counts.columns]
        age_counts = age_counts[common_age_ratings]
        
        # Calculate proportions (100% stacked bar)
        age_proportions = age_counts.apply(lambda x: x / x.sum(), axis=1).sort_values(by='18+', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        age_proportions.plot(kind='barh', stacked=True, ax=ax, colormap='coolwarm_r')
        
        ax.set_title("Proportion of Content by Age Rating")
        ax.set_xlabel("Proportion")
        ax.set_ylabel("Platform")
        ax.legend(title='Age Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlim(0, 1)
        # Format x-axis as percentage
        ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("Insights")
    st.markdown("""
    * **Library Age (Left Plot):**
        * **Prime Video** has the most diverse library, with a large collection of older titles (peak around 2010-2015) but also many new ones.
        * **Netflix** and **Hulu** are very similar, with libraries heavily skewed towards newer content (peak after 2015).
        * **Disney+** has two distinct peaks: one for its classic animated films (pre-2000) and one for its modern blockbusters (2010-2020).
        
    * **Content Maturity (Right Plot):**
        * **Disney+** is, by far, the most family-friendly platform, with a library dominated by 'all' and '7+' ratings.
        * **Netflix** and **Hulu** have the most mature libraries, with the largest proportions of '18+' and '16+' content.
        * **Prime Video** sits in the middle, offering a large block of '13+' content in addition to its mature and family-friendly titles.
    """)