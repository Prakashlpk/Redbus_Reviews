import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sqlalchemy import create_engine
import pymysql
import os
import traceback

# -----------------------
# CONFIG
# -----------------------
DEBUG = False

DB_CONFIG = {
    'host': "gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
    'user': "3843tbpvmio3tiw.root",
    'password': "BvhwR5z1itdG9CMJ",
    'database': "redbus_reviews",
    'port': 4000,
    'ssl': {"ca": r"D:\prakash\redbus_scraper\isrgrootx1 (3).pem"},
}

DB_CONFIG['host'] = os.getenv('DB_HOST', DB_CONFIG['host'])
DB_CONFIG['user'] = os.getenv('DB_USER', DB_CONFIG['user'])
DB_CONFIG['password'] = os.getenv('DB_PASSWORD', DB_CONFIG['password'])
DB_CONFIG['database'] = os.getenv('DB_NAME', DB_CONFIG['database'])
DB_CONFIG['port'] = int(os.getenv('DB_PORT', DB_CONFIG['port']))

st.set_page_config(
    page_title="RedBus Reviews Analytics",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# SQLAlchemy Engine
# -----------------------
pymysql.install_as_MySQLdb()

connect_url = (
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

engine = create_engine(
    connect_url,
    connect_args={"ssl": DB_CONFIG.get("ssl")},
    pool_pre_ping=True,
    pool_recycle=3600,
    future=True
)

# -----------------------
# Helpers & Data Loaders
# -----------------------
def analyze_review_text_sentiment(text):
    """
    Analyze review text for negative, positive, and neutral keywords.
    Returns a sentiment score based on keyword analysis.
    """
    if pd.isna(text) or str(text).strip() == '':
        return 0.0
    
    text = str(text).lower()
    
    # Comprehensive negative keywords
    negative_keywords = [
        'insufficient', 'not working', 'not properly', 'poorly', 'bad', 'worst',
        'terrible', 'horrible', 'awful', 'pathetic', 'useless', 'waste',
        'never', 'no', 'not', 'late', 'delay', 'delayed', 'old', 'dirty',
        'uncomfortable', 'rude', 'unprofessional', 'unhygienic', 'smelly',
        'broken', 'damaged', 'defective', 'poor', 'disappointing', 'disappointed',
        'issue', 'problem', 'complaint', 'unsafe', 'dangerous', 'won\'t',
        'wouldn\'t', 'couldn\'t', 'shouldn\'t', 'failed', 'failure', 'inadequate',
        'substandard', 'unacceptable', 'worst', 'avoid', 'never again'
    ]
    
    # Comprehensive positive keywords
    positive_keywords = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'best', 'love', 'perfect', 'clean', 'comfortable', 'smooth', 'nice',
        'friendly', 'professional', 'helpful', 'polite', 'timely', 'punctual',
        'on time', 'recommend', 'satisfied', 'happy', 'pleased', 'enjoyed',
        'appreciated', 'thank', 'awesome', 'superb', 'outstanding', 'impressed'
    ]
    
    # Neutral/mixed indicators
    neutral_keywords = [
        'but', 'however', 'although', 'except', 'otherwise', 'overall'
    ]
    
    # Count occurrences
    negative_count = sum(1 for word in negative_keywords if word in text)
    positive_count = sum(1 for word in positive_keywords if word in text)
    neutral_count = sum(1 for word in neutral_keywords if word in text)
    
    # Calculate sentiment score
    total_count = negative_count + positive_count
    
    if total_count == 0:
        return 0.0
    
    # If mixed indicators present, reduce the magnitude
    if neutral_count > 0:
        sentiment_score = (positive_count - negative_count) / total_count * 0.5
    else:
        sentiment_score = (positive_count - negative_count) / total_count
    
    return sentiment_score

def fix_sentiment_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced sentiment classification using both star ratings AND review text analysis.
    This addresses issues where:
    1. 5-star reviews were incorrectly labeled as Neutral
    2. 3-4 star reviews with purely negative text were labeled as Neutral/Positive
    """
    if 'Star_Rating' not in df.columns:
        return df
    
    # Create a backup of original sentiment if it doesn't exist
    if 'Original_Sentiment' not in df.columns and 'Sentiment_Label' in df.columns:
        df['Original_Sentiment'] = df['Sentiment_Label'].copy()
    
    # Analyze review text sentiment if available
    if 'Review_Text' in df.columns:
        df['Text_Sentiment_Score'] = df['Review_Text'].apply(analyze_review_text_sentiment)
    else:
        df['Text_Sentiment_Score'] = 0.0
    
    # Apply improved sentiment logic
    def classify_sentiment(row):
        star_rating = row.get('Star_Rating')
        text_sentiment = row.get('Text_Sentiment_Score', 0.0)
        original_sentiment_score = row.get('Sentiment_Score', 0.0)
        
        # If star rating is missing, keep original sentiment
        if pd.isna(star_rating):
            return row.get('Sentiment_Label', 'Neutral')
        
        # CLEAR POSITIVE: 5 stars OR 4 stars with positive/neutral text
        if star_rating == 5:
            return 'Positive'
        elif star_rating == 4:
            # If text is clearly negative, mark as Neutral
            if text_sentiment < -0.3:
                return 'Neutral'
            return 'Positive'
        
        # CLEAR NEGATIVE: 1-2 stars
        elif star_rating <= 2:
            return 'Negative'
        
        # NUANCED CASE: 3 stars - requires text analysis
        else:  # star_rating == 3
            # If text is purely negative (no positive aspects), mark as Negative
            if text_sentiment < -0.3:
                return 'Negative'
            # If text has mixed sentiment or positive aspects, mark as Neutral
            elif -0.3 <= text_sentiment <= 0.3:
                return 'Neutral'
            # If text is mostly positive despite 3 stars, mark as Neutral (not Positive)
            else:
                return 'Neutral'
    
    df['Sentiment_Label'] = df.apply(classify_sentiment, axis=1)
    
    # Update sentiment score based on new classification
    if 'Sentiment_Score' in df.columns:
        # For 5-star reviews
        df.loc[df['Star_Rating'] == 5, 'Sentiment_Score'] = df.loc[
            df['Star_Rating'] == 5, 'Text_Sentiment_Score'
        ].fillna(0.8)
        
        # For 4-star reviews
        df.loc[df['Star_Rating'] == 4, 'Sentiment_Score'] = df.loc[
            df['Star_Rating'] == 4, 'Text_Sentiment_Score'
        ].fillna(0.6)
        
        # For 3-star reviews with negative text
        mask_3_star_negative = (df['Star_Rating'] == 3) & (df['Text_Sentiment_Score'] < -0.3)
        df.loc[mask_3_star_negative, 'Sentiment_Score'] = -0.5
        
        # For 1-2 star reviews
        df.loc[df['Star_Rating'] <= 2, 'Sentiment_Score'] = df.loc[
            df['Star_Rating'] <= 2, 'Text_Sentiment_Score'
        ].fillna(-0.8)
    
    return df

def safe_postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce commonly used columns to proper dtypes and clean strings."""
    if df is None or df.empty:
        return pd.DataFrame()

    string_cols = ['Route', 'Bus_Name', 'Bus_Type', 'Name_of_passenger', 'Review_Text', 'Sentiment_Label']
    for c in string_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: x.decode() if isinstance(x, (bytes, bytearray)) else x)
            df[c] = df[c].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '': np.nan})

    integer_cols = ['Punctuality', 'Cleanliness', 'Staff_Behaviour', 'Driving',
                    'AC', 'Rest_stop_hygiene', 'Seat_comfort', 'Live_tracking']
    for c in integer_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: x.decode() if isinstance(x, (bytes, bytearray)) else x)
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = df[c].astype('Int64')

    float_cols = ['Price', 'Star_Rating', 'Sentiment_Score']
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: x.decode() if isinstance(x, (bytes, bytearray)) else x)
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if 'Date_of_Review' in df.columns:
        df['Date_of_Review'] = df['Date_of_Review'].apply(
            lambda x: x.decode() if isinstance(x, (bytes, bytearray)) else x
        )
        df['Date_of_Review'] = pd.to_datetime(df['Date_of_Review'], errors='coerce')

    if 'Sentiment_Label' in df.columns:
        df.loc[df['Sentiment_Label'].notna(), 'Sentiment_Label'] = df.loc[
            df['Sentiment_Label'].notna(), 'Sentiment_Label'
        ].str.title()

    # **FIX SENTIMENT CLASSIFICATION - This is the key addition**
    df = fix_sentiment_classification(df)

    return df

@st.cache_data(ttl=600)
def load_data(table_name):
    """Production loader using SQLAlchemy engine (cached)."""
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(sql=query, con=engine)
        df = safe_postprocess(df)
        return df
    except Exception as e:
        if DEBUG:
            st.error("Error reading DB (full traceback below):")
            st.error(traceback.format_exc())
        else:
            st.error(f"Database read failed: {e}")
        return pd.DataFrame()

def load_data_debug(table_name, limit_rows=1000):
    """Verbose loader for debugging (not cached)."""
    try:
        q = f"SELECT * FROM {table_name} LIMIT {limit_rows}"
        df = pd.read_sql_query(sql=q, con=engine)
        st.write("DEBUG: df.shape:", df.shape)
        st.write("DEBUG: df.columns repr:", [repr(c) for c in df.columns])
        st.write("DEBUG: head:"); st.dataframe(df.head(5))
        df = safe_postprocess(df)
        st.write("DEBUG: after postprocess dtypes:"); st.write(df.dtypes)
        return df
    except Exception:
        st.error("Debug read failed:")
        st.error(traceback.format_exc())
        return pd.DataFrame()

# -----------------------
# Charts & UI Components
# -----------------------
def home_page(df, operator_name):
    st.title(f"🚌 RedBus Reviews Analytics Dashboard - {operator_name}")
    
    # Show sentiment fix notification
    if 'Original_Sentiment' in df.columns:
        changes = (df['Sentiment_Label'] != df['Original_Sentiment']).sum()
        if changes > 0:
            st.info(f"ℹ️ Sentiment classification has been corrected for {changes:,} reviews based on star ratings")
    
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Reviews", f"{len(df):,}")
    with col2:
        if 'Star_Rating' in df.columns and df['Star_Rating'].notna().sum() > 0:
            avg_rating = df['Star_Rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f}/5")
        else:
            st.metric("Average Rating", "N/A")
    with col3:
        if 'Route' in df.columns and df['Route'].notna().sum() > 0:
            st.metric("Total Routes", f"{df['Route'].nunique():,}")
        else:
            st.metric("Total Routes", "N/A")
    with col4:
        if 'Bus_Name' in df.columns and df['Bus_Name'].notna().sum() > 0:
            st.metric("Total Buses", f"{df['Bus_Name'].nunique():,}")
        else:
            st.metric("Total Buses", "N/A")
    with col5:
        if 'Sentiment_Label' in df.columns and df['Sentiment_Label'].notna().sum() > 0:
            positive_count = (df['Sentiment_Label'].str.lower() == 'positive').sum()
            positive_pct = (positive_count / len(df)) * 100
            st.metric("Positive Reviews", f"{positive_pct:.1f}%")
        else:
            st.metric("Positive Reviews", "N/A")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Rating Distribution")
        if 'Star_Rating' in df.columns and df['Star_Rating'].notna().sum() > 0:
            rating_counts = df['Star_Rating'].value_counts().sort_index()
            fig = px.bar(
                x=rating_counts.index, y=rating_counts.values,
                labels={'x': 'Star Rating', 'y': 'Number of Reviews'},
                color=rating_counts.values, color_continuous_scale='RdYlGn'
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⚠️ Rating column not found")

    with col2:
        st.subheader("😊 Sentiment Analysis")
        if 'Sentiment_Label' in df.columns and df['Sentiment_Label'].notna().sum() > 0:
            df_sent = df[df['Sentiment_Label'].notna()].copy()
            sentiment_counts = df_sent['Sentiment_Label'].value_counts()
            colors = {'Positive': '#4caf50', 'Neutral': '#ff9800', 'Negative': '#f44336'}
            fig = px.pie(
                values=sentiment_counts.values, names=sentiment_counts.index,
                color=sentiment_counts.index, color_discrete_map=colors, hole=0.4
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⚠️ Sentiment_Label not found")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Reviews Over Time")
        if 'Date_of_Review' in df.columns and df['Date_of_Review'].notna().sum() > 0:
            df_temp = df[df['Date_of_Review'].notna()].copy()
            reviews_by_month = df_temp.groupby(df_temp['Date_of_Review'].dt.to_period('M')).size()
            reviews_by_month.index = reviews_by_month.index.astype(str)
            fig = px.line(x=reviews_by_month.index, y=reviews_by_month.values,
                          labels={'x': 'Month', 'y': 'Number of Reviews'})
            fig.update_traces(line_color='#d84e55', line_width=3)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⚠️ Date information not available")

    with col2:
        st.subheader("🏆 Bus Operators With the Highest Number of Reviews")
        if 'Bus_Name' in df.columns and df['Bus_Name'].notna().sum() > 0:
            top_buses = df['Bus_Name'].value_counts().head(10)
            fig = px.bar(
                x=top_buses.values, y=top_buses.index, orientation='h',
                labels={'x': 'Number of Reviews', 'y': 'Bus Operator'},
                color=top_buses.values, color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⚠️ Bus name data not available")

def find_best_buses_page(df):
    st.title("🎯 Find Best Buses by Route")
    st.markdown("### Filter buses based on comprehensive review metrics to find the best option for your journey")
    st.markdown("---")
    
    with st.sidebar:
        st.header("🔍 Smart Filters")
        st.markdown("**Route Selection**")
        
        if 'Route' in df.columns and df['Route'].notna().sum() > 0:
            routes = sorted([r for r in df['Route'].dropna().unique() if str(r).strip()])
            selected_route = st.selectbox("Select Your Route", ['All Routes'] + routes, 
                                         help="Choose the route you want to travel")
        else:
            selected_route = 'All Routes'
            st.warning("⚠️ Route data not available")
        
        st.markdown("---")
        st.markdown("**Bus Preferences**")
        
        if 'Bus_Type' in df.columns and df['Bus_Type'].notna().sum() > 0:
            bus_types = sorted([b for b in df['Bus_Type'].dropna().unique() if str(b).strip()])
            selected_bus_types = st.multiselect("Bus Type", bus_types, default=bus_types,
                                               help="Select one or more bus types")
        else:
            selected_bus_types = None
            st.warning("⚠️ Bus type not available")
        
        if 'Price' in df.columns and df['Price'].notna().sum() > 0:
            try:
                min_price = int(df['Price'].min())
                max_price = int(df['Price'].max())
                price_range = st.slider("Price Range (₹)", min_price, max_price, 
                                       (min_price, max_price),
                                       help="Set your budget range")
            except Exception:
                price_range = None
                st.warning("⚠️ Price filter unavailable")
        else:
            price_range = None
        
        st.markdown("---")
        st.markdown("**Review-Based Filters** ⭐")
        
        if 'Star_Rating' in df.columns and df['Star_Rating'].notna().sum() > 0:
            min_rating = st.slider("Minimum Star Rating", 1.0, 5.0, 3.0, 0.5,
                                  help="Filter buses with minimum rating")
        else:
            min_rating = None
        
        if 'Sentiment_Label' in df.columns and df['Sentiment_Label'].notna().sum() > 0:
            available_sentiments = sorted(df['Sentiment_Label'].dropna().unique().tolist())
            sentiment_options = st.multiselect("Sentiment", options=available_sentiments, 
                                             default=available_sentiments,
                                             help="Filter by customer sentiment (corrected)")
        else:
            sentiment_options = None
        
        st.markdown("---")
        st.markdown("**Service Quality Filters** 🎯")
        
        quality_metrics = {
            'Punctuality': 'On-time performance',
            'Cleanliness': 'Bus cleanliness',
            'Staff_Behaviour': 'Staff behavior',
            'Driving': 'Driving quality',
            'AC': 'AC quality',
            'Rest_stop_hygiene': 'Rest stop hygiene',
            'Seat_comfort': 'Seat comfort',
            'Live_tracking': 'Live tracking'
        }
        
        quality_filters = {}
        for metric, description in quality_metrics.items():
            if metric in df.columns and df[metric].notna().sum() > 0:
                max_val = min(500, int(df[metric].max()))
                quality_filters[metric] = st.slider(
                    f"Min {metric.replace('_', ' ')}",
                    0, max_val, 0, 1,
                    help=f"Minimum likes for {description} (0–{max_val})"
                )
        
        st.markdown("---")
        st.markdown("**Sentiment Score**")
        if 'Sentiment_Score' in df.columns and df['Sentiment_Score'].notna().sum() > 0:
            min_sentiment_score = st.slider("Minimum Sentiment Score", -1.0, 1.0, -1.0, 0.1,
                                           help="Filter by sentiment score (-1 to 1)")
        else:
            min_sentiment_score = None

    filtered_df = df.copy()
    filter_messages = []
    
    if 'Route' in df.columns and selected_route != 'All Routes':
        filtered_df = filtered_df[filtered_df['Route'] == selected_route]
        filter_messages.append(f"Route: {selected_route}")
    
    if selected_bus_types and 'Bus_Type' in df.columns:
        filtered_df = filtered_df[filtered_df['Bus_Type'].isin(selected_bus_types)]
        filter_messages.append(f"Bus Types: {len(selected_bus_types)}")
    
    if price_range and 'Price' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['Price'] >= price_range[0]) & 
            (filtered_df['Price'] <= price_range[1])
        ]
        filter_messages.append(f"Price: ₹{price_range[0]}-₹{price_range[1]}")
    
    if min_rating and 'Star_Rating' in df.columns:
        filtered_df = filtered_df[filtered_df['Star_Rating'] >= min_rating]
        filter_messages.append(f"Min Rating: {min_rating}⭐")
    
    if sentiment_options and 'Sentiment_Label' in df.columns:
        filtered_df = filtered_df[filtered_df['Sentiment_Label'].isin(sentiment_options)]
        filter_messages.append(f"Sentiments: {', '.join(sentiment_options)}")
    
    if min_sentiment_score and 'Sentiment_Score' in df.columns:
        filtered_df = filtered_df[filtered_df['Sentiment_Score'] >= min_sentiment_score]
        filter_messages.append(f"Min Sentiment Score: {min_sentiment_score}")
    
    for metric, min_value in quality_filters.items():
        if min_value > 0:
            filtered_df = filtered_df[filtered_df[metric] >= min_value]
            filter_messages.append(f"Min {metric.replace('_', ' ')}: {min_value}")
    
    if len(filtered_df) > 0:
        st.success(f"✅ Found **{len(filtered_df):,}** reviews matching your criteria")
        
        if filter_messages:
            with st.expander("🔍 Active Filters", expanded=False):
                for msg in filter_messages:
                    st.write(f"• {msg}")
        
        if 'Bus_Name' in filtered_df.columns:
            st.markdown("---")
            st.subheader("🏆 Top Buses Ranked by Reviews")
            
            agg_dict = {'review_id': 'count'}
            agg_columns = ['Star_Rating', 'Price', 'Sentiment_Score'] + list(quality_metrics.keys())
            
            for col in agg_columns:
                if col in filtered_df.columns:
                    agg_dict[col] = 'mean'
            
            bus_stats = filtered_df.groupby('Bus_Name').agg(agg_dict).round(2)
            bus_stats = bus_stats.rename(columns={'review_id': 'Total_Reviews'})

            int_display_cols = [
                'Punctuality', 'Cleanliness', 'Staff_Behaviour', 'Driving',
                'AC', 'Rest_stop_hygiene', 'Seat_comfort', 'Live_tracking',
                'Price', 'Star_Rating'
            ]
            for col in int_display_cols:
                if col in bus_stats.columns:
                    bus_stats[col] = bus_stats[col].round().astype('Int64')

            if 'Star_Rating' in bus_stats.columns:
                bus_stats = bus_stats.sort_values(
                    ['Star_Rating', 'Total_Reviews'], 
                    ascending=[False, False]
                ).head(20)
            else:
                bus_stats = bus_stats.sort_values('Total_Reviews', ascending=False).head(20)
            
            st.dataframe(
                bus_stats.style.background_gradient(
                    subset=[col for col in ['Star_Rating'] if col in bus_stats.columns],
                    cmap='RdYlGn'
                ),
                use_container_width=True
            )
            
            if 'Star_Rating' in bus_stats.columns and len(bus_stats) > 0:
                st.markdown("---")
                st.subheader("📊 Visual Comparison of Top Buses")
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Average Rating',
                    x=bus_stats.index[:10],
                    y=bus_stats['Star_Rating'][:10],
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title="Top 10 Buses by Average Rating",
                    xaxis_title="Bus Name",
                    yaxis_title="Average Rating",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("💬 Customer Reviews")
        
        if 'Star_Rating' in filtered_df.columns:
            display_df = filtered_df.sort_values('Star_Rating', ascending=False)
        else:
            display_df = filtered_df
        
        reviews_per_page = 10
        total_pages = max(1, (len(display_df) - 1) // reviews_per_page + 1)
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page - 1) * reviews_per_page
        end_idx = start_idx + reviews_per_page
        
        for idx, row in display_df.iloc[start_idx:end_idx].iterrows():
            bus_name = row.get('Bus_Name', 'N/A')
            route = row.get('Route', 'N/A')
            rating = row.get('Star_Rating', 'N/A')
            sentiment = row.get('Sentiment_Label', 'N/A')
            review_text = row.get('Review_Text', 'No review available')
            price = row.get('Price', 'N/A')
            
            # Show original sentiment if different
            original_sent = row.get('Original_Sentiment', sentiment)
            sentiment_note = ""
            if original_sent != sentiment:
                sentiment_note = f" (was: {original_sent})"
            
            sentiment_color = {
                'Positive': '#d4edda',
                'Neutral': '#fff3cd',
                'Negative': '#f8d7da'
            }.get(sentiment, '#ffffff')
            
            quality_display = []
            for metric in quality_metrics.keys():
                if metric in row and pd.notna(row[metric]):
                    quality_display.append(f"{metric.replace('_', ' ')}: {row[metric]:.1f}⭐")
            
            quality_str = " | ".join(quality_display) if quality_display else "No quality metrics available"
            
            st.markdown(f"""
            <div style="background-color:{sentiment_color};padding:16px;border-radius:10px;margin-bottom:12px;border-left:4px solid #007bff;">
                <h4 style="margin:0;color:#333">{bus_name}</h4>
                <p style="margin:4px 0;color:#666"><strong>Route:</strong> {route} | <strong>Price:</strong> ₹{price}</p>
                <p style="margin:4px 0"><strong>Rating:</strong> {rating}/5 ⭐ | <strong>Sentiment:</strong> {sentiment}{sentiment_note}</p>
                <p style="margin:4px 0;font-size:0.9em;color:#555"><strong>Quality Metrics:</strong> {quality_str}</p>
                <p style="margin:8px 0 0 0;color:#333"><strong>Review:</strong> {review_text}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.info(f"Showing reviews {start_idx + 1} to {min(end_idx, len(display_df))} of {len(display_df)} total")
        
    else:
        st.error("❌ No buses found matching your criteria")
        st.info("💡 Try relaxing some filters to see more results")
        
        if filter_messages:
            st.markdown("**Current Filters:**")
            for msg in filter_messages:
                st.write(f"• {msg}")

def business_insights_page(df):
    st.title("📈 Business Use Cases & Insights")
    st.markdown("### Data-driven insights for travelers and bus operators")
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "👥 Customer Insights", 
        "😊 Sentiment Analysis", 
        "🔧 Service Improvement", 
        "🎯 Recommendations"
    ])
    
    # ----------------------
    # TAB 1: Customer Insights
    # ----------------------
    with tab1:
        st.header("1. Customer Insights: Compare Bus Services")
        st.markdown("""
        Travelers can make informed decisions by comparing different bus services based on comprehensive customer reviews.
        """)
        
        if 'Bus_Name' in df.columns and 'Star_Rating' in df.columns:
            st.subheader("📊 Comparative Analysis of Bus Operators")
            
            bus_comparison = df.groupby('Bus_Name').agg({
                'review_id': 'count',
                'Star_Rating': 'mean',
                'Sentiment_Score': 'mean' if 'Sentiment_Score' in df.columns else 'count',
                'Price': 'mean' if 'Price' in df.columns else 'count'
            }).round(2)
            
            bus_comparison = bus_comparison.rename(columns={'review_id': 'Total_Reviews'})
            bus_comparison = bus_comparison.sort_values('Star_Rating', ascending=False).head(15)
            
            st.markdown("**Top Rated Bus Operators**")
            fig = px.bar(
                bus_comparison.reset_index(),
                x='Bus_Name',
                y='Star_Rating',
                color='Star_Rating',
                color_continuous_scale='Greens',
                title="Top 15 Bus Operators by Rating"
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown("**Bubble Chart: Rating Quality by Review Volume**")
            
            fig = px.scatter(
                bus_comparison.head(15).reset_index(),
                x='Bus_Name',
                y='Star_Rating',
                size='Total_Reviews',
                color='Star_Rating',
                hover_data={
                    'Bus_Name': True,
                    'Star_Rating': ':.2f',
                    'Total_Reviews': True
                },
                color_continuous_scale='RdYlGn',
                title="Top 15 Operators: Rating Quality by Review Volume",
                labels={'Star_Rating': 'Average Rating', 'Bus_Name': 'Bus Operator'}
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                height=400,
                yaxis_range=[0, 5.2]
            )
            fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown("**Key Insights:**")
            top_bus = bus_comparison.index[0]
            top_rating = bus_comparison.iloc[0]['Star_Rating']
            st.success(f"🏆 **{top_bus}** leads with an average rating of **{top_rating}/5**")
            
            if 'Price' in bus_comparison.columns:
                avg_price = bus_comparison['Price'].mean()
                st.info(f"💰 Average price among top operators: **₹{avg_price:.2f}**")
    
    # ----------------------
    # TAB 2: Sentiment Analysis
    # ----------------------
    with tab2:
        st.header("2. Sentiment Analysis: Identify Top-Rated Services")
        st.markdown("""
        Analyzing customer sentiment helps identify the best-performing buses and improve overall customer satisfaction.
        **Note:** Sentiment labels have been corrected based on star ratings.
        """)
        
        if 'Sentiment_Label' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Overall Sentiment Distribution")
                sentiment_counts = df['Sentiment_Label'].value_counts()
                colors = {'Positive': '#4caf50', 'Neutral': '#ff9800', 'Negative': '#f44336'}
                
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    color=sentiment_counts.index,
                    color_discrete_map=colors,
                    title="Customer Sentiment Breakdown (Corrected)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                positive_pct = (sentiment_counts.get('Positive', 0) / len(df)) * 100
                if positive_pct > 70:
                    st.success(f"✅ **{positive_pct:.1f}%** positive reviews indicate high customer satisfaction")
                elif positive_pct > 50:
                    st.info(f"ℹ️ **{positive_pct:.1f}%** positive reviews - room for improvement")
                else:
                    st.warning(f"⚠️ Only **{positive_pct:.1f}%** positive reviews - significant improvements needed")
            
            with col2:
                st.subheader("Sentiment by Bus Operator")
                if 'Bus_Name' in df.columns and 'Sentiment_Score' in df.columns:
                    temp = df[df['Sentiment_Score'].notna()].copy()

                    if 'review_id' in temp.columns:
                        bus_sentiment = (
                            temp.groupby('Bus_Name')
                            .agg(
                                Avg_Sentiment=('Sentiment_Score', 'mean'),
                                Total_Reviews=('review_id', 'count')
                            )
                            .reset_index()
                        )
                    else:
                        bus_sentiment = (
                            temp.groupby('Bus_Name')
                            .agg(
                                Avg_Sentiment=('Sentiment_Score', 'mean'),
                                Total_Reviews=('Sentiment_Score', 'size')
                            )
                            .reset_index()
                        )

                    bus_sentiment = bus_sentiment.sort_values(
                        'Avg_Sentiment', ascending=False
                    )

                    fig = px.bar(
                        bus_sentiment,
                        x='Bus_Name',
                        y='Avg_Sentiment',
                        color='Avg_Sentiment',
                        color_continuous_scale='RdYlGn',
                        title="Average Sentiment Score by Bus Operator",
                        hover_data={
                            'Bus_Name': True,
                            'Avg_Sentiment': ':.3f',
                            'Total_Reviews': True
                        },
                        labels={
                            'Bus_Name': 'Bus Operator',
                            'Avg_Sentiment': 'Sentiment Score'
                        }
                    )

                    fig.update_layout(
                        xaxis_tickangle=-90,
                        height=550,
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("⚠️ Bus_Name or Sentiment_Score column not found")

            
            if 'Date_of_Review' in df.columns and df['Date_of_Review'].notna().sum() > 0:
                st.subheader("📈 Sentiment Trends Over Time")
                df_time = df[df['Date_of_Review'].notna()].copy()
                df_time['Month'] = df_time['Date_of_Review'].dt.to_period('M').astype(str)
                
                sentiment_trend = df_time.groupby(['Month', 'Sentiment_Label']).size().unstack(fill_value=0)
                
                fig = go.Figure()
                for sentiment in sentiment_trend.columns:
                    color_map = {'Positive': '#4caf50', 'Neutral': '#ff9800', 'Negative': '#f44336'}
                    fig.add_trace(go.Scatter(
                        x=sentiment_trend.index,
                        y=sentiment_trend[sentiment],
                        name=sentiment,
                        mode='lines+markers',
                        line=dict(color=color_map.get(sentiment, '#999'), width=2)
                    ))
                
                fig.update_layout(
                    title="Sentiment Evolution Over Time",
                    xaxis_title="Month",
                    yaxis_title="Number of Reviews",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ----------------------
    # TAB 3: Service Improvement
    # ----------------------
    with tab3:
        st.header("3. Service Improvement: Identify Strengths & Weaknesses")
        st.markdown("""
        Bus operators can analyze customer feedback to understand which aspects of their service need improvement.
        """)
        
        quality_metrics = ['Punctuality', 'Cleanliness', 'Staff_Behaviour', 'Driving',
                          'AC', 'Rest_stop_hygiene', 'Seat_comfort', 'Live_tracking']
        available_quality = [m for m in quality_metrics if m in df.columns and df[m].notna().sum() > 0]
        
        if available_quality:
            st.subheader("🔍 Service Quality Breakdown")
            
            quality_avg = {metric: df[metric].mean() for metric in available_quality}
            quality_df = pd.DataFrame(list(quality_avg.items()), columns=['Metric', 'Average_Rating'])
            quality_df = quality_df.sort_values('Average_Rating', ascending=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    quality_df,
                    x='Average_Rating',
                    y='Metric',
                    orientation='h',
                    title="Average Rating by Service Metric",
                    color='Average_Rating',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                weak_areas = quality_df[quality_df['Average_Rating'] < 3.5]
                if not weak_areas.empty:
                    st.warning("⚠️ **Areas Needing Improvement:**")
                    for _, row in weak_areas.iterrows():
                        st.write(f"• **{row['Metric'].replace('_', ' ')}**: {row['Average_Rating']:.2f}/5")
            
            with col2:
                if 'Bus_Name' in df.columns:
                    st.markdown("**Quality Metrics by Top Operators**")
                    
                    top_buses = df['Bus_Name'].value_counts().head(5).index
                    quality_by_bus = df[df['Bus_Name'].isin(top_buses)].groupby('Bus_Name')[available_quality].mean()
                    
                    fig = go.Figure()
                    for metric in available_quality[:4]:
                        fig.add_trace(go.Bar(
                            name=metric.replace('_', ' '),
                            x=quality_by_bus.index,
                            y=quality_by_bus[metric]
                        ))
                    
                    fig.update_layout(
                        title="Service Quality Comparison (Top 5 Operators)",
                        barmode='group',
                        height=400,
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🔗 Impact on Overall Rating")
            if 'Star_Rating' in df.columns:
                correlations = []
                for metric in available_quality:
                    corr = df[['Star_Rating', metric]].corr().iloc[0, 1]
                    correlations.append({'Metric': metric, 'Correlation': corr})
                
                corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
                
                fig = px.bar(
                    corr_df,
                    x='Correlation',
                    y='Metric',
                    orientation='h',
                    title="Correlation Between Service Metrics and Overall Rating",
                    color='Correlation',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
                
                strongest = corr_df.iloc[0]
                st.info(f"💡 **Key Insight:** {strongest['Metric'].replace('_', ' ')} has the strongest impact on overall rating (correlation: {strongest['Correlation']:.2f})")
        
        if 'Sentiment_Label' in df.columns and 'Review_Text' in df.columns:
            st.subheader("📝 Common Complaints Analysis")
            negative_reviews = df[df['Sentiment_Label'].str.lower() == 'negative']
            
            if not negative_reviews.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Negative Reviews", f"{len(negative_reviews):,}")
                    negative_pct = (len(negative_reviews) / len(df)) * 100
                    st.metric("Percentage", f"{negative_pct:.1f}%")
                
                with col2:
                    if 'Bus_Name' in negative_reviews.columns:
                        most_complaints = negative_reviews['Bus_Name'].value_counts().head(5)
                        st.markdown("**Operators with Most Complaints:**")
                        for bus, count in most_complaints.items():
                            st.write(f"• {bus}: {count} complaints")
    
    # ----------------------
    # TAB 4: Recommendations
    # ----------------------
    with tab4:
        st.header("4. Recommendation System: Personalized Bus Suggestions")
        st.markdown("""
        Based on comprehensive review analysis, we can recommend the best buses for different traveler preferences.
        """)
        
        st.subheader("🎯 Find Your Perfect Bus")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Budget Traveler**")
            st.write("Looking for affordable options with good reviews")
            if 'Price' in df.columns and 'Star_Rating' in df.columns:
                budget_buses = df[df['Star_Rating'] >= 3.5].nsmallest(10, 'Price')
                if not budget_buses.empty and 'Bus_Name' in budget_buses.columns:
                    unique_budget = budget_buses.drop_duplicates(subset=['Bus_Name']).head(5)
                    st.markdown("**Recommended:**")
                    for _, row in unique_budget.iterrows():
                        st.write(f"✓ {row['Bus_Name']} - ₹{row['Price']:.0f} ({row['Star_Rating']:.1f}⭐)")
        
        with col2:
            st.markdown("**Comfort Seeker**")
            st.write("Prioritizing comfort and amenities")
            if 'Seat_comfort' in df.columns and 'AC' in df.columns:
                comfort_buses = df[(df['Seat_comfort'] >= 4) & (df['AC'] >= 4)]
                if not comfort_buses.empty and 'Bus_Name' in comfort_buses.columns:
                    top_comfort = comfort_buses.groupby('Bus_Name').agg({
                        'Seat_comfort': 'mean',
                        'AC': 'mean',
                        'Star_Rating': 'mean'
                    }).nlargest(5, 'Seat_comfort')
                    st.markdown("**Recommended:**")
                    for bus, row in top_comfort.iterrows():
                        st.write(f"✓ {bus} - Comfort: {row['Seat_comfort']:.1f}⭐")
        
        with col3:
            st.markdown("**Reliability Focused**")
            st.write("Need punctuality and consistent service")
            if 'Punctuality' in df.columns and 'Star_Rating' in df.columns:
                reliable_buses = df[df['Punctuality'] >= 4]
                if not reliable_buses.empty and 'Bus_Name' in reliable_buses.columns:
                    top_reliable = reliable_buses.groupby('Bus_Name').agg({
                        'Punctuality': 'mean',
                        'Star_Rating': 'mean',
                        'review_id': 'count'
                    }).nlargest(5, 'Punctuality')
                    st.markdown("**Recommended:**")
                    for bus, row in top_reliable.iterrows():
                        st.write(f"✓ {bus} - Punctuality: {row['Punctuality']:.1f}⭐")
        
        st.markdown("---")
        st.subheader("🤖 Smart Recommendation Engine")
        
        st.markdown("**Get Personalized Recommendations**")
        
        pref_col1, pref_col2 = st.columns(2)
        
        with pref_col1:
            priorities = st.multiselect(
                "What matters most to you?",
                options=['Price', 'Rating', 'Punctuality', 'Cleanliness', 'Comfort', 'Staff Behavior'],
                default=['Rating']
            )
        
        with pref_col2:
            if 'Route' in df.columns:
                routes = sorted([r for r in df['Route'].dropna().unique() if str(r).strip()])
                pref_route = st.selectbox("Preferred Route", ['Any'] + routes)
            else:
                pref_route = 'Any'
        
        if st.button("🔍 Get Recommendations", type="primary"):
            recommendations_df = df.copy()
            
            if pref_route != 'Any' and 'Route' in recommendations_df.columns:
                recommendations_df = recommendations_df[recommendations_df['Route'] == pref_route]
            
            recommendations_df['recommendation_score'] = 0.0
            
            priority_map = {
                'Price': ('Price', -1),
                'Rating': ('Star_Rating', 1),
                'Punctuality': ('Punctuality', 1),
                'Cleanliness': ('Cleanliness', 1),
                'Comfort': ('Seat_comfort', 1),
                'Staff Behavior': ('Staff_Behaviour', 1)
            }
            
            for priority in priorities:
                if priority in priority_map:
                    col, direction = priority_map[priority]
                    if col in recommendations_df.columns:
                        col_min = recommendations_df[col].min()
                        col_max = recommendations_df[col].max()
                        if pd.notna(col_min) and pd.notna(col_max) and col_max > col_min:
                            normalized = (recommendations_df[col] - col_min) / (col_max - col_min)
                            recommendations_df['recommendation_score'] += normalized.fillna(0) * direction
            
            if 'Bus_Name' in recommendations_df.columns:
                top_recommendations = recommendations_df.groupby('Bus_Name').agg({
                    'recommendation_score': 'mean',
                    'Star_Rating': 'mean',
                    'Price': 'mean',
                    'review_id': 'count'
                }).nlargest(10, 'recommendation_score')
                
                st.success(f"✅ Found {len(top_recommendations)} recommended buses for you!")
                
                st.dataframe(
                    top_recommendations.style.background_gradient(
                        subset=['recommendation_score'],
                        cmap='Greens'
                    ),
                    use_container_width=True
                )

# -----------------------
# MAIN APP
# -----------------------
def main():
    # Add table selection at the very top of the sidebar
    st.sidebar.title("🚌 RedBus Analytics")
    st.sidebar.markdown("---")
    
    # TABLE SELECTOR - MAIN FEATURE
    st.sidebar.subheader("📋 Select Data Source")
    table_options = {
        'APSRTC': 'redbus_reviews_apsrtc',
        'TGSRTC': 'redbus_reviews_tgsrtc'
    }
    
    selected_operator = st.sidebar.selectbox(
        "Bus Operator",
        options=list(table_options.keys()),
        help="Select which bus operator's data to analyze"
    )
    
    selected_table = table_options[selected_operator]
    
    # Load data based on selected table
    if DEBUG:
        df = load_data_debug(selected_table, limit_rows=1000)
    else:
        with st.spinner(f"Loading {selected_operator} data from database..."):
            df = load_data(selected_table)

    if df.empty:
        st.error(f"❌ Unable to load data from {selected_table}. Check DB connection and table existence.")
        return

    st.sidebar.markdown("---")
    st.sidebar.title("📍 Navigation")
    page = st.sidebar.radio(
        "Go to", 
        ["🏠 Dashboard", "🎯 Find Best Buses", "📈 Business Insights"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Show sentiment correction stats
    if 'Original_Sentiment' in df.columns:
        changes = (df['Sentiment_Label'] != df['Original_Sentiment']).sum()
        if changes > 0:
            st.sidebar.success(f"✅ Corrected {changes:,} sentiment labels")
    
    st.sidebar.info(f"""
    **Database Stats - {selected_operator}:**
    - Total Records: {len(df):,}
    - Routes: {df['Route'].nunique() if 'Route' in df.columns else 'N/A'}
    - Buses: {df['Bus_Name'].nunique() if 'Bus_Name' in df.columns else 'N/A'}
    - Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    """)

    if page == "🏠 Dashboard":
        home_page(df, selected_operator)
    elif page == "🎯 Find Best Buses":
        find_best_buses_page(df)
    else:
        business_insights_page(df)

if __name__ == "__main__":
    main()