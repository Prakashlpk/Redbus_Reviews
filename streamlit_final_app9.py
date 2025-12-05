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
    st.markdown("### Comprehensive analysis of scraped bus reviews from RedBus")
    st.markdown("---")

    # Key Metrics
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
        if 'Price' in df.columns and df['Price'].notna().sum() > 0:
            avg_price = df['Price'].mean()
            st.metric("Average Price", f"₹{avg_price:.2f}")
        else:
            st.metric("Average Price", "N/A")

    st.markdown("---")

    # Main visualizations
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
            
            # Show distribution percentages
            st.markdown("**Rating Breakdown:**")
            for rating in sorted(rating_counts.index, reverse=True):
                count = rating_counts[rating]
                percentage = (count / len(df)) * 100
                st.write(f"⭐ {rating} Stars: {count:,} reviews ({percentage:.1f}%)")
        else:
            st.info("⚠️ Rating column not found")

    with col2:
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
            
            # Show peak month
            peak_month = reviews_by_month.idxmax()
            peak_count = reviews_by_month.max()
            st.info(f"📅 Peak month: **{peak_month}** with **{peak_count}** reviews")
        else:
            st.info("⚠️ Date information not available")

    st.markdown("---")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Bus Operators by Review Count")
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
    
    with col2:
        st.subheader("🛣️ Most Reviewed Routes")
        if 'Route' in df.columns and df['Route'].notna().sum() > 0:
            top_routes = df['Route'].value_counts().head(10)
            fig = px.bar(
                x=top_routes.values, y=top_routes.index, orientation='h',
                labels={'x': 'Number of Reviews', 'y': 'Route'},
                color=top_routes.values, color_continuous_scale='Greens'
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⚠️ Route data not available")

    # NOTE: Service Quality Metrics Overview removed from the dashboard as requested.

def find_best_buses_page(df):
    st.title("🎯 Find Best Buses by Route")
    st.markdown("### Filter buses based on review metrics and service quality")
    st.markdown("---")
    
    with st.sidebar:
        st.header("🔍 Search Filters")
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
                    help=f"Minimum score for {description} (0–{max_val})"
                )

    # Apply filters
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
    
    for metric, min_value in quality_filters.items():
        if min_value > 0:
            filtered_df = filtered_df[filtered_df[metric] >= min_value]
            filter_messages.append(f"Min {metric.replace('_', ' ')}: {min_value}")
    
    # Display results
    if len(filtered_df) > 0:
        st.success(f"✅ Found **{len(filtered_df):,}** reviews matching your criteria")
        
        if filter_messages:
            with st.expander("🔍 Active Filters", expanded=False):
                for msg in filter_messages:
                    st.write(f"• {msg}")
        
        if 'Bus_Name' in filtered_df.columns:
            st.markdown("---")
            st.subheader("🏆 Top Buses Ranked by Reviews")
            
            agg_dict = {}
            if 'review_id' in filtered_df.columns:
                agg_dict['review_id'] = 'count'
            agg_columns = ['Star_Rating', 'Price'] + list(quality_metrics.keys())
            
            for col in agg_columns:
                if col in filtered_df.columns:
                    agg_dict[col] = 'mean'
            
            # If review_id is missing, aggregate by count of rows
            if 'review_id' in agg_dict:
                bus_stats = filtered_df.groupby('Bus_Name').agg(agg_dict).round(2)
                bus_stats = bus_stats.rename(columns={'review_id': 'Total_Reviews'})
            else:
                bus_stats = filtered_df.groupby('Bus_Name').agg(agg_dict).round(2)
                bus_stats['Total_Reviews'] = filtered_df.groupby('Bus_Name').size()
            
            # Format integer columns
            int_display_cols = list(quality_metrics.keys()) + ['Price']
            for col in int_display_cols:
                if col in bus_stats.columns:
                    try:
                        bus_stats[col] = bus_stats[col].round().astype('Int64')
                    except Exception:
                        pass
            
            # Round Star_Rating to 2 decimal places
            if 'Star_Rating' in bus_stats.columns:
                bus_stats['Star_Rating'] = bus_stats['Star_Rating'].round(2)
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
            review_text = row.get('Review_Text', 'No review available')
            price = row.get('Price', 'N/A')
            date = row.get('Date_of_Review', 'N/A')
            
            # Create quality metrics display
            quality_display = []
            for metric in quality_metrics.keys():
                if metric in row and pd.notna(row[metric]):
                    quality_display.append(f"{metric.replace('_', ' ')}: {row[metric]:.1f}⭐")
            
            quality_str = " | ".join(quality_display) if quality_display else "No quality metrics available"
            
            # Color code by rating
            if pd.notna(rating):
                try:
                    if rating >= 4:
                        bg_color = '#d4edda'  # Green
                    elif rating >= 3:
                        bg_color = '#fff3cd'  # Yellow
                    else:
                        bg_color = '#f8d7da'  # Red
                except Exception:
                    bg_color = '#ffffff'
            else:
                bg_color = '#ffffff'
            
            st.markdown(f"""
            <div style="background-color:{bg_color};padding:16px;border-radius:10px;margin-bottom:12px;border-left:4px solid #007bff;">
                <h4 style="margin:0;color:#333">{bus_name}</h4>
                <p style="margin:4px 0;color:#666"><strong>Route:</strong> {route} | <strong>Price:</strong> ₹{price}</p>
                <p style="margin:4px 0"><strong>Rating:</strong> {rating}/5 ⭐ | <strong>Date:</strong> {date}</p>
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
    st.markdown("### Data-driven insights from scraped RedBus reviews")
    st.markdown("---")
    
    st.markdown("**Business Use Cases:**")
    st.markdown("""
    1. Customer Insights: Travelers can compare different bus services based on customer reviews.  
    2. Sentiment Analysis: Analyzing reviews can help identify top-rated buses, improving customer satisfaction.  
    3. Service Improvement: Bus operators can identify strengths and weaknesses in their services by analyzing customer feedback.  
    4. Recommendation System: Future enhancements can include personalized bus recommendations based on user preferences.
    """)
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "👥 Customer Insights", 
        "😊 Sentiment Analysis", 
        "🔧 Service Improvement", 
        "🎯 Recommendation System"
    ])
    
    # ----------------------
    # TAB 1: Customer Insights (Only Best Operators by Route)
    # ----------------------
    with tab1:
        st.header("👥 Customer Insights")
        st.markdown("Travelers can compare different bus services based on customer reviews.")
        if 'Route' in df.columns and 'Bus_Name' in df.columns:
            st.subheader("🏆 Best Operators by Route")
            
            # route selector
            popular_routes = df['Route'].value_counts().head(10).index.tolist()
            selected_route = st.selectbox("Select a route to analyze", popular_routes, key="ci_route_select")
            
            if selected_route:
                route_df = df[df['Route'] == selected_route]
                
                if 'Star_Rating' in route_df.columns and 'Bus_Name' in route_df.columns:
                    aggs = {}
                    if 'review_id' in route_df.columns:
                        aggs['review_id'] = 'count'
                    aggs['Star_Rating'] = 'mean'
                    if 'Price' in route_df.columns:
                        aggs['Price'] = 'mean'
                    operator_performance = route_df.groupby('Bus_Name').agg(aggs).round(2)
                    if 'review_id' in operator_performance.columns:
                        operator_performance = operator_performance.rename(columns={'review_id': 'Reviews'})
                    else:
                        operator_performance['Reviews'] = route_df.groupby('Bus_Name').size()
                    operator_performance = operator_performance.sort_values('Star_Rating', ascending=False).head(10)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Performance on {selected_route}**")
                        fig = px.bar(
                            operator_performance.reset_index(),
                            x='Bus_Name',
                            y='Star_Rating',
                            color='Star_Rating',
                            title=f"Top Operators on {selected_route}",
                            color_continuous_scale='Greens'
                        )
                        fig.update_layout(xaxis_tickangle=-45, height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Operator Statistics**")
                        st.dataframe(
                            operator_performance.style.background_gradient(
                                subset=['Star_Rating'] if 'Star_Rating' in operator_performance.columns else None,
                                cmap='RdYlGn'
                            ),
                            use_container_width=True
                        )
                    
                    # Key insights
                    st.markdown("---")
                    best_operator = operator_performance.index[0]
                    best_rating = operator_performance.iloc[0]['Star_Rating']
                    best_reviews = operator_performance.iloc[0]['Reviews']
                    st.success(f"✅ **Best Operator:** {best_operator} with {best_rating}/5 rating ({best_reviews} reviews)")
                    
                    if 'Price' in operator_performance.columns:
                        avg_price = operator_performance['Price'].mean()
                        st.info(f"💰 **Average Price on this route:** ₹{avg_price:.2f}")
                else:
                    st.info("Not enough data to display operator performance for this route.")
        else:
            st.warning("⚠️ Route or bus operator data not available")

    # ----------------------
    # TAB 2: Sentiment Analysis (unchanged content from prior Sentiment Overview)
    # ----------------------
    with tab2:
        st.header("2. Sentiment Analysis: Customer Satisfaction Trends")
        st.markdown("""
        **Use Case:** Understanding overall customer sentiment helps identify satisfaction levels and areas of concern.
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
                    title="Customer Sentiment Breakdown"
                )
                st.plotly_chart(fig, use_container_width=True)
                positive_pct = (sentiment_counts.get('Positive', 0) / len(df)) * 100
                neutral_pct = (sentiment_counts.get('Neutral', 0) / len(df)) * 100
                negative_pct = (sentiment_counts.get('Negative', 0) / len(df)) * 100
                st.markdown("**Sentiment Breakdown:**")
                st.write(f"✅ Positive: {positive_pct:.1f}% ({sentiment_counts.get('Positive', 0):,} reviews)")
                st.write(f"➖ Neutral: {neutral_pct:.1f}% ({sentiment_counts.get('Neutral', 0):,} reviews)")
                st.write(f"❌ Negative: {negative_pct:.1f}% ({sentiment_counts.get('Negative', 0):,} reviews)")
            with col2:
                st.subheader("Sentiment Distribution by Operator")
                if 'Bus_Name' in df.columns:
                    top_operators = df['Bus_Name'].value_counts().head(10).index
                    sentiment_by_bus = df[df['Bus_Name'].isin(top_operators)].groupby(
                        ['Bus_Name', 'Sentiment_Label']
                    ).size().unstack(fill_value=0)
                    fig = px.bar(
                        sentiment_by_bus.reset_index(),
                        x='Bus_Name',
                        y=['Positive', 'Neutral', 'Negative'],
                        title="Sentiment Distribution - Top 10 Operators",
                        labels={'value': 'Number of Reviews', 'Bus_Name': 'Bus Operator'},
                        color_discrete_map={'Positive': '#4caf50', 'Neutral': '#ff9800', 'Negative': '#f44336'}
                    )
                    fig.update_layout(xaxis_tickangle=-45, height=400, barmode='stack')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("⚠️ Bus operator data not available")
        else:
            st.warning("⚠️ Sentiment data not available in scraped reviews")

    # ----------------------
    # TAB 3: Service Improvement (same analysis as prior Service Quality Analysis)
    # ----------------------
    with tab3:
        st.header("3. Service Improvement: Identify Strengths & Weaknesses")
        st.markdown("""
        **Use Case:** Bus operators can analyze quality metrics from customer reviews to understand which aspects of their service need improvement.
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
                threshold = quality_df['Average_Rating'].median()
                weak_areas = quality_df[quality_df['Average_Rating'] < threshold]
                if not weak_areas.empty:
                    st.warning("⚠️ **Areas Needing Improvement (Below Median):**")
                    for _, row in weak_areas.iterrows():
                        st.write(f"• **{row['Metric'].replace('_', ' ')}**: {row['Average_Rating']:.2f}/5")
                strong_areas = quality_df[quality_df['Average_Rating'] >= threshold]
                if not strong_areas.empty:
                    st.success("✅ **Strong Performance Areas:**")
                    for _, row in strong_areas.head(3).iterrows():
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
        else:
            st.warning("⚠️ Quality metrics not available in scraped data")
        if 'Sentiment_Label' in df.columns:
            st.subheader("📝 Review Analysis by Sentiment")
            negative_reviews = df[df['Sentiment_Label'].str.lower() == 'negative']
            if not negative_reviews.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Negative Reviews", f"{len(negative_reviews):,}")
                    negative_pct = (len(negative_reviews) / len(df)) * 100
                    st.metric("Percentage", f"{negative_pct:.1f}%")
                with col2:
                    if 'Bus_Name' in negative_reviews.columns:
                        most_complaints = negative_reviews['Bus_Name'].value_counts().head(1)
                        if len(most_complaints) > 0:
                            st.metric("Most Complaints", most_complaints.index[0][:20])
                            st.metric("Count", f"{most_complaints.values[0]:,}")
                with col3:
                    if 'Route' in negative_reviews.columns:
                        problematic_routes = negative_reviews['Route'].value_counts().head(1)
                        if len(problematic_routes) > 0:
                            st.metric("Most Complained Route", problematic_routes.index[0][:20])
                            st.metric("Complaints", f"{problematic_routes.values[0]:,}")

    # ----------------------
    # TAB 4: Recommendation System (new, uses user's provided layout)
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
                        price_str = f"₹{row['Price']:.0f}" if pd.notna(row.get('Price')) else "₹N/A"
                        rating_str = f"({row['Star_Rating']:.1f}⭐)" if pd.notna(row.get('Star_Rating')) else ""
                        st.write(f"✓ {row['Bus_Name']} - {price_str} {rating_str}")
        
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
                        'review_id': 'count' if 'review_id' in reliable_buses.columns else (lambda s: s.size)
                    })
                    top_reliable = top_reliable.sort_values('Punctuality', ascending=False).head(5)
                    st.markdown("**Recommended:**")
                    for bus, row in top_reliable.iterrows():
                        punct = row['Punctuality'] if pd.notna(row.get('Punctuality')) else 0
                        st.write(f"✓ {bus} - Punctuality: {punct:.1f}⭐")

# ----------------------
# MAIN: Table selector, sidebar, routing
# ----------------------
def main():
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

    # Database statistics
    try:
        date_min = df['Date_of_Review'].min().strftime('%Y-%m-%d') if 'Date_of_Review' in df.columns and df['Date_of_Review'].notna().sum() > 0 else 'N/A'
        date_max = df['Date_of_Review'].max().strftime('%Y-%m-%d') if 'Date_of_Review' in df.columns and df['Date_of_Review'].notna().sum() > 0 else 'N/A'
    except Exception:
        date_min, date_max = 'N/A', 'N/A'

    st.sidebar.info(f"""
**Database Stats - {selected_operator}:**
- Total Records: {len(df):,}
- Routes: {df['Route'].nunique() if 'Route' in df.columns else 'N/A'}
- Buses: {df['Bus_Name'].nunique() if 'Bus_Name' in df.columns else 'N/A'}
- Date Range: {date_min} to {date_max}
""")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
**About This Project:**

This application demonstrates:
- ✅ Web scraping with Selenium
- ✅ Data storage in SQL database
- ✅ Interactive visualization with Streamlit
- ✅ Data filtering and analysis
""")

    # Route to appropriate page
    if page == "🏠 Dashboard":
        home_page(df, selected_operator)
    elif page == "🎯 Find Best Buses":
        find_best_buses_page(df)
    else:
        business_insights_page(df)

if __name__ == "__main__":
    main()
