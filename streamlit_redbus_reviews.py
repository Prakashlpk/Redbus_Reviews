# full_streamlit_app_with_sql_queries.py
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
from sqlalchemy import text

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
def home_page(df, operator_name, table_name=None, engine=None):
    """
    Dashboard home page that by default uses SQL to compute aggregates.
    If `Use SQL` toggle is off, uses the in-memory pandas `df`.
    Requires table_name and engine when using SQL.
    """
    st.title(f"🚌 RedBus Reviews Analytics Dashboard - {operator_name}")
    st.markdown("### Comprehensive analysis of scraped bus reviews from RedBus")
    st.markdown("---")

    # Sidebar toggle to choose SQL vs Pandas for dashboard aggregations
    with st.sidebar:
        st.markdown("### Dashboard compute mode")
        use_sql = st.checkbox("Use SQL for dashboard aggregations (recommended)", value=True)
        st.markdown("Tip: SQL mode pushes aggregations to the database, which is efficient for large datasets.")

    # Helper to execute SQL safely and return dataframe (applies safe_postprocess)
    def run_sql(q, params=None):
        try:
            if engine is None:
                raise RuntimeError("DB engine is not provided.")
            df_q = pd.read_sql_query(text(q), con=engine, params=params or {})
            df_q = safe_postprocess(df_q)
            return df_q
        except Exception as e:
            # Show an unobtrusive warning and return empty DF so code can fallback to pandas
            st.warning(f"SQL query failed or unavailable: {e}. Falling back to pandas for this widget.")
            return pd.DataFrame()

    # Key metrics (Total reviews, avg rating, total routes/buses, avg price)
    try:
        if use_sql and table_name and engine:
            q_total = f"SELECT COUNT(*) AS total_reviews FROM {table_name};"
            df_total = run_sql(q_total)
            total_reviews = int(df_total['total_reviews'].iloc[0]) if not df_total.empty else len(df)
        else:
            total_reviews = len(df)
    except Exception:
        total_reviews = len(df)

    # avg rating
    try:
        if use_sql and table_name and engine:
            q_avg = f"SELECT AVG(Star_Rating) AS avg_rating FROM {table_name} WHERE Star_Rating IS NOT NULL;"
            df_avg = run_sql(q_avg)
            avg_rating = float(df_avg['avg_rating'].iloc[0]) if not df_avg.empty and pd.notna(df_avg['avg_rating'].iloc[0]) else None
        else:
            avg_rating = df['Star_Rating'].mean() if 'Star_Rating' in df.columns and df['Star_Rating'].notna().sum() > 0 else None
    except Exception:
        avg_rating = None

    # total routes and buses and avg price
    try:
        if use_sql and table_name and engine:
            q_routes = f"SELECT COUNT(DISTINCT Route) AS n_routes, COUNT(DISTINCT Bus_Name) AS n_buses, AVG(Price) AS avg_price FROM {table_name};"
            df_stats = run_sql(q_routes)
            n_routes = int(df_stats['n_routes'].iloc[0]) if not df_stats.empty else (df['Route'].nunique() if 'Route' in df.columns else 0)
            n_buses = int(df_stats['n_buses'].iloc[0]) if not df_stats.empty else (df['Bus_Name'].nunique() if 'Bus_Name' in df.columns else 0)
            avg_price = float(df_stats['avg_price'].iloc[0]) if not df_stats.empty and pd.notna(df_stats['avg_price'].iloc[0]) else None
        else:
            n_routes = df['Route'].nunique() if 'Route' in df.columns else 0
            n_buses = df['Bus_Name'].nunique() if 'Bus_Name' in df.columns else 0
            avg_price = df['Price'].mean() if 'Price' in df.columns and df['Price'].notna().sum() > 0 else None
    except Exception:
        n_routes = df['Route'].nunique() if 'Route' in df.columns else 0
        n_buses = df['Bus_Name'].nunique() if 'Bus_Name' in df.columns else 0
        avg_price = df['Price'].mean() if 'Price' in df.columns and df['Price'].notna().sum() > 0 else None

    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Reviews", f"{total_reviews:,}")
    with col2:
        if avg_rating:
            st.metric("Average Rating", f"{avg_rating:.2f}/5")
        else:
            st.metric("Average Rating", "N/A")
    with col3:
        st.metric("Total Routes", f"{n_routes:,}" if n_routes is not None else "N/A")
    with col4:
        st.metric("Total Buses", f"{n_buses:,}" if n_buses is not None else "N/A")
    with col5:
        st.metric("Average Price", f"₹{avg_price:.2f}" if avg_price is not None else "N/A")

    st.markdown("---")

    # -----------------------------
    # Rating Distribution (bar)
    # -----------------------------
    st.subheader("📊 Rating Distribution")

    if use_sql and table_name and engine:
        q = f"""
            SELECT Star_Rating, COUNT(*) AS cnt
            FROM {table_name}
            WHERE Star_Rating IS NOT NULL
            GROUP BY Star_Rating
            ORDER BY Star_Rating;
        """
        rating_df = run_sql(q)
        if rating_df.empty:
            # fallback to pandas
            rating_counts = df['Star_Rating'].value_counts().sort_index() if 'Star_Rating' in df.columns else pd.Series([])
        else:
            # ensure numeric
            rating_df['Star_Rating'] = pd.to_numeric(rating_df['Star_Rating'], errors='coerce')
            rating_df = rating_df.dropna(subset=['Star_Rating']).sort_values('Star_Rating')
            rating_counts = pd.Series(data=rating_df['cnt'].values, index=rating_df['Star_Rating'].astype(int).values)
    else:
        rating_counts = df['Star_Rating'].value_counts().sort_index() if 'Star_Rating' in df.columns else pd.Series([])

    if rating_counts is not None and len(rating_counts) > 0:
        fig = px.bar(
            x=rating_counts.index, y=rating_counts.values,
            labels={'x': 'Star Rating', 'y': 'Number of Reviews'},
            color=rating_counts.values, color_continuous_scale='RdYlGn'
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

        # Show distribution percentages
        st.markdown("**Rating Breakdown:**")
        total_for_pct = rating_counts.sum() if rating_counts.sum() > 0 else max(total_reviews, 1)
        for rating in sorted(rating_counts.index, reverse=True):
            count = int(rating_counts[rating])
            percentage = (count / total_for_pct) * 100
            st.write(f"⭐ {rating} Stars: {count:,} reviews ({percentage:.1f}%)")
    else:
        st.info("⚠️ Rating data not available")

    st.markdown("---")

    # -----------------------------
    # Reviews Over Time (monthly)
    # -----------------------------
    st.subheader("📈 Reviews Over Time")
    if use_sql and table_name and engine:
        # MySQL/TiDB date month formatting; works if Date_of_Review is a date/datetime
        q = f"""
            SELECT DATE_FORMAT(Date_of_Review, '%Y-%m') AS ym, COUNT(*) AS cnt
            FROM {table_name}
            WHERE Date_of_Review IS NOT NULL
            GROUP BY ym
            ORDER BY ym;
        """
        time_df = run_sql(q)
        if time_df.empty:
            # fallback
            if 'Date_of_Review' in df.columns:
                df_temp = df[df['Date_of_Review'].notna()].copy()
                reviews_by_month = df_temp.groupby(df_temp['Date_of_Review'].dt.to_period('M')).size()
                x = reviews_by_month.index.astype(str)
                y = reviews_by_month.values
            else:
                x, y = [], []
        else:
            x = time_df['ym'].astype(str)
            y = time_df['cnt'].astype(int)
    else:
        if 'Date_of_Review' in df.columns and df['Date_of_Review'].notna().sum() > 0:
            df_temp = df[df['Date_of_Review'].notna()].copy()
            reviews_by_month = df_temp.groupby(df_temp['Date_of_Review'].dt.to_period('M')).size()
            x = reviews_by_month.index.astype(str)
            y = reviews_by_month.values
        else:
            x, y = [], []

    if len(x) > 0:
        fig = px.line(x=x, y=y, labels={'x': 'Month', 'y': 'Number of Reviews'})
        fig.update_traces(line_color='#d84e55', line_width=3)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        # peak month info
        try:
            peak_idx = int(np.argmax(y))
            st.info(f"📅 Peak month: **{x[peak_idx]}** with **{int(y[peak_idx])}** reviews")
        except Exception:
            pass
    else:
        st.info("⚠️ Date information not available")

    st.markdown("---")

    # -----------------------------
    # Top Bus Operators by review count
    # -----------------------------
    st.subheader("🏆 Top Bus Operators by Review Count")
    if use_sql and table_name and engine:
        q = f"""
            SELECT Bus_Name, COUNT(*) AS total_reviews
            FROM {table_name}
            WHERE Bus_Name IS NOT NULL
            GROUP BY Bus_Name
            ORDER BY total_reviews DESC
            LIMIT 10;
        """
        top_buses_df = run_sql(q)
        if top_buses_df.empty:
            if 'Bus_Name' in df.columns:
                top_buses = df['Bus_Name'].value_counts().head(10)
            else:
                top_buses = pd.Series([])
        else:
            top_buses_df = top_buses_df.set_index('Bus_Name')
            top_buses = pd.Series(data=top_buses_df['total_reviews'].values, index=top_buses_df.index)
    else:
        top_buses = df['Bus_Name'].value_counts().head(10) if 'Bus_Name' in df.columns else pd.Series([])

    if len(top_buses) > 0:
        fig = px.bar(
            x=top_buses.values, y=top_buses.index, orientation='h',
            labels={'x': 'Number of Reviews', 'y': 'Bus Operator'},
            color=top_buses.values, color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⚠️ Bus name data not available")

    st.markdown("---")

    # -----------------------------
    # Most reviewed routes
    # -----------------------------
    st.subheader("🛣️ Most Reviewed Routes")
    if use_sql and table_name and engine:
        q = f"""
            SELECT Route, COUNT(*) AS total_reviews
            FROM {table_name}
            WHERE Route IS NOT NULL
            GROUP BY Route
            ORDER BY total_reviews DESC
            LIMIT 10;
        """
        top_routes_df = run_sql(q)
        if top_routes_df.empty:
            if 'Route' in df.columns:
                top_routes = df['Route'].value_counts().head(10)
            else:
                top_routes = pd.Series([])
        else:
            top_routes_df = top_routes_df.set_index('Route')
            top_routes = pd.Series(data=top_routes_df['total_reviews'].values, index=top_routes_df.index)
    else:
        top_routes = df['Route'].value_counts().head(10) if 'Route' in df.columns else pd.Series([])

    if len(top_routes) > 0:
        fig = px.bar(
            x=top_routes.values, y=top_routes.index, orientation='h',
            labels={'x': 'Number of Reviews', 'y': 'Route'},
            color=top_routes.values, color_continuous_scale='Greens'
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⚠️ Route data not available")

    st.markdown("---")
    st.caption("Note: For large datasets SQL mode is recommended — it pushes aggregation to the database and transfers only small summarized results.")


# --- existing run_filtered_query and find_best_buses_page functions unchanged ---
def run_filtered_query(
    table_name: str,
    engine,
    route_filter: str,
    bus_categories: list,
    price_range,
    min_rating,
    quality_filters: dict,
    like_bus_name: str,
    route_regex: str,
) -> pd.DataFrame:
    
    # CASE expression to normalize Bus_Type into 6 buckets
    bus_category_case = """
        CASE
            WHEN UPPER(Bus_Type) REGEXP 'AC' AND UPPER(Bus_Type) REGEXP 'SLEEPER' AND UPPER(Bus_Type) REGEXP 'SEMI'
                THEN 'AC Semi-Sleeper'
            WHEN UPPER(Bus_Type) REGEXP 'AC' AND UPPER(Bus_Type) REGEXP 'SLEEPER'
                THEN 'AC Sleeper'
            WHEN UPPER(Bus_Type) REGEXP 'AC' AND UPPER(Bus_Type) REGEXP 'SEMI'
                THEN 'AC Semi-Sleeper'
            WHEN UPPER(Bus_Type) REGEXP 'AC' AND UPPER(Bus_Type) REGEXP 'SEATER'
                THEN 'AC Seater'

            WHEN UPPER(Bus_Type) NOT REGEXP 'AC' AND UPPER(Bus_Type) REGEXP 'SLEEPER' AND UPPER(Bus_Type) REGEXP 'SEMI'
                THEN 'Non-AC Semi-Sleeper'
            WHEN UPPER(Bus_Type) NOT REGEXP 'AC' AND UPPER(Bus_Type) REGEXP 'SLEEPER'
                THEN 'Non-AC Sleeper'
            WHEN UPPER(Bus_Type) NOT REGEXP 'AC' AND UPPER(Bus_Type) REGEXP 'SEMI'
                THEN 'Non-AC Semi-Sleeper'
            WHEN UPPER(Bus_Type) NOT REGEXP 'AC' AND UPPER(Bus_Type) REGEXP 'SEATER'
                THEN 'Non-AC Seater'

            ELSE 'Non-AC Seater'
        END
    """

    # Subquery so we can filter on Bus_Category alias
    base_query = f"""
        SELECT *
        FROM (
            SELECT
                t.*,
                {bus_category_case} AS Bus_Category
            FROM {table_name} t
        ) AS sub
        WHERE 1=1
    """

    params = {}

    # Route filter
    if route_filter and route_filter != "All Routes":
        base_query += " AND Route = :route"
        params["route"] = route_filter

    # Bus category filter (our 6 buckets)
    if bus_categories:
        placeholders = []
        for idx, bc in enumerate(bus_categories):
            key = f"bc{idx}"
            placeholders.append(f":{key}")
            params[key] = bc
        base_query += f" AND Bus_Category IN ({', '.join(placeholders)})"

    # Price range
    if price_range is not None:
        base_query += " AND Price BETWEEN :p_min AND :p_max"
        params["p_min"] = float(price_range[0])
        params["p_max"] = float(price_range[1])

    # Minimum star rating
    if min_rating is not None:
        base_query += " AND Star_Rating >= :min_rating"
        params["min_rating"] = float(min_rating)

    # Quality metric filters
    for metric, min_val in quality_filters.items():
        if min_val and min_val > 0:
            key = f"q_{metric}"
            base_query += f" AND {metric} >= :{key}"
            params[key] = int(min_val)

    # LIKE filter on Bus_Name
    if like_bus_name:
        base_query += " AND Bus_Name LIKE :bus_like"
        params["bus_like"] = like_bus_name

    # REGEXP filter on Route
    if route_regex:
        base_query += " AND Route REGEXP :route_regex"
        params["route_regex"] = route_regex

    sql = text(base_query)
    df_filtered = pd.read_sql_query(sql, con=engine, params=params)
    df_filtered = safe_postprocess(df_filtered)
    return df_filtered

def find_best_buses_page(df, table_name: str, engine):
    st.title("🎯 Find Best Buses by Route")
    st.markdown("### Filter buses based on review metrics and service quality (SQL-powered)")
    st.markdown("---")
    
    # These are the ONLY categories we use
    BUS_CATEGORY_OPTIONS = [
        "AC Sleeper",
        "AC Semi-Sleeper",
        "AC Seater",
        "Non-AC Sleeper",
        "Non-AC Semi-Sleeper",
        "Non-AC Seater",
    ]
    
    # ---------- SIDEBAR FILTERS (UI) ----------
    with st.sidebar:
        st.header("🔍 Search Filters")
        st.markdown("**Route Selection**")
        
        # Route options from in-memory df (for dropdown)
        if 'Route' in df.columns and df['Route'].notna().sum() > 0:
            routes = sorted([r for r in df['Route'].dropna().unique() if str(r).strip()])
            selected_route = st.selectbox(
                "Select Your Route",
                ['All Routes'] + routes, 
                help="Choose the route you want to travel"
            )
        else:
            selected_route = 'All Routes'
            st.warning("⚠️ Route data not available")
        
        st.markdown("---")
        st.markdown("**Bus Category (Normalized)**")
        selected_bus_categories = st.multiselect(
            "Bus Category",
            BUS_CATEGORY_OPTIONS,
            default=BUS_CATEGORY_OPTIONS,
            help="Bus types are grouped into these 6 standard categories using SQL CASE + REGEXP"
        )
        
        # Price range slider
        if 'Price' in df.columns and df['Price'].notna().sum() > 0:
            try:
                min_price = int(df['Price'].min())
                max_price = int(df['Price'].max())
                price_range = st.slider(
                    "Price Range (₹)", 
                    min_price, 
                    max_price, 
                    (min_price, max_price),
                    help="Set your budget range"
                )
            except Exception:
                price_range = None
                st.warning("⚠️ Price filter unavailable")
        else:
            price_range = None
        
        st.markdown("---")
        st.markdown("**Review-Based Filters** ⭐")
        
        if 'Star_Rating' in df.columns and df['Star_Rating'].notna().sum() > 0:
            min_rating = st.slider(
                "Minimum Star Rating",
                1.0, 5.0, 3.0, 0.5,
                help="Filter buses with minimum rating"
            )
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

        st.markdown("---")
        st.markdown("**Text Search (SQL LIKE / REGEXP)**")
        like_bus_name = st.text_input(
            "Bus Name contains (SQL LIKE)",
            value="",
            placeholder="e.g. VOLVO, SLEEPER"
        )
        route_regex = st.text_input(
            "Route pattern (SQL REGEXP)",
            value="",
            placeholder="e.g. Hyderabad.*Bangalore"
        )

    # ---------- RUN SQL QUERY WITH FILTERS ----------
    with st.spinner("Fetching filtered buses from database using SQL..."):
        filtered_df = run_filtered_query(
            table_name=table_name,
            engine=engine,
            route_filter=selected_route,
            bus_categories=selected_bus_categories,
            price_range=price_range,
            min_rating=min_rating,
            quality_filters=quality_filters,
            like_bus_name=like_bus_name.strip(),
            route_regex=route_regex.strip(),
        )

    # ---------- ACTIVE FILTERS LIST ----------
    filter_messages = []
    if selected_route != 'All Routes':
        filter_messages.append(f"Route: {selected_route}")
    if selected_bus_categories:
        filter_messages.append("Bus Categories: " + ", ".join(selected_bus_categories))
    if price_range:
        filter_messages.append(f"Price: ₹{price_range[0]}–₹{price_range[1]}")
    if min_rating:
        filter_messages.append(f"Min Rating: {min_rating}⭐")
    for metric, min_value in quality_filters.items():
        if min_value and min_value > 0:
            filter_messages.append(f"Min {metric.replace('_', ' ')}: {min_value}")
    if like_bus_name:
        filter_messages.append(f"Bus_Name LIKE '%{like_bus_name}%'")
    if route_regex:
        filter_messages.append(f"Route REGEXP '{route_regex}'")

    # ---------- DISPLAY RESULTS ----------
    if len(filtered_df) > 0:
        st.success(f"✅ Found **{len(filtered_df):,}** reviews matching your criteria")
        
        if filter_messages:
            with st.expander("🔍 Active Filters (SQL WHERE / LIKE / REGEXP / CASE)", expanded=False):
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
            
            if 'review_id' in agg_dict:
                bus_stats = filtered_df.groupby('Bus_Name').agg(agg_dict).round(2)
                bus_stats = bus_stats.rename(columns={'review_id': 'Total_Reviews'})
            else:
                bus_stats = filtered_df.groupby('Bus_Name').agg(agg_dict).round(2)
                bus_stats['Total_Reviews'] = filtered_df.groupby('Bus_Name').size()
            
            int_display_cols = list(quality_metrics.keys()) + ['Price']
            for col in int_display_cols:
                if col in bus_stats.columns:
                    try:
                        bus_stats[col] = bus_stats[col].round().astype('Int64')
                    except Exception:
                        pass
            
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
            
            quality_display = []
            for metric in quality_metrics.keys():
                if metric in row and pd.notna(row[metric]):
                    quality_display.append(f"{metric.replace('_', ' ')}: {row[metric]:.1f}⭐")
            
            quality_str = " | ".join(quality_display) if quality_display else "No quality metrics available"
            
            if pd.notna(rating):
                try:
                    if rating >= 4:
                        bg_color = '#d4edda'
                    elif rating >= 3:
                        bg_color = '#fff3cd'
                    else:
                        bg_color = '#f8d7da'
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

# ----------------------
# NEW: SQL Queries Page
# ----------------------
def sql_queries_page(table_name: str, engine):
    st.title("📊SQL Queries")
    st.markdown("Select a query from the dropdown and press **Run** to view results.")
    st.markdown("---")

    # List of prebuilt queries (label -> SQL text)
    # All queries use the provided `table_name` and are read-only SELECTs.
    queries = {
        "Top 10 operators by number of reviews": text(f"""
            SELECT Bus_Name, COUNT(*) AS total_reviews
            FROM {table_name}
            WHERE Bus_Name IS NOT NULL
            GROUP BY Bus_Name
            ORDER BY total_reviews DESC
            LIMIT 10;
        """),
        "Top 10 operators by average punctuality (highest avg punctuality)": text(f"""
            SELECT Bus_Name, AVG(Punctuality) AS avg_punctuality, COUNT(*) AS reviews
            FROM {table_name}
            WHERE Punctuality IS NOT NULL AND Bus_Name IS NOT NULL
            GROUP BY Bus_Name
            HAVING reviews >= 5
            ORDER BY avg_punctuality DESC
            LIMIT 10;
        """),
        "Top 10 operators by average cleanliness": text(f"""
            SELECT Bus_Name, AVG(Cleanliness) AS avg_cleanliness, COUNT(*) AS reviews
            FROM {table_name}
            WHERE Cleanliness IS NOT NULL AND Bus_Name IS NOT NULL
            GROUP BY Bus_Name
            HAVING reviews >= 5
            ORDER BY avg_cleanliness DESC
            LIMIT 10;
        """),
        "Top 20 reviews with Positive sentiment (highest rating first)": text(f"""
            SELECT review_id, Bus_Name, Route, Star_Rating, Sentiment_Label, Review_Text, Date_of_Review
            FROM {table_name}
            WHERE Sentiment_Label = 'Positive'
            ORDER BY Star_Rating DESC, Date_of_Review DESC
            LIMIT 20;
        """),
        "Top 20 reviews with Negative sentiment (most recent first)": text(f"""
            SELECT review_id, Bus_Name, Route, Star_Rating, Sentiment_Label, Review_Text, Date_of_Review
            FROM {table_name}
            WHERE LOWER(Sentiment_Label) = 'negative'
            ORDER BY Date_of_Review DESC
            LIMIT 20;
        """),
        "Average rating per route (top 10 routes by avg rating, min 10 reviews)": text(f"""
            SELECT Route, AVG(Star_Rating) AS avg_rating, COUNT(*) AS reviews
            FROM {table_name}
            WHERE Route IS NOT NULL AND Star_Rating IS NOT NULL
            GROUP BY Route
            HAVING reviews >= 10
            ORDER BY avg_rating DESC
            LIMIT 10;
        """),
        "Most reviewed routes (top 10)": text(f"""
            SELECT Route, COUNT(*) AS total_reviews
            FROM {table_name}
            WHERE Route IS NOT NULL
            GROUP BY Route
            ORDER BY total_reviews DESC
            LIMIT 10;
        """),
        "Average price by operator (top 10 cheapest by avg price with min 5 reviews)": text(f"""
            SELECT Bus_Name, AVG(Price) AS avg_price, COUNT(*) AS reviews
            FROM {table_name}
            WHERE Price IS NOT NULL AND Bus_Name IS NOT NULL
            GROUP BY Bus_Name
            HAVING reviews >= 5
            ORDER BY avg_price ASC
            LIMIT 10;
        """),
        "Operators with highest avg seat comfort (top 10, min 5 reviews)": text(f"""
            SELECT Bus_Name, AVG(Seat_comfort) AS avg_seat_comfort, COUNT(*) AS reviews
            FROM {table_name}
            WHERE Seat_comfort IS NOT NULL AND Bus_Name IS NOT NULL
            GROUP BY Bus_Name
            HAVING reviews >= 5
            ORDER BY avg_seat_comfort DESC
            LIMIT 10;
        """),
        "Operators with highest AC quality (top 10, min 5 reviews)": text(f"""
            SELECT Bus_Name, AVG(AC) AS avg_ac, COUNT(*) AS reviews
            FROM {table_name}
            WHERE AC IS NOT NULL AND Bus_Name IS NOT NULL
            GROUP BY Bus_Name
            HAVING reviews >= 5
            ORDER BY avg_ac DESC
            LIMIT 10;
        """),
        "Routes with most negative reviews (top 10)": text(f"""
            SELECT Route, SUM(CASE WHEN LOWER(Sentiment_Label) = 'negative' THEN 1 ELSE 0 END) AS negative_count,
                   COUNT(*) AS total_reviews
            FROM {table_name}
            WHERE Route IS NOT NULL
            GROUP BY Route
            ORDER BY negative_count DESC
            LIMIT 10;
        """),
        "Sentiment distribution counts (Positive / Neutral / Negative)": text(f"""
            SELECT Sentiment_Label, COUNT(*) AS cnt
            FROM {table_name}
            WHERE Sentiment_Label IS NOT NULL
            GROUP BY Sentiment_Label;
        """),
        "Top 20 highest rated reviews overall": text(f"""
            SELECT review_id, Bus_Name, Route, Star_Rating, Review_Text, Date_of_Review
            FROM {table_name}
            WHERE Star_Rating IS NOT NULL
            ORDER BY Star_Rating DESC, Date_of_Review DESC
            LIMIT 20;
        """),
        "Operators with most 5-star reviews (top 10)": text(f"""
            SELECT Bus_Name,
                   SUM(CASE WHEN Star_Rating = 5 THEN 1 ELSE 0 END) AS five_star_count,
                   COUNT(*) AS total_reviews
            FROM {table_name}
            WHERE Bus_Name IS NOT NULL
            GROUP BY Bus_Name
            ORDER BY five_star_count DESC
            LIMIT 10;
        """),
        "Top 10 operators by average driving quality (min 5 reviews)": text(f"""
            SELECT Bus_Name, AVG(Driving) AS avg_driving, COUNT(*) AS reviews
            FROM {table_name}
            WHERE Driving IS NOT NULL AND Bus_Name IS NOT NULL
            GROUP BY Bus_Name
            HAVING reviews >= 5
            ORDER BY avg_driving DESC
            LIMIT 10;
        """),
    }

    # UI: query selector
    query_names = list(queries.keys())
    selected_query_name = st.selectbox("Choose a query", query_names)
    st.markdown("---")

    # Optionally allow a small text filter for Bus_Name or Route if the selected query supports it
    # (we keep queries simple and read-only by default)
    run_button = st.button("Run Query")

    if run_button:
        sql_text = queries[selected_query_name]
        try:
            df_res = pd.read_sql_query(sql_text, con=engine)
            df_res = safe_postprocess(df_res)
            st.success(f"Query executed — returned {len(df_res):,} rows.")
            st.dataframe(df_res, use_container_width=True)

            # Download CSV
            csv = df_res.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name=f"query_result_{selected_query_name[:30].replace(' ','_')}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Query execution failed: {e}")
            if DEBUG:
                st.error(traceback.format_exc())

# ----------------------
# business_insights_page (unchanged)
# (I kept the business_insights_page you already had — for brevity it is not repeated here.)
# For completeness in your file, include your existing business_insights_page function implementation.
# ----------------------

# (Insert your existing business_insights_page function here. If you used the SQL-powered
#  version from your last draft, keep it unchanged. For brevity in this snippet I assume it
#  remains as you previously wrote.)

def business_insights_page(df, table_name: str, engine):
    st.title("📈 Business Use Cases & Insights")
    st.markdown("### Data-driven SQL-powered insights from scraped RedBus reviews")
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
    # TAB 1: Customer Insights
    # ----------------------
    with tab1:
        st.header("👥 Customer Insights")
        st.markdown("Travelers can compare different bus services based on customer reviews.")

        # Top 10 routes from SQL
        try:
            routes_sql = text(f"""
                SELECT Route, COUNT(*) AS reviews_count
                FROM {table_name}
                WHERE Route IS NOT NULL AND Route <> ''
                GROUP BY Route
                ORDER BY reviews_count DESC
                LIMIT 10;
            """)
            routes_df = pd.read_sql_query(routes_sql, con=engine)
        except Exception as e:
            st.error(f"Failed to load routes from database: {e}")
            routes_df = pd.DataFrame()

        if not routes_df.empty and "Route" in routes_df.columns:
            popular_routes = routes_df["Route"].tolist()
            selected_route = st.selectbox(
                "Select a route to analyze",
                popular_routes,
                key="ci_route_select"
            )

            if selected_route:
                # Operator performance for selected route (SQL)
                op_sql = text(f"""
                    SELECT
                        Bus_Name,
                        AVG(Star_Rating) AS avg_rating,
                        COUNT(*) AS Reviews,
                        AVG(Price) AS avg_price
                    FROM {table_name}
                    WHERE Route = :route
                      AND Bus_Name IS NOT NULL
                    GROUP BY Bus_Name
                    ORDER BY avg_rating DESC, Reviews DESC
                    LIMIT 10;
                """)
                operator_performance = pd.read_sql_query(
                    op_sql, con=engine, params={"route": selected_route}
                )

                if not operator_performance.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Performance on {selected_route}**")
                        fig = px.bar(
                            operator_performance,
                            x='Bus_Name',
                            y='avg_rating',
                            color='avg_rating',
                            title=f"Top Operators on {selected_route}",
                            labels={'avg_rating': 'Average Rating', 'Bus_Name': 'Bus Operator'},
                            color_continuous_scale='Greens'
                        )
                        fig.update_layout(xaxis_tickangle=-45, height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Operator Statistics using SQL Aggregation**")
                        show_df = operator_performance.copy()
                        show_df.rename(
                            columns={
                                'avg_rating': 'Average Rating',
                                'avg_price': 'Average Price'
                            },
                            inplace=True
                        )
                        st.dataframe(
                            show_df.style.background_gradient(
                                subset=['Average Rating'] if 'Average Rating' in show_df.columns else None,
                                cmap='RdYlGn'
                            ),
                            use_container_width=True
                        )
                    
                    # Key insights
                    st.markdown("---")
                    best_operator = operator_performance.iloc[0]['Bus_Name']
                    best_rating = operator_performance.iloc[0]['avg_rating']
                    best_reviews = operator_performance.iloc[0]['Reviews']
                    st.success(
                        f"✅ **Best Operator on this route:** {best_operator} "
                        f"with {best_rating:.2f}/5 rating ({best_reviews} reviews)"
                    )
                    
                    if 'avg_price' in operator_performance.columns:
                        avg_price = operator_performance['avg_price'].mean()
                        st.info(f"💰 **Average Price on this route:** ₹{avg_price:.2f}")
                else:
                    st.info("Not enough data from database to display operator performance for this route.")
        else:
            st.warning("⚠️ Route data not available from database")

    # ----------------------
    # TAB 2: Sentiment Analysis
    # ----------------------
    with tab2:
        st.header("2. Sentiment Analysis: Customer Satisfaction Trends")
        st.markdown("""
        **Use Case:** Understanding overall customer sentiment helps identify satisfaction levels and areas of concern.
        """)

        # Overall sentiment distribution from SQL
        try:
            sentiment_sql = text(f"""
                SELECT
                    Sentiment_Label,
                    COUNT(*) AS count_reviews
                FROM {table_name}
                WHERE Sentiment_Label IS NOT NULL
                GROUP BY Sentiment_Label;
            """)
            sentiment_df = pd.read_sql_query(sentiment_sql, con=engine)
        except Exception as e:
            st.error(f"Failed to load sentiment data from database: {e}")
            sentiment_df = pd.DataFrame()

        # Fixed colour mapping for sentiment
        sentiment_colors = {
            "Positive": "darkblue",   # dark blue
            "Neutral": "lightblue",   # light blue
            "Negative": "red"         # red
        }

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Overall Sentiment Distribution")
            if not sentiment_df.empty and "Sentiment_Label" in sentiment_df.columns:
                total_reviews = sentiment_df["count_reviews"].sum()
                fig = px.pie(
                    sentiment_df,
                    values='count_reviews',
                    names='Sentiment_Label',
                    title="Customer Sentiment Breakdown",
                    color='Sentiment_Label',
                    color_discrete_map=sentiment_colors
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Sentiment Breakdown:**")
                for _, row in sentiment_df.iterrows():
                    label = row['Sentiment_Label']
                    cnt = row['count_reviews']
                    pct = (cnt / total_reviews * 100) if total_reviews > 0 else 0
                    l = str(label).lower()
                    if l == "positive":
                        icon = "✅"
                    elif l == "negative":
                        icon = "❌"
                    else:
                        icon = "➖"
                    st.write(f"{icon} {label}: {pct:.1f}% ({cnt:,} reviews)")
            else:
                st.info("⚠️ Sentiment data not available from database")

        with col2:
            st.subheader("Sentiment Distribution by Operator")
            try:
                top_ops_sql = text(f"""
                    WITH top_operators AS (
                        SELECT Bus_Name
                        FROM {table_name}
                        WHERE Bus_Name IS NOT NULL
                        GROUP BY Bus_Name
                        ORDER BY COUNT(*) DESC
                        LIMIT 10
                    )
                    SELECT
                        r.Bus_Name,
                        r.Sentiment_Label,
                        COUNT(*) AS count_reviews
                    FROM {table_name} r
                    JOIN top_operators t ON r.Bus_Name = t.Bus_Name
                    WHERE r.Sentiment_Label IS NOT NULL
                    GROUP BY r.Bus_Name, r.Sentiment_Label
                    ORDER BY r.Bus_Name, r.Sentiment_Label;
                """)
                sent_bus_df = pd.read_sql_query(top_ops_sql, con=engine)
            except Exception as e:
                st.error(f"Failed to load sentiment-by-operator data from database: {e}")
                sent_bus_df = pd.DataFrame()

            if not sent_bus_df.empty:
                fig = px.bar(
                    sent_bus_df,
                    x='Bus_Name',
                    y='count_reviews',
                    color='Sentiment_Label',
                    title="Sentiment Distribution - Top 10 Operators",
                    labels={'count_reviews': 'Number of Reviews', 'Bus_Name': 'Bus Operator'},
                    barmode='stack',
                    color_discrete_map=sentiment_colors
                )
                fig.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("⚠️ Not enough sentiment data by operator from database")

    # ----------------------
    # TAB 3: Service Improvement
    # ----------------------
    with tab3:
        st.header("3. Service Improvement: Identify Strengths & Weaknesses")
        st.markdown("""
        **Use Case:** Bus operators can analyze quality metrics from customer reviews to understand which aspects of their service need improvement.
        """)

        # Average rating by service metric from SQL
        try:
            quality_sql = text(f"""
                SELECT
                    AVG(Punctuality)       AS avg_Punctuality,
                    AVG(Cleanliness)       AS avg_Cleanliness,
                    AVG(Staff_Behaviour)   AS avg_Staff_Behaviour,
                    AVG(Driving)           AS avg_Driving,
                    AVG(AC)                AS avg_AC,
                    AVG(Rest_stop_hygiene) AS avg_Rest_stop_hygiene,
                    AVG(Seat_comfort)      AS avg_Seat_comfort,
                    AVG(Live_tracking)     AS avg_Live_tracking
                FROM {table_name};
            """)
            quality_avg_df = pd.read_sql_query(quality_sql, con=engine)
        except Exception as e:
            st.error(f"Failed to load quality metrics from database: {e}")
            quality_avg_df = pd.DataFrame()

        quality_metrics = [
            'Punctuality', 'Cleanliness', 'Staff_Behaviour', 'Driving',
            'AC', 'Rest_stop_hygiene', 'Seat_comfort', 'Live_tracking'
        ]

        available_quality = []
        if not quality_avg_df.empty:
            for col in quality_avg_df.columns:
                if col.startswith("avg_") and not quality_avg_df[col].isna().all():
                    available_quality.append(col)

        if available_quality:
            st.subheader("🔍 Service Quality Breakdown")
            metric_rows = []
            row = quality_avg_df.iloc[0]
            for col in available_quality:
                metric_name = col.replace("avg_", "")
                metric_rows.append({
                    "Metric": metric_name,
                    "Average_Rating": row[col]
                })
            quality_long_df = pd.DataFrame(metric_rows)
            quality_long_df = quality_long_df.sort_values('Average_Rating', ascending=True)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    quality_long_df,
                    x='Average_Rating',
                    y='Metric',
                    orientation='h',
                    title="Average Rating by Service Metric",
                    color='Average_Rating',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                threshold = quality_long_df['Average_Rating'].median()
                weak_areas = quality_long_df[quality_long_df['Average_Rating'] < threshold]
                strong_areas = quality_long_df[quality_long_df['Average_Rating'] >= threshold]

                if not weak_areas.empty:
                    st.warning("⚠️ **Areas Needing Improvement (Below Median):**")
                    for _, r in weak_areas.iterrows():
                        st.write(f"• **{r['Metric'].replace('_', ' ')}**: {r['Average_Rating']:.2f}/5")

                if not strong_areas.empty:
                    st.success("✅ **Strong Performance Areas:**")
                    for _, r in strong_areas.head(3).iterrows():
                        st.write(f"• **{r['Metric'].replace('_', ' ')}**: {r['Average_Rating']:.2f}/5")

            # Quality metrics by top operators from SQL
            with col2:
                st.markdown("**Quality Metrics by Top Operators**")
                try:
                    quality_by_bus_sql = text(f"""
                        WITH top_buses AS (
                            SELECT Bus_Name
                            FROM {table_name}
                            WHERE Bus_Name IS NOT NULL
                            GROUP BY Bus_Name
                            ORDER BY COUNT(*) DESC
                            LIMIT 5
                        )
                        SELECT
                            r.Bus_Name,
                            AVG(r.Punctuality)       AS avg_Punctuality,
                            AVG(r.Cleanliness)       AS avg_Cleanliness,
                            AVG(r.Staff_Behaviour)   AS avg_Staff_Behaviour,
                            AVG(r.Driving)           AS avg_Driving,
                            AVG(r.AC)                AS avg_AC,
                            AVG(r.Rest_stop_hygiene) AS avg_Rest_stop_hygiene,
                            AVG(r.Seat_comfort)      AS avg_Seat_comfort,
                            AVG(r.Live_tracking)     AS avg_Live_tracking
                        FROM {table_name} r
                        JOIN top_buses t ON r.Bus_Name = t.Bus_Name
                        GROUP BY r.Bus_Name
                        ORDER BY r.Bus_Name;
                    """)
                    quality_by_bus_df = pd.read_sql_query(quality_by_bus_sql, con=engine)
                except Exception as e:
                    st.error(f"Failed to load quality-by-bus data from database: {e}")
                    quality_by_bus_df = pd.DataFrame()

                if not quality_by_bus_df.empty:
                    fig = go.Figure()
                    # use first 4 metrics for readability
                    metric_cols = [
                        "avg_Punctuality", "avg_Cleanliness",
                        "avg_Seat_comfort", "avg_AC"
                    ]
                    for m in metric_cols:
                        if m in quality_by_bus_df.columns:
                            fig.add_trace(go.Bar(
                                name=m.replace("avg_", "").replace("_", " "),
                                x=quality_by_bus_df['Bus_Name'],
                                y=quality_by_bus_df[m]
                            ))
                    fig.update_layout(
                        title="Service Quality Comparison (Top 5 Operators)",
                        barmode='group',
                        height=400,
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("⚠️ No quality-by-operator data from database")

        else:
            st.warning("⚠️ Quality metrics not available in database results")

        # Negative reviews analysis from SQL
        if 'Sentiment_Label' in df.columns:
            st.subheader("📝 Review Analysis by Sentiment")
            try:
                neg_count_sql = text(f"""
                    SELECT COUNT(*) AS negative_reviews
                    FROM {table_name}
                    WHERE LOWER(Sentiment_Label) = 'negative';
                """)
                total_count_sql = text(f"""
                    SELECT COUNT(*) AS total_reviews
                    FROM {table_name};
                """)
                neg_df = pd.read_sql_query(neg_count_sql, con=engine)
                total_df = pd.read_sql_query(total_count_sql, con=engine)
                negative_count = int(neg_df.iloc[0]['negative_reviews']) if not neg_df.empty else 0
                total_count = int(total_df.iloc[0]['total_reviews']) if not total_df.empty else 0
                negative_pct = (negative_count / total_count * 100) if total_count > 0 else 0.0
            except Exception as e:
                st.error(f"Failed to load negative review counts from database: {e}")
                negative_count, total_count, negative_pct = 0, 0, 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Negative Reviews", f"{negative_count:,}")
                st.metric("Percentage", f"{negative_pct:.1f}%")

            # Bus with most negative reviews
            with col2:
                try:
                    neg_bus_sql = text(f"""
                        SELECT Bus_Name, COUNT(*) AS neg_count
                        FROM {table_name}
                        WHERE LOWER(Sentiment_Label) = 'negative'
                          AND Bus_Name IS NOT NULL
                        GROUP BY Bus_Name
                        ORDER BY neg_count DESC
                        LIMIT 1;
                    """)
                    neg_bus_df = pd.read_sql_query(neg_bus_sql, con=engine)
                    if not neg_bus_df.empty:
                        st.metric("Most Complaints (Bus)", neg_bus_df.iloc[0]['Bus_Name'][:20])
                        st.metric("Count", f"{int(neg_bus_df.iloc[0]['neg_count']):,}")
                except Exception:
                    st.info("No bus-level negative review data")

            # Route with most negative reviews
            with col3:
                try:
                    neg_route_sql = text(f"""
                        SELECT Route, COUNT(*) AS neg_count
                        FROM {table_name}
                        WHERE LOWER(Sentiment_Label) = 'negative'
                          AND Route IS NOT NULL
                        GROUP BY Route
                        ORDER BY neg_count DESC
                        LIMIT 1;
                    """)
                    neg_route_df = pd.read_sql_query(neg_route_sql, con=engine)
                    if not neg_route_df.empty:
                        st.metric("Most Complained Route", str(neg_route_df.iloc[0]['Route'])[:20])
                        st.metric("Complaints", f"{int(neg_route_df.iloc[0]['neg_count']):,}")
                except Exception:
                    st.info("No route-level negative review data")

    # ----------------------
    # TAB 4: Recommendation System
    # ----------------------
    with tab4:
        st.header("4. Recommendation System: Personalized Bus Suggestions")
        st.markdown("""
        Based on comprehensive review analysis, we can recommend the best buses for different traveler preferences.
        """)
        
        st.subheader("🎯 Find Your Perfect Bus")
        
        col1, col2, col3 = st.columns(3)
        
        # Budget Traveler
        with col1:
            st.markdown("**Budget Traveler**")
            st.write("Looking for affordable options with good reviews")
            try:
                budget_sql = text(f"""
                    SELECT
                        Bus_Name,
                        AVG(Star_Rating) AS avg_rating,
                        AVG(Price)       AS avg_price
                    FROM {table_name}
                    WHERE Star_Rating >= 3.5
                      AND Bus_Name IS NOT NULL
                    GROUP BY Bus_Name
                    ORDER BY avg_price ASC, avg_rating DESC
                    LIMIT 5;
                """)
                budget_df = pd.read_sql_query(budget_sql, con=engine)
            except Exception as e:
                st.error(f"Failed to load budget recommendations: {e}")
                budget_df = pd.DataFrame()

            if not budget_df.empty:
                st.markdown("**Recommended:**")
                for _, row in budget_df.iterrows():
                    price_str = f"₹{row['avg_price']:.0f}" if pd.notna(row.get('avg_price')) else "₹N/A"
                    rating_str = f"({row['avg_rating']:.1f}⭐)" if pd.notna(row.get('avg_rating')) else ""
                    st.write(f"✓ {row['Bus_Name']} - {price_str} {rating_str}")
            else:
                st.info("No suitable budget recommendations from database.")

        # Comfort Seeker
        with col2:
            st.markdown("**Comfort Seeker**")
            st.write("Prioritizing comfort and amenities")
            try:
                comfort_sql = text(f"""
                    SELECT
                        Bus_Name,
                        AVG(Seat_comfort) AS avg_seat_comfort,
                        AVG(AC)           AS avg_AC,
                        AVG(Star_Rating)  AS avg_rating
                    FROM {table_name}
                    WHERE Seat_comfort >= 4
                      AND AC >= 4
                      AND Bus_Name IS NOT NULL
                    GROUP BY Bus_Name
                    ORDER BY avg_seat_comfort DESC, avg_rating DESC
                    LIMIT 5;
                """)
                comfort_df = pd.read_sql_query(comfort_sql, con=engine)
            except Exception as e:
                st.error(f"Failed to load comfort recommendations: {e}")
                comfort_df = pd.DataFrame()

            if not comfort_df.empty:
                st.markdown("**Recommended:**")
                for _, row in comfort_df.iterrows():
                    seat_str = f"{row['avg_seat_comfort']:.1f}⭐" if pd.notna(row.get('avg_seat_comfort')) else "N/A"
                    st.write(f"✓ {row['Bus_Name']} - Seat Comfort: {seat_str}")
            else:
                st.info("No suitable comfort recommendations from database.")

        # Reliability Focused
        with col3:
            st.markdown("**Reliability Focused**")
            st.write("Need punctuality and consistent service")
            try:
                reliable_sql = text(f"""
                    SELECT
                        Bus_Name,
                        AVG(Punctuality)  AS avg_punctuality,
                        AVG(Star_Rating)  AS avg_rating,
                        COUNT(*)          AS reviews_count
                    FROM {table_name}
                    WHERE Punctuality >= 4
                      AND Bus_Name IS NOT NULL
                    GROUP BY Bus_Name
                    ORDER BY avg_punctuality DESC, reviews_count DESC
                    LIMIT 5;
                """)
                reliable_df = pd.read_sql_query(reliable_sql, con=engine)
            except Exception as e:
                st.error(f"Failed to load reliability recommendations: {e}")
                reliable_df = pd.DataFrame()

            if not reliable_df.empty:
                st.markdown("**Recommended:**")
                for _, row in reliable_df.iterrows():
                    punct = row['avg_punctuality'] if pd.notna(row.get('avg_punctuality')) else 0
                    st.write(f"✓ {row['Bus_Name']} - Punctuality: {punct:.1f}⭐")
            else:
                st.info("No suitable reliability recommendations from database.")

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
        ["🏠 Dashboard", "🎯 Find Best Buses", "📊 SQL Queries", "📈 Business Insights"],
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
        home_page(df, selected_operator, table_name=selected_table, engine=engine)
    elif page == "🎯 Find Best Buses":
        find_best_buses_page(df, selected_table, engine)
    elif page == "📊 SQL Queries":
        sql_queries_page(selected_table, engine)
    else:
        business_insights_page(df, selected_table, engine)


if __name__ == "__main__":
    main()
