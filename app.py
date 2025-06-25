import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import json
import seaborn as sns
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename
from datetime import datetime
import pickle

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# UPDATED Configuration - Increased limits to 1GB
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1GB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Add these new configurations for larger files
app.config['MAX_FORM_MEMORY_SIZE'] = 1024 * 1024 * 1024  # 1GB
app.config['MAX_FORM_PARTS'] = 10000
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Clean up matplotlib on app shutdown
import atexit

def cleanup_matplotlib():
    plt.close('all')

atexit.register(cleanup_matplotlib)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Add this import at the top
from utils.data_processor import LargeDataProcessor

def validate_file_contents(sentiment_df, trader_df):
    """Validate that files contain expected data types"""
    errors = []
    
    # Check sentiment file
    sentiment_indicators = ['classification', 'sentiment', 'fear', 'greed', 'value']
    has_sentiment = any(indicator in col.lower() for col in sentiment_df.columns for indicator in sentiment_indicators)
    
    if not has_sentiment:
        errors.append("Sentiment file doesn't appear to contain sentiment data")
    
    # Check trader file  
    trader_indicators = ['pnl', 'profit', 'loss', 'trade', 'position', 'account', 'execution']
    has_trading = any(indicator in col.lower() for col in trader_df.columns for indicator in trader_indicators)
    
    if not has_trading:
        errors.append("Trader file doesn't appear to contain trading data")
    
    # Check for file swap
    if has_sentiment and not has_trading:
        errors.append("⚠️  FILES MAY BE SWAPPED! The 'trader file' looks like sentiment data.")
    
    return errors

def process_uploaded_data(sentiment_file, trader_file):
    """Process uploaded CSV files with enhanced date parsing"""
    try:
        # Initialize large data processor
        from utils.data_processor import LargeDataProcessor
        processor = LargeDataProcessor(max_memory_mb=500)
        
        # Process files efficiently
        print("Processing sentiment file...")
        sentiment_df = processor.process_large_csv(sentiment_file, sample_size=20000)
        
        print("Processing trader file...")
        trader_df = processor.process_large_csv(trader_file, sample_size=50000)
        
        print(f"Loaded {len(sentiment_df)} sentiment records and {len(trader_df)} trader records")
        
        # DEBUG: Print column info
        print("Sentiment file columns:", sentiment_df.columns.tolist())
        print("Trader file columns:", trader_df.columns.tolist())
        
        # Auto-detect date columns
        sentiment_date_col = detect_date_column(sentiment_df)
        trader_date_col = detect_date_column(trader_df)
        
        if not sentiment_date_col or not trader_date_col:
            # Manual fallback - look for first column that might be a date
            print("Auto-detection failed, trying manual detection...")
            
            if not sentiment_date_col:
                for col in sentiment_df.columns:
                    sample_val = str(sentiment_df[col].iloc[0]) if len(sentiment_df) > 0 else ""
                    if any(char in sample_val for char in ['-', '/', ':']):
                        sentiment_date_col = col
                        print(f"Manual detection found sentiment date column: {col}")
                        break
            
            if not trader_date_col:
                for col in trader_df.columns:
                    sample_val = str(trader_df[col].iloc[0]) if len(trader_df) > 0 else ""
                    if any(char in sample_val for char in ['-', '/', ':']):
                        trader_date_col = col
                        print(f"Manual detection found trader date column: {col}")
                        break
        
        if not sentiment_date_col or not trader_date_col:
            raise ValueError(f"Could not detect date columns. Sentiment columns: {sentiment_df.columns.tolist()}, Trader columns: {trader_df.columns.tolist()}")
        
        print(f"Using date columns: sentiment='{sentiment_date_col}', trader='{trader_date_col}'")
        
        # DEBUG: Print sample date values
        print(f"Sample sentiment dates: {sentiment_df[sentiment_date_col].head(3).tolist()}")
        print(f"Sample trader dates: {trader_df[trader_date_col].head(3).tolist()}")
        
        # ENHANCED DATE PARSING with European format priority
        print("Parsing sentiment dates...")
        try:
            # Try European format first (DD-MM-YYYY)
            sentiment_df['parsed_date'] = pd.to_datetime(
                sentiment_df[sentiment_date_col], 
                dayfirst=True, 
                errors='coerce'
            )
            
            # Check success rate
            success_rate = sentiment_df['parsed_date'].notna().sum() / len(sentiment_df)
            print(f"Sentiment date parsing success rate: {success_rate:.2%}")
            
            if success_rate < 0.5:
                # Try without dayfirst
                sentiment_df['parsed_date'] = pd.to_datetime(
                    sentiment_df[sentiment_date_col], 
                    errors='coerce'
                )
                success_rate = sentiment_df['parsed_date'].notna().sum() / len(sentiment_df)
                print(f"Sentiment date parsing (attempt 2) success rate: {success_rate:.2%}")
                
        except Exception as e:
            print(f"Sentiment date parsing error: {e}")
            raise ValueError(f"Could not parse sentiment dates: {e}")
        
        print("Parsing trader dates...")
        try:
            # Try European format first (DD-MM-YYYY)
            trader_df['parsed_date'] = pd.to_datetime(
                trader_df[trader_date_col], 
                dayfirst=True, 
                errors='coerce'
            )
            
            # Check success rate
            success_rate = trader_df['parsed_date'].notna().sum() / len(trader_df)
            print(f"Trader date parsing success rate: {success_rate:.2%}")
            
            if success_rate < 0.5:
                # Try without dayfirst
                trader_df['parsed_date'] = pd.to_datetime(
                    trader_df[trader_date_col], 
                    errors='coerce'
                )
                success_rate = trader_df['parsed_date'].notna().sum() / len(trader_df)
                print(f"Trader date parsing (attempt 2) success rate: {success_rate:.2%}")
                
        except Exception as e:
            print(f"Trader date parsing error: {e}")
            raise ValueError(f"Could not parse trader dates: {e}")
        
        # Remove rows where date parsing failed
        initial_sentiment_count = len(sentiment_df)
        initial_trader_count = len(trader_df)
        
        sentiment_df = sentiment_df.dropna(subset=['parsed_date'])
        trader_df = trader_df.dropna(subset=['parsed_date'])
        
        print(f"After removing invalid dates:")
        print(f"  Sentiment: {initial_sentiment_count} -> {len(sentiment_df)} records")
        print(f"  Trader: {initial_trader_count} -> {len(trader_df)} records")
        
        if len(sentiment_df) == 0:
            raise ValueError("No valid dates found in sentiment file")
        if len(trader_df) == 0:
            raise ValueError("No valid dates found in trader file")
        
        # Create common date column (date only, no time)
        sentiment_df['date'] = sentiment_df['parsed_date'].dt.date
        trader_df['date'] = trader_df['parsed_date'].dt.date
        
        # Print date ranges for debugging
        print(f"Sentiment date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
        print(f"Trader date range: {trader_df['date'].min()} to {trader_df['date'].max()}")
        
        # Auto-detect other important columns
        pnl_col = detect_pnl_column(trader_df)
        sentiment_col = detect_sentiment_column(sentiment_df)
        leverage_col = detect_leverage_column(trader_df)
        trader_id_col = detect_trader_id_column(trader_df)
        
        print(f"Detected columns - PnL: {pnl_col}, Sentiment: {sentiment_col}, Leverage: {leverage_col}, Trader ID: {trader_id_col}")
        
        # Standardize column names with better error handling
        if pnl_col:
            trader_df['pnl'] = pd.to_numeric(trader_df[pnl_col], errors='coerce')
            # Remove rows with invalid PnL
            trader_df = trader_df.dropna(subset=['pnl'])
        else:
            raise ValueError(f"Could not detect PnL/profit column in trader data. Available columns: {trader_df.columns.tolist()}")
            
        if sentiment_col:
            sentiment_df['sentiment_value'] = pd.to_numeric(sentiment_df[sentiment_col], errors='coerce')
        
        if leverage_col:
            trader_df['leverage'] = pd.to_numeric(trader_df[leverage_col], errors='coerce')
        else:
            # Create default leverage if not found
            trader_df['leverage'] = 1.0
            
        if trader_id_col:
            trader_df['trader_id'] = trader_df[trader_id_col].astype(str)
        else:
            # Create trader IDs if not present
            trader_df['trader_id'] = 'trader_' + (trader_df.index + 1).astype(str)
        
        # Create sentiment classification if not exists
        if 'classification' not in sentiment_df.columns:
            if 'sentiment_value' in sentiment_df.columns:
                sentiment_df['classification'] = sentiment_df['sentiment_value'].apply(classify_sentiment)
            else:
                # Try to find classification column with different names
                class_cols = [col for col in sentiment_df.columns 
                             if any(keyword in col.lower() for keyword in ['class', 'sentiment', 'fear', 'greed'])]
                if class_cols:
                    sentiment_df['classification'] = sentiment_df[class_cols[0]]
                    print(f"Using classification column: {class_cols[0]}")
                else:
                    # Create default classification
                    sentiment_df['classification'] = 'Neutral'
                    print("Warning: No sentiment classification found, using default 'Neutral'")
        
        # Merge datasets
        print("Merging datasets...")
        merged_df = pd.merge(trader_df, sentiment_df, on='date', how='inner')
        
        if merged_df.empty:
            # Try outer join to see what dates we have
            outer_merge = pd.merge(trader_df, sentiment_df, on='date', how='outer', indicator=True)
            left_only = outer_merge[outer_merge['_merge'] == 'left_only']['date'].nunique()
            right_only = outer_merge[outer_merge['_merge'] == 'right_only']['date'].nunique()
            
            raise ValueError(f"No matching dates found between datasets. "
                           f"Trader-only dates: {left_only}, Sentiment-only dates: {right_only}. "
                           f"Check if date ranges overlap.")
        
        print(f"Successfully merged {len(merged_df)} records")
        print(f"Merged date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
        print(f"Unique dates in merged data: {merged_df['date'].nunique()}")
        
        # Clean up memory
        del sentiment_df, trader_df
        import gc
        gc.collect()
        
        return merged_df, None
        
    except Exception as e:
        print(f"Error in process_uploaded_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, str(e)

def detect_date_column(df):
    """Enhanced date column detection with multiple format support"""
    potential_date_cols = []
    
    # Look for columns with date-like names
    date_keywords = ['date', 'time', 'timestamp', 'created', 'updated', 'day', 'dt']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in date_keywords):
            potential_date_cols.append(col)
    
    # If no obvious date columns, check first few columns
    if not potential_date_cols:
        potential_date_cols = df.columns[:3].tolist()
    
    # Test each potential column
    for col in potential_date_cols:
        if is_date_column(df[col]):
            print(f"Detected date column: {col}")
            return col
    
    return None

def is_date_column(series):
    """Check if a series contains date-like data"""
    if series.dtype == 'object':
        # Sample first 10 non-null values
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False
        
        date_count = 0
        for value in sample:
            if is_date_string(str(value)):
                date_count += 1
        
        # If more than 70% look like dates, consider it a date column
        return (date_count / len(sample)) > 0.7
    
    return False

def is_date_string(date_str):
    """Check if string looks like a date"""
    date_patterns = [
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # DD-MM-YYYY or MM-DD-YYYY
        r'\d{2,4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\s+\d{1,2}:\d{2}',  # With time
        r'\d{2,4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}',  # YYYY-MM-DD with time
    ]
    
    for pattern in date_patterns:
        if re.match(pattern, date_str.strip()):
            return True
    
    return False

def parse_dates_flexible(series):
    """Parse dates with multiple format attempts"""
    print(f"Parsing dates from column with sample values: {series.head(3).tolist()}")
    
    # Common date formats to try
    date_formats = [
        # European formats (DD-MM-YYYY)
        '%d-%m-%Y %H:%M',
        '%d-%m-%Y',
        '%d/%m/%Y %H:%M',
        '%d/%m/%Y',
        '%d-%m-%y %H:%M',
        '%d-%m-%y',
        '%d/%m/%y %H:%M',
        '%d/%m/%y',
        
        # American formats (MM-DD-YYYY)
        '%m-%d-%Y %H:%M',
        '%m-%d-%Y',
        '%m/%d/%Y %H:%M',
        '%m/%d/%Y',
        '%m-%d-%y %H:%M',
        '%m-%d-%y',
        '%m/%d/%y %H:%M',
        '%m/%d/%y',
        
        # ISO formats (YYYY-MM-DD)
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d',
        '%Y/%m/%d %H:%M:%S',
        '%Y/%m/%d %H:%M',
        '%Y/%m/%d',
        
        # Other common formats
        '%d %b %Y',
        '%d %B %Y',
        '%b %d, %Y',
        '%B %d, %Y',
    ]
    
    # Try pandas built-in parser first with different settings
    for dayfirst in [True, False]:
        try:
            print(f"Trying pandas parser with dayfirst={dayfirst}")
            result = pd.to_datetime(series, dayfirst=dayfirst, errors='coerce')
            
            # Check if parsing was successful (less than 50% NaT values)
            success_rate = (result.notna().sum() / len(result))
            print(f"Success rate with dayfirst={dayfirst}: {success_rate:.2%}")
            
            if success_rate > 0.5:
                print(f"Successfully parsed dates with dayfirst={dayfirst}")
                return result
        except Exception as e:
            print(f"Pandas parser failed with dayfirst={dayfirst}: {e}")
            continue
    
    # Try specific formats
    for fmt in date_formats:
        try:
            print(f"Trying format: {fmt}")
            result = pd.to_datetime(series, format=fmt, errors='coerce')
            
            # Check success rate
            success_rate = (result.notna().sum() / len(result))
            print(f"Success rate with format {fmt}: {success_rate:.2%}")
            
            if success_rate > 0.5:
                print(f"Successfully parsed dates with format: {fmt}")
                return result
        except Exception as e:
            continue
    
    # Try mixed format parsing (pandas 2.0+)
    try:
        print("Trying mixed format parsing")
        result = pd.to_datetime(series, format='mixed', dayfirst=True, errors='coerce')
        success_rate = (result.notna().sum() / len(result))
        print(f"Success rate with mixed format: {success_rate:.2%}")
        
        if success_rate > 0.5:
            print("Successfully parsed dates with mixed format")
            return result
    except Exception as e:
        print(f"Mixed format parsing failed: {e}")
    
    # Last resort: try to infer format from first valid entry
    try:
        print("Attempting format inference from sample data")
        sample_value = series.dropna().iloc[0] if len(series.dropna()) > 0 else None
        
        if sample_value:
            inferred_format = infer_date_format(str(sample_value))
            if inferred_format:
                print(f"Inferred format: {inferred_format}")
                result = pd.to_datetime(series, format=inferred_format, errors='coerce')
                success_rate = (result.notna().sum() / len(result))
                
                if success_rate > 0.5:
                    print(f"Successfully parsed with inferred format: {inferred_format}")
                    return result
    except Exception as e:
        print(f"Format inference failed: {e}")
    
    print("All date parsing attempts failed")
    return None

def infer_date_format(date_string):
    """Infer date format from a sample string"""
    date_string = date_string.strip()
    
    # Common patterns and their formats
    patterns = [
        (r'^\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}:\d{2}$', '%d-%m-%Y %H:%M'),
        (r'^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}$', '%d/%m/%Y %H:%M'),
        (r'^\d{1,2}-\d{1,2}-\d{4}$', '%d-%m-%Y'),
        (r'^\d{1,2}/\d{1,2}/\d{4}$', '%d/%m/%Y'),
        (r'^\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}:\d{2}$', '%Y-%m-%d %H:%M:%S'),
        (r'^\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}$', '%Y-%m-%d %H:%M'),
        (r'^\d{4}-\d{1,2}-\d{1,2}$', '%Y-%m-%d'),
        (r'^\d{1,2}-\d{1,2}-\d{2}\s+\d{1,2}:\d{2}$', '%d-%m-%y %H:%M'),
        (r'^\d{1,2}-\d{1,2}-\d{2}$', '%d-%m-%y'),
    ]
    
    for pattern, fmt in patterns:
        if re.match(pattern, date_string):
            return fmt
    
    return None

def detect_pnl_column(df):
    """Auto-detect PnL column"""
    pnl_keywords = ['pnl', 'profit', 'loss', 'return', 'gain', 'closed pnl', 'closedpnl', 'net pnl']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in pnl_keywords):
            return col
    return None

def detect_sentiment_column(df):
    """Auto-detect sentiment value column"""
    sentiment_keywords = ['sentiment', 'fear', 'greed', 'index', 'value', 'score']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in sentiment_keywords):
            if df[col].dtype in ['int64', 'float64']:
                return col
    return None

def detect_leverage_column(df):
    """Auto-detect leverage column"""
    leverage_keywords = ['leverage', 'margin', 'multiplier']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in leverage_keywords):
            return col
    return None

def detect_trader_id_column(df):
    """Auto-detect trader ID column"""
    trader_keywords = ['trader', 'user', 'id', 'account']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in trader_keywords):
            return col
    return None

def classify_sentiment(value):
    """Classify numeric sentiment values into categories"""
    try:
        val = float(value)
        if val < 30:
            return 'Fear'
        elif val > 70:
            return 'Greed'
        else:
            return 'Neutral'
    except:
        return 'Unknown'

def generate_insights(df):
    """Generate insights from merged dataset with JSON-serializable output"""
    try:
        insights = {}
        
        # Basic statistics - convert to native Python types
        insights['total_records'] = int(len(df))
        insights['unique_traders'] = int(df['trader_id'].nunique()) if 'trader_id' in df.columns else 0
        insights['date_range'] = {
            'start': str(df['date'].min()),
            'end': str(df['date'].max())
        }
        
        # PnL analysis
        if 'pnl' in df.columns:
            insights['pnl_stats'] = {
                'total_pnl': float(df['pnl'].sum()),
                'avg_pnl': float(df['pnl'].mean()),
                'median_pnl': float(df['pnl'].median()),
                'std_pnl': float(df['pnl'].std()),
                'min_pnl': float(df['pnl'].min()),
                'max_pnl': float(df['pnl'].max()),
                'profitable_trades': int((df['pnl'] > 0).sum()),
                'losing_trades': int((df['pnl'] < 0).sum()),
                'win_rate': float((df['pnl'] > 0).mean() * 100)
            }
        
        # Sentiment analysis
        if 'sentiment_value' in df.columns:
            insights['sentiment_stats'] = {
                'avg_sentiment': float(df['sentiment_value'].mean()),
                'min_sentiment': float(df['sentiment_value'].min()),
                'max_sentiment': float(df['sentiment_value'].max()),
                'std_sentiment': float(df['sentiment_value'].std())
            }
        
        if 'classification' in df.columns:
            sentiment_counts = df['classification'].value_counts()
            insights['sentiment_distribution'] = {
                str(k): int(v) for k, v in sentiment_counts.items()
            }
        
        # Correlation analysis
        if 'pnl' in df.columns and 'sentiment_value' in df.columns:
            correlation = df['pnl'].corr(df['sentiment_value'])
            insights['pnl_sentiment_correlation'] = float(correlation) if not pd.isna(correlation) else 0.0
        
        # Performance by sentiment
        if 'pnl' in df.columns and 'classification' in df.columns:
            perf_by_sentiment = df.groupby('classification')['pnl'].agg(['mean', 'sum', 'count']).round(4)
            insights['performance_by_sentiment'] = {}
            for sentiment in perf_by_sentiment.index:
                insights['performance_by_sentiment'][str(sentiment)] = {
                    'avg_pnl': float(perf_by_sentiment.loc[sentiment, 'mean']),
                    'total_pnl': float(perf_by_sentiment.loc[sentiment, 'sum']),
                    'trade_count': int(perf_by_sentiment.loc[sentiment, 'count'])
                }
        
        # Leverage analysis (if available)
        if 'leverage' in df.columns:
            insights['leverage_stats'] = {
                'avg_leverage': float(df['leverage'].mean()),
                'max_leverage': float(df['leverage'].max()),
                'min_leverage': float(df['leverage'].min())
            }
            
            # High leverage performance
            high_leverage = df[df['leverage'] > df['leverage'].quantile(0.75)]
            if len(high_leverage) > 0 and 'pnl' in df.columns:
                insights['high_leverage_performance'] = {
                    'avg_pnl': float(high_leverage['pnl'].mean()),
                    'win_rate': float((high_leverage['pnl'] > 0).mean() * 100),
                    'trade_count': int(len(high_leverage))
                }
        
        # Time-based insights
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['month'] = df_copy['date'].dt.month
        df_copy['weekday'] = df_copy['date'].dt.dayofweek
        
        if 'pnl' in df.columns:
            monthly_perf = df_copy.groupby('month')['pnl'].mean()
            insights['monthly_performance'] = {
                str(month): float(pnl) for month, pnl in monthly_perf.items()
            }
            
            weekday_perf = df_copy.groupby('weekday')['pnl'].mean()
            weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            insights['weekday_performance'] = {
                weekday_names[day]: float(pnl) for day, pnl in weekday_perf.items()
            }
        
        return insights
        
    except Exception as e:
        print(f"Error generating insights: {e}")
        return {
            'error': str(e),
            'total_records': int(len(df)) if df is not None else 0
        }

def create_visualizations(df):
    """Create visualizations and save them as files"""
    try:
        # Ensure the static directory exists
        os.makedirs('static/plots', exist_ok=True)
        
        # Set style for better looking plots
        plt.style.use('default')
        
        # 1. PnL over time
        if 'pnl' in df.columns and 'date' in df.columns:
            plt.figure(figsize=(12, 6))
            df_plot = df.copy()
            df_plot['date'] = pd.to_datetime(df_plot['date'])
            daily_pnl = df_plot.groupby('date')['pnl'].sum().reset_index()
            
            plt.plot(daily_pnl['date'], daily_pnl['pnl'], linewidth=2, color='blue')
            plt.title('Daily PnL Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('PnL', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('static/plots/pnl_over_time.png', dpi=300, bbox_inches='tight')
            plt.close()  # Important: close the figure
        
        # 2. Sentiment distribution
        if 'classification' in df.columns:
            plt.figure(figsize=(10, 6))
            sentiment_counts = df['classification'].value_counts()
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
            
            plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                   colors=colors[:len(sentiment_counts)], startangle=90)
            plt.title('Sentiment Distribution', fontsize=16, fontweight='bold')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig('static/plots/sentiment_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. PnL by sentiment
        if 'pnl' in df.columns and 'classification' in df.columns:
            plt.figure(figsize=(10, 6))
            sentiment_pnl = df.groupby('classification')['pnl'].mean().sort_values(ascending=True)
            
            colors = ['red' if x < 0 else 'green' for x in sentiment_pnl.values]
            bars = plt.bar(sentiment_pnl.index, sentiment_pnl.values, color=colors, alpha=0.7)
            
            plt.title('Average PnL by Sentiment', fontsize=16, fontweight='bold')
            plt.xlabel('Sentiment', fontsize=12)
            plt.ylabel('Average PnL', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top')
            
            plt.tight_layout()
            plt.savefig('static/plots/pnl_by_sentiment.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Correlation heatmap
        if len(df.select_dtypes(include=[np.number]).columns) > 1:
            plt.figure(figsize=(10, 8))
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlation_matrix = df[numeric_cols].corr()
            
            # Create heatmap
            im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar(im)
            
            # Add labels
            plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha='right')
            plt.yticks(range(len(numeric_cols)), numeric_cols)
            plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
            
            # Add correlation values
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                            ha='center', va='center', 
                            color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')
            
            plt.tight_layout()
            plt.savefig('static/plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Cumulative PnL
        if 'pnl' in df.columns and 'date' in df.columns:
            plt.figure(figsize=(12, 6))
            df_plot = df.copy()
            df_plot['date'] = pd.to_datetime(df_plot['date'])
            df_plot = df_plot.sort_values('date')
            df_plot['cumulative_pnl'] = df_plot['pnl'].cumsum()
            
            plt.plot(df_plot['date'], df_plot['cumulative_pnl'], linewidth=2, color='green')
            plt.title('Cumulative PnL Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Cumulative PnL', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('static/plots/cumulative_pnl.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("Visualizations created successfully")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

def train_prediction_model(df):
    """Train prediction model with JSON-serializable output"""
    try:
        if 'pnl' in df.columns and 'sentiment_value' in df.columns:
            # Prepare features
            features = ['sentiment_value']
            if 'leverage' in df.columns:
                features.append('leverage')
            
            X = df[features].fillna(0)
            y = (df['pnl'] > 0).astype(int)  # Binary classification: profitable or not
            
            if len(X) < 10:
                return None, "Not enough data for model training"
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics - convert to native Python types
            accuracy = float(accuracy_score(y_test, y_pred))
            
            # Feature importance
            feature_importance = {}
            for i, feature in enumerate(features):
                feature_importance[feature] = float(model.feature_importances_[i])
            
            # Save model
            model_path = 'models/trading_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            results = {
                'accuracy': accuracy,
                'feature_importance': feature_importance,
                'model_saved': True,
                'test_samples': int(len(y_test)),
                'features_used': features
            }
            
            return results, None
            
    except Exception as e:
        print(f"Model training error: {e}")
        return None, str(e)
    
    return None, "Insufficient data for model training"

@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')

# Add file size validation function
def validate_file_size(file):
    """Check if file size is within limits"""
    if file:
        file.seek(0, 2)  # Seek to end of file
        size = file.tell()
        file.seek(0)  # Reset to beginning
        
        # Convert to MB
        size_mb = size / (1024 * 1024)
        
        if size_mb > 1024:  # 1GB limit per file
            return False, f"File size ({size_mb:.1f}MB) exceeds 1GB limit"
        
        return True, f"File size: {size_mb:.1f}MB"
    
    return False, "No file provided"

# Add this after your app configuration


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads with better validation"""
    try:
        # Check if files were uploaded
        if 'sentiment_file' not in request.files or 'trader_file' not in request.files:
            flash('Please upload both CSV files', 'error')
            return redirect(url_for('index'))
        
        sentiment_file = request.files['sentiment_file']
        trader_file = request.files['trader_file']
        
        # Check if files are selected
        if sentiment_file.filename == '' or trader_file.filename == '':
            flash('Please select both files', 'error')
            return redirect(url_for('index'))
        
        # Check file extensions
        if not (allowed_file(sentiment_file.filename) and allowed_file(trader_file.filename)):
            flash('Only CSV files are allowed', 'error')
            return redirect(url_for('index'))
        
        # Validate file sizes
        sentiment_valid, sentiment_msg = validate_file_size(sentiment_file)
        trader_valid, trader_msg = validate_file_size(trader_file)
        
        if not sentiment_valid:
            flash(f'Sentiment file error: {sentiment_msg}', 'error')
            return redirect(url_for('index'))
        
        if not trader_valid:
            flash(f'Trader file error: {trader_msg}', 'error')
            return redirect(url_for('index'))
        
        flash(f'Files validated - {sentiment_msg}, {trader_msg}', 'info')
        
        # Save uploaded files
        sentiment_filename = secure_filename(sentiment_file.filename)
        trader_filename = secure_filename(trader_file.filename)
        
        sentiment_path = os.path.join(app.config['UPLOAD_FOLDER'], sentiment_filename)
        trader_path = os.path.join(app.config['UPLOAD_FOLDER'], trader_filename)
        
        sentiment_file.save(sentiment_path)
        trader_file.save(trader_path)
        
        flash('Files uploaded successfully, processing data...', 'info')
        
        # Load files for validation
        try:
            sentiment_df_preview = pd.read_csv(sentiment_path, nrows=5)
            trader_df_preview = pd.read_csv(trader_path, nrows=5)
        except Exception as e:
            flash(f'Error reading files: {str(e)}', 'error')
            return redirect(url_for('index'))
        
        # Validate file contents
        validation_errors = validate_file_contents(sentiment_df_preview, trader_df_preview)
        if validation_errors:
            for error in validation_errors:
                flash(error, 'error')
            flash('Please check your files and try again.', 'warning')
            return redirect(url_for('index'))
        
        # Process data (no signal timeout needed)
        merged_df, error = process_uploaded_data(sentiment_path, trader_path)
        
        if error:
            flash(f'Error processing data: {error}', 'error')
            return redirect(url_for('index'))
        
        # Generate insights
        print("Generating insights...")
        insights = generate_insights(merged_df)
        
        # Create visualizations
        print("Creating visualizations...")
        create_visualizations(merged_df)
        
        # Train prediction model
        print("Training model...")
        model_results, model_error = train_prediction_model(merged_df)
        
        # Convert to JSON-serializable format before storing in session
        print("Converting data for session storage...")
        insights_serializable = make_json_serializable(insights)
        model_results_serializable = make_json_serializable(model_results) if model_results else None
        
        # Store results in session with error handling
        try:
            session['insights'] = insights_serializable
            session['model_results'] = model_results_serializable
            session['data_processed'] = True
            print("Data stored in session successfully")
        except Exception as session_error:
            print(f"Session storage error: {session_error}")
            # Try to store minimal data
            session['data_processed'] = True
            session['insights'] = {
                'total_records': int(len(merged_df)),
                'error': 'Full insights could not be stored due to data size'
            }
        
        # Clean up uploaded files
        try:
            os.remove(sentiment_path)
            os.remove(trader_path)
        except:
            pass
        
        if model_error:
            flash(f'Model training warning: {model_error}', 'warning')
        
        flash('Data processed successfully!', 'success')
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        flash(f'Error processing files: {str(e)}', 'error')
        return redirect(url_for('index'))
    finally:
        # Ensure matplotlib figures are closed
        plt.close('all')

@app.route('/dashboard')
def dashboard():
    """Dashboard with results"""
    if not session.get('data_processed'):
        flash('Please upload and process data first', 'warning')
        return redirect(url_for('index'))
    
    insights = session.get('insights', {})
    model_results = session.get('model_results', {})
    
    return render_template('dashboard.html', insights=insights, model_results=model_results)

@app.route('/insights')
def insights_page():
    """Detailed insights page"""
    if not session.get('data_processed'):
        flash('Please upload and process data first', 'warning')
        return redirect(url_for('index'))
    
    insights = session.get('insights', {})
    model_results = session.get('model_results', {})
    
    return render_template('insights.html', insights=insights, model_results=model_results)

@app.route('/api/insights')
def api_insights():
    """API endpoint for insights data"""
    if not session.get('data_processed'):
        return jsonify({'error': 'No data processed'}), 400
    
    return jsonify({
        'insights': session.get('insights', {}),
        'model_results': session.get('model_results', {})
    })

@app.route('/reset')
def reset_session():
    """Reset session and clear uploaded files"""
    session.clear()
    
    # Clean up uploaded files
    try:
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"Error cleaning up files: {e}")
    
    flash('Session reset successfully', 'info')
    return redirect(url_for('index'))

@app.route('/debug-upload', methods=['POST'])
def debug_upload():
    """Debug endpoint to analyze file structure"""
    try:
        if 'sentiment_file' not in request.files or 'trader_file' not in request.files:
            return jsonify({'error': 'Missing files'}), 400
        
        sentiment_file = request.files['sentiment_file']
        trader_file = request.files['trader_file']
        
        # Save files temporarily
        sentiment_path = os.path.join(app.config['UPLOAD_FOLDER'], 'debug_sentiment.csv')
        trader_path = os.path.join(app.config['UPLOAD_FOLDER'], 'debug_trader.csv')
        
        sentiment_file.save(sentiment_path)
        trader_file.save(trader_path)
        
        # Analyze file structure
        debug_info = {}
        
        # Analyze sentiment file
        try:
            sentiment_df = pd.read_csv(sentiment_path, nrows=5)
            debug_info['sentiment'] = {
                'columns': sentiment_df.columns.tolist(),
                'sample_data': sentiment_df.to_dict('records'),
                'dtypes': sentiment_df.dtypes.to_dict()
            }
        except Exception as e:
            debug_info['sentiment'] = {'error': str(e)}
        
        # Analyze trader file
        try:
            trader_df = pd.read_csv(trader_path, nrows=5)
            debug_info['trader'] = {
                'columns': trader_df.columns.tolist(),
                'sample_data': trader_df.to_dict('records'),
                'dtypes': trader_df.dtypes.to_dict()
            }
        except Exception as e:
            debug_info['trader'] = {'error': str(e)}
        
        # Clean up
        os.remove(sentiment_path)
        os.remove(trader_path)
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def make_json_serializable(obj):
    """Simple JSON serialization fix"""
    import numpy as np
    import pandas as pd
    
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    elif pd.isna(obj):
        return None
    else:
        return obj

if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='0.0.0.0', port=5000)
else:
    # For production deployment
    import logging
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)