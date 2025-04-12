from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

app = Flask(__name__)

def create_visualizations():
    # Read the CSV file
    df = pd.read_csv('Air_Quality.csv')
    
    # Convert Start_Date to datetime if it's not already
    df['Start_Date'] = pd.to_datetime(df['Start_Date'])
    
    # 1. Air Quality Distribution Across Cities
    # Check if 'Geo Place Name' exists (city information)
    if 'Geo Place Name' in df.columns:
        city_df = df.groupby('Geo Place Name')['Data Value'].mean().reset_index()
        city_df = city_df.sort_values('Data Value', ascending=False).head(10)  # Top 10 cities
        
        fig_cities = px.bar(
            city_df, 
            x='Geo Place Name', 
            y='Data Value',
            title='Average Air Quality by City',
            labels={'Geo Place Name': 'City', 'Data Value': 'Average AQI Value'},
            color='Data Value',
            color_continuous_scale=px.colors.sequential.Reds_r
        )
    else:
        # Fallback if city info not available
        fig_cities = go.Figure()
        fig_cities.add_annotation(text="City data not available in dataset", showarrow=False, font=dict(size=20))
    
    # 2. Outlier Detection in Pollutant Levels
    # Create box plots for each pollutant type
    if 'Measure' in df.columns:
        fig_outliers = px.box(
            df, 
            x='Measure', 
            y='Data Value',
            title='Pollutant Levels Distribution and Outliers',
            color='Measure'
        )
    else:
        fig_outliers = go.Figure()
        fig_outliers.add_annotation(text="Measure data not available", showarrow=False, font=dict(size=20))
    
    # 3. Trend of PM2.5 Over Time
    # Check if PM2.5 measure exists
    pm25_exists = 'PM2.5' in df['Measure'].unique() if 'Measure' in df.columns else False
    
    if pm25_exists:
        # Create PM2.5 trend (filtering for PM2.5 measure)
        pm25_df = df[df['Measure'] == 'PM2.5'].copy()
        pm25_df['Month'] = pm25_df['Start_Date'].dt.to_period('M')
        monthly_pm25 = pm25_df.groupby('Month')['Data Value'].mean().reset_index()
        monthly_pm25['Month'] = monthly_pm25['Month'].astype(str)
        
        fig_pm25 = px.line(
            monthly_pm25, 
            x='Month', 
            y='Data Value',
            title='PM2.5 Monthly Trend',
            labels={'Month': 'Month-Year', 'Data Value': 'PM2.5 Value'},
            markers=True
        )
    else:
        # If PM2.5 doesn't exist, create a chart with the first measure available
        if 'Measure' in df.columns:
            first_measure = df['Measure'].unique()[0]
            measure_df = df[df['Measure'] == first_measure].copy()
            measure_df['Month'] = measure_df['Start_Date'].dt.to_period('M')
            monthly_measure = measure_df.groupby('Month')['Data Value'].mean().reset_index()
            monthly_measure['Month'] = monthly_measure['Month'].astype(str)
            
            fig_pm25 = px.line(
                monthly_measure, 
                x='Month', 
                y='Data Value', 
                title=f'{first_measure} Monthly Trend',
                labels={'Month': 'Month-Year', 'Data Value': f'{first_measure} Value'},
                markers=True
            )
        else:
            fig_pm25 = go.Figure()
            fig_pm25.add_annotation(text="Measure data not available", showarrow=False, font=dict(size=20))
    
    # 4. Pollutant Correlation Heatmap
    if 'Measure' in df.columns:
        # Get available measures
        available_measures = df['Measure'].unique()
        
        # Check if we have at least 2 measures for correlation
        if len(available_measures) >= 2:
            # Use all available measures
            use_measures = available_measures
            
            # Create pivot table with available measures
            pivot_df = df.pivot_table(
                values='Data Value',
                index='Start_Date',
                columns='Measure',
                aggfunc='mean'
            ).dropna(axis=1)  # Drop columns with all NaN values
            
            # Get actual columns that were created
            actual_columns = pivot_df.columns.tolist()
            
            if len(actual_columns) >= 2:
                # Create correlation matrix with available columns
                correlation_matrix = pivot_df.corr()
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    colorscale='RdBu_r',
                    zmin=-1, zmax=1
                ))
                fig_heatmap.update_layout(
                    title='Pollutant Correlation Heatmap',
                    xaxis_title='Pollutant',
                    yaxis_title='Pollutant'
                )
            else:
                # Not enough measures for correlation, create placeholder
                fig_heatmap = go.Figure()
                fig_heatmap.add_annotation(
                    text="Not enough measures for correlation analysis",
                    showarrow=False,
                    font=dict(size=20)
                )
        else:
            # No correlation possible, create placeholder
            fig_heatmap = go.Figure()
            fig_heatmap.add_annotation(
                text="Not enough measures for correlation analysis",
                showarrow=False,
                font=dict(size=20)
            )
    else:
        fig_heatmap = go.Figure()
        fig_heatmap.add_annotation(text="Measure data not available", showarrow=False, font=dict(size=20))
    
    # 5. Daily Pollution Level Breakdown
    # Define AQI categories
    def categorize_aqi(value):
        if value <= 50:
            return 'Good'
        elif value <= 100:
            return 'Moderate'
        elif value <= 150:
            return 'Unhealthy for Sensitive Groups'
        elif value <= 200:
            return 'Unhealthy'
        elif value <= 300:
            return 'Very Unhealthy'
        else:
            return 'Hazardous'
    
    # Apply category function
    if 'Data Value' in df.columns:
        df['AQI_Category'] = df['Data Value'].apply(categorize_aqi)
        
        # Count days in each category
        category_counts = df.groupby('AQI_Category').size().reset_index(name='Count')
        category_order = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 
                         'Unhealthy', 'Very Unhealthy', 'Hazardous']
        category_counts['AQI_Category'] = pd.Categorical(
            category_counts['AQI_Category'], 
            categories=category_order, 
            ordered=True
        )
        category_counts = category_counts.sort_values('AQI_Category')
        
        # Create pie chart
        fig_categories = px.pie(
            category_counts, 
            values='Count', 
            names='AQI_Category',
            title='Air Quality Category Distribution',
            color='AQI_Category',
            color_discrete_map={
                'Good': 'green',
                'Moderate': 'yellow',
                'Unhealthy for Sensitive Groups': 'orange',
                'Unhealthy': 'red',
                'Very Unhealthy': 'purple',
                'Hazardous': 'maroon'
            }
        )
    else:
        fig_categories = go.Figure()
        fig_categories.add_annotation(text="AQI data not available", showarrow=False, font=dict(size=20))
    
    # Add data summary stats
    df_stats = pd.DataFrame({
        'Measure': df['Measure'].unique(),
        'Count': [df[df['Measure'] == m]['Data Value'].count() for m in df['Measure'].unique()],
        'Average': [round(df[df['Measure'] == m]['Data Value'].mean(), 2) for m in df['Measure'].unique()],
        'Min': [df[df['Measure'] == m]['Data Value'].min() for m in df['Measure'].unique()],
        'Max': [df[df['Measure'] == m]['Data Value'].max() for m in df['Measure'].unique()]
    })
    
    fig_stats = go.Figure(data=[go.Table(
        header=dict(values=list(df_stats.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df_stats[col] for col in df_stats.columns],
                fill_color='lavender',
                align='left'))
    ])
    fig_stats.update_layout(title="Air Quality Measures Summary")
    
    # Return the figures directly
    return {
        'cities': fig_cities,
        'outliers': fig_outliers,
        'pm25': fig_pm25,
        'heatmap': fig_heatmap,
        'categories': fig_categories,
        'stats': fig_stats
    }

@app.route('/')
def dashboard():
    # Create visualizations
    figures = create_visualizations()
    
    # Convert figures to JSON
    graphJSON = {}
    for name, fig in figures.items():
        graphJSON[name] = fig.to_json()
    
    return render_template('dashboard.html', graphJSON=graphJSON)

if __name__ == '__main__':
    app.run(debug=True) 