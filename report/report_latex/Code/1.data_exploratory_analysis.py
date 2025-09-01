import pandas as pd
import numpy as np
from pathlib import Path

def analyze_csv_file(file_path):
    """Analyze CSV file structure and content"""
    df = pd.read_csv(file_path, skiprows=11, header=0)
    
    # Basic information analysis
    column_names = df.columns.tolist()
    total_rows = len(df)
    total_columns = len(df.columns)
    
    # Missing values analysis
    missing_values = df.isnull().sum()
    missing_summary = {col: missing for col, missing in missing_values.items() if missing > 0}
    
    # Time coverage analysis
    time_coverage = {}
    if 'Academic Year' in df.columns:
        time_coverage = {
            'earliest_year': df['Academic Year'].min(),
            'latest_year': df['Academic Year'].max(),
            'year_count': df['Academic Year'].nunique()
        }
    
    # Institution analysis
    institution_stats = {}
    if 'HE Provider' in df.columns:
        institution_stats = {
            'total_institutions': df['HE Provider'].nunique(),
            'institution_list': df['HE Provider'].unique().tolist()
        }
        
        if 'Region of HE provider' in df.columns:
            region_counts = df['Region of HE provider'].value_counts().to_dict()
            ne_institutions = df[df['Region of HE provider'] == 'North East']['HE Provider'].unique().tolist()
            institution_stats.update({
                'regional_distribution': region_counts,
                'north_east_institutions': ne_institutions
            })
    
    # Durham University specific analysis
    durham_analysis = {}
    if 'HE Provider' in df.columns:
        durham_data = df[df['HE Provider'] == 'Durham University']
        if not durham_data.empty:
            durham_analysis = {
                'total_records': len(durham_data),
                'value_stats': durham_data['Value'].describe().to_dict() if 'Value' in df.columns else {}
            }
    
    return {
        'columns': column_names,
        'dimensions': {'rows': total_rows, 'columns': total_columns},
        'missing_values': missing_summary,
        'time_coverage': time_coverage,
        'institution_stats': institution_stats,
        'durham_analysis': durham_analysis
    }

def handle_missing_values(df, file_name):
    """Handle missing values based on file type and column characteristics"""
    df_processed = df.copy()
    
    # Handle Value column missing values
    if 'Value' in df_processed.columns:
        if file_name in ['table-1.csv', 'table-2b.csv', 'table-3.csv', 'table-4c.csv', 'table-4d.csv']:
            df_processed['Value'] = df_processed['Value'].fillna(0)
        else:
            df_processed['Value'] = df_processed['Value'].fillna(0)
    
    # Handle Unit column missing values for table-2a
    if 'Unit' in df_processed.columns and 'Number/Value Marker' in df_processed.columns:
        df_processed['Unit'] = df_processed.apply(
            lambda row: '£000s' if pd.notna(row['Number/Value']) and isinstance(row['Number/Value'], (int, float)) 
            else 'Count' if pd.notna(row['Number/Value']) 
            else row['Unit'], 
            axis=1
        )
    
    # Handle Number/Value column missing values in table-2a
    if file_name == 'table-2a.csv' and 'Number/Value' in df_processed.columns:
        df_processed['Number/Value'] = df_processed['Number/Value'].fillna(0)
    
    return df_processed

def create_dataset_summary(all_data):
    """Create comprehensive dataset summary statistics"""
    summary = {}
    
    # Missing values comparison across files
    missing_data = []
    for file_name, df in all_data.items():
        missing_count = df.isnull().sum().sum()
        missing_data.append({'File': file_name, 'Missing Values': missing_count})
    
    summary['missing_values_comparison'] = missing_data
    
    # Regional distribution analysis
    first_df = next((df for df in all_data.values() if 'Region of HE provider' in df.columns), None)
    if first_df is not None:
        region_counts = first_df['Region of HE provider'].value_counts().to_dict()
        summary['regional_distribution'] = region_counts
    
    # North East institutions value trends
    ne_institutions = ['University of Durham', 'Newcastle University', 
                      'University of Northumbria at Newcastle', 
                      'The University of Sunderland', 'Teesside University']
    
    ne_trends = {}
    for inst in ne_institutions:
        yearly_means = []
        for df in all_data.values():
            if 'HE Provider' in df.columns and 'Value' in df.columns and 'Academic Year' in df.columns:
                inst_data = df[df['HE Provider'] == inst]
                if not inst_data.empty:
                    yearly_mean = inst_data.groupby('Academic Year')['Value'].mean()
                    yearly_means.append(yearly_mean)
        
        if yearly_means:
            combined_means = pd.concat(yearly_means).groupby(level=0).mean()
            ne_trends[inst] = combined_means.to_dict()
    
    summary['north_east_trends'] = ne_trends
    
    # Overall dataset statistics
    dataset_stats = {
        'records_per_file': {name: len(df) for name, df in all_data.items()},
        'avg_values_per_file': {name: df['Value'].mean() if 'Value' in df.columns else 0 
                               for name, df in all_data.items()},
        'unique_institutions_per_file': {name: df['HE Provider'].nunique() if 'HE Provider' in df.columns else 0 
                                       for name, df in all_data.items()},
        'year_coverage_per_file': {name: df['Academic Year'].nunique() if 'Academic Year' in df.columns else 0 
                                  for name, df in all_data.items()}
    }
    
    summary['dataset_statistics'] = dataset_stats
    
    return summary
