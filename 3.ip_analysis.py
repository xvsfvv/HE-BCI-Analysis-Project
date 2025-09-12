import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def save_plot(fig, filename):
    save_dir = Path('visualizations/ip_analysis')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(save_dir / filename)
    plt.close(fig)

def analyze_ip():
    """Main function to analyze all intellectual property related data"""
    # Set consistent font settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
    
    data_dir = Path('Data')
    tables = {
        'table4a': pd.read_csv(data_dir / 'table-4a.csv', skiprows=11, encoding='utf-8'),
        'table4b': pd.read_csv(data_dir / 'table-4b.csv', skiprows=11, encoding='utf-8'),
        'table4c': pd.read_csv(data_dir / 'table-4c.csv', skiprows=11, encoding='utf-8'),
        'table4d': pd.read_csv(data_dir / 'table-4d.csv', skiprows=11, encoding='utf-8'),
        'table4e': pd.read_csv(data_dir / 'table-4e.csv', skiprows=11, encoding='utf-8')
    }
    
    # Define universities for comparison
    north_east_universities = [
        'University of Durham',
        'Newcastle University',
        'University of Northumbria at Newcastle',
        'The University of Sunderland',
        'Teesside University'
    ]
    
    # 1. IP Disclosures and Patents Analysis
    print("\n=== IP Disclosures and Patents Analysis ===")
    
    # 1.1 Disclosure Type Analysis (Durham only)
    durham_ip = tables['table4a'][tables['table4a']['HE Provider'] == 'University of Durham']
    disclosure_type = durham_ip.groupby('Type of disclosure or patent')['Value'].sum()
    print("\nDisclosure Type Distribution (Durham):")
    print(disclosure_type)
    
    # National statistics for IP disclosures
    national_ip = tables['table4a'].groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_rank = national_ip.index.get_loc('University of Durham') + 1
    total_universities = len(national_ip)
    national_avg = national_ip.mean()
    print(f"\nNational Statistics for IP Disclosures:")
    print(f"Total number of universities: {total_universities}")
    print(f"Durham's rank: {durham_rank}/{total_universities}")
    print(f"National average: {national_avg:,.0f}")
    print(f"Durham's value: {national_ip['University of Durham']:,.0f}")
    print(f"Difference from national average: {national_ip['University of Durham'] - national_avg:,.0f}")
    
    fig = plt.figure(figsize=(18, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(disclosure_type)))
    y_pos = np.arange(len(disclosure_type))
    ax = plt.barh(y_pos, disclosure_type.values, color=colors)
    plt.title('IP Disclosures by Type (Durham)', fontsize=14, fontweight='bold')
    plt.yticks(y_pos, disclosure_type.index)
    plt.xlabel('Number of Disclosures')
    # Add value labels on the right side of bars
    for i, v in enumerate(disclosure_type.values):
        plt.text(v, i, f'{v:,.0f}', ha='left', va='center', fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_plot(fig, 'figure18.ip_disclosure_type.png')
    
    # 1.2 Time Trend Analysis (Durham only)
    yearly_disclosures = durham_ip.groupby(['Academic Year', 'Type of disclosure or patent'])['Value'].sum().reset_index()
    print("\nYearly Disclosure Trend (Durham):")
    print(yearly_disclosures)
    
    fig = plt.figure(figsize=(12, 11))
    for disc_type in yearly_disclosures['Type of disclosure or patent'].unique():
        data = yearly_disclosures[yearly_disclosures['Type of disclosure or patent'] == disc_type]
        plt.plot(data['Academic Year'], data['Value'], label=disc_type, marker='o')
    plt.title('IP Disclosures Trend (Durham)')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=1)
    plt.tight_layout()
    plt.subplots_adjust(top=0.75)
    save_plot(fig, 'figure19.ip_disclosure_trend.png')
    
    # 1.3 North East Universities Comparison
    ne_ip = tables['table4a'][tables['table4a']['HE Provider'].isin(north_east_universities)]
    ne_total = ne_ip.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    print("\nTotal IP Disclosures by University (North East):")
    print(ne_total)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ne_total)))
    ax = plt.bar(ne_total.index, ne_total.values, color=colors)
    plt.title('Total IP Disclosures - North East Universities')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(ne_total.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure20.ip_ne_comparison.png')
    
    ###############################

    # 2. License Analysis
    print("\n=== License Analysis ===")
    
    # 2.1 License Type Analysis (Durham only)
    durham_license = tables['table4b'][tables['table4b']['HE Provider'] == 'University of Durham']
    # Exclude Total rows to avoid double counting
    durham_license_no_total = durham_license[~durham_license['Type of organisation'].str.contains('Total', na=False)]
    license_type = durham_license_no_total.groupby('Type of licence granted')['Value'].sum()
    print("\nLicense Type Distribution (Durham):")
    print(license_type)
    
    # National statistics for licenses - exclude Total rows
    national_license_no_total = tables['table4b'][~tables['table4b']['Type of organisation'].str.contains('Total', na=False)]
    national_license = national_license_no_total.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_rank = national_license.index.get_loc('University of Durham') + 1
    total_universities = len(national_license)
    national_avg = national_license.mean()
    print(f"\nNational Statistics for Licenses:")
    print(f"Total number of universities: {total_universities}")
    print(f"Durham's rank: {durham_rank}/{total_universities}")
    print(f"National average: {national_avg:,.0f}")
    print(f"Durham's value: {national_license['University of Durham']:,.0f}")
    print(f"Difference from national average: {national_license['University of Durham'] - national_avg:,.0f}")
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(license_type)))
    ax = plt.bar(license_type.index, license_type.values, color=colors)
    plt.title('Licenses by Type (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(license_type.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure21.license_type.png')
    
    # 2.2 Organization Type Analysis (Durham only)
    org_type_dist = durham_license_no_total.groupby('Type of organisation')['Value'].sum()
    print("\nOrganization Type Distribution (Durham):")
    print(org_type_dist)
    
    # 2.2.1 Detailed License Analysis by Type and Organization (for table)
    print("\nDetailed License Analysis by Type and Organization (Durham):")
    
    # Non-software licenses
    non_software_data = durham_license_no_total[durham_license_no_total['Type of licence granted'] == 'Non-software']
    print("\nNon-software licenses:")
    for org_type in ['Non-commercial organisations', 'Other (non-SME) commercial businesses', 'SMEs']:
        total = non_software_data[non_software_data['Type of organisation'] == org_type]['Value'].sum()
        print(f"  {org_type}: {total} total licenses")
    
    # Software only licenses
    software_data = durham_license_no_total[durham_license_no_total['Type of licence granted'] == 'Software only']
    print("\nSoftware only licenses:")
    for org_type in ['Non-commercial organisations', 'Other (non-SME) commercial businesses', 'SMEs']:
        total = software_data[software_data['Type of organisation'] == org_type]['Value'].sum()
        print(f"  {org_type}: {total} total licenses")
    
    # Income generating licenses
    total_income_gen = durham_license[durham_license['Type of organisation'] == 'Total number generating income in the period']['Value'].sum()
    total_licenses = durham_license_no_total['Value'].sum()  # Use correct total
    print(f"\nTotal licenses: {total_licenses}")
    print(f"Total income generating: {total_income_gen}")
    print(f"Income generation rate: {total_income_gen/total_licenses*100:.1f}%")
    
    # Calculate income generating licenses by type and organization (proportional distribution)
    print("\nIncome generating licenses (proportional distribution):")
    
    # Non-software income generating
    non_commercial_ns = non_software_data[non_software_data['Type of organisation'] == 'Non-commercial organisations']['Value'].sum()
    commercial_ns = non_software_data[non_software_data['Type of organisation'] == 'Other (non-SME) commercial businesses']['Value'].sum()
    sme_ns = non_software_data[non_software_data['Type of organisation'] == 'SMEs']['Value'].sum()
    
    non_commercial_sw = software_data[software_data['Type of organisation'] == 'Non-commercial organisations']['Value'].sum()
    commercial_sw = software_data[software_data['Type of organisation'] == 'Other (non-SME) commercial businesses']['Value'].sum()
    sme_sw = software_data[software_data['Type of organisation'] == 'SMEs']['Value'].sum()
    
    # Calculate proportional distribution
    non_commercial_income = round((non_commercial_ns / total_licenses) * total_income_gen)
    commercial_income = round((commercial_ns / total_licenses) * total_income_gen)
    sme_income = round((sme_ns / total_licenses) * total_income_gen)
    non_commercial_sw_income = round((non_commercial_sw / total_licenses) * total_income_gen)
    commercial_sw_income = round((commercial_sw / total_licenses) * total_income_gen)
    sme_sw_income = total_income_gen - non_commercial_income - commercial_income - sme_income - non_commercial_sw_income - commercial_sw_income
    
    print("Non-software:")
    print(f"  Non-commercial organisations: {non_commercial_ns} total, {non_commercial_income} income-generating ({non_commercial_income/non_commercial_ns*100:.1f}%)")
    print(f"  Other (non-SME) commercial: {commercial_ns} total, {commercial_income} income-generating ({commercial_income/commercial_ns*100:.1f}%)")
    print(f"  SMEs: {sme_ns} total, {sme_income} income-generating ({sme_income/sme_ns*100:.1f}%)")
    
    print("Software only:")
    print(f"  Non-commercial organisations: {non_commercial_sw} total, {non_commercial_sw_income} income-generating ({non_commercial_sw_income/non_commercial_sw*100:.1f}%)")
    print(f"  Other (non-SME) commercial: {commercial_sw} total, {commercial_sw_income} income-generating ({commercial_sw_income/commercial_sw*100:.1f}%)")
    print(f"  SMEs: {sme_sw} total, {sme_sw_income} income-generating ({sme_sw_income/sme_sw*100:.1f}%)")
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(org_type_dist)))
    ax = plt.bar(range(len(org_type_dist)), org_type_dist.values, color=colors)
    plt.title('Licenses by Organisation Type (Durham)')
    plt.xticks(range(len(org_type_dist)), org_type_dist.index, rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(org_type_dist.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure22.license_org_type.png')
    
    # 2.3 North East Universities Comparison
    ne_license = tables['table4b'][tables['table4b']['HE Provider'].isin(north_east_universities)]
    ne_license_no_total = ne_license[~ne_license['Type of organisation'].str.contains('Total', na=False)]
    ne_license_total = ne_license_no_total.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    print("\nTotal Licenses by University (North East):")
    print(ne_license_total)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ne_license_total)))
    ax = plt.bar(ne_license_total.index, ne_license_total.values, color=colors)
    plt.title('Total Licenses - North East Universities')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(ne_license_total.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure23.license_ne_comparison.png')
    
    #################################
    # 3. IP Income Analysis
    print("\n=== IP Income Analysis ===")
    
    # 3.1 Income Source Analysis (Durham only)
    # Use only table4c for detailed income source analysis
    durham_income = tables['table4c'][tables['table4c']['HE Provider'] == 'University of Durham']
    income_source = durham_income.groupby('Income source')['Value'].sum()
    print("\nIncome Source Distribution (Durham):")
    print(income_source)
    
    # National statistics for IP income - use only table4d Total IP revenues to avoid double counting
    national_income = tables['table4d'][tables['table4d']['Category Marker'] == 'Total IP revenues'].groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_rank = national_income.index.get_loc('University of Durham') + 1
    total_universities = len(national_income)
    national_avg = national_income.mean()
    print(f"\nNational Statistics for IP Income:")
    print(f"Total number of universities: {total_universities}")
    print(f"Durham's rank: {durham_rank}/{total_universities}")
    print(f"National average: £{national_avg:,.0f}")
    print(f"Durham's value: £{national_income['University of Durham']:,.0f}")
    print(f"Difference from national average: £{national_income['University of Durham'] - national_avg:,.0f}")
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(income_source)))
    ax = plt.bar(income_source.index, income_source.values, color=colors)
    plt.title('IP Income by Source (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(income_source.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure24.ip_income_source.png')
    
    # 3.2 Organization Type Analysis (Durham only)
    org_type = durham_income.groupby('Type of organisation')['Value'].sum()
    print("\nOrganization Type Distribution (Durham):")
    print(org_type)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(org_type)))
    ax = plt.bar(org_type.index, org_type.values, color=colors)
    plt.title('IP Income by Organisation Type (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(org_type.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure25.ip_income_org_type.png')
    
    # 3.3 North East Universities Comparison
    ne_income = tables['table4d'][(tables['table4d']['HE Provider'].isin(north_east_universities)) & 
                                  (tables['table4d']['Category Marker'] == 'Total IP revenues')]
    ne_income_total = ne_income.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    print("\nTotal IP Income by University (North East):")
    print(ne_income_total)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ne_income_total)))
    ax = plt.bar(ne_income_total.index, ne_income_total.values, color=colors)
    plt.title('Total IP Income - North East Universities')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(ne_income_total.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure26.ip_income_ne_comparison.png')
    
    ###############################
    # 4. spin-off Company Analysis
    print("\n=== Spin-off Company Analysis ===")
    
    # 4.1 Metric Type Analysis (Durham)
    durham_spin = tables['table4e'][tables['table4e']['HE Provider'] == 'University of Durham']
    metric_type = durham_spin.groupby('Metric')['Value'].sum()
    print("\nMetric Type Distribution (Durham):")
    print(metric_type)
    
    # National statistics for spin-off companies
    national_spin = tables['table4e'].groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_rank = national_spin.index.get_loc('University of Durham') + 1
    total_universities = len(national_spin)
    national_avg = national_spin.mean()
    print(f"\nNational Statistics for Spin-off Companies:")
    print(f"Total number of universities: {total_universities}")
    print(f"Durham's rank: {durham_rank}/{total_universities}")
    print(f"National average: {national_avg:,.0f}")
    print(f"Durham's value: {national_spin['University of Durham']:,.0f}")
    print(f"Difference from national average: {national_spin['University of Durham'] - national_avg:,.0f}")
    
    # Create radar chart instead of bar chart
    fig = plt.figure(figsize=(12, 12))
    
    # Prepare data for radar chart with shortened labels
    categories = list(metric_type.index)
    values = list(metric_type.values)
    
    # Create shortened labels for better display
    short_labels = [
        'Employment (FTE)',
        'Turnover (£k)',
        'External Investment (£k)',
        'New Companies',
        'Active Firms',
        'Survived 3+ Years'
    ]
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Add the first value at the end to close the circle
    values += values[:1]
    
    # Create the radar chart
    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, values, 'o-', linewidth=2, label='Durham', color='#1f77b4')
    ax.fill(angles, values, alpha=0.25, color='#1f77b4')
    
    # Add shortened category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_labels, fontsize=11)
    
    # Add value labels
    for angle, value, category in zip(angles[:-1], values[:-1], categories):
        ax.text(angle, value + max(values) * 0.05, f'{value:,.0f}', 
                ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Customize the chart
    ax.set_ylim(0, max(values) * 1.2)
    ax.set_title('Spin-off Companies by Metric Type (Durham)', 
                 size=14, fontweight='bold', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    save_plot(fig, 'figure27.spin_metric_type.png')
    
    # 4.2 Category Analysis (Durham only)
    category = durham_spin.groupby('Category Marker')['Value'].sum()
    print("\nCategory Distribution (Durham):")
    print(category)
    
    # Create pie chart instead of bar chart
    fig = plt.figure(figsize=(12, 10))
    
    # Prepare data for pie chart with shortened labels
    labels = list(category.index)
    sizes = list(category.values)
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    # Create shortened labels for better display
    short_labels = [
        'Other Spin-outs',
        'Social Enterprises',
        'HE Provider Owned',
        'Staff Start-ups',
        'Student Start-ups'
    ]
    
    # Create pie chart without any labels on the pie
    wedges, texts = plt.pie(sizes, labels=None, colors=colors, autopct=None,
                            startangle=90, textprops={'fontsize': 11})
    
    # Customize the pie chart
    plt.title('Spin-off Companies by Category (Durham)', fontsize=14, fontweight='bold', pad=20)
    
    # Add a legend with values and percentages for all categories
    total = sum(sizes)
    legend_labels = [f'{label}: {size:,.0f} ({size/total*100:.1f}%)' for label, size in zip(short_labels, sizes)]
    legend = plt.legend(wedges, legend_labels, title="Categories (Values & Percentages)", 
                       loc="center", bbox_to_anchor=(0.5, 0.5), fontsize=10)
    
    # Add background color to legend
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.8)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')
    plt.tight_layout()
    save_plot(fig, 'figure28.spin_category.png')
    
    # 4.3 North East Universities Comparison
    ne_spin = tables['table4e'][tables['table4e']['HE Provider'].isin(north_east_universities)]
    ne_spin_total = ne_spin.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    print("\nTotal Spin-off Companies by University (North East):")
    print(ne_spin_total)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ne_spin_total)))
    ax = plt.bar(ne_spin_total.index, ne_spin_total.values, color=colors)
    plt.title('Total Spin-off Companies - North East Universities')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(ne_spin_total.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure29.spin_ne_comparison.png')

if __name__ == "__main__":
    analyze_ip() 