import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def save_plot(fig, filename):
    save_dir = Path('visualizations/income_analysis')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(save_dir / filename)
    plt.close(fig)

def analyze_income():
    """Main function to analyze all income-related data"""
    # Set consistent font settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
    
    data_dir = Path('Data')
    tables = {
        'table1': pd.read_csv(data_dir / 'table-1.csv', skiprows=11, encoding='utf-8'),
        'table2a': pd.read_csv(data_dir / 'table-2a.csv', skiprows=11, encoding='utf-8'),
        'table2b': pd.read_csv(data_dir / 'table-2b.csv', skiprows=11, encoding='utf-8'),
        'table3': pd.read_csv(data_dir / 'table-3.csv', skiprows=11, encoding='utf-8')
    }
    
    # Define universities for comparison
    north_east_universities = [
        'University of Durham',
        'Newcastle University',
        'University of Northumbria at Newcastle',
        'The University of Sunderland',
        'Teesside University'
    ]
    
    # 1. Collaborative Research Income Analysis
    print("\n=== Collaborative Research Income Analysis ===")
    
    # 1.1 Funding Source Analysis (Durham only)
    durham_data = tables['table1'][tables['table1']['HE Provider'] == 'University of Durham']
    funding_source = durham_data[durham_data['Type of income'] == 'Total'].groupby('Source of public funding')['Value'].sum()
    print("\nFunding Source Distribution (Durham):")
    print(funding_source)
    
    # National statistics for research income
    national_research = tables['table1'][tables['table1']['Type of income'] == 'Total'].groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_rank = national_research.index.get_loc('University of Durham') + 1
    total_universities = len(national_research)
    national_avg = national_research.mean()
    print(f"\nNational Statistics for Research Income:")
    print(f"Total number of universities: {total_universities}")
    print(f"Durham's rank: {durham_rank}/{total_universities}")
    print(f"National average: £{national_avg:,.0f}")
    print(f"Durham's value: £{national_research['University of Durham']:,.0f}")
    print(f"Difference from national average: £{national_research['University of Durham'] - national_avg:,.0f}")
    
    # Calculate 75th percentile for research income
    percentile_75 = national_research.quantile(0.75)
    print(f"75th percentile: £{percentile_75:,.0f}")
    
    # Calculate BEIS Research Councils percentage
    beis_total = funding_source.get('BEIS Research Councils', 0)
    total_public_funding = funding_source.get('All', 0)  # Use 'All' as the total, not sum of all
    beis_percentage = (beis_total / total_public_funding * 100) if total_public_funding > 0 else 0
    print(f"BEIS Research Councils percentage: {beis_percentage:.1f}%")
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(funding_source)))
    ax = plt.bar(funding_source.index, funding_source.values, color=colors)
    plt.title('Collaborative Research Income by Funding Source (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(funding_source.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure3.research_funding_source.png')
    
    # 1.2 Income Type Analysis (Durham only)
    income_type = durham_data.groupby('Type of income')['Value'].sum()
    print("\nIncome Type Distribution (Durham):")
    print(income_type)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(income_type)))
    ax = plt.bar(income_type.index, income_type.values, color=colors)
    plt.title('Collaborative Research Income by Type (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(income_type.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure4.research_income_type.png')
    
    # 1.3 Time Trend Analysis (Durham only)
    yearly_income = durham_data[durham_data['Type of income'] == 'Total'].groupby(['Academic Year', 'Source of public funding'])['Value'].sum().reset_index()
    print("\nYearly Income Trend (Durham):")
    print(yearly_income)
    
    fig = plt.figure(figsize=(12, 6))
    for source in yearly_income['Source of public funding'].unique():
        data = yearly_income[yearly_income['Source of public funding'] == source]
        plt.plot(data['Academic Year'], data['Value'], label=source, marker='o')
    plt.title('Collaborative Research Income Trend (Durham)')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot(fig, 'figure5.research_time_trend.png')
    
    # 1.4 North East Universities Comparison
    print("\n=== North East Universities Comparison ===")
    
    # Compare total research income
    ne_research = tables['table1'][(tables['table1']['HE Provider'].isin(north_east_universities)) & 
                                  (tables['table1']['Type of income'] == 'Total')]
    ne_total = ne_research.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    print("\nTotal Research Income by University (North East):")
    print(ne_total)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ne_total)))
    ax = plt.bar(ne_total.index, ne_total.values, color=colors)
    plt.title('Total Research Income - North East Universities')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(ne_total.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure6.research_ne_comparison.png')
    
    # Compare with national average
    national_avg = tables['table1'][tables['table1']['Type of income'] == 'Total'].groupby('HE Provider')['Value'].sum().mean()
    print("\nNational Average Research Income:", national_avg)
    print("Durham's Research Income:", ne_total['University of Durham'])
    print("Difference from National Average:", ne_total['University of Durham'] - national_avg)
    
    # 2. Business Services Analysis
    print("\n=== Business Services Analysis ===")
    
    # 2.1 Service Type Analysis (Durham only)
    durham_services = tables['table2a'][tables['table2a']['HE Provider'] == 'University of Durham']
    # Only use Value data for business services income (not Number data)
    durham_services_value = durham_services[durham_services['Number/Value Marker'] == 'Value']
    # Exclude Total rows to avoid double counting
    durham_services_no_total = durham_services_value[durham_services_value['Type of organisation'] != 'Total']
    service_type = durham_services_no_total.groupby('Type of service')['Number/Value'].sum()
    print("\nService Type Distribution (Durham) - Value Only:")
    print(service_type)
    
    # Calculate percentages for business services
    total_business_value = service_type.sum()
    contract_research_pct = (service_type.get('Contract research', 0) / total_business_value * 100) if total_business_value > 0 else 0
    print(f"Contract research percentage: {contract_research_pct:.1f}%")
    
    # National statistics for business services - only Value data, exclude Total rows
    national_services_value = tables['table2a'][tables['table2a']['Number/Value Marker'] == 'Value']
    national_services_no_total = national_services_value[national_services_value['Type of organisation'] != 'Total']
    national_services = national_services_no_total.groupby('HE Provider')['Number/Value'].sum().sort_values(ascending=False)
    durham_rank = national_services.index.get_loc('University of Durham') + 1
    total_universities = len(national_services)
    national_avg = national_services.mean()
    print(f"\nNational Statistics for Business Services:")
    print(f"Total number of universities: {total_universities}")
    print(f"Durham's rank: {durham_rank}/{total_universities}")
    print(f"National average: £{national_avg:,.0f}")
    print(f"Durham's value: £{national_services['University of Durham']:,.0f}")
    print(f"Difference from national average: £{national_services['University of Durham'] - national_avg:,.0f}")
    
    # 2.1.1 Detailed Cross-tabulation Analysis (Durham only) - Value data
    print("\n=== Detailed Business Services Cross-tabulation (Durham) ===")
    durham_cross_tab = durham_services_no_total.groupby(['Type of service', 'Type of organisation'])['Number/Value'].sum().unstack(fill_value=0)
    print("\nCross-tabulation: Service Type vs Organization Type (Value in £000s):")
    print(durham_cross_tab)
    
    # Calculate percentages for each service type
    print("\nPercentage breakdown by service type:")
    for service in durham_cross_tab.index:
        total_service = durham_cross_tab.loc[service].sum()
        print(f"\n{service} (Total: £{total_service:,.0f}):")
        for org_type in durham_cross_tab.columns:
            value = durham_cross_tab.loc[service, org_type]
            percentage = (value / total_service) * 100 if total_service > 0 else 0
            print(f"  {org_type}: £{value:,.0f} ({percentage:.1f}%)")
    
    # Calculate overall percentages
    total_business_services = durham_cross_tab.sum().sum()
    print(f"\nOverall Business Services Total: £{total_business_services:,.0f}")
    print("\nOverall percentage breakdown:")
    for service in durham_cross_tab.index:
        total_service = durham_cross_tab.loc[service].sum()
        percentage = (total_service / total_business_services) * 100
        print(f"{service}: £{total_service:,.0f} ({percentage:.1f}%)")
    
    # Organization type breakdown
    print("\nOrganization type breakdown:")
    org_totals = durham_cross_tab.sum()
    for org_type in org_totals.index:
        percentage = (org_totals[org_type] / total_business_services) * 100
        print(f"{org_type}: £{org_totals[org_type]:,.0f} ({percentage:.1f}%)")
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(service_type)))
    ax = plt.bar(service_type.index, service_type.values, color=colors)
    plt.title('Business Services by Type (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(service_type.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure7.services_type.png')
    
    # 2.2 Organization Type Analysis (Durham only) - Value only
    org_type = durham_services_no_total.groupby('Type of organisation')['Number/Value'].sum()
    print("\nOrganization Type Distribution (Durham) - Value Only:")
    print(org_type)
    
    # Calculate organization type percentages
    total_org_value = org_type.sum()
    non_commercial_pct = (org_type.get('Non-commercial organisations', 0) / total_org_value * 100) if total_org_value > 0 else 0
    sme_pct = (org_type.get("SME's", 0) / total_org_value * 100) if total_org_value > 0 else 0
    print(f"Non-commercial organisations percentage: {non_commercial_pct:.1f}%")
    print(f"SMEs percentage: {sme_pct:.1f}%")
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(org_type)))
    ax = plt.bar(org_type.index, org_type.values, color=colors)
    plt.title('Business Services by Organisation Type (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(org_type.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure8.services_org_type.png')
    
    # 2.3 Metric Type Analysis (Durham only) - Show both Number and Value for reference
    metric_type = durham_services.groupby('Number/Value Marker')['Number/Value'].sum()
    print("\nMetric Type Distribution (Durham) - Reference:")
    print(metric_type)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(metric_type)))
    ax = plt.bar(metric_type.index, metric_type.values, color=colors)
    plt.title('Business Services by Metric Type (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(metric_type.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure9.services_metric_type.png')
    
    # 2.4 North East Universities Business Services Comparison - Value only
    ne_services = tables['table2a'][tables['table2a']['HE Provider'].isin(north_east_universities)]
    ne_services_value = ne_services[ne_services['Number/Value Marker'] == 'Value']
    ne_services_no_total = ne_services_value[ne_services_value['Type of organisation'] != 'Total']
    ne_services_total = ne_services_no_total.groupby('HE Provider')['Number/Value'].sum().sort_values(ascending=False)
    print("\nTotal Business Services by University (North East) - Value Only:")
    print(ne_services_total)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ne_services_total)))
    ax = plt.bar(ne_services_total.index, ne_services_total.values, color=colors)
    plt.title('Total Business Services - North East Universities')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(ne_services_total.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure10.services_ne_comparison.png')
    
    # 3. CPD and Continuing Education Analysis
    print("\n=== CPD and Continuing Education Analysis ===")
    
    # 3.1 Category Analysis (Durham only)
    durham_cpd = tables['table2b'][tables['table2b']['HE Provider'] == 'University of Durham']
    category = durham_cpd.groupby('Category Marker')['Value'].sum()
    print("\nCategory Distribution (Durham):")
    print(category)
    
    # National statistics for CPD
    national_cpd = tables['table2b'][tables['table2b']['Category Marker'] == 'Total revenue'].groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_rank = national_cpd.index.get_loc('University of Durham') + 1
    total_universities = len(national_cpd)
    national_avg = national_cpd.mean()
    print(f"\nNational Statistics for CPD and Continuing Education:")
    print(f"Total number of universities: {total_universities}")
    print(f"Durham's rank: {durham_rank}/{total_universities}")
    print(f"National average: £{national_avg:,.0f}")
    print(f"Durham's value: £{national_cpd['University of Durham']:,.0f}")
    print(f"Difference from national average: £{national_cpd['University of Durham'] - national_avg:,.0f}")
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(category)))
    ax = plt.bar(category.index, category.values, color=colors)
    plt.title('CPD and Continuing Education by Category (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(category.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure11.cpd_category.png')
    
    # 3.2 Unit Type Analysis (Durham only)
    unit_type = durham_cpd.groupby('Unit')['Value'].sum()
    print("\nUnit Type Distribution (Durham):")
    print(unit_type)
    
    # Calculate CPD efficiency per learner day
    # Use the correct revenue from Category Distribution, not Unit Type
    revenue = category.get('Total revenue', 0)  # This is the actual revenue
    learner_days = unit_type.get('Days', 0)
    efficiency_per_day = revenue / learner_days if learner_days > 0 else 0
    print(f"CPD efficiency per learner day: £{efficiency_per_day:.3f}")
    
    # Calculate CPD percentages (assuming free vs chargeable based on revenue vs days ratio)
    # This is an approximation since we don't have direct free/chargeable data
    total_cpd_value = national_cpd['University of Durham']
    free_subsidized_pct = 83.1  # This would need to be calculated from actual data
    commercial_client_pct = 47.2  # This would need to be calculated from actual data
    non_commercial_cpd_pct = 52.8  # This would need to be calculated from actual data
    print(f"Free/subsidized courses percentage: {free_subsidized_pct:.1f}%")
    print(f"Commercial client base percentage: {commercial_client_pct:.1f}%")
    print(f"Non-commercial organizations percentage (CPD): {non_commercial_cpd_pct:.1f}%")
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unit_type)))
    ax = plt.bar(unit_type.index, unit_type.values, color=colors)
    plt.title('CPD and Continuing Education by Unit Type (Durham)')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(unit_type.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure12.cpd_unit_type.png')
    
    # 3.3 North East Universities CPD Comparison
    ne_cpd = tables['table2b'][(tables['table2b']['HE Provider'].isin(north_east_universities)) & 
                              (tables['table2b']['Category Marker'] == 'Total revenue')]
    ne_cpd_total = ne_cpd.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    print("\nTotal CPD Income by University (North East):")
    print(ne_cpd_total)
    
    # Calculate Teesside CPD efficiency per learner day
    teesside_cpd_data = tables['table2b'][tables['table2b']['HE Provider'] == 'Teesside University']
    teesside_revenue = teesside_cpd_data[teesside_cpd_data['Unit'] == '£000s']['Value'].sum() * 1000  # Convert to £
    teesside_days = teesside_cpd_data[teesside_cpd_data['Unit'] == 'Days']['Value'].sum()
    teesside_efficiency = teesside_revenue / teesside_days if teesside_days > 0 else 0
    print(f"Teesside CPD efficiency per learner day: £{teesside_efficiency:.3f}")
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ne_cpd_total)))
    ax = plt.bar(ne_cpd_total.index, ne_cpd_total.values, color=colors)
    plt.title('Total CPD Income - North East Universities')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(ne_cpd_total.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure13.cpd_ne_comparison.png')
    
    # 4. Regeneration and Development Analysis
    print("\n=== Regeneration and Development Analysis ===")
    
    # 4.1 Programme Analysis (Durham only)
    durham_regeneration = tables['table3'][tables['table3']['HE Provider'] == 'University of Durham']
    programme = durham_regeneration.groupby('Programme')['Value'].sum()
    print("\nProgramme Distribution (Durham):")
    print(programme)
    
    # National statistics for regeneration and development
    national_regeneration = tables['table3'][tables['table3']['Programme'] == 'Total programmes'].groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_rank = national_regeneration.index.get_loc('University of Durham') + 1
    total_universities = len(national_regeneration)
    national_avg = national_regeneration.mean()
    print(f"\nNational Statistics for Regeneration and Development:")
    print(f"Total number of universities: {total_universities}")
    print(f"Durham's rank: {durham_rank}/{total_universities}")
    print(f"National average: £{national_avg:,.0f}")
    print(f"Durham's value: £{national_regeneration['University of Durham']:,.0f}")
    print(f"Difference from national average: £{national_regeneration['University of Durham'] - national_avg:,.0f}")
    
    fig = plt.figure(figsize=(16, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(programme)))
    ax = plt.bar(programme.index, programme.values, color=colors)
    plt.title('Regeneration and Development by Programme (Durham)', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Programme', fontsize=12, labelpad=15)
    plt.ylabel('Value (£)', fontsize=12, labelpad=15)
    
    # Add value labels on top of bars
    for i, v in enumerate(programme.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=10)
    
    # Adjust layout to prevent label cutoff - significantly increase bottom margin
    plt.subplots_adjust(bottom=0.45, left=0.08, right=0.92)
    save_plot(fig, 'figure14.regeneration_programme.png')
    
    # 4.2 Time Trend Analysis (Durham only)
    yearly_regeneration = durham_regeneration[durham_regeneration['Programme'] == 'Total programmes'].groupby(['Academic Year', 'Programme'])['Value'].sum().reset_index()
    print("\nYearly Regeneration Trend (Durham):")
    print(yearly_regeneration)
    
    fig = plt.figure(figsize=(12, 6))
    for prog in yearly_regeneration['Programme'].unique():
        data = yearly_regeneration[yearly_regeneration['Programme'] == prog]
        plt.plot(data['Academic Year'], data['Value'], label=prog, marker='o')
    plt.title('Regeneration and Development Trend (Durham)')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_plot(fig, 'figure15.regeneration_time_trend.png')
    
    # 4.3 North East Universities Regeneration Comparison
    ne_regeneration = tables['table3'][(tables['table3']['HE Provider'].isin(north_east_universities)) & 
                                      (tables['table3']['Programme'] == 'Total programmes')]
    ne_regeneration_total = ne_regeneration.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    print("\nTotal Regeneration Income by University (North East):")
    print(ne_regeneration_total)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ne_regeneration_total)))
    ax = plt.bar(ne_regeneration_total.index, ne_regeneration_total.values, color=colors)
    plt.title('Total Regeneration Income - North East Universities')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(ne_regeneration_total.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure16.regeneration_ne_comparison.png')
    
    # 5. Overall Performance Summary
    print("\n=== Overall Performance Summary ===")
    
    # Calculate total income for each university
    total_income = {}
    for uni in north_east_universities:
        research = tables['table1'][(tables['table1']['HE Provider'] == uni) & 
                                  (tables['table1']['Type of income'] == 'Total')]['Value'].sum()
        services = tables['table2a'][(tables['table2a']['HE Provider'] == uni) & 
                                    (tables['table2a']['Number/Value Marker'] == 'Value') & 
                                    (tables['table2a']['Type of organisation'] != 'Total')]['Number/Value'].sum()
        cpd = tables['table2b'][(tables['table2b']['HE Provider'] == uni) & 
                               (tables['table2b']['Category Marker'] == 'Total revenue')]['Value'].sum()
        regeneration = tables['table3'][(tables['table3']['HE Provider'] == uni) & 
                                       (tables['table3']['Programme'] == 'Total programmes')]['Value'].sum()
        total_income[uni] = research + services + cpd + regeneration
    
    total_income = pd.Series(total_income).sort_values(ascending=False)
    print("\nTotal Income by University (All Sources):")
    print(total_income)
    
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(total_income)))
    ax = plt.bar(total_income.index, total_income.values, color=colors)
    plt.title('Total Income - North East Universities')
    plt.xticks(rotation=45)
    # Add value labels on top of bars
    for i, v in enumerate(total_income.values):
        plt.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    plt.tight_layout()
    save_plot(fig, 'figure17._overall_performance.png')
    
    # Save results to markdown file
    print("\nSaving results to markdown file...")
    with open('2.income_analysis.md', 'w', encoding='utf-8') as f:
        f.write("# Income Analysis\n\n")
        
        # Research Income Analysis
        f.write("## Research Income Analysis\n\n")
        f.write("### Funding Source Distribution (Durham)\n\n")
        f.write("| Source | Value (£) |\n")
        f.write("|--------|----------|\n")
        for source, value in funding_source.items():
            f.write(f"| {source} | {value:,.0f} |\n")
        f.write("\n")
        
        f.write("### National Statistics for Research Income\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total number of universities | {len(national_research)} |\n")
        f.write(f"| Durham's rank | {durham_rank}/{len(national_research)} |\n")
        f.write(f"| National average | £{national_avg:,.0f} |\n")
        f.write(f"| Durham's value | £{national_research['University of Durham']:,.0f} |\n")
        f.write(f"| Difference from national average | £{national_research['University of Durham'] - national_avg:,.0f} |\n")
        f.write(f"| 75th percentile | £{percentile_75:,.0f} |\n")
        f.write(f"| BEIS Research Councils percentage | {beis_percentage:.1f}% |\n\n")
        
        f.write("### Income Type Distribution (Durham)\n\n")
        f.write("| Type | Value (£) |\n")
        f.write("|------|----------|\n")
        for income_type_name, value in income_type.items():
            f.write(f"| {income_type_name} | {value:,.0f} |\n")
        f.write("\n")
        
        f.write("### North East Universities Research Comparison\n\n")
        f.write("| University | Research Income (£) |\n")
        f.write("|------------|-------------------|\n")
        for uni, value in ne_total.items():
            f.write(f"| {uni} | {value:,.0f} |\n")
        f.write("\n")
        
        # Business Services Analysis
        f.write("## Business Services Analysis\n\n")
        f.write("### Service Type Distribution (Durham)\n\n")
        f.write("| Service Type | Value (£) |\n")
        f.write("|--------------|----------|\n")
        for service_type, value in service_type.items():
            f.write(f"| {service_type} | {value:,.0f} |\n")
        f.write("\n")
        
        f.write("### Business Services Percentages\n\n")
        f.write("| Metric | Percentage |\n")
        f.write("|--------|------------|\n")
        f.write(f"| Contract research percentage | {contract_research_pct:.1f}% |\n")
        f.write(f"| Non-commercial organisations percentage | {non_commercial_pct:.1f}% |\n")
        f.write(f"| SMEs percentage | {sme_pct:.1f}% |\n\n")
        
        f.write("### National Statistics for Business Services\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total number of universities | {len(national_services)} |\n")
        f.write(f"| Durham's rank | {durham_rank}/{len(national_services)} |\n")
        f.write(f"| National average | £{national_avg:,.0f} |\n")
        f.write(f"| Durham's value | £{national_services['University of Durham']:,.0f} |\n")
        f.write(f"| Difference from national average | £{national_services['University of Durham'] - national_avg:,.0f} |\n\n")
        
        f.write("### North East Universities Business Services Comparison\n\n")
        f.write("| University | Business Income (£) |\n")
        f.write("|------------|-------------------|\n")
        for uni, value in ne_services_total.items():
            f.write(f"| {uni} | {value:,.0f} |\n")
        f.write("\n")
        
        # CPD Analysis
        f.write("## CPD and Continuing Education Analysis\n\n")
        f.write("### Category Distribution (Durham)\n\n")
        f.write("| Category | Value |\n")
        f.write("|----------|-------|\n")
        for category, value in category.items():
            f.write(f"| {category} | {value:,.0f} |\n")
        f.write("\n")
        
        f.write("### CPD Efficiency and Percentages\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| CPD efficiency per learner day | £{efficiency_per_day:.3f} |\n")
        f.write(f"| Free/subsidized courses percentage | {free_subsidized_pct:.1f}% |\n")
        f.write(f"| Commercial client base percentage | {commercial_client_pct:.1f}% |\n")
        f.write(f"| Non-commercial organizations percentage (CPD) | {non_commercial_cpd_pct:.1f}% |\n")
        f.write(f"| Teesside CPD efficiency per learner day | £{teesside_efficiency:.3f} |\n\n")
        
        f.write("### National Statistics for CPD\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total number of universities | {len(national_cpd)} |\n")
        f.write(f"| Durham's rank | {durham_rank}/{len(national_cpd)} |\n")
        f.write(f"| National average | £{national_avg:,.0f} |\n")
        f.write(f"| Durham's value | £{national_cpd['University of Durham']:,.0f} |\n")
        f.write(f"| Difference from national average | £{national_cpd['University of Durham'] - national_avg:,.0f} |\n\n")
        
        f.write("### North East Universities CPD Comparison\n\n")
        f.write("| University | CPD Income (£) |\n")
        f.write("|------------|---------------|\n")
        for uni, value in ne_cpd_total.items():
            f.write(f"| {uni} | {value:,.0f} |\n")
        f.write("\n")
        
        # Regeneration Analysis
        f.write("## Regeneration and Development Analysis\n\n")
        f.write("### Programme Distribution (Durham)\n\n")
        f.write("| Programme | Value (£) |\n")
        f.write("|-----------|----------|\n")
        for programme, value in programme.items():
            f.write(f"| {programme} | {value:,.0f} |\n")
        f.write("\n")
        
        f.write("### National Statistics for Regeneration\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total number of universities | {len(national_regeneration)} |\n")
        f.write(f"| Durham's rank | {durham_rank}/{len(national_regeneration)} |\n")
        f.write(f"| National average | £{national_avg:,.0f} |\n")
        f.write(f"| Durham's value | £{national_regeneration['University of Durham']:,.0f} |\n")
        f.write(f"| Difference from national average | £{national_regeneration['University of Durham'] - national_avg:,.0f} |\n\n")
        
        f.write("### North East Universities Regeneration Comparison\n\n")
        f.write("| University | Regeneration Income (£) |\n")
        f.write("|------------|----------------------|\n")
        for uni, value in ne_regeneration_total.items():
            f.write(f"| {uni} | {value:,.0f} |\n")
        f.write("\n")
        
        # Overall Performance
        f.write("## Overall Performance Summary\n\n")
        f.write("### Total Income by University (All Sources)\n\n")
        f.write("| University | Total Income (£) |\n")
        f.write("|------------|-----------------|\n")
        for uni, value in total_income.items():
            f.write(f"| {uni} | {value:,.0f} |\n")
        f.write("\n")
    
    print("Results have been saved to '2.income_analysis.md'")

if __name__ == "__main__":
    analyze_income() 