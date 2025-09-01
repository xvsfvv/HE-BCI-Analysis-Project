def calculate_efficiency_metrics(data, ne_universities):
    """Calculate efficiency metrics for all universities"""
    # Prepare efficiency metrics
    efficiency_data = {}

    # 1. Research Income Efficiency (per unit of staff time)
    research_by_univ = data['research'].groupby(['HE Provider', 'Academic Year'])['Value'].sum().reset_index()
    
    # Get staff time from public engagement data
    staff_time = data['public_engagement'][data['public_engagement']['Metric'] == 'Academic staff time (days)']
    staff_time_by_univ = staff_time.groupby(['HE Provider', 'Academic Year'])['Value'].sum().reset_index()
    
    # Merge research income with staff time
    research_efficiency = research_by_univ.merge(staff_time_by_univ, 
                                                on=['HE Provider', 'Academic Year'], 
                                                how='left', 
                                                suffixes=('_income', '_staff_time'))
    
    # Calculate efficiency (income per staff day)
    research_efficiency['efficiency'] = research_efficiency['Value_income'] / research_efficiency['Value_staff_time']
    research_efficiency = research_efficiency.replace([np.inf, -np.inf], np.nan)
    
    efficiency_data['Research'] = research_efficiency

    # 2. Business Services Efficiency
    business_by_univ = data['business'].groupby(['HE Provider', 'Academic Year'])[data['business'].columns[-1]].sum().reset_index()
    business_efficiency = business_by_univ.merge(staff_time_by_univ, 
                                               on=['HE Provider', 'Academic Year'], 
                                               how='left')
    business_efficiency['efficiency'] = business_efficiency[data['business'].columns[-1]] / business_efficiency['Value']
    business_efficiency = business_efficiency.replace([np.inf, -np.inf], np.nan)
    
    efficiency_data['Business'] = business_efficiency

    # 3. IP Income Efficiency
    ip_by_univ = data['ip_income'].groupby(['HE Provider', 'Academic Year'])['Value'].sum().reset_index()
    ip_efficiency = ip_by_univ.merge(staff_time_by_univ, 
                                    on=['HE Provider', 'Academic Year'], 
                                    how='left')
    ip_efficiency['efficiency'] = ip_efficiency['Value_x'] / ip_efficiency['Value_y']
    ip_efficiency = ip_efficiency.replace([np.inf, -np.inf], np.nan)
    
    efficiency_data['IP'] = ip_efficiency

    return efficiency_data

def analyze_ne_efficiency(efficiency_data, ne_universities):
    """Analyze efficiency within North East universities"""
    # Calculate average efficiency for each university across all metrics
    ne_efficiency_summary = {}
    
    for metric_name, data in efficiency_data.items():
        ne_data = data[data['HE Provider'].isin(ne_universities)]
        
        # Calculate average efficiency by university
        avg_efficiency = ne_data.groupby('HE Provider')['efficiency'].mean().sort_values(ascending=False)
        ne_efficiency_summary[metric_name] = avg_efficiency

    return ne_efficiency_summary

def analyze_national_efficiency_ranking(efficiency_data, ne_universities):
    """Analyze national efficiency rankings"""
    national_rankings = {}
    
    for metric_name, data in efficiency_data.items():
        # Calculate average efficiency for all universities
        avg_efficiency = data.groupby('HE Provider')['efficiency'].mean().sort_values(ascending=False)
        
        # Find NE universities' rankings
        ne_rankings = {}
        for univ in ne_universities:
            if univ in avg_efficiency.index:
                rank = avg_efficiency.index.get_loc(univ) + 1
                efficiency = avg_efficiency[univ]
                ne_rankings[univ] = {'rank': rank, 'efficiency': efficiency}
        
        national_rankings[metric_name] = ne_rankings

    return national_rankings

def analyze_efficiency_trends(efficiency_data, ne_universities):
    """Analyze efficiency trends over time"""
    # Calculate trend statistics
    trend_summary = {}
    for metric_name, data in efficiency_data.items():
        ne_data = data[data['HE Provider'].isin(ne_universities)]
        
        for univ in ne_universities:
            univ_data = ne_data[ne_data['HE Provider'] == univ]
            if len(univ_data) > 1:
                # Calculate trend (simple linear trend)
                years = range(len(univ_data))
                efficiencies = univ_data['efficiency'].values
                if not np.all(np.isnan(efficiencies)):
                    trend = np.polyfit(years, efficiencies, 1)[0]
                    trend_summary[f"{metric_name}_{univ}"] = trend

    return trend_summary
