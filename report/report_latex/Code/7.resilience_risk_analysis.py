
def calculate_income_volatility(data, ne_universities):
    """Calculate income volatility metrics for all universities"""
    # Prepare volatility metrics
    volatility_data = {}

    # 1. Research Income Volatility
    research_by_univ = data['research'].groupby(['HE Provider', 'Academic Year'])['Value'].sum().reset_index()
    
    # Calculate volatility (coefficient of variation) for each university
    research_volatility = research_by_univ.groupby('HE Provider')['Value'].agg(['mean', 'std']).reset_index()
    research_volatility['cv'] = research_volatility['std'] / research_volatility['mean']
    research_volatility = research_volatility.sort_values('cv', ascending=True)  # Lower CV = more stable
    
    volatility_data['Research'] = research_volatility

    # 2. Business Services Volatility
    business_by_univ = data['business'].groupby(['HE Provider', 'Academic Year'])[data['business'].columns[-1]].sum().reset_index()
    
    business_volatility = business_by_univ.groupby('HE Provider')[data['business'].columns[-1]].agg(['mean', 'std']).reset_index()
    business_volatility['cv'] = business_volatility['std'] / business_volatility['mean']
    business_volatility = business_volatility.sort_values('cv', ascending=True)
    
    volatility_data['Business'] = business_volatility

    # 3. CPD Income Volatility
    cpd_by_univ = data['cpd'].groupby(['HE Provider', 'Academic Year'])['Value'].sum().reset_index()
    
    cpd_volatility = cpd_by_univ.groupby('HE Provider')['Value'].agg(['mean', 'std']).reset_index()
    cpd_volatility['cv'] = cpd_volatility['std'] / cpd_volatility['mean']
    cpd_volatility = cpd_volatility.sort_values('cv', ascending=True)
    
    volatility_data['CPD'] = cpd_volatility

    # 4. IP Income Volatility
    ip_by_univ = data['ip_income'].groupby(['HE Provider', 'Academic Year'])['Value'].sum().reset_index()
    
    ip_volatility = ip_by_univ.groupby('HE Provider')['Value'].agg(['mean', 'std']).reset_index()
    ip_volatility['cv'] = ip_volatility['std'] / ip_volatility['mean']
    ip_volatility = ip_volatility.sort_values('cv', ascending=True)
    
    volatility_data['IP'] = ip_volatility

    return volatility_data

def analyze_income_concentration(data, ne_universities):
    """Analyze income concentration and diversification"""
    concentration_data = {}

    # Calculate income concentration for each university
    for univ in ne_universities:
        # Get all income sources for this university
        univ_income = {}
        
        # Research income by source
        research_data = data['research'][data['research']['HE Provider'] == univ]
        if len(research_data) > 0:
            research_by_source = research_data.groupby('Source of public funding')['Value'].sum()
            univ_income['Research'] = research_by_source
        
        # Business income by type
        business_data = data['business'][data['business']['HE Provider'] == univ]
        if len(business_data) > 0:
            business_by_type = business_data.groupby('Type of service')[data['business'].columns[-1]].sum()
            univ_income['Business'] = business_by_type
        
        # CPD income
        cpd_data = data['cpd'][data['cpd']['HE Provider'] == univ]
        if len(cpd_data) > 0:
            cpd_total = cpd_data['Value'].sum()
            univ_income['CPD'] = pd.Series([cpd_total], index=['CPD Total'])
        
        # IP income
        ip_data = data['ip_income'][data['ip_income']['HE Provider'] == univ]
        if len(ip_data) > 0:
            ip_total = ip_data['Value'].sum()
            univ_income['IP'] = pd.Series([ip_total], index=['IP Total'])
        
        # Regeneration income
        regen_data = data['regeneration'][data['regeneration']['HE Provider'] == univ]
        if len(regen_data) > 0:
            regen_total = regen_data['Value'].sum()
            univ_income['Regeneration'] = pd.Series([regen_total], index=['Regeneration Total'])
        
        concentration_data[univ] = univ_income
        
        # Calculate concentration metrics
        total_income = sum([income.sum() for income in univ_income.values()])
        if total_income > 0:
            # Calculate HHI for income concentration
            shares = []
            for category, income in univ_income.items():
                for source, value in income.items():
                    if value > 0:
                        shares.append((value / total_income) ** 2)
            
            hhi = sum(shares)
            max_source_share = max([income.max() / total_income for income in univ_income.values() if income.sum() > 0])
            
            concentration_data[univ]['metrics'] = {
                'total_income': total_income,
                'hhi_index': hhi,
                'max_source_share': max_source_share,
                'num_income_sources': len([s for s in shares if s > 0])
            }

    return concentration_data

def analyze_ne_volatility(volatility_data, ne_universities):
    """Analyze volatility within North East universities"""
    # Calculate volatility rankings for NE universities
    ne_volatility_summary = {}
    
    for metric_name, data in volatility_data.items():
        ne_data = data[data['HE Provider'].isin(ne_universities)].copy()
        
        # Sort by CV (lower is more stable)
        ne_data = ne_data.sort_values('cv')
        ne_volatility_summary[metric_name] = ne_data

    return ne_volatility_summary

def analyze_national_volatility_ranking(volatility_data, ne_universities):
    """Analyze national volatility rankings"""
    national_rankings = {}
    
    for metric_name, data in volatility_data.items():
        # Find NE universities' rankings
        ne_rankings = {}
        for univ in ne_universities:
            if univ in data['HE Provider'].values:
                univ_data = data[data['HE Provider'] == univ].iloc[0]
                rank = data.index.get_loc(data[data['HE Provider'] == univ].index[0]) + 1
                cv = univ_data['cv']
                ne_rankings[univ] = {'rank': rank, 'cv': cv}
        
        national_rankings[metric_name] = ne_rankings

    return national_rankings

def analyze_volatility_trends(data, ne_universities):
    """Analyze volatility trends over time"""
    # Create summary tables for each metric
    trend_summary = {}
    
    for metric_name, table_name in [('Research', 'research'), ('Business', 'business'), 
                                   ('CPD', 'cpd'), ('IP', 'ip_income')]:
        
        metric_data = data[table_name]
        if table_name == 'business':
            value_col = metric_data.columns[-1]
        else:
            value_col = 'Value'
        
        # Create summary table for this metric
        summary_data = []
        
        for univ in ne_universities:
            # For CPD data, we need to aggregate by year
            if metric_name == 'CPD':
                univ_data = metric_data[metric_data['HE Provider'] == univ]
                if len(univ_data) > 0:
                    univ_data = univ_data.groupby('Academic Year')['Value'].sum().reset_index()
                    univ_data = univ_data.sort_values('Academic Year')
            else:
                univ_data = metric_data[metric_data['HE Provider'] == univ]
                if len(univ_data) > 0:
                    univ_data = univ_data.sort_values('Academic Year')
            
            if len(univ_data) > 0:
                # Calculate basic statistics
                total_income = univ_data[value_col].sum()
                avg_income = univ_data[value_col].mean()
                min_income = univ_data[value_col].min()
                max_income = univ_data[value_col].max()
                
                # Calculate year-over-year changes
                univ_data['pct_change'] = univ_data[value_col].pct_change() * 100
                valid_changes = univ_data['pct_change'].dropna()
                
                if len(valid_changes) > 0:
                    # Handle infinite values
                    valid_changes_clean = valid_changes.replace([np.inf, -np.inf], np.nan)
                    non_nan_changes = valid_changes_clean.dropna()
                    
                    if len(non_nan_changes) > 0:
                        avg_change = non_nan_changes.mean()
                        max_increase = non_nan_changes.max()
                        max_decrease = non_nan_changes.min()
                        volatility = non_nan_changes.std()
                    else:
                        avg_change = max_increase = max_decrease = volatility = 0.0
                else:
                    avg_change = max_increase = max_decrease = volatility = np.nan
                
                # Calculate trend if enough data points
                if len(valid_changes) > 1:
                    years = range(len(valid_changes))
                    changes = valid_changes.values
                    trend = np.polyfit(years, changes, 1)[0]
                    trend_summary[f"{metric_name}_{univ}"] = trend
                else:
                    trend = np.nan
                
                summary_data.append({
                    'University': univ,
                    'Total Income': total_income,
                    'Avg Income': avg_income,
                    'Min Income': min_income,
                    'Max Income': max_income,
                    'Avg Change': avg_change,
                    'Max Increase': max_increase,
                    'Max Decrease': max_decrease,
                    'Volatility': volatility,
                    'Trend': trend
                })

    return trend_summary
