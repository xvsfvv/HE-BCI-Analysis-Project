
def calculate_correlations(data):
    """Calculate correlations between different metrics"""
    # Prepare metrics for correlation analysis
    metrics = {}

    # Research income
    metrics['Research Income'] = data['research'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()

    # Business services income
    metrics['Business Income'] = data['business'].groupby(['HE Provider', 'Academic Year'])[
        data['business'].columns[-1]].sum()

    # CPD income
    metrics['CPD Income'] = data['cpd'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()

    # Regeneration income
    metrics['Regeneration Income'] = data['regeneration'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()

    # IP metrics
    metrics['IP Disclosures'] = data['ip_disclosures'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()
    metrics['IP Licenses'] = data['ip_licenses'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()
    metrics['IP Income'] = data['ip_income'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()

    # Spin-out employment
    metrics['Spin-out Employment'] = data['spinouts'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()

    # Public engagement
    metrics['Public Engagement'] = data['public_engagement'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()

    # Create correlation matrix
    corr_matrix = pd.DataFrame(metrics).corr()
    
    # Get top correlations
    corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
    corr_pairs = corr_pairs[corr_pairs != 1.0]  # Remove self-correlations
    
    return corr_matrix, corr_pairs.head()

def analyze_rankings(data):
    """Analyze ranking changes over time"""
    metrics = {}
    metrics['Research Income'] = data['research'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()
    metrics['Business Income'] = data['business'].groupby(['HE Provider', 'Academic Year'])[
        data['business'].columns[-1]].sum()
    metrics['IP Disclosures'] = data['ip_disclosures'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()
    metrics['IP Income'] = data['ip_income'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()
    metrics['Public Engagement'] = data['public_engagement'].groupby(['HE Provider', 'Academic Year'])['Value'].sum()
    
    rankings = {}
    for metric_name, metric_data in metrics.items():
        # Calculate rank for each year
        ranking = metric_data.groupby('Academic Year', group_keys=False).rank(ascending=False, method='min')
        ranking_df = ranking.to_frame(name='Rank').reset_index()
        ranking_pivot = ranking_df.pivot(index='Academic Year', columns='HE Provider', values='Rank')
        rankings[metric_name] = ranking_pivot
    
    return rankings

def analyze_specialization(data):
    """Analyze specialization patterns using HHI"""
    specialization = {}
    
    research_by_source = data['research'].pivot_table(
        index=['HE Provider', 'Academic Year'],
        columns='Type of income',
        values='Value',
        aggfunc='sum'
    ).fillna(0)
    
    business_by_type = data['business'].pivot_table(
        index=['HE Provider', 'Academic Year'],
        columns='Type of service',
        values=data['business'].columns[-1],
        aggfunc='sum'
    ).fillna(0)
    
    ip_by_type = data['ip_disclosures'].pivot_table(
        index=['HE Provider', 'Academic Year'],
        columns='Type of disclosure or patent',
        values='Value',
        aggfunc='sum'
    ).fillna(0)
    
    engagement_by_type = data['public_engagement'].pivot_table(
        index=['HE Provider', 'Academic Year'],
        columns='Type of event',
        values='Value',
        aggfunc='sum'
    ).fillna(0)

    def calculate_hhi(df):
        shares = df.div(df.sum(axis=1), axis=0)
        return (shares ** 2).sum(axis=1)

    specialization['Research'] = calculate_hhi(research_by_source).unstack('HE Provider')
    specialization['Business'] = calculate_hhi(business_by_type).unstack('HE Provider')
    specialization['IP'] = calculate_hhi(ip_by_type).unstack('HE Provider')
    specialization['Engagement'] = calculate_hhi(engagement_by_type).unstack('HE Provider')
    
    return specialization

