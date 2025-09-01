def analyze_ip():
    
    # 1. IP Disclosures and Patents Analysis
    durham_ip = tables['table4a'][tables['table4a']['HE Provider'] == 'University of Durham']
    disclosure_type = durham_ip.groupby('Type of disclosure or patent')['Value'].sum()
    yearly_disclosures = durham_ip.groupby(['Academic Year', 'Type of disclosure or patent'])['Value'].sum().reset_index()
    
    # National statistics for IP disclosures
    national_ip = tables['table4a'].groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_ip_rank = national_ip.index.get_loc('University of Durham') + 1
    national_ip_avg = national_ip.mean()
    
    # North East universities IP comparison
    ne_ip = tables['table4a'][tables['table4a']['HE Provider'].isin(north_east_universities)]
    ne_total = ne_ip.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    
    # 2. License Analysis
    durham_license = tables['table4b'][tables['table4b']['HE Provider'] == 'University of Durham']
    license_type = durham_license.groupby('Type of licence granted')['Value'].sum()
    org_type_license = durham_license.groupby('Type of organisation')['Value'].sum()
    
    # National statistics for licenses
    national_license = tables['table4b'].groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_license_rank = national_license.index.get_loc('University of Durham') + 1
    national_license_avg = national_license.mean()
    
    # North East universities license comparison
    ne_license = tables['table4b'][tables['table4b']['HE Provider'].isin(north_east_universities)]
    ne_license_total = ne_license.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    
    # 3. IP Income Analysis
    durham_income = pd.concat([
        tables['table4c'][tables['table4c']['HE Provider'] == 'University of Durham'],
        tables['table4d'][tables['table4d']['HE Provider'] == 'University of Durham']
    ])
    income_source = durham_income.groupby('Income source')['Value'].sum()
    org_type_income = durham_income.groupby('Type of organisation')['Value'].sum()
    
    # National statistics for IP income
    national_income = pd.concat([
        tables['table4c'],
        tables['table4d']
    ]).groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_income_rank = national_income.index.get_loc('University of Durham') + 1
    national_income_avg = national_income.mean()
    
    # North East universities IP income comparison
    ne_income = pd.concat([
        tables['table4c'][tables['table4c']['HE Provider'].isin(north_east_universities)],
        tables['table4d'][tables['table4d']['HE Provider'].isin(north_east_universities)]
    ])
    ne_income_total = ne_income.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    
    # 4. Spin-off Company Analysis
    durham_spin = tables['table4e'][tables['table4e']['HE Provider'] == 'University of Durham']
    metric_type = durham_spin.groupby('Metric')['Value'].sum()
    category = durham_spin.groupby('Category Marker')['Value'].sum()
    
    # National statistics for spin-off companies
    national_spin = tables['table4e'].groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_spin_rank = national_spin.index.get_loc('University of Durham') + 1
    national_spin_avg = national_spin.mean()
    
    # North East universities spin-off comparison
    ne_spin = tables['table4e'][tables['table4e']['HE Provider'].isin(north_east_universities)]
    ne_spin_total = ne_spin.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    
    # Compile analysis results
    analysis_results = {
        'ip_disclosures': {
            'durham_disclosure_type': disclosure_type.to_dict(),
            'durham_yearly_trend': yearly_disclosures.to_dict('records'),
            'national_rank': durham_ip_rank,
            'national_average': national_ip_avg,
            'north_east_comparison': ne_total.to_dict()
        },
        'licenses': {
            'durham_license_type': license_type.to_dict(),
            'durham_org_type': org_type_license.to_dict(),
            'national_rank': durham_license_rank,
            'national_average': national_license_avg,
            'north_east_comparison': ne_license_total.to_dict()
        },
        'ip_income': {
            'durham_income_source': income_source.to_dict(),
            'durham_org_type': org_type_income.to_dict(),
            'national_rank': durham_income_rank,
            'national_average': national_income_avg,
            'north_east_comparison': ne_income_total.to_dict()
        },
        'spin_off_companies': {
            'durham_metric_type': metric_type.to_dict(),
            'durham_category': category.to_dict(),
            'national_rank': durham_spin_rank,
            'national_average': national_spin_avg,
            'north_east_comparison': ne_spin_total.to_dict()
        }
    }
    
    return analysis_results
