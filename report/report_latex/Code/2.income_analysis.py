def analyze_income():

    # 1. Collaborative Research Income Analysis
    durham_data = tables['table1'][tables['table1']['HE Provider'] == 'University of Durham']
    funding_source = durham_data.groupby('Source of public funding')['Value'].sum()
    income_type = durham_data.groupby('Type of income')['Value'].sum()
    yearly_income = durham_data.groupby(['Academic Year', 'Source of public funding'])['Value'].sum().reset_index()
    
    # National statistics for research income
    national_research = tables['table1'].groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_rank = national_research.index.get_loc('University of Durham') + 1
    total_universities = len(national_research)
    national_avg = national_research.mean()
    
    # North East universities research comparison
    ne_research = tables['table1'][tables['table1']['HE Provider'].isin(north_east_universities)]
    ne_total = ne_research.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    
    # 2. Business Services Analysis
    durham_services = tables['table2a'][tables['table2a']['HE Provider'] == 'University of Durham']
    service_type = durham_services.groupby('Type of service')['Number/Value'].sum()
    org_type = durham_services.groupby('Type of organisation')['Number/Value'].sum()
    metric_type = durham_services.groupby('Number/Value Marker')['Number/Value'].sum()
    
    # National statistics for business services
    national_services = tables['table2a'].groupby('HE Provider')['Number/Value'].sum().sort_values(ascending=False)
    durham_services_rank = national_services.index.get_loc('University of Durham') + 1
    national_services_avg = national_services.mean()
    
    # North East universities business services comparison
    ne_services = tables['table2a'][tables['table2a']['HE Provider'].isin(north_east_universities)]
    ne_services_total = ne_services.groupby('HE Provider')['Number/Value'].sum().sort_values(ascending=False)
    
    # 3. CPD and Continuing Education Analysis
    durham_cpd = tables['table2b'][tables['table2b']['HE Provider'] == 'University of Durham']
    category = durham_cpd.groupby('Category Marker')['Value'].sum()
    unit_type = durham_cpd.groupby('Unit')['Value'].sum()
    
    # National statistics for CPD
    national_cpd = tables['table2b'].groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_cpd_rank = national_cpd.index.get_loc('University of Durham') + 1
    national_cpd_avg = national_cpd.mean()
    
    # North East universities CPD comparison
    ne_cpd = tables['table2b'][tables['table2b']['HE Provider'].isin(north_east_universities)]
    ne_cpd_total = ne_cpd.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    
    # 4. Regeneration and Development Analysis
    durham_regeneration = tables['table3'][tables['table3']['HE Provider'] == 'University of Durham']
    programme = durham_regeneration.groupby('Programme')['Value'].sum()
    yearly_regeneration = durham_regeneration.groupby(['Academic Year', 'Programme'])['Value'].sum().reset_index()
    
    # National statistics for regeneration and development
    national_regeneration = tables['table3'].groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    durham_regeneration_rank = national_regeneration.index.get_loc('University of Durham') + 1
    national_regeneration_avg = national_regeneration.mean()
    
    # North East universities regeneration comparison
    ne_regeneration = tables['table3'][tables['table3']['HE Provider'].isin(north_east_universities)]
    ne_regeneration_total = ne_regeneration.groupby('HE Provider')['Value'].sum().sort_values(ascending=False)
    
    # 5. Overall Performance Summary
    total_income = {}
    for uni in north_east_universities:
        research = tables['table1'][tables['table1']['HE Provider'] == uni]['Value'].sum()
        services = tables['table2a'][tables['table2a']['HE Provider'] == uni]['Number/Value'].sum()
        cpd = tables['table2b'][tables['table2b']['HE Provider'] == uni]['Value'].sum()
        regeneration = tables['table3'][tables['table3']['HE Provider'] == uni]['Value'].sum()
        total_income[uni] = research + services + cpd + regeneration
    
    total_income = pd.Series(total_income).sort_values(ascending=False)
    
    # Compile analysis results
    analysis_results = {
        'research_income': {
            'durham_funding_source': funding_source.to_dict(),
            'durham_income_type': income_type.to_dict(),
            'durham_yearly_trend': yearly_income.to_dict('records'),
            'national_rank': durham_rank,
            'national_average': national_avg,
            'north_east_comparison': ne_total.to_dict()
        },
        'business_services': {
            'durham_service_type': service_type.to_dict(),
            'durham_org_type': org_type.to_dict(),
            'durham_metric_type': metric_type.to_dict(),
            'national_rank': durham_services_rank,
            'national_average': national_services_avg,
            'north_east_comparison': ne_services_total.to_dict()
        },
        'cpd_education': {
            'durham_category': category.to_dict(),
            'durham_unit_type': unit_type.to_dict(),
            'national_rank': durham_cpd_rank,
            'national_average': national_cpd_avg,
            'north_east_comparison': ne_cpd_total.to_dict()
        },
        'regeneration_development': {
            'durham_programme': programme.to_dict(),
            'durham_yearly_trend': yearly_regeneration.to_dict('records'),
            'national_rank': durham_regeneration_rank,
            'national_average': national_regeneration_avg,
            'north_east_comparison': ne_regeneration_total.to_dict()
        },
        'overall_performance': {
            'total_income_by_university': total_income.to_dict()
        }
    }
    
    return analysis_results
