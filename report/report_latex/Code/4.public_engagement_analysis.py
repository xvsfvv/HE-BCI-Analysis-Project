def analyze_public_engagement():
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Filter for Durham University
    durham_data = df[df['HE Provider'] == 'University of Durham']
    
    # 1. Analysis by Nature of Event
    nature_dist = durham_data.groupby('Nature of Event')['Value'].sum()
    
    # Calculate national statistics
    total_universities = df['HE Provider'].nunique()
    durham_total = nature_dist.sum()
    national_avg = df.groupby('HE Provider')['Value'].sum().mean()
    durham_rank = (df.groupby('HE Provider')['Value'].sum() > durham_total).sum() + 1
    
    # 2. Analysis by Type of Event
    type_dist = durham_data.groupby('Type of event')['Value'].sum()
    
    # 3. Attendees Analysis
    attendees_data = durham_data[durham_data['Metric'] == 'Attendees']
    attendees_by_type = attendees_data.groupby('Type of event')['Value'].sum()
    
    # 4. Academic Staff Time Analysis
    staff_time_data = durham_data[durham_data['Metric'] == 'Academic staff time (days)']
    staff_time_by_type = staff_time_data.groupby('Type of event')['Value'].sum()
    
    # 5. North East Universities Comparison
    ne_universities = ['University of Durham', 'Newcastle University', 
                      'University of Northumbria at Newcastle', 'Teesside University',
                      'The University of Sunderland']
    ne_data = df[df['HE Provider'].isin(ne_universities)]
    ne_totals = ne_data.groupby('HE Provider')['Value'].sum()
    
    # Compile analysis results
    analysis_results = {
        'nature_of_event': {
            'durham_distribution': nature_dist.to_dict(),
            'national_rank': durham_rank,
            'total_universities': total_universities,
            'national_average': national_avg,
            'durham_total': durham_total
        },
        'type_of_event': {
            'durham_distribution': type_dist.to_dict()
        },
        'attendees': {
            'by_event_type': attendees_by_type.to_dict()
        },
        'academic_staff_time': {
            'by_event_type': staff_time_by_type.to_dict()
        },
        'north_east_comparison': {
            'university_totals': ne_totals.to_dict()
        }
    }
    
    return analysis_results
