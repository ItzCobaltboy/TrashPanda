import pandas as pd
import holidays

def add_holiday_and_vacation_flag(df, date_column="timestamp", country='IN',
                                   vacation_threshold=10, min_vacation_days=3):
    """
    Adds two columns:
    - 'is_holiday': 1 if date is a public holiday
    - 'is_vacation': 1 if bin is unused (low fill) for consecutive days

    Parameters:
    - df: DataFrame with ['timestamp', 'trashcan_id', 'fill_level']
    - vacation_threshold: Max fill % to consider bin "inactive"
    - min_vacation_days: Min consecutive inactive days to call it vacation

    Returns:
    - DataFrame with new columns
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Public holidays
    years = df[date_column].dt.year.unique()
    holiday_map = {}
    for year in years:
        holiday_map.update(holidays.country_holidays(country, years=year))
    df['is_holiday'] = df[date_column].dt.date.apply(lambda x: 1 if x in holiday_map else 0)

    # Vacation detection
    df['is_vacation'] = 0
    for bin_id in df['trashcan_id'].unique():
        bin_data = df[df['trashcan_id'] == bin_id].sort_values(date_column)
        low_fill = bin_data['fill_level'] <= vacation_threshold
        streak = 0

        for idx in bin_data.index:
            if low_fill.loc[idx]:
                streak += 1
            else:
                streak = 0

            if streak >= min_vacation_days:
                df.loc[idx, 'is_vacation'] = 1

    return df
