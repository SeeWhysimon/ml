import pandas as pd

s = pd.date_range('2024-08-14', '2024-08-15', freq='3h')

df = pd.DataFrame({
    'dayofweek': s.dayofweek,
    'dayofyear': s.dayofyear,
    'hour': s.hour,
    'is_leap_year': s.is_leap_year,
    'quarter': s.quarter,
    'weekofyear': s.isocalendar().week
})

features = {
    'dayofweek': df['dayofweek'].values,
    'dayofyear': df['dayofyear'].values,
    'hour': df['hour'].values,
    'is_leap_year': df['is_leap_year'].values,
    'quarter': df['quarter'].values,
    'weekofyear': df['weekofyear'].values
}

print(features)