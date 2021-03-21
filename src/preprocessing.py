import pandas as pd
import numpy as np
import missingno as msno

orders_df = pd.read_csv('../data/hackathon_order.csv')
orders_fix_df = pd.read_csv('../data/hackathon_order_fix2.csv', error_bad_lines=False)
comments = orders_df['comment']

orders_fix_df.adult.value_counts()
# msno.matrix(orders_fix_df)
# orders_df['email'].value_counts()
# orders_df['address'].value_counts()
# for column in orders_df.columns:
#     print(orders_df[column].value_counts())
