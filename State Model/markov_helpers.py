import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
from sklearn.utils.class_weight import compute_sample_weight
import seaborn as sns
import joblib
import os
from pandas.tseries.offsets import DateOffset


def add_next_state(df, months_forward=12):
    """
    Adds a 'next_state' column based on the most severe state that occurs
    in the future months_forward period after each row's reporting_period.

    Priority: Foreclosure > Default > Prepaid > Current

    Parameters:
        df (pd.DataFrame): Must include ['Loan Number', 'reporting_period', 'state']
        months_forward (int): How far ahead (in months) to look for state changes

    Returns:
        pd.DataFrame: Original DataFrame with added 'next_state' column
    """
    df = df.copy()
    df['Reporting Period Date'] = pd.to_datetime(df['Reporting Period Date'], errors='coerce')
    df = df.sort_values(by=['Loan Number', 'Reporting Period Date'])

    priority_order = ['Foreclosure', 'Default', 'Prepaid', 'Current']

    # Group by loan
    grouped = df.groupby('Loan Number')
    next_states = []

    for _, group in grouped:
        group = group.reset_index(drop=True)
        dates = group['Reporting Period Date']
        states = group['state']

        for i, this_date in enumerate(dates):
            window_end = this_date + DateOffset(months=months_forward)
            future_mask = (dates > this_date) & (dates <= window_end)
            future_states = states[future_mask].dropna().values

            # Start with the current state as fallback
            current_state = states[i]
            assigned_state = current_state

            for candidate in priority_order:
                if candidate in future_states:
                    assigned_state = candidate
                    break

            next_states.append(assigned_state)

    df['Next State'] = next_states
    return df

def encode_col(df, column, mapping_dict, new_column):
    """
    Maps values in a specified column to numeric values using a dictionary,
    and stores the result in a new column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): Name of the column to encode (e.g. 'state').
        mapping_dict (dict): Dictionary mapping string values to numbers.
                             e.g. {'Current': 0, 'Default': 1, 'Foreclosure': 2}
        new_column (str): Name of the new column to store the encoded values.

    Returns:
        pd.DataFrame: DataFrame with the new encoded column added.
    """
    df = df.copy()
    df[new_column] = df[column].map(mapping_dict)
    return df