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
from sklearn.metrics import classification_report



# def add_next_state(df, months_forward=12):
#     """
#     Adds a 'next_state' column based on the most severe state that occurs
#     in the future months_forward period after each row's reporting_period.

#     Priority: Foreclosure > Default > Prepaid > Current

#     Parameters:
#         df (pd.DataFrame): Must include ['Loan Number', 'reporting_period', 'state']
#         months_forward (int): How far ahead (in months) to look for state changes

#     Returns:
#         pd.DataFrame: Original DataFrame with added 'next_state' column
#     """
#     df = df.copy()
#     df['Reporting Period Date'] = pd.to_datetime(df['Reporting Period Date'], errors='coerce')
#     df = df.sort_values(by=['Loan Number', 'Reporting Period Date'])

#     priority_order = ['Foreclosure', 'Fully Paid, Matured', 'Fully Paid, Prepaid', 'Default', 'Current']

#     # Group by loan
#     grouped = df.groupby('Loan Number')
#     next_states = []

#     for _, group in grouped:
#         group = group.reset_index(drop=True)
#         dates = group['Reporting Period Date']
#         states = group['State']

#         for i, this_date in enumerate(dates):
#             window_end = this_date + DateOffset(months=months_forward)
#             future_mask = (dates > this_date) & (dates <= window_end)
#             future_states = states[future_mask].dropna().values

#             # Start with the current state as fallback
#             current_state = states[i]
#             assigned_state = current_state

#             for candidate in priority_order:
#                 if candidate in future_states:
#                     assigned_state = candidate
#                     break

#             next_states.append(assigned_state)

#     df['Next State'] = next_states
#     return df

def add_next_state(df, months_forward = 12):
    """
    Vectorised, faster version of add_next_state with the IndexError fixed.
    """
    out = df.copy()
    out['Reporting Period Date'] = pd.to_datetime(
        out['Reporting Period Date'], errors='coerce'
    )
    out.sort_values(['Loan Number', 'Reporting Period Date'], inplace=True)

    priority = ['Foreclosure',
                'Fully Paid, Matured',
                'Fully Paid, Prepaid',
                'Default',
                'Current']

    # integer “year-month” representation
    month_no = (out['Reporting Period Date'].dt.year * 12
                + out['Reporting Period Date'].dt.month).to_numpy()

    next_state_all = np.empty(len(out), dtype=object)

    start = 0
    for _, grp in out.groupby('Loan Number', sort=False):
        n = len(grp)
        idx = slice(start, start + n)

        states = grp['State'].to_numpy()
        months = month_no[idx]

        candidate_flags = np.zeros((len(priority), n), dtype=bool)

        for p, cand in enumerate(priority):
            is_cand = states == cand

            next_idx = np.full(n, n, dtype=int)
            current_next = n
            for i in range(n - 1, -1, -1):
                next_idx[i] = current_next
                if is_cand[i]:
                    current_next = i

            # --- SAFE difference calculation (avoids out-of-bounds) ----------
            valid = next_idx < n
            within = np.zeros(n, dtype=bool)
            within[valid] = (months[next_idx[valid]] - months[valid]) <= months_forward
            candidate_flags[p] = within

        chosen = np.empty(n, dtype=object)
        chosen[:] = None
        for p, cand in enumerate(priority):
            mask = (chosen == None) & candidate_flags[p]
            chosen[mask] = cand

        # fall back to current state
        mask_none = chosen == None
        chosen[mask_none] = states[mask_none]

        next_state_all[idx] = chosen
        start += n

    out['Next State'] = next_state_all
    return out

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

def downsample(df, n=750):
    df = df.copy()
    random_loans = df['Loan Number'].dropna().unique()
    random_loans = pd.Series(random_loans).sample(n, random_state=42)  # optional random_state for reproducibility
    df_random = df[df['Loan Number'].isin(random_loans)]
    return df_random

def train_model(X, y, X_train, y_train, X_test, y_test, le_state):
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train) #Punishing for getting wrong the more rare classes


    model = XGBClassifier(
        objective='multi:softprob',
        num_class=len(y.unique()),
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_jobs=-1,
        max_depth=6,             # ↓ slightly, avoids overfitting rare paths
        n_estimators=700,        # ↑ if using lower learning rate
        learning_rate=0.05,      # ↓ improves generalization
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,                 # ↑ only split if meaningful gain
        min_child_weight=10      # ↑ avoids overfitting noisy splits
    )
    '''

    model = XGBClassifier(
        objective='multi:softprob',  # returns full probability distribution
        num_class=len(y.unique()),   # number of distinct loan states
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_jobs=-1                    # parallel training
    )
    '''
    print(type(sample_weights))
    print(sample_weights.shape if hasattr(sample_weights, 'shape') else len(sample_weights))
    print(sample_weights[:5])

    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le_state.classes_))

    print("Feature Importances:")
    print(model.feature_importances_)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=le_state.classes_, yticklabels=le_state.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    plt.show()

    # 2. Classification Report
    report = classification_report(y_test, y_pred, target_names=le_state.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("Classification Report:")
    print(report_df.round(3))

    # 3. Feature Importance Plot
    importances = model.feature_importances_
    features = X.columns
    feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=feat_imp_df.head(20), x='Importance', y='Feature')
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.show()

    # 4. Transition-level Accuracy Matrix (if you have State and Next State columns in your dataset)
    # Optional: if you originally had these columns for each row
    if 'State' in X_test.columns and 'Next State' in y_test.index.names:
        actual_transitions = pd.DataFrame({
            'State': X_test['State'].values,
            'Next State True': le_state.inverse_transform(y_test),
            'Next State Pred': le_state.inverse_transform(y_pred)
        })

        trans_summary = actual_transitions.groupby(['State', 'Next State True']).size().reset_index(name='Total')
        trans_correct = actual_transitions[actual_transitions['Next State True'] == actual_transitions['Next State Pred']]
        trans_correct_summary = trans_correct.groupby(['State', 'Next State True']).size().reset_index(name='Correct')

        merged = pd.merge(trans_summary, trans_correct_summary, on=['State', 'Next State True'], how='left')
        merged['Correct'] = merged['Correct'].fillna(0)
        merged['Accuracy'] = merged['Correct'] / merged['Total']

        # Pivot to heatmap
        matrix = merged.pivot(index='State', columns='Next State True', values='Accuracy')
        plt.figure(figsize=(12, 8))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title("Transition Prediction Accuracy by State to Next State")
        plt.xlabel("True Next State")
        plt.ylabel("Current State")
        plt.show()