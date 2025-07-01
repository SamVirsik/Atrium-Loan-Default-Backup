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
from sklearn.metrics import accuracy_score
import seaborn as sns
import joblib
import os
import re
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report

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
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    model = XGBClassifier(
        objective='multi:softprob',
        num_class=len(y.unique()),
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_jobs=-1,
        max_depth=4,               # ↓ Lower tree depth = less overfitting
        n_estimators=500,          # ↓ Fewer trees = faster, less prone to overfit
        learning_rate=0.07,        # ↑ Slightly higher to compensate for fewer trees
        subsample=0.7,             # ↓ Force more randomness
        colsample_bytree=0.7,      # ↓ Force more randomness
        gamma=5,                   # ↑ Require higher gain to split
        min_child_weight=30        # ↑ Split only when large sample supports it
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model

def train_model_old_2(X, y, X_train, y_train, X_test, y_test, le_state):
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

    model.fit(X_train, y_train, sample_weight=sample_weights)
    return model

def evaluate_model(model, X_test, y_test, le_state):
    """
    Prints accuracy and shows feature importances for a trained XGBoost model.
    """
    # Predict
    y_pred = model.predict(X_test)
    
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {acc:.2%}")
    
    # Feature importances
    importance_values = model.feature_importances_
    feature_names = X_test.columns
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    }).sort_values("Importance", ascending=False).head(20)  # top 20

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.show()

def predict_one(model, x_row):
    """
    Predicts class probabilities for a single row (loan).
    Returns the full probability vector.
    """
    if isinstance(x_row, pd.Series):
        x_row = x_row.to_frame().T
    return model.predict_proba(x_row)[0]

def predict_all(model, X_test):
    """
    Predicts class probabilities for all loans in X_test at once.
    Returns:
      - prob_df: DataFrame of class probabilities (each column = state)
      - predicted_classes: Series of predicted classes (encoded)
    """
    all_probs = model.predict_proba(X_test)
    prob_df = pd.DataFrame(all_probs, index=X_test.index)
    predicted_classes = all_probs.argmax(axis=1)
    return prob_df, predicted_classes

def decode_classes(predicted_classes, le_state):
    """
    Decode numeric predictions using the label encoder.
    """
    return le_state.inverse_transform(predicted_classes)

def add_predicted_class(prob_df, le_state):
    """
    Given a DataFrame of probabilities and the label encoder used on y,
    return a Series of predicted classes (decoded).
    """
    predicted_classes = prob_df.values.argmax(axis=1)
    return le_state.inverse_transform(predicted_classes)


def train_model_old(X, y, X_train, y_train, X_test, y_test, le_state):
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

    # y_pred = model.predict(X_test)
    # print(classification_report(y_test, y_pred, target_names=le_state.classes_))

    # print("Feature Importances:")
    # print(model.feature_importances_)

    # # 1. Confusion Matrix
    # cm = confusion_matrix(y_test, y_pred)
    # cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')

    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
    #             xticklabels=le_state.classes_, yticklabels=le_state.classes_)
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.title('Normalized Confusion Matrix')
    # plt.show()

    # # 2. Classification Report
    # report = classification_report(y_test, y_pred, target_names=le_state.classes_, output_dict=True)
    # report_df = pd.DataFrame(report).transpose()
    # print("Classification Report:")
    # print(report_df.round(3))

    # # 3. Feature Importance Plot
    # importances = model.feature_importances_
    # features = X.columns
    # feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    # feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

    # plt.figure(figsize=(12, 6))
    # sns.barplot(data=feat_imp_df.head(20), x='Importance', y='Feature')
    # plt.title("Top 20 Feature Importances")
    # plt.tight_layout()
    # plt.show()

    # # 4. Transition-level Accuracy Matrix (if you have State and Next State columns in your dataset)
    # # Optional: if you originally had these columns for each row
    # if 'State' in X_test.columns and 'Next State' in y_test.index.names:
    #     actual_transitions = pd.DataFrame({
    #         'State': X_test['State'].values,
    #         'Next State True': le_state.inverse_transform(y_test),
    #         'Next State Pred': le_state.inverse_transform(y_pred)
    #     })

    #     trans_summary = actual_transitions.groupby(['State', 'Next State True']).size().reset_index(name='Total')
    #     trans_correct = actual_transitions[actual_transitions['Next State True'] == actual_transitions['Next State Pred']]
    #     trans_correct_summary = trans_correct.groupby(['State', 'Next State True']).size().reset_index(name='Correct')

    #     merged = pd.merge(trans_summary, trans_correct_summary, on=['State', 'Next State True'], how='left')
    #     merged['Correct'] = merged['Correct'].fillna(0)
    #     merged['Accuracy'] = merged['Correct'] / merged['Total']

    #     # Pivot to heatmap
    #     matrix = merged.pivot(index='State', columns='Next State True', values='Accuracy')
    #     plt.figure(figsize=(12, 8))
    #     sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu")
    #     plt.title("Transition Prediction Accuracy by State to Next State")
    #     plt.xlabel("True Next State")
    #     plt.ylabel("Current State")
    #     plt.show()
    return model 

def join_on_date(df1, df2, on_col1='date', on_col2='date', how='left', exact=True):
    """
    Joins df1 (more granular, e.g., monthly) with df2 (less granular, e.g., quarterly) on date columns.

    Parameters:
    - df1: DataFrame with more granular datetime values
    - df2: DataFrame with less granular datetime values
    - on_col1: column name in df1 to join on (this will be retained in output)
    - on_col2: column name in df2 to join on
    - how: type of join ('left', 'inner', etc.)
    - exact: if True, join only on exact matches;
             if False, join each row in df1 with the most recent date in df2 that is <= the df1 date

    Returns:
    - Merged DataFrame
    """
    df1 = df1.copy()
    df2 = df2.copy()

    df1[on_col1] = pd.to_datetime(df1[on_col1])
    df2[on_col2] = pd.to_datetime(df2[on_col2])

    if exact:
        merged = pd.merge(df1, df2, left_on=on_col1, right_on=on_col2, how=how)
    else:
        df1 = df1.sort_values(on_col1)
        df2 = df2.sort_values(on_col2)
        merged = pd.merge_asof(
            df1,
            df2,
            left_on=on_col1,
            right_on=on_col2,
            direction='backward'
        )

    # Drop the df2 join column if it's different
    if on_col1 != on_col2 and on_col2 in merged.columns:
        merged = merged.drop(columns=[on_col2])

    return merged

def save_model(model, name):
    path = name + '.pkl'
    joblib.dump(model, path)
    model_size = os.path.getsize(path) / (1024 * 1024)  # in megabytes
    print(f"Model size: {model_size:.2f} MB")

def clean_msa(msa):
    msa = str(msa).strip().upper()

    if "METROPOLITAN STATISTICAL AREA" not in msa:
        return msa.title()

    # Remove suffix
    msa = msa.replace(" METROPOLITAN STATISTICAL AREA", "")

    # Check for comma to separate city and state
    if ',' in msa:
        city_part, state_part = msa.rsplit(',', 1)
        # Replace hyphens in both parts with spaces
        city_clean = re.sub(r'-+', ' ', city_part.strip()).title()
        state_clean = re.sub(r'-+', ' ', state_part.strip()).upper()
        return f"{city_clean}, {state_clean}"
    else:
        # Fallback
        return re.sub(r'-+', ' ', msa).title()

def clean_MSA_naming(df, msa_col):
    df['MSA_clean'] = df[msa_col].apply(clean_msa)
    msa_corrections = {
        'Aguadilla Isabela, PR': 'Aguadilla, PR',
        'Albany Lebanon, OR': 'Albany, OR',
        'Atlanta Sandy Springs Alpharetta, GA': 'Atlanta Sandy Springs Roswell, GA', #Not perfect
        'Austin Round Rock Georgetown, TX': 'Austin Round Rock San Marcos, TX', #Not perfect
        'Bakersfield, CA': 'Bakersfield Delano, CA',
        'Birmingham Hoover, AL': 'Birmingham, AL',
        'Blacksburg Christiansburg, VA': 'Blacksburg Christiansburg Radford, VA',
        'Bloomsburg Berwick, PA': 'Scranton  Wilkes Barre, PA', #GPT beleives this is closest
        'Bridgeport Stamford Norwalk, CT': 'Bridgeport Stamford Danbury, CT', 
        'Brunswick, GA': 'Brunswick St. Simons, GA', 
        'California Lexington Park, MD': 'Lexington Park, MD', 
        'Carbondale Marion, IL': 'Cape Girardeau, MO IL', #GPT Believes this is closest
        'Chambersburg Waynesboro, PA': 'Chambersburg, PA',
        'Chicago Naperville Elgin, IL IN WI': 'Chicago Naperville Elgin, IL IN',
        'Cleveland Elyria, OH': 'Cleveland, OH',
        "Coeur D'Alene, ID": "Coeur d'Alene, ID",
        'Cumberland, MD WV': 'Hagerstown Martinsburg, MD WV', #GPT
        'Danville, IL': 'Champaign Urbana, IL', #GPT
        'Dayton Kettering, OH': 'Dayton Kettering Beavercreek, OH',
        'Denver Aurora Lakewood, CO': 'Denver Aurora Centennial, CO',
        'East Stroudsburg, PA': 'Allentown Bethlehem Easton, PA NJ', #GPT
        'Elizabethtown Fort Knox, KY': 'Elizabethtown, KY',
        'Evansville, IN KY': 'Evansville, IN',
        'Fairbanks, AK': 'Fairbanks College, AK',
        'Fond Du Lac, WI': 'Fond du Lac, WI',
        'Fort Collins, CO': 'Fort Collins Loveland, CO',
        'Grand Rapids Kentwood, MI': 'Grand Rapids Wyoming Kentwood, MI',
        'Greenville Anderson, SC': 'Greenville Anderson Greer, SC',
        'Hartford East Hartford Middletown, CT': 'Hartford West Hartford East Hartford, CT',
        'Hilton Head Island Bluffton, SC': 'Hilton Head Island Bluffton Port Royal, SC',
        'Houma Thibodaux, LA': 'Houma Bayou Cane Thibodaux, LA',
        'Houston The Woodlands Sugar Land, TX': 'Houston Pasadena The Woodlands, TX',
        'Indianapolis Carmel Anderson, IN': 'Indianapolis Carmel Greenwood, IN',
        'Joplin, MO': 'Joplin, MO KS',
        'Kahului Wailuku Lahaina, HI': 'Kahului Wailuku, HI',
        'Las Vegas Henderson Paradise, NV': 'Las Vegas Henderson North Las Vegas, NV',
        'Longview, WA': 'Longview Kelso, WA',
        'Madera, CA': 'Fresno, CA', #GPT
        'Mcallen Edinburg Mission, TX': 'McAllen Edinburg Mission, TX',
        'Miami Fort Lauderdale Pompano Beach, FL': 'Miami Fort Lauderdale West Palm Beach, FL', 
        'Multiple Properties': 'Multiple Properties', #STILL AN ISSUE
        'Muskegon, MI': 'Muskegon Norton Shores, MI',
        'Myrtle Beach Conway North Myrtle Beach, SC NC': 'Myrtle Beach Conway North Myrtle Beach, SC',
        'Nashville Davidson Murfreesboro Franklin, TN': 'Nashville Davidson  Murfreesboro  Franklin, TN',
        'New Bern, NC': 'Greenville, NC', #GPT
        'New Haven Milford, CT': 'New Haven, CT',
        'New York Newark Jersey City, NY NJ PA': 'New York Newark Jersey City, NY NJ',
        'Non Msa': 'Non Msa', #STILL AN ISSUE
        'North Port Sarasota Bradenton, FL': 'North Port Bradenton Sarasota, FL',
        'Norwich New London, CT': 'Norwich New London Willimantic, CT',
        'Ocean City, NJ': 'Atlantic City Hammonton, NJ', #GPT
        'Ogden Clearfield, UT': 'Ogden, UT',
        'Omaha Council Bluffs, NE IA': 'Omaha, NE IA',
        'Panama City, FL': 'Panama City Panama City Beach, FL',
        'Pine Bluff, AR': 'Little Rock North Little Rock Conway, AR', #GPT
        'Poughkeepsie Newburgh Middletown, NY': 'Kiryas Joel Poughkeepsie Newburgh, NY', 
        'Provo Orem, UT': 'Provo Orem Lehi, UT',
        'Racine, WI': 'Racine Mount Pleasant, WI',
        'Salisbury, MD DE': 'Salisbury, MD',
        'Salt Lake City, UT': 'Salt Lake City Murray, UT',
        'San Francisco Oakland Berkeley, CA': 'San Francisco Oakland Fremont, CA',
        'San German, PR': 'Mayagüez, PR', #GPT
        'San Juan Bayamon Caguas, PR': 'San Juan Bayamón Caguas, PR',
        'Scranton Wilkes Barre, PA': 'Scranton  Wilkes Barre, PA', 
        'Sebastian Vero Beach, FL': 'Sebastian Vero Beach West Vero Corridor, FL',
        'Sebring Avon Park, FL': 'Sebring, FL',
        'Sioux Falls, SD': 'Sioux Falls, SD MN',
        'Staunton, VA': 'Staunton Stuarts Draft, VA',
        'Stockton, CA': 'Stockton Lodi, CA',
        'The Villages, FL': 'Wildwood The Villages, FL',
        'Vineland Bridgeton, NJ': 'Vineland, NJ',
        'Virginia Beach Norfolk Newport News, VA NC': 'Virginia Beach Chesapeake Norfolk, VA NC',
        'Wausau Weston, WI': 'Wausau, WI',
        'Wenatchee, WA': 'Wenatchee East Wenatchee, WA',
        'Worcester, MA CT': 'Worcester, MA',
        'Youngstown Warren Boardman, OH PA': 'Youngstown Warren, OH'
    }

    df['MSA_clean'] = df['MSA_clean'].replace(msa_corrections)
    return df