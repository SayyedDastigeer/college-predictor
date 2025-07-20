# jee.py
import os
from flask import Blueprint, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# Create a Blueprint for JEE
jee_bp = Blueprint('jee', __name__)

@jee_bp.route('/jee-form', methods=["GET", "POST"])
def jee_form():
    if request.method == "POST":
        user_rank = request.form.get("rank")
        user_c = request.form.get("category")
        user_round = request.form.get("round")
        user_q = request.form.get("quota")
        user_p = request.form.get("pool")

        # Load and preprocess the data
        dataframe = pd.read_csv('jee_data.csv')
        dataframe = dataframe.drop(['id', 'Unnamed: 0'], axis=1)

        # Convert categorical variables to numerical codes
        category = np.unique(dataframe['category'])
        code = [i + 1 for i in range(len(category))]
        dataframe['category'] = dataframe['category'].replace(category, code).astype(int)

        df = dataframe.copy()
        df.drop(columns=["year", "institute_type", "institute_short", "program_name", "program_duration", "degree_short", "is_preparatory", "opening_rank"], inplace=True)

        college = np.unique(df['College'])
        code = [i + 1 for i in range(len(college))]
        df['College'] = df['College'].replace(college, code)

        quota = np.unique(df['quota'])
        code = [i + 1 for i in range(len(quota))]
        df['quota'] = df['quota'].replace(quota, code)

        pool = np.unique(df['pool'])
        code = [i + 1 for i in range(len(pool))]
        df['pool'] = df['pool'].replace(pool, code)

        # Map user inputs to the corresponding numerical values
        user_quota = next((i + 1 for i in range(len(quota)) if user_q == quota[i]), None)
        user_pool = next((i + 1 for i in range(len(pool)) if user_p == pool[i]), None)
        user_category = next((i + 1 for i in range(len(category)) if user_c == category[i]), None)

        # Decision Tree function
        def Decision_Tree(df, dataframe, user_round, user_quota, user_pool, user_rank):
            X = df.drop(['College', 'category'], axis=1)
            y = pd.DataFrame({'College': df['College']})
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=51)
            classifier = DecisionTreeClassifier(criterion='gini')
            classifier.fit(X_train, y_train)
            accuracy = classifier.score(X_test, y_test) * 100

            arr = np.array(dataframe['College'])
            arr1 = np.array(df['College'])
            dict1 = {arr1[i]: arr[i] for i in range(len(arr))}

            input_data = pd.DataFrame([[user_round, user_quota, user_pool, user_rank]], columns=X.columns)
            arr2 = classifier.predict(input_data)

            global list1
            list1 = [dict1[i] for i in arr2 if i in dict1]
            return df, arr2

        # Creating DataFrames based on user categories
        df_by_category = {i: (df[df.category == i], dataframe[dataframe.category == i]) for i in range(1, 11)}

        dfcluster, dfdataframe = df_by_category.get(user_category, (None, None))
        if dfcluster is not None:
            dfcluster, arr2 = Decision_Tree(dfcluster, dfdataframe, user_round, user_quota, user_pool, user_rank)

            # KMeans clustering
            X = dfcluster.drop(['College'], axis=1)
            y = dfcluster['College']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=51)
            kmeans = KMeans(n_clusters=100, random_state=51).fit(X_train)

            cluster1 = kmeans.predict(X)
            dfcluster.loc[:, 'cluster'] = cluster1

            dict2 = {dfcluster['College'].iloc[i]: dfcluster['cluster'].iloc[i] for i in range(len(dfcluster))}

            if arr2[0] in dict2:
                tempdf = dfcluster[dfcluster['cluster'] == dict2[arr2[0]]].sort_values('closing_rank')
            else:
                return "Error: Cluster not found for predicted college.", 400

            # Replace codes back with actual values
            tempdf['quota'] = tempdf['quota'].replace(list(range(1, len(quota) + 1)), quota)
            tempdf['pool'] = tempdf['pool'].replace(list(range(1, len(pool) + 1)), pool)
            tempdf['College'] = tempdf['College'].replace(list(range(1, len(college) + 1)), college)
            tempdf['category'] = tempdf['category'].replace(list(range(1, len(category) + 1)), category)

            final = pd.DataFrame(tempdf['College']).drop_duplicates(keep="first")
            final_1 = pd.DataFrame(final.head(10))

            # Save final result to CSV
            final_1.to_csv(os.path.join('static', 'jee_final.csv'), index=False)

            return redirect(url_for('jee.jee_result'))

    return render_template('jee_form.html')

@jee_bp.route('/jee-result')
def jee_result():
    final_data = pd.read_csv(os.path.join('static', 'jee_final.csv'))
    final_data_list = final_data.to_dict(orient='records')
    return render_template('jee_result.html', tables=final_data_list)
