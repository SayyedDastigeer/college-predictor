import os
from flask import Blueprint, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Create a Blueprint for MHT CET
mht_cet_bp = Blueprint('mht_cet', __name__)

# Load your cleaned data once at the start
dataframe = pd.read_csv('mhtcet_data.csv')

# Initialize label encoders for categorical columns
le_stage = LabelEncoder()
le_category = LabelEncoder()

# Fit label encoders on the data
dataframe['Stage'] = le_stage.fit_transform(dataframe['Stage'])
dataframe['Category'] = le_category.fit_transform(dataframe['Category'])

@mht_cet_bp.route('/mht-cet-form', methods=["GET", "POST"])
def form():
    if request.method == "POST":
        user_percentile = float(request.form.get('percentile'))
        user_category = request.form.get('category')
        user_course = request.form.get('course')  # Get the selected course from the form

        # Filter the DataFrame based on user inputs (percentile first)
        filtered_df = dataframe[dataframe['Percent'] <= user_percentile]

        # If user selected a specific category, filter by that category
        if user_category != 'ALL':  # 'ALL' means no category filtering
            category_code = le_category.transform([user_category])[0]  # Convert category to its encoded form
            filtered_df = filtered_df[filtered_df['Category'] == category_code]

        # Filter by course name if a specific course is selected
        if user_course != 'ALL':  # 'ALL' means no course filtering
            filtered_df = filtered_df[filtered_df['Course Name'] == user_course]

        # Prepare the data for the decision tree (excluding irrelevant columns)
        X = filtered_df.drop(columns=['College Code', 'College Name', 'College Type', 'Choice Code', 'Course Name', 'Year', 'Category', 'Percent'])
        y = filtered_df['Category']  # Assuming 'Category' is the target variable

        # Ensure X contains only numerical data, dropping any non-numeric columns
        X = X.apply(pd.to_numeric, errors='coerce').dropna(axis=1)

        # Create and fit the decision tree classifier
        classifier = DecisionTreeClassifier()
        classifier.fit(X, y)

        # Predict using the filtered data
        predicted = classifier.predict(X)

        # Get predicted college names and corresponding course names
        predictions = filtered_df[filtered_df['Category'].isin(predicted)][['College Name', 'Course Name', 'Percent', 'Category']]

        # Convert numerical categories back to original values
        predictions['Category'] = le_category.inverse_transform(predictions['Category'])

        # Calculate the absolute difference and sort by it
        predictions['Difference'] = abs(predictions['Percent'] - user_percentile)
        top_colleges_courses = predictions.sort_values(by='Difference').drop_duplicates().head(10)  # Get top 10 results

        # Save the results to a CSV file
        top_colleges_courses[['College Name', 'Course Name', 'Percent', 'Category']].to_csv('static/final_mht_cet.csv', index=False)

        # Pass the updated DataFrame to the template
        return render_template('mht_cet_result.html', colleges_courses=top_colleges_courses[['College Name', 'Course Name', 'Percent', 'Category']].values)

    return render_template('mht_cet_form.html')

@mht_cet_bp.route('/mht-cet-result')
def result():
    # Read the saved final results for MHT CET
    top_colleges_courses = pd.read_csv('static/final_mht_cet.csv')
    return render_template('mht_cet_result.html', colleges_courses=top_colleges_courses.values)
