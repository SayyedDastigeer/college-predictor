
# 🎓 College Predictor for MHT-CET and JEE (Flask + ML)

A web-based application to help students predict potential colleges and courses based on their **MHT-CET percentile** or **JEE rank**, category, quota, and pool. Built using **Flask**, **Pandas**, and **scikit-learn**, this project leverages **Decision Trees** and **KMeans Clustering** to generate tailored predictions.

---

## 🚀 Features

* 🔍 Predict suitable colleges and courses for **MHT-CET** applicants.
* 🎯 Estimate probable colleges for **JEE** aspirants using clustering and decision trees.
* 📊 Data preprocessing with label encoding and category mapping.
* 📁 Result exports to CSV and dynamic HTML rendering of top recommendations.

---

## 🧠 Technologies Used

* Python
* Flask (Blueprint-based modular routing)
* Pandas, NumPy
* scikit-learn (DecisionTreeClassifier, KMeans, LabelEncoder)
* HTML (Jinja2 templates)
* CSV for result persistence

---

## 📂 Project Structure

```
project/
│
├── static/
│   ├── jee_final.csv
│   └── final_mht_cet.csv
│
├── templates/
│   ├── jee_form.html
│   ├── jee_result.html
│   ├── mht_cet_form.html
│   └── mht_cet_result.html
│
├── jee.py
├── mht_cet.py
├── cleaned_combined_data.csv
├── updated1.csv
└── app.py  # Main Flask app that registers both Blueprints
```

---

## 📝 How It Works

### 🧪 MHT-CET Prediction

1. User submits **percentile**, **category**, and optionally a **course**.
2. Data is filtered based on cutoff percentiles and encoded category/course.
3. A Decision Tree is trained and used to predict matching college-category combinations.
4. Results are sorted by proximity to user's percentile and top 10 are shown.

### 📐 JEE Prediction

1. User provides **rank**, **category**, **round**, **quota**, and **pool**.
2. Dataset is cleaned, encoded, and split based on the user category.
3. A Decision Tree is trained on this subset.
4. Predicted college is clustered with KMeans.
5. Top 10 colleges from the same cluster are returned, sorted by **closing rank**.

---

## 🛠 Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/college-predictor.git
   cd college-predictor
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Make sure CSV files are placed**

   * `updated1.csv` (JEE dataset)
   * `cleaned_combined_data.csv` (MHT-CET dataset)

4. **Run the Flask App**

   ```bash
   python app.py
   ```

5. **Access via browser**

   ```
   http://localhost:5000/jee-form
   http://localhost:5000/mht-cet-form
   ```

---


## 📊 Sample Input

### JEE Form

* Rank: `12000`
* Category: `OBC`
* Quota: `AI`
* Pool: `Gender-Neutral`

### MHT-CET Form

* Percentile: `95.5`
* Category: `OPEN`
* Course: `Computer Engineering`

---

## 🙋‍♂️ Author

**Dastigeer Sayyed**

If you found this project helpful, feel free to ⭐ the repo and share your feedback!
