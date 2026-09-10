# Student Performance Prediction.

🔗 **[Try the live app here](https://student-performance-nti-jjh47t35afjhuteksfmgzo.streamlit.app/)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-performance-nti-jjh47t35afjhuteksfmgzo.streamlit.app/)

A beginner-friendly machine learning project that predicts a student's final grade (A-F) from their study habits — no advanced techniques, just the basics done cleanly.

## What it does

Given info about a student (study time, attendance, sleep, etc.), it predicts what final grade they're likely to get — **without** using their actual exam score. That makes it useful as an early check-in: you could estimate a student's grade before the final exam even happens.

## Files

```
ML_Project_Simple/
├── dataset.csv        # The data
├── notebook.ipynb      # Step-by-step: explore data, train models, save the best one
├── app.py               # Simple web app to try predictions yourself
├── requirements.txt
└── README.md
```

## How the notebook works (step by step)

1. **Load the data** and take a first look (shape, missing values, duplicates).
2. **Drop two columns**:
   - `student_id` — just a row number, not useful.
   - `final_exam_score` — it turns out this score *directly* determines the grade (90+ = A, etc.), so keeping it would be "cheating". We predict from other info instead.
3. **Explore the data** with a few simple charts (grade counts, histograms, a correlation heatmap).
4. **Prepare the data**: scale the number columns, and turn text columns (like "Male"/"Female") into 0/1 columns using one-hot encoding.
5. **Train 5-6 common models**: Logistic Regression, KNN, SVM, Decision Tree, Random Forest, and XGBoost (if installed).
6. **Compare them** using accuracy, precision, recall, and F1-score, and pick the best one automatically.
7. **Look at feature importance** — which factors matter most for predicting a grade (usually study time and previous grade!).
8. **Save everything** (the trained model + how it processes data) into one file: `best_model.pkl`.

## How to run it

**1. Install the libraries:**
```bash
pip install -r requirements.txt
```

**2. Run the notebook** (this creates `best_model.pkl`):
```bash
jupyter notebook notebook.ipynb
```
Then just run all the cells from top to bottom.

**3. Run the web app:**
```bash
streamlit run app.py
```
This opens a page in your browser where you can enter a student's info and click "Predict Final Grade" to see the result.

## A couple of simple things worth knowing

- **Class imbalance**: there are far fewer "F" grade students than "B" grade students in the data. This is why we look at more than just accuracy (precision, recall, F1) when judging the models.
- **No fancy tricks used**: no manual hyperparameter tuning, no SMOTE, no advanced feature engineering — just clean, standard scikit-learn code that's easy to read and follow.

## 👥 Team Members :

- Ahemd Ibrahim
- Karim Mohamed
- Maria Fayek
- Nour Alaa
- Seif Eldein