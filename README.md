# 📚 Book Recommendation System

This project is a comprehensive **Book Analysis and Recommendation System** built using **Python** and **Jupyter Notebook**. It includes data cleaning, visualization, and a content-based recommendation engine using **KNN (K-Nearest Neighbors)**.

---

## 🚀 Features

- 🔍 Data analysis on book dataset (average ratings, ratings count, number of pages)
- 📊 Visualizations with **Seaborn** and **Matplotlib**
- 👤 Most popular authors and most rated books analysis
- 🤖 Content-based recommender using:
  - Average rating bins
  - Ratings count
  - Language encoding
- 📈 Normalization with **MinMaxScaler**
- 🔁 Recommendation based on **KNN** model (Ball Tree algorithm)

---

## 📁 Dataset

- `books.csv`: Contains book metadata such as title, author, rating, number of pages, and language.

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn (KNN, MinMaxScaler)

---

## 🧠 How It Works

1. Dataset is cleaned and null values handled.
2. Data visualizations provide insights into:
   - Top rated books
   - Top authors
   - Ratings distribution
3. Data is prepared by encoding categorical features (`language_code`, `rating_between`) and scaling numerical features.
4. A **K-Nearest Neighbors model** is trained to recommend similar books based on selected features.
5. Use the `BookRecommender("Book Title")` function to get suggestions.

---

## 🔎 Example Usage

```python
BookNames = BookRecommender('Harry Potter and the Half-Blood Prince (Harry Potter  #6)')
print(BookNames)
