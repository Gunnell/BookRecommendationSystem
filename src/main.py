import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn import model_selection
from numpy import count_nonzero
from sklearn.metrics import mean_squared_error
from math import sqrt

import tkinter as tk
import tkinter.font as tkFont
import warnings

warnings.filterwarnings('ignore')


file_path_rt = 'data/BX-Book-Ratings.csv'
file_path_bk = 'data/BX-Books.csv'
file_path_us = 'data/BX-Users.csv'

# Load datasets
df_rt = pd.read_csv(file_path_rt, sep=';', encoding="latin-1", engine='python', quotechar='"', on_bad_lines='skip')
df_rt.columns = ['userID', 'ISBN', 'bookRating']

df_bk = pd.read_csv(file_path_bk, sep=';', encoding="latin-1", engine='python', quotechar='"', on_bad_lines='skip')
df_bk.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']

df_us = pd.read_csv(file_path_us, sep=';', encoding="latin-1", engine='python', quotechar='"', on_bad_lines='skip')
df_us.columns = ['userID', 'Location', 'Age']

# Shape of Data
print("*********Shape of Data*********")
print(f"books: {df_bk.shape}")
print(f"users: {df_us.shape}")
print(f"ratings: {df_rt.shape}\n")

# Data Preprocessing
print("*********Data Preprocessing*********")
books = df_bk.drop(['imageUrlS', 'imageUrlM', 'imageUrlL'], axis=1)
print(books['yearOfPublication'].unique())
print("\n")
print(books.describe())
print("\n")

# Handle invalid publication years
books.loc[(books.yearOfPublication > 2022) | (books.yearOfPublication == 0), 'yearOfPublication'] = np.nan
books.yearOfPublication.fillna(round(books.yearOfPublication.mean()), inplace=True)
books.yearOfPublication = books.yearOfPublication.astype(np.int32)

print(books.loc[books.publisher.isnull(), :])
books.loc[books.publisher.isnull(), 'publisher'] = 'other'
print("\n")

# Outlier Analysis for Age attribute in Users
Q1 = df_us.Age.quantile(0.25)
Q3 = df_us.Age.quantile(0.75)
IQR = Q3 - Q1
df_us.loc[((df_us.Age > Q3 + 1.5 * IQR) | (df_us.Age <= 0)), 'Age'] = np.nan

print("Before processing Age attr: ")
print(df_us.describe())
df_us.Age = df_us.Age.fillna(df_us.Age.mean())
df_us.Age = df_us.Age.astype(np.int32)
users = df_us
print("\nAfter processing Age attr: ")
print(users.describe())

# Filter ratings to include only books and users present in the respective datasets
ratings = df_rt[df_rt.ISBN.isin(books.ISBN)]
ratings = ratings[ratings.userID.isin(users.userID)]
print(f"\n ratings before : {df_rt.shape}")
print(f" ratings after : {ratings.shape}")

# Split the data into train and test sets
train_data_rt, test_data_rt = model_selection.train_test_split(ratings, test_size=0.20)
rated_books = train_data_rt[train_data_rt.bookRating != 0]
unrated_books = train_data_rt[train_data_rt.bookRating == 0]

test_rated_books = test_data_rt[test_data_rt.bookRating != 0]
test_unrated_books = test_data_rt[test_data_rt.bookRating == 0]

rated_users = users[users.userID.isin(rated_books.userID)]
unrated_users = users[users.userID.isin(unrated_books.userID)]

test_rated_users = users[users.userID.isin(test_rated_books.userID)]
test_unrated_books = users[users.userID.isin(test_unrated_books.userID)]

# Collaborative Filtering
counts1 = rated_books['userID'].value_counts()
test_counts1 = test_rated_books['userID'].value_counts()

# Filter users with less than 100 ratings (Because of RAM issues, if you have powerful PC u can comment these 2 lines)
rated_books = rated_books[rated_books['userID'].isin(counts1[counts1 >= 100].index)]
test_rated_books = test_rated_books[test_rated_books['userID'].isin(test_counts1[test_counts1 >= 100].index)]

# Filter books with less than 100 ratings
counts = rated_books['bookRating'].value_counts()
test_counts = test_rated_books['bookRating'].value_counts()

rated_books = rated_books[rated_books['bookRating'].isin(counts[counts >= 100].index)]
test_rated_books = test_rated_books[test_rated_books['bookRating'].isin(counts[counts >= 100].index)]

print(rated_books.describe())

# Create ratings matrix
ratings_matrix = rated_books.pivot(index='userID', columns='ISBN', values='bookRating')
test_ratings_matrix = test_rated_books.pivot(index='userID', columns='ISBN', values='bookRating')

print(f"rating matrix shape : {ratings_matrix.shape}")

ratings_matrix.fillna(0, inplace=True)
test_ratings_matrix.fillna(0, inplace=True)

# Calculate sparsity
sparsity = 1.0 - count_nonzero(ratings_matrix) / float(ratings_matrix.size)
print(f'Sparsity level of Rating Matrix: {float(sparsity * 100)}%')
print(ratings_matrix.iloc[:100, :100])

k = 10

# User Based Recommendation

def button1():
    listbox.delete('0', 'end')
    inp = entry1.get()
    preference = "User User-Based"
    recommend_col(inp, preference)

def button2():
    listbox.delete('0', 'end')
    inp = entry1.get()
    preference = "User Item-Based"
    recommend_col(inp, preference)

def find_k_similar_users(user_id, ratings_matrix, metric, k=k):
    # Find k similar users using NearestNeighbors
    model_knn = NearestNeighbors(metric=metric, algorithm='brute')
    model_knn.fit(ratings_matrix)
    loc = ratings_matrix.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors(ratings_matrix.iloc[loc, :].values.reshape(1, -1), n_neighbors=k + 1)
    similarities = 1 - distances.flatten()
    return similarities, indices

def predict_userbased(user_id, item_id, ratings_matrix, metric):
    user_loc = ratings_matrix.index.get_loc(user_id)
    item_loc = ratings_matrix.columns.get_loc(item_id)
    similarities, indices = find_k_similar_users(user_id, ratings_matrix, metric, k)

    mean_rating = ratings_matrix.iloc[user_loc, :].mean()
    sum_wt = np.sum(similarities) - 1
    wtd_sum = 0

    for i in range(len(indices.flatten())):
        if indices.flatten()[i] == user_loc:
            continue
        ratings_diff = ratings_matrix.iloc[indices.flatten()[i], item_loc] - np.mean(
            ratings_matrix.iloc[indices.flatten()[i], :])
        wtd_sum += ratings_diff * similarities[i]

    prediction = mean_rating + (wtd_sum / sum_wt)
    prediction = max(min(int(round(prediction)), 10), 1)

    return prediction

# Item Based Recommendation

def find_k_similar_items(item_id, ratings_matrix, metric):
    # Find k similar items using NearestNeighbors
    ratings_matrix = ratings_matrix.T
    loc = ratings_matrix.index.get_loc(item_id)
    model_knn = NearestNeighbors(metric=metric, algorithm='brute')
    model_knn.fit(ratings_matrix)
    distances, indices = model_knn.kneighbors(ratings_matrix.iloc[loc, :].values.reshape(1, -1), n_neighbors=k + 1)
    similarities = 1 - distances.flatten()
    return similarities, indices

def predict_itembased(user_id, item_id, ratings_matrix, metric):
    user_loc = ratings_matrix.index.get_loc(user_id)
    item_loc = ratings_matrix.columns.get_loc(item_id)
    similarities, indices = find_k_similar_items(item_id, ratings_matrix, metric)

    sum_wt = np.sum(similarities) - 1
    wtd_sum = sum(ratings_matrix.iloc[user_loc, indices.flatten()[i]] * similarities[i]
                  for i in range(len(indices.flatten())) if indices.flatten()[i] != item_loc)

    prediction = max(min(int(round(wtd_sum / sum_wt)), 10), 1)
    return prediction

def recommend_col(inp, preference):
    # Recommend books based on user ID and filtering preference
    user_id = int(inp)
    predictions = []

    if user_id not in ratings_matrix.index.values or not isinstance(user_id, int):
        print("Warning: user id not in matrix or not an int")
        return

    metric = "cosine"
    for i in range(ratings_matrix.shape[1]):
        item_id = str(ratings_matrix.columns[i])
        if ratings_matrix[item_id][user_id] != 0:
            if preference == "User User-Based":
                predictions.append(predict_userbased(user_id, item_id, ratings_matrix, metric))
            else:
                predictions.append(predict_itembased(user_id, item_id, ratings_matrix, metric))
        else:
            predictions.append(-1)

    predictions = pd.Series(predictions).sort_values(ascending=False)
    recommended = predictions[:10]

    for i, book_id in enumerate(recommended.index):
        listbox.insert(i, books.bookTitle[book_id])
    listbox.insert(len(recommended), " ")

def rmse(prediction, test_matrix):
    # Calculate root mean squared error
    prediction = prediction[test_matrix.nonzero()].flatten()
    test_matrix = test_matrix[test_matrix.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, test_matrix))

# GUI
root = tk.Tk()
root.title("Book Recommender")
canvas = tk.Canvas(root, height=800, width=800, bg='#9799BA')
canvas.grid(columnspan=20, rowspan=20)

lbl = tk.Label(root, text="Choose Filtering Method", font=("Raleway", 23), height=2, bg='#9799BA', fg="white")
lbl.place(relx=0.37, rely=0.09)

lbl2 = tk.Label(root, text="User ID:", font=("Raleway", 18), height=2, bg='#9799BA', fg="white")
lbl2.place(relx=0.15, rely=0.01)

entry1 = tk.Entry(root, font=tkFont.Font(family='Times', size=26))
entry1.place(relx=0.37, rely=0.02, width=250, height=40)

ub_cos_btn = tk.Button(root, text="User User-Based", font="Raleway", height=2, bg='#8dbbc0', fg="#2e5054",
                       width=15, command=button1)
ub_cos_btn.place(relx=0.4, rely=0.2, width=200, height=50)

ib_cos_btn = tk.Button(root, text="User Item-Based", font="Raleway", height=2, bg='#8dbbc0', fg="#2e5054",
                       width=15, command=button2)
ib_cos_btn.place(relx=0.4, rely=0.30, width=200, height=50)

listbox = tk.Listbox(root, height=10, width=15, bg="grey", activestyle='dotbox', font="Helvetica", fg="yellow")
listbox.place(relx=0.07, rely=0.4, width=700, height=400)

root.mainloop()
