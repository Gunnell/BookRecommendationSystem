# Book Recommender System

This project is a book recommender system that uses collaborative filtering to suggest books to users. The system employs both user-based and item-based filtering techniques to generate recommendations. The GUI is built using Tkinter.

## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Collaborative Filtering](#collaborative-filtering)
  - [User-Based Filtering](#user-based-filtering)
  - [Item-Based Filtering](#item-based-filtering)
- [Usage](#usage)
- [License](#license)

## Introduction

This project aims to recommend books to users based on their past ratings and the ratings of similar users. The recommendation system uses collaborative filtering methods to predict the ratings for books that a user has not yet rated.

## Data

The datasets used in this project are:

1. `BX-Books.csv`: Contains book information.
2. `BX-Book-Ratings.csv`: Contains user ratings for books.
3. `BX-Users.csv`: Contains user demographic information.

## Collaborative Filtering

Collaborative filtering is a method of making automatic predictions about the interests of a user by collecting preferences from many users. The system assumes that if User A has the same opinion as User B on a book, User A is more likely to have User B's opinion on a different book than that of a randomly chosen user. 

Collaborative filtering can be divided into two main categories:

1. **User-Based Collaborative Filtering**:
   - **Description**: This method finds users who are similar to the target user and recommends books that these similar users have liked.
   - **How It Works**: The system calculates similarity measures between users based on their ratings of books. It then suggests books to the target user based on the preferences of users with similar tastes.
   - **Use in Project**: This method is implemented in the project to recommend books based on user-user similarity.

2. **Item-Based Collaborative Filtering**:
   - **Description**: This method finds items (books) that are similar to the books that the target user has liked in the past.
   - **How It Works**: The system calculates similarity measures between books based on the ratings given by users. It then recommends books that are similar to those the user has liked.
   - **Use in Project**: This method is implemented in the project to recommend books based on item-item similarity.

In this project, user-based filtering and item-based filtering are implemented using the k-Nearest Neighbors (k-NN) algorithm to find similar users.


## Usage

To use the recommender system, follow these steps:

1. **Install Dependencies**: Ensure you have all the required Python packages installed.
2. **Run the Application**: Execute the script to start the Tkinter GUI.
```bash
    python main.py
```


