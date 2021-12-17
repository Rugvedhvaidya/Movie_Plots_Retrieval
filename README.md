# MOVIE PLOT RETRIEVAL SYSTEM.

This is a **Information Retrieval** Project which is based on the searching of the movie plot. The goal of our project is to analyze and extract meaningful information from Wikipedia movie plots dataset. Our information retrieval algorithm returns similar movie titles based on an input plot description.


## Dataset Used in the Project

 **[https://www.kaggle.com/jrobischon/wikipedia-movie-plots](https://www.kaggle.com/jrobischon/wikipedia-movie-plots)**

## Project  Description


We got a dataset from **Kaggle**. The dataset contains descriptions of 34,886 movies from around the world.

**The Column Descriptions are listed as below**:-

-   **Release Year**:- In which year the movie was released.
    
-   **Movie Title**:- Title of the Movie.
    
-   **Movie Origin**:- From where the movie was released.
    
-   **Director**:- Who directed the film.
    
-   **Cast**:- Main Actor and Actress and the movies.
    
There are so many Null and Unknown values present in the dataset.

Our dataset was too huge with more than **30K** movies data, we need to remove 		   null 	value rows. So, data preprocessing was very important.

There are 2191 unique genres in the dataset. Then we also removed all unimportant columns like director, cast, etc.

We made our model by finding plot’s Tf-Idf vector. At last, we made top 10 genre and remove other genre movies.  Top genre will get 0 and it will be followed for all genre. Tasks Involved and Steps Implemented Configuring Spark Understanding problem statement Understanding the algorithm Fetching the data Data Preprocessing Implementing Index, Query Index, TF-IDF, Cosine Similarity etc.

By Following the  **Required Steps** we can complete the main goal of this Project.


## Components of this Movie Plot Retrieval System

 
 - **Collection of documents**- The dataset Collection and other needed Documents for the Project.

- **Preprocessing**- It the most important task in information retrieval for extracting important information from unstructured text data.

- **Indexing system** - Indexing and searching methods and procedures (an indexing system can be human or automated).

 - **Defined set of queries**- which are input into the system, with or without the involvement of a human searcher.
 
 - **Evaluation** - The specified measures by which each system is evaluated, for example ‘precision’ and ‘recall’ as measures of relevance. Recall is the proportion of relevant documents in the collection retrieved in response to the query. Precision is the proportion of relevant documents amongst the set of documents retrieved in response to the query.
 
 - **Readme**-  Writing the Project Description of the project, and the components used and how the run the Project.

 - **Report**- Mentioning the  Tasks and the Sub tasks of the Information Retrieval System and mentioning the contributions of the team members and their details.

## How to Run The Project

- Open the ipynb in some editor like Jupyter Notebook
- Run all the cells in the file.
- Head to the query cell and enter the query in the input box.
- The top 10 documents will be retrieved and shown.
- You can give feedback for each document listed above this cell
- Using the above entered feedback the next cell will show a p-r graph.
