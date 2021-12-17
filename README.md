
# MOVIE PLOT RETRIEVAL SYSTEM.

This is a **Information Retrieval** Project which is based on the searching of the movie plot in a given dataset. The goal of our project is to analyze and extract meaningful information from Wikipedia movie plots dataset. Our information retrieval algorithm returns similar movie titles based on an input query.


## Dataset Used in the Project

 **[https://www.kaggle.com/jrobischon/wikipedia-movie-plots](https://www.kaggle.com/jrobischon/wikipedia-movie-plots)**

## Project Description


We got a dataset from **Kaggle**. The dataset contains descriptions of 34,886 movies from around the world.

**The Column Descriptions are listed as below**: -

-   **Release Year**: - In which year the movie was released.
    
-   **Movie Title**: - Title of the Movie.
    
-   **Movie Origin**: - From where the movie was released.
    
-   **Director**: - Who directed the film.
    
-   **Cast**: - Main Actor and Actress and the movies.
    
There are so many Null and Unknown values present in the dataset.

Our dataset was too huge with more than **30K** movies data, we need to remove 		   null 	value rows. So, data preprocessing was very important.

There are 2191 unique genres in the dataset. Then we also removed all unimportant columns like director, cast, etc.

We made our model by finding plot’s Tf-Idf vector. At last, we made top 10 genre and remove other genre movies.  Top genre will get 0 and it will be followed for all genre. Tasks Involved and Steps Implemented Configuring Spark Understanding problem statement Understanding the algorithm Fetching the Data Preprocessing Implementing Index, Query Index, TF-IDF, Cosine Similarity etc.

By Following the **Required Steps** we can complete the main goal of this Project.

## Components in Movie Plot Retrieval System

 
 - **Collection of documents**- The dataset is collected and imported in the file.

- **Preprocessing**- It the most important task in information retrieval for extracting important information from unstructured text data.
	```py 
	def  preprocess(data):
				return remove_stop_words(nltk.word_tokenize(text_cleaner(data)))
``

- **Indexing** - Indexing and searching methods and procedures (an indexing system can be human or automated).

	```py 
	inverted_index = {}
	def  add_term_to_inverted_index(term,documentID):
		try:
			for document in inverted_index[term]['posting_list'].copy():
			if document['docID']>documentID:
				inverted_index[term]['posting_list'].append({'docID':documentID,'count':1})
				inverted_index[term]['count']=len(inverted_index[term]['posting_list'])
				inverted_index[term]['posting_list'] = sorted(inverted_index[term ['posting_list'],key=lambda x:x['docID'])
				break
			elif document['docID']==documentID:
			document['count']+=1
			break
		else:
			inverted_index[term]['posting_list'].append({'docID':documentID,'count':1})
			inverted_index[term]['count']=len(inverted_index[term]['posting_list'])
			inverted_index[term]['posting_list'] = sorted(inverted_index[term]['posting_list'],key=lambda x:x['docID'])

- **Ranked Retrieval** - The documents are ranked using cosine similarity and top 10 documents are retrieved.

 - **Query Input**- A query can be asked by user.
 
 - **Evaluation** - Using users' feedback precision and recall are calculated. p-r curve is drawn at last.
```py 
def  evaluate(feedback):
relevant_retrieved=feedback.count('1')
relavent_count=0
for ct in  range(10):
	if feedback[ct]==1:
		relavent_count+=1;
	recall.append(relavent_count/relevant)
	precision.append(relavent_count/(ct+1))
print("RECALL",recall)
print("PRECISION",precision)
plt.xlabel("RECALL")
plt.ylabel("PRECISION")
plt.title("P-R Curve")
plt.plot(recall,precision)
```
## How to Run the Project

- Open the ipynb in some editor like Jupyter Notebook
- Run all the cells in the file.
- Head to the query cell and enter the query in the input box.
- The top 10 documents will be retrieved and shown.
- You can give feedback for each document listed above this cell
- Using the above entered feedback the next cell will show a p-r graph.
