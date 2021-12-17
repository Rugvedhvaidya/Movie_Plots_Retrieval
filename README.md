
# MOVIE PLOT RETRIEVAL SYSTEM.

This is a **Information Retrieval** Project which is based on the searching of the movie plot in a given dataset. The goal of our project is to analyze and extract meaningful information from Wikipedia movie plots dataset. Our information retrieval algorithm returns similar movie titles based on an input query.


## Dataset Used in the Project
- Kaggle
 **[https://www.kaggle.com/jrobischon/wikipedia-movie-plots](https://www.kaggle.com/jrobischon/wikipedia-movie-plots)**
- Github
 **[https://github.com/Rugvedhvaidya/Movie_Plots_Retrieval/blob/main/wiki_movie_plots_deduped.csv](https://github.com/Rugvedhvaidya/Movie_Plots_Retrieval/blob/main/wiki_movie_plots_deduped.csv)**

# Project Description


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

# How to run the Project

- Open the ipynb in some editor like Jupyter Notebook, Google Colab
- Run all the cells in the file.
- Head to the query cell and enter the query in the input box.
- The top 10 documents will be retrieved and shown.
- You can give feedback for each document listed above this cell
- Using the above entered feedback the next cell will show a p-r graph.

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
	```
	
# Ranked Retrieval - 
The documents are ranked using cosine similarity and top 10 documents are retrieved.
## Calculating TF
We are calculating the tf score by using logarthermic TF factor as it is the best method to calculate
```py
(1+math.log2(t['count']))
```
Here **count** is Term frequency if the term is not present then we are taking TF as 1
## Calculated idf
```py
	def idf_func(c):
  	return math.log2(total_doc/c)
```

## Term Weighting

### Calculating TF-IDF score 

```py
tfidf_dict={}
tfidf_list=[]
for term in inverted_index.keys():

tfidf_list.append(term)

temp=inverted_index[term]['posting_list']

for t in temp:

tfidf_dict[t['docID'],term]=inverted_index[term]['idf']*(1+math.log2(t['count']))
```

Here we are taking Idf which we calculated and added to inverted index 
and TF and multiplying both of them and add it to **tfidf_dict**

## Cosine similarity
Here the tf-idf score are taken and stored into numpy arrays.
```py
	D = np.zeros((total_doc,len(inverted_index)))
	for i in tfidf_dict:
  		k=tfidf_list.index(i[1])
  		D[i[0]][k] = tfidf_dict[i]
```
Here the function for cosine similarity is written
```py
def cosine_similarity1(query_dict):
 
  Q=  np.zeros((len(inverted_index)))
  res=[]
  for i in inverted_index.keys():
     k=tfidf_list.index(i)    
     if i in query_dict.keys():
       Q[k]=query_dict[i]
  for row in range(D.shape[0]):
    result= 1 - spatial.distance.cosine(D[row], Q)
    res.append([row,result])
  return res
 ```

## Query Input- 
A query can be asked by user.

## Query processing-
Here we removed the stopwords and special characters in the query because they are not going to be used in term weighting
```py
query=input("Enter your query:-")
q_list= preprocess(query)
```
## Evaluation- 
Using users' feedback precision and recall are calculated. p-r curve is drawn at last.
- Recall = #(relevant retrieved)/#(relevant)-> tp/tp+fn
- Precision = #(relevant retrieved)/#(retrieved) -> tp/tp+fp
- p-r graphs are drawn using plot
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
## Non-trivial tasks
### Task-1
For Non trivial task we are doing filtering by year where we are retrieving all the documents which are released in the same year

### Task-2
For Non trivial task we are doing filtering by Genre where we are retrieving all the documents which are released in the same year

## Team Members
- **Rugvedh**
- **Likhith**
- **Pranay**
- **Bhaskar**
