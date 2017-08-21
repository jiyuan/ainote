# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 19:56:30 2017

@author: jiyuan
"""

import numpy as np
import pandas as pd
# import seaborn as sns

import timeit

## example of pandas.Series, 1-dim data structure
# pd.Series([1, 90, 'hey', np.nan], index=['a', 'B', 'C', 'd'])

## example of pandas.DataFrame, 2-dim data structure
pd.DataFrame({'day': [17, 30], 'month': [1, 12], 'year': [2010, 2017]})

## example of pandas.Panels, 3-dim data structure


## test the time to execute the following line
%timeit -n1000 titanic['is_old'] = titanic.apply(is_old_func, axis='columns')

## read data from file
data = pd.read_csv('file_name.csv', names = [
        "Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", 
        "Magnesium", "Total_phenols", "Falvanoids", "Nonflavanoid_phenols",
        "Proanthocyanins", "Color_intensity", "Hue", "OD280", "Proline"
        ])
data.describe().transpose()


## prepare data for train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


## 1. Data cleaning
## Homogenize missing values and different types of in the same feature, fix
## input errors, types, etc.

## 1.1 Inputation for missing values
## + Datasets contain missing values, often encoded as blanks, NaNs or other
##   placeholders
## + Ignoring rows and/or columns with missing values is possible, but at the
##   price of loosing data which might be valuable
## + Better strategy is to infer them from the known part of data
## + Strategies: Mean, Median, Mode, Using a model

from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
#imp.fit(X)
#imp.transform(Y)
## X 是用来训练缺失值的数据集，Y是需要填充的数据集


## 2. Aggregating
## Necessary when the entity to model is an aggregation from the provided data.


## 3. Pivoting
## Necessary when the entity to model is an aggregation from the provided data.



## 4. Feature Deriving


## 4.1 Numerical Features

## 4.1.1 Binning: Rounding, Binarization, Binning

## + Binarization
from sklearn import preprocessing
binarizer = preprocessing.Binarizer(threshold=1.0)
binarizer.transform(X)

## + Binning
df['xx'].quantile([.1,.2,.3,.4,.5,.6,.7,.8,.9])

## 4.1.2 Transformation: log trans, Scaling(MinMax, Standard_Z), Normalization

## + MinMax
from sklearn import preprocessing
minmax_scaler = preprocessing.MinMaxScaler()
x_minmax = minmax_scaler.fit_transform(x)

## + Standard_Z Scaling
from sklearn import preprocessing
from numpy as np
x_scaled = preprocessing.scale(x)

## + Normalization
from sklearn import preprocessing
x_normalized = preprocessing.normalize(x, norm='l2')

## 4.2 Categorical Features

## 4.2.1 One-Hot Encoding, 实际上是哑变量

from pyspark.ml.feature import OneHotEncoder, StringIndexer

string_indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
model = string_indexer.fit(df)
indexed = model.transform(df)

encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
encoded = encoder.transform(indexed)
encoded.show()

## 4.2.2 Large Categorical Variables, means category number is large.

## 4.2.3 Feature Hashing
## + Hashes categorical values into vectors with fixed-length.
## + lower sparsity and higher compression compared to OHE
## + Deals with new and rare categorical values(eg: new user-agents)
## + May introduce collisions

## + by python
import hashlib
def hashstr(s, nr_bins):
    return int(hashlib.md5(s.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1
categorical_value = 'ad_id=354424'  # original category
max_bins = 100000
hashstr(categorical_value, max_bins)

## + by tensorflow
import tensorflow as tf
ad_id_hashed = tf.contrib.layers.sparse_column_with_hash_bucket(
        'ad_id',
        hash_bucket_size=250000,
        dtype=tf.int64,
        combiner="sum"
        )

## 4.2.4 Bin-counting
## + Instead of using the actual categorical value, use a global statistic of
##   this category on historical data.
## + Useful for both linear and non-linear algorithms
## + May give collisions(save encoding for different categories)
## + Be careful about leakage
## + Strategies: Count, Average CTR(Click-Through Rate)


## 4.2.5 LabelCount encoding
## + Rank categorical variables by count in train set
## + Useful for both linear and non-linear algorithms(eg: decision trees)
## + Not sensitive to outliers
## + won't give same encoding to different variables


## 4.2.6 Category Embedding
## + Use a Neural Network to create dense embeddings from categorical variables.
## + Map categorical variables in a function approximation problem Euclidean spaces
## + Faster model training.
## + Less memory overhead.
## + Can give better accuracy than 1-hot encoded.

import tensorflow as tf

def get_embedding_size(unique_val_count):
    return int(math.floor(6*unique_val_count**0.25))

ad_id_hashed_feature = tf.contrib.layers.sparse_column_with_hash_bucket(
        'ad_id', hash_bucket_size=250000, dtype=tf.int64, combiner="sum")
embedding_size = get_embedding_size(ad_id_hashed_feature.length)
ad_embedding_feature = tf.contrib.layers.embedding_column(
        ad_id_hashed_feature, dimension=embedding_size, combiner="sum")

## 4.3 Temporal Features

## 4.3.1 Time Zone conversion
## + Factors to consider:
##   - Multiple time zones in some countries
##   - Daylight Saving Time(DST): Start and end DST dates

## 4.3.2 Time binning
## + Apply binning on time data to make it categorial and more general.
## + Binning a time in hours or periods of day.
## + Extraction: weekday/weekend, weeks, months, quarters, years.


## 4.3.3 Trendlines
## + Instead of encoding: total spend, encode things like:
##   Spend in last week, spend in last month, spend in last year.
## + Gives a trend to the algorithm:
##   two custoers with equal spend, can have wildly different behavior -- one
##   customer may be starting to spend more, while the other is starting to
##   decline spending.


## 4.3.4 Closeness to major events
## + Hardcode categorical features from dates
## + Example: Factors that might have major influence on spending behavior
## + Proximity to major events(holidays, major sports events)
##   - Eg. date_X_days_before_holidays
## + Proximity to wages payment date(monthly seasonality)
##   - Eg. first_saturday_of_the_month





## 4.3.5 Time differences
## + Differences between dates might be relevant
## + Example:
##   - user_interaction_date - published_doc_date
##     To model how recent was the ad when the user viewed it.
##     Hypothesis: user interests on a topic may decay over time
##   - last_user_interaction_date - user_interaction_date
##     To model how old was a given user interaction compared to his last
##     interaction




## 4.4 Spatial Features

## 4.4.1 Spatial Variables
## + Spatial variables encode a location in space, like:
##   - GPS-coordinates(lat./long.) - sometimes require projection to a different
##     coordinate system
##   - Street Addresses - require geocoding
##   - ZipCodes, Cities, States, Countries - usually enriched with the centroid
##     coordinate of the polygon(from external GIS data)
## + Derived features
##   - Distance between a user location and searched hotels(Expedia competition)
##   - Impossible travel speed(fraud detection)



## 4.4.2 Spatial Enrichment
## Usually useful to enrich with external geographic data(eg. Census demographics)



## 4.5 Textual data

## 4.5.1 Natural Language Processing
## + Cleaning: Lowercasing, Convert accented characters, Removing non-alphanumeric,
##             Repairing
## + Removing: Stopwords, Rare words, Common words
## + Tokenizing: 
##   - Encode punctuation marks
##   - Tokenize
##   - N-Grams
##   - Skip-grams
##   - Char-grams
##   - Affixes
## + Roots: Spelling correction, Chop, Stem, Lemmatize
## + Enrich: Entity Insertion/Extraction, Parse Trees, Reading Level


## 4.5.2 Text vectorization
## Represent each document as a feature vector in the vector space, where each
## position represents a word (token) and the contained value is its relevance
## in the document.
## + BoW(bag of words)
## + TF-IDF(Term Frequency - Inverse Document Frequency)
## + Embeddings(eg. Words2Vec, Glove)
## + Topic models(eg. LDA)

## + TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
        max_df=0.5, max_features=1000, min_df=2, stop_words='english')
tfidf_corpus = vectorizer.fit_transform(text_corpus)

## + Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)


## 4.5.3 Textual Similarities
## + Token similarity: Count number of tokens that appear in two texts.
## + Levenshtein/Hamming/Jaccard Distance: Check similarity between two strings,
##   by looking at number of operations needed to transform one in the other.
## + Word2Vec/Glove: Check cosine similarity between two word embedding vectors.


## 4.5.4 Topic Modeling
## + Latent Dirichlet Allocation(LDA) -> Probabilistic
## + Latent Semantic Indexing / Analysis(LSI/LSA) -> Matrix Factorization
## + Non-Negative Matrix Factorization(NMF) -> Matrix Factorization








## 4.6 Interaction Features



## clean data and the replace the missing value





## merge all xdata
