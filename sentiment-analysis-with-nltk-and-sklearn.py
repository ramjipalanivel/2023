#!/usr/bin/env python
# coding: utf-8

# #### Sentiment Analysis with NLTK and ScikitLearn: movie reviews and tweets
# Sentiment analysis on movie reviews and tweets is performed using the NLTK datasets:
# * movie_review corpus
# * twitter_samples corpus
# For each dataset, perform sentiment analysis using:
# 1. NLTK's pre-trained built-in sentiment analyser
# 2. Customised NLTK classifier
# 3. ScikitLearn classifiers
# 4. Compare the performance and results of all three methods
# 
# 

# In[1]:


# Language resources from NLTK
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download(["names",
               "stopwords",
               "twitter_samples",
               "movie_reviews",
               "averaged_perceptron_tagger",
               "vader_lexicon",
               "punkt"
              ])

# Sklearn ML Classifiers
import sklearn
from sklearn import naive_bayes, neighbors, tree, ensemble, linear_model, neural_network, discriminant_analysis


# Data processing and visualisation
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from random import shuffle

# Pandas and print rounding
pd.set_option('precision', 3) 
get_ipython().run_line_magic('precision', '3')


# In[3]:


# Load twitter_samples data
tweets = nltk.corpus.twitter_samples
print(tweets.fileids())

# To check methods for twitter_samples
# tweets.ensure_loaded()
# help(tweets)


# twitter_samples contain 3 files, with positive and negative tweets store separately, and also together in one file.
# 
# ##### Read in the positive and negative tweets; store separately. Print out a few in each category.

# In[4]:


pos_tweets = tweets.strings(fileids = 'positive_tweets.json')
neg_tweets = tweets.strings('negative_tweets.json')

print('\033[94m',"Number of positive tweets: ",len(pos_tweets))
for i in range(5):
    print('\033[90m',pos_tweets[i],end = "\n\n\n")
    
print('\033[91m',"Number of negative tweets: ",len(neg_tweets))
for i in range(5):
    print('\033[90m',neg_tweets[i],end = "\n\n\n")


# In[5]:


# Tokenising the tweets splits words and punctuation; emoticons lost
'''
print('\033[91m',"Number of negative tweets: ",len(neg_tweets))
for i in range(5):
    print('\033[90m',nltk.word_tokenize(pos_tweets[i]),end = "\n\n\n")
'''


# In[6]:


pos_tweet_tokens = tweets.tokenized(fileids = 'positive_tweets.json')
neg_tweet_tokens = tweets.tokenized(fileids = 'negative_tweets.json')

pos_tokens = [len(tweet) for tweet in pos_tweet_tokens]
neg_tokens = [len(tweet) for tweet in neg_tweet_tokens]

len_fd_pos = nltk.FreqDist(pos_tokens).most_common(20)
len_pos_series = pd.Series(dict(len_fd_pos))

len_fd_neg = nltk.FreqDist(neg_tokens).most_common(20)
len_neg_series = pd.Series(dict(len_fd_neg))

frame = { 'Positive': len_pos_series, 'Negative': len_neg_series }
#Creating DataFrame by passing Dictionary
word_fd = pd.DataFrame(frame)

tokens = pd.DataFrame.from_dict({"Positive":pos_tokens,"Negative": neg_tokens})
tokens.describe()


# In[7]:


fig = plt.figure() # create figure

ax0 = fig.add_subplot(1,2,1) # add subplot 1 (1 row, 2 columns, first plot)
ax1 = fig.add_subplot(1,2,2) # add subplot 2 (1 row, 2 columns, second plot). See tip below**

# Subplot 1: Line plot
word_fd.plot(xlabel = "Number of tokens", ylabel = "Frequency", title = "Number of tokens per tweet", figsize = (14,6),xlim=(0,20), ax=ax0) # add to subplot 2
ax0.set_title ('Number of tokens per tweet')
ax0.set_ylabel('Frequency')
ax0.set_xlabel('Number of Tokens')

# Subplot 2: Box plot
tokens.plot(kind='box', vert=False, figsize=(20, 6), ax=ax1, xlim = (0,41), xticks = np.arange(0, 41, 10)) # add to subplot 1
ax1.set_title('Number of tokens per tweet')
ax1.set_xlabel('Number of Tokens')
#ax1.set_ylabel()
plt.savefig('Tweet token count')
plt.show()


# In[8]:


pos_tweet_token_words = []
for tweet in pos_tweet_tokens:
    token_words = [token for token in tweet if token.isalpha()]
    pos_tweet_token_words.append(token_words)

neg_tweet_token_words = []
for tweet in neg_tweet_tokens:
    token_words = [token for token in tweet if token.isalpha()]
    neg_tweet_token_words.append(token_words)
    
# Print first five tweet word tokens for positive (blue) and negative (red)  
for i in range(5):
    print('\033[94m',pos_tweet_token_words[i])
    
for i in range(5):
    print('\033[91m',neg_tweet_token_words[i])


# In[8]:


pos_words = [len(tweet) for tweet in pos_tweet_token_words]
neg_words = [len(tweet) for tweet in neg_tweet_token_words]

len_words_fd_pos = nltk.FreqDist(pos_words).most_common(20)
len_words_pos_series = pd.Series(dict(len_words_fd_pos))
#print(len_pos_series)

len_words_fd_neg = nltk.FreqDist(neg_words).most_common(20)
len_words_neg_series = pd.Series(dict(len_words_fd_neg))
# sns.lineplot(len_pos_series, len_neg_series)

frame = { 'Positive': len_words_pos_series, 'Negative': len_words_neg_series }
#Creating DataFrame by passing Dictionary
word_fd = pd.DataFrame(frame)
word_fd.head()

words = pd.DataFrame.from_dict({"Positive":pos_words,"Negative": neg_words})
words.describe()


# In[9]:


fig = plt.figure() # create figure

ax0 = fig.add_subplot(1,2,1) # add subplot 1 (1 row, 2 columns, first plot)
ax1 = fig.add_subplot(1,2,2) # add subplot 2 (1 row, 2 columns, second plot). See tip below**

# Subplot 1: Line plot
word_fd.plot(xlabel = "Number of word tokens", ylabel = "Frequency", title = "Number of words per tweet", figsize = (14,6),xlim=(0,20), ax=ax0) # add to subplot 2
ax0.set_title ('Number of punctuation tokens per tweet')
ax0.set_ylabel('Frequency')
ax0.set_xlabel('Number of Tokens')

# Subplot 2: Box plot
words.plot(kind='box', vert=False, figsize=(20, 6), ax=ax1) # add to subplot 1
ax1.set_title('Number of word tokens per tweet')
ax1.set_xlabel('Number of Tokens')
#ax1.set_ylabel()
plt.savefig('Tweet word count')
plt.show()


# ##### The number of words per tweet is similar for positive and negative tweets.

# In[10]:


pos_tweet_token_punct = []
for tweet in pos_tweet_tokens:
    token_punct = [token for token in tweet if not token.isalpha()]
    pos_tweet_token_punct.append(token_punct)

neg_tweet_token_punct = []
for tweet in neg_tweet_tokens:
    token_punct = [token for token in tweet if not token.isalpha()]
    neg_tweet_token_punct.append(token_punct)
    
# Print first five tweet word tokens for positive (blue) and negative (red)  
for i in range(5):
    print('\033[94m',pos_tweet_token_punct[i])
    
for i in range(5):
    print('\033[91m',neg_tweet_token_punct[i])


# * **Will need to remove social media tags starting with @**
# * **Build freq dist of punct used in positive and negative tweets**

# In[11]:


pos_punct = [len(tweet) for tweet in pos_tweet_token_punct]
neg_punct = [len(tweet) for tweet in neg_tweet_token_punct]

len_punct_fd_pos = nltk.FreqDist(pos_punct).most_common()
len_punct_pos_series = pd.Series(dict(len_punct_fd_pos))
#print(len_pos_series)

len_punct_fd_neg = nltk.FreqDist(neg_punct).most_common()
len_punct_neg_series = pd.Series(dict(len_punct_fd_neg))
# sns.lineplot(len_pos_series, len_neg_series)

frame = { 'Positive': len_punct_pos_series, 'Negative': len_punct_neg_series }
#Creating DataFrame by passing Dictionary
punct_fd = pd.DataFrame(frame)
punct_fd.head()

punct = pd.DataFrame.from_dict({"Positive":pos_punct,"Negative": neg_punct})
punct.describe()


# In[9]:


fig = plt.figure() # create figure

ax0 = fig.add_subplot(1,2,1) # add subplot 1 (1 row, 2 columns, first plot)
ax1 = fig.add_subplot(1,2,2) # add subplot 2 (1 row, 2 columns, second plot). See tip below**

# Subplot 1: Line plot
punct_fd.plot(xlabel = "Number of tokens", ylabel = "Frequency", title = "Number of punctuation per tweet", figsize = (14,6), xlim=(0,16), ax=ax0) # add to subplot 2
ax0.set_title ('Number of punctuation tokens per tweet')
ax0.set_ylabel('Frequency')
ax0.set_xlabel('Number of Tokens')

# Subplot 2: Box plot
ax1 = punct.plot(kind='box', vert=False, figsize=(20, 6), ax=ax1, xlim = (0,17), xticks = np.arange(0, 17, 2)) # add to subplot 1
ax1.set_title('Number of punctuation tokens per tweet')
ax1.set_xlabel('Number of Tokens')
#ax1.set_ylabel()
plt.savefig('Tweet punctuation token count')
plt.show()


# In[13]:


#help(nltk.collocations.BigramCollocationFinder)


# ### Examine common tokens in positive and negative tweets

# In[14]:


pos_tweet_tokens = tweets.tokenized(fileids = 'positive_tweets.json')
neg_tweet_tokens = tweets.tokenized(fileids = 'negative_tweets.json')


# In[15]:


# Make lists of positive and negative words
pos_words_all = []
for tweet in pos_tweet_tokens:
    for token in tweet:
        pos_words_all.append(token.lower())
    pos_words_all.append("\t\t\t")
        
neg_words_all = []
for tweet in neg_tweet_tokens:
    for token in tweet:
        neg_words_all.append(token.lower())
    neg_words_all.append("\t\t\t")


# In[16]:


N = 30
pos_words = nltk.FreqDist(pos_words_all)
top_pos_words = pos_words.most_common(N)

neg_words = nltk.FreqDist(neg_words_all)
top_neg_words = neg_words.most_common(N)

print("COMMON TOKENS \n",'\033[94m',"Positive:",'\t\t\t','\033[91m',"Negative:")
for i in range(1,N):
    print('\033[94m',top_pos_words[i],'\t\t\t','\033[91m',top_neg_words[i])


# #### Examine top words

# In[10]:


# Make lists of positive and negative words
pos_words_all = []
for tweet in pos_tweet_tokens:
    for token in tweet:
        if token.isalpha():
            pos_words_all.append(token.lower())
    pos_words_all.append("\t\t\t")
        
neg_words_all = []
for tweet in neg_tweet_tokens:
    for token in tweet:
        if token.isalpha():
            neg_words_all.append(token.lower())
    neg_words_all.append("\t\t\t")


# In[18]:


N = 30
pos_words = nltk.FreqDist(pos_words_all)
top_pos_words = pos_words.most_common(N)

neg_words = nltk.FreqDist(neg_words_all)
top_neg_words = neg_words.most_common(N)

print("COMMON WORDS \n",'\033[94m',"Positive:",'\t\t\t','\033[91m',"Negative:")
for i in range(1,N):
    print('\033[94m',top_pos_words[i],'\t\t\t','\033[91m',top_neg_words[i])


# #### Examine top punctuation

# In[11]:


# Make lists of positive and negative words
pos_words_all = []
for tweet in pos_tweet_tokens:
    for token in tweet:
        if not token.isalpha():
            pos_words_all.append(token.lower())
    pos_words_all.append("\t\t\t")
        
neg_words_all = []
for tweet in neg_tweet_tokens:
    for token in tweet:
        if not token.isalpha():
            neg_words_all.append(token.lower())
    neg_words_all.append("\t\t\t")


# In[20]:


N = 30
pos_words = nltk.FreqDist(pos_words_all)
top_pos_words = pos_words.most_common(N)

neg_words = nltk.FreqDist(neg_words_all)
top_neg_words = neg_words.most_common(N)

print("COMMON PUNCTUATION TOKENS \n",'\033[94m',"Positive:",'\t\t\t','\033[91m',"Negative:")
for i in range(1,N):
    print('\033[94m',top_pos_words[i],'\t\t\t','\033[91m',top_neg_words[i])


# ### Examine common bigrams in positive and negative tweets

# In[12]:


# N - number of top bigrams to find
N = 30

# Find N positive bigrams
pos_bigrams = nltk.collocations.BigramCollocationFinder.from_words(pos_words_all)
pos_top_bigrams = pos_bigrams.ngram_fd.most_common(N)

# Find N negative bigrams
neg_bigrams = nltk.collocations.BigramCollocationFinder.from_words(neg_words_all)
neg_top_bigrams = neg_bigrams.ngram_fd.most_common(N)

print("COMMON BIGRAMS \n",'\033[94m',"Positive:",'\t\t\t\t\t','\033[91m',"Negative:")
for i in range(N):
    print('\033[94m',pos_top_bigrams[i],"\t\t\t\t\t",'\033[91m',neg_top_bigrams[i])


# ### Examine common trigrams in positive and negative tweets 

# In[13]:


# N - number of top trigrams to find
N = 30

# Find N positive trigrams
pos_trigrams = nltk.collocations.TrigramCollocationFinder.from_words(pos_words_all)
pos_top_trigrams = pos_trigrams.ngram_fd.most_common(N)

# Find N negative trigrams
neg_trigrams = nltk.collocations.TrigramCollocationFinder.from_words(neg_words_all)
neg_top_trigrams = neg_trigrams.ngram_fd.most_common(N)

print("COMMON TRIGRAMS \n",'\033[94m',"Positive:",'\t\t\t\t\t','\033[91m',"Negative:")
for i in range(N):
    print('\033[94m',pos_top_trigrams[i],"\t\t\t",'\033[91m',neg_top_trigrams[i])


# In[23]:


# N - number of top quadgrams to find
N = 30

# Find N positive quadgrams
pos_quadgrams = nltk.collocations.QuadgramCollocationFinder.from_words(pos_words_all)
pos_top_quadgrams = pos_quadgrams.ngram_fd.most_common(N)

# Find N negative quadgrams
neg_quadgrams = nltk.collocations.QuadgramCollocationFinder.from_words(neg_words_all)
neg_top_quadgrams = neg_quadgrams.ngram_fd.most_common(N)

print("COMMON QUADGRAMS \n",'\033[94m',"Positive:",'\t\t\t\t\t','\033[91m',"Negative:")
for i in range(N):
    print('\033[94m',pos_top_quadgrams[i],"\t",'\033[91m',neg_top_quadgrams[i])


# ### Preprocessing required 
# 
# #### Remove
# * tags
# * common words
# * links
# 
# #### Add features
# * presence of emoticons 
# * collocations
# * tokens/ per tweet
# * punctuation / per tweet

# ## NLTK built-in pre-trained sentiment analyser
# ### 1. Load datasets
# ### 2. Preprocess
# ### 3. Train-test-split
# ### 4. Analyse sentiment
# ### 5. Evaluate

# In[14]:


# Instantiate NLTK SentimentIntensityAnalyzer (imported as Sia)
sia = SentimentIntensityAnalyzer()
# help(sia)
sia.polarity_scores("I like that")


# In[15]:


from random import shuffle
def is_positive(tweet:str) -> bool:
    """True if tweet has positive compound sentiment, False otherwise."""
    return sia.polarity_scores(tweet)["compound"] > 0


# In[16]:


# Load twitter data as strings
pos_tweets = tweets.strings('positive_tweets.json')
neg_tweets = tweets.strings('negative_tweets.json')

# Label each tweet and store in dataframes with labels
df_pos_tweets = pd.DataFrame({"Tweet":[tweet for tweet in pos_tweets],"Label":[1 for tweet in pos_tweets]})
df_neg_tweets = pd.DataFrame({"Tweet":[tweet for tweet in neg_tweets],"Label":[0 for tweet in neg_tweets]})

# Stack the two DataFrames together to make one labelled dataset
# 1 - positive, 0 - negative
df_all_tweets = pd.concat([df_pos_tweets,df_neg_tweets], ignore_index=True)


# In[17]:


from numpy import random


# In[18]:


# Copy the dataframe containing all labelled tweets and add a column to store sia prediction
sia_all_tweets = df_all_tweets.copy()
sia_all_tweets["Random"] = random.binomial(n=1, p=.5, size=10000)
sia_all_tweets["Random_true"] = -1
sia_all_tweets["Sia"] = -1
sia_all_tweets["Sia_true"] = -1

for index, tweet in enumerate(sia_all_tweets.Tweet):
    prediction = is_positive(tweet)
    sia_all_tweets.at[index,'Sia'] = int(prediction)
    sia_all_tweets.at[index,'Sia_true'] = int(prediction==sia_all_tweets.loc[index,"Label"])
    sia_all_tweets.at[index, 'Random_true'] = int(sia_all_tweets.loc[index,"Random"]==sia_all_tweets.loc[index,"Label"])

sia_all_tweets
#print(sia_all_tweets.describe())
#print(sia_all_tweets.head())
#for tweet in sia_all_tweets:


# The accuracy of the NLTK built in sentiment analyser for tweet_samples data is 85.7% (n = 10,000)

# #### Built in sentiment analyser on movie reviews

# In[19]:


movies = nltk.corpus.movie_reviews
print(movies.categories())

# Store fileids for positive and negative movie reviews
pos_movie_ids = movies.fileids(categories = "pos")
neg_movie_ids = movies.fileids(categories = "neg")

# Combine fileids and create lables for each fileid
all_movie_ids = pos_movie_ids + neg_movie_ids
all_movie_labels = [1 if label[:3] == "pos" else 0 for label in all_movie_ids]

print("There are {} positive and {} negative movie reviews.".format(len(pos_movie_ids),len(neg_movie_ids)))

# print(type(movies.raw(fileids= pos_movie_ids[0])))
# To get the movie review as string
# movies.raw(fileids= pos_movie_ids[0])

# Make a dataframe with movie ids and label for each id
df_movies = pd.DataFrame({"Movie_id":all_movie_ids,"Label":all_movie_labels})


# In[21]:


from statistics import mean, median, mode
def is_positive_review(review_id: str) -> bool:
    """True if the average of all sentence compound scores is positive."""
    text = movies.raw(review_id)
    scores = [sia.polarity_scores(sentence)["compound"] for sentence in nltk.sent_tokenize(text)]
    return median(scores) > 0


# In[22]:


# Make a copy of the dataframe and intialise columns for sia classifier
sia_movies = df_movies.copy()
sia_movies["Sia"] = -1
sia_movies["Sia_true"] = -1

for index, movie_id in enumerate(sia_movies.Movie_id):
    prediction = is_positive_review(movie_id)
    sia_movies.at[index,'Sia'] = int(prediction)
    sia_movies.at[index,'Sia_true'] = int(prediction==sia_movies.loc[index,"Label"])
    #sia_movies.at[index, 'Random_true'] = int(sia_movies.loc[index,"Random"]==sia_movies.loc[index,"Label"])

sia_movies.describe()


# ## Comparing Additional Classifiers

# In[23]:


# Instantiate classifiers, make a mapping of names to instances
classifiers = {
    "BernoulliNB": sklearn.naive_bayes.BernoulliNB(),
    "ComplementNB": sklearn.naive_bayes.ComplementNB(),
    "MultinomialNB": sklearn.naive_bayes.ComplementNB(),
    "KNeighborsClassifier": sklearn.neighbors.KNeighborsClassifier(),
    "DecisionTreeClassifier": sklearn.tree.DecisionTreeClassifier(),
    "RandomForestClassifier": sklearn.ensemble.RandomForestClassifier(),
    "AdaBoostClassifier": sklearn.ensemble.AdaBoostClassifier(),
    "LogisticRegression": sklearn.linear_model.LogisticRegression(),
    "MLPClassifier": sklearn.neural_network.MLPClassifier(),
    "QuadraticDiscriminantAnalysis": sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()
}


# In[24]:


#movie_y.columns
#movie_y = movie_data.Label
#movie_features = ["Sia","",""]
#movie_X  = movie_data[movie_feaures]
#movie_X.describe()
#movie_X.head()


# In[26]:


unwanted = nltk.corpus.stopwords.words("english")
unwanted.extend([w.lower() for w in nltk.corpus.names.words()])

def skip_unwanted(pos_tuple):
    word, tag = pos_tuple
    if not word.isalpha() or word in unwanted:
        return False
    if tag.startswith("NN"):
        return False
    return True

pos_movie_words = [word for word, tag in filter(skip_unwanted, nltk.pos_tag(nltk.corpus.movie_reviews.words(categories = ["pos"])))]
neg_movie_words = [word for word, tag in filter(skip_unwanted, nltk.pos_tag(nltk.corpus.movie_reviews.words(categories = ["neg"])))]

pos_movie_word_fd = nltk.FreqDist(pos_movie_words)
neg_movie_word_fd = nltk.FreqDist(neg_movie_words)

common_movie_word_set = set(pos_movie_word_fd).intersection(neg_movie_word_fd)

for word in common_movie_word_set:
    del pos_movie_word_fd[word]
    del neg_movie_word_fd[word]

# Top N words
N = 500
top_100_pos_movie_words = {word for word, count in pos_movie_word_fd.most_common(N)}
top_100_neg_movie_words = {word for word, count in neg_movie_word_fd.most_common(N)}


# In[27]:


def extract_movie_features(text):
    movie_features = {}
    wordcount = 0
    compound_scores = list()
    positive_scores = list()
    n_tokens = len(list(nltk.word_tokenize(text)))
    n_types = len(set(nltk.word_tokenize(text)))
    lexical_diversity = n_types/n_tokens
    
    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            if word.lower() in top_100_pos_movie_words:
                #wordcount += 1
                wordcount += pos_movie_word_fd[word]
                # here can modify to increment by frequency of word in FreqDist
    
    compound_scores.append(sia.polarity_scores(sentence)["compound"])
    positive_scores.append(sia.polarity_scores(sentence)["pos"])
    
    # Add 1 to final compound score to ensure it is always positive
    # to satisfy some classifiers requirements later.
    movie_features["mean_compound"] = mean(compound_scores) + 1
    movie_features["mean_positive"] = mean(positive_scores)
    movie_features["wordcount"] = wordcount
    movie_features["token_count"] = n_tokens
    movie_features["type_count"] = n_types
    movie_features["lexical_diversity"] = lexical_diversity
    
    return movie_features


# In[29]:


# movies_feautres is a list of tuples, 
# each tuple contains a dictionary of features and a label
movie_ids = nltk.corpus.movie_reviews.fileids()
movies_features = [
    (extract_movie_features(nltk.corpus.movie_reviews.raw(fileid)),fileid[:3])
    for fileid in movie_ids
]


# In[30]:


# Train and test split is 4:1
train_count = len(movies_features)//5



# Create dictionary and initialise keys, and empty lists as values to store accuracy scores
movie_accuracy = {key: [] for key in classifiers.keys()}

# repeat M times
M = 10
for i in range(M):
    # Features are shuffled
    shuffle(movies_features)
    # For each sklearn classifier train and test the model, store accuracy
    for name, sklearn_classifier in classifiers.items():
        classifier = nltk.classify.SklearnClassifier(sklearn_classifier, sparse=False)
        classifier.train(movies_features[:train_count])
        accuracy = nltk.classify.accuracy(classifier, movies_features[train_count:])

        # Accuracy as percentage 2dp
        movie_accuracy[name].append(round(100*accuracy,2))

# Results
print("Analysis complete")
for key,value in movie_accuracy.items():
    print(key, ": ",value)


# In[38]:


movie_experiment_3 = pd.DataFrame(movie_accuracy)


# In[31]:


pd.set_option('precision',2)
movie_experiment_3.describe()
#movie_experiment_1


# In[40]:


# Create lists for the plot
x_pos = np.arange(len(classifiers))

# Get the mean accuracy scores
accuracy_mean = list(movie_experiment_3.describe().loc["mean",:])

# Get the standard deviation of accuracy scores
accuracy_sd = list(movie_experiment_3.describe().loc["std",:])


# In[41]:


df_movie_experiment_3 = pd.DataFrame(columns = classifiers.keys())
df_movie_experiment_3.loc[len(df_movie_experiment_3)] = accuracy_mean
df_movie_experiment_3.loc[len(df_movie_experiment_3)] = accuracy_sd
df_movie_experiment_3.index = ["Mean","SD"]
df_movie_experiment_3 = df_movie_experiment_3.sort_values(by = 'Mean', axis = 1, ascending = True)
df_movie_experiment_3


# In[42]:


df_movie_experiment_3.to_csv('Movie_experiment_3_summary.csv')
df_movie_experiment_3.to_pickle('Movie_experiment_3_summary.pkl')


# In[43]:


# Build the plot
fig, ax = plt.subplots(figsize = (14,7))
ax.barh(x_pos, df_movie_experiment_3.loc["Mean",:], xerr=df_movie_experiment_3.loc["SD",:], align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_xlim((55,85))
ax.set_xlabel('Mean Accuracy (%)')
ax.set_yticks(x_pos)
ax.set_yticklabels(df_movie_experiment_3.columns)
ax.set_title('Comparison of Classifier Accuracy for Sentiment Analysis of Movie Reviews')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('Movie_experiment_3_accuracy.png')
plt.show()


# ## Conclusion
# * The Logistic regression classifier is the best performing model for sentiment analysis of movie reviews.
# * Building a frequency distribution of common words in positive reviews increases model performance. Increasing the number of words from 100 to 500 gives a 10% improvement in model performance.
# 
# 

# In[44]:


'''
# Build frequency distributio of words 
fd = nltk.FreqDist(words)
common_words = fd.most_common(50)
plt.figure(figsize = (15,6))
fd.plot(50,cumulative=True)
plt.show()
print(common_words)
'''


# In[45]:


#custom_stopwords = [w.lower() for (w,freq) in common_words]
#print(custom_stopwords)
#w_out_custom_stopwords = [w for w in words if w.lower() not in custom_stopwords]
#print(len(w_out_custom_stopwords))


# In[46]:


# Maybe remove some step words that could be useful in sentiment analysis
#print(sorted(stopwords))
# nor not no against aren't won't wouldn't would shouldn't souldn off down
# above


# # REFERENCES
# 
# * [Text Classification using NLTK](https://mylearningsinaiml.wordpress.com/nlp/text-classification-using-nltk/)
# * [Learning to Classify Text](https://www.nltk.org/book/ch06.html)
