import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from tqdm import tqdm 
import re 
import nltk 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Read the CSV file into a DataFrame
data = pd.read_csv('News.csv', index_col=0) 

# Drop specified columns from the DataFrame
data = data.drop(["title", "subject", "date"], axis=1)

# Shuffle the rows of the DataFrame
data = data.sample(frac=1)

# Reset the index of the DataFrame
data.reset_index(inplace=True)
data.drop(["index"], axis=1, inplace=True)

# Display the first few rows of the DataFrame
print(data.head())

# Display the shape of the DataFrame
print(data.shape)

# Print the count of missing values for each column
print(data.isnull().sum())

# Plot a count plot
sns.countplot(data=data, x='class', order=data['class'].value_counts().index)
plt.show()

def preprocess_text(text_data): 
    preprocessed_text = [] 
    
    for sentence in tqdm(text_data): 
        sentence = re.sub(r'[^\w\s]', '', sentence) 
        preprocessed_text.append(' '.join(token.lower() 
                                for token in str(sentence).split() 
                                if token not in stopwords.words('english'))) 

    return preprocessed_text

# Preprocess text data
data['text'] = preprocess_text(data['text'].values)

# Generate word clouds for Real and Fake news
def generate_word_cloud(data, classification):
    consolidated = ' '.join(word for word in data['text'][data['class'] == classification].astype(str))
    wordCloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110, collocations=False) 
    plt.figure(figsize=(15, 10)) 
    plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear') 
    plt.axis('off') 
    plt.show()

generate_word_cloud(data, 1)  # Real news
generate_word_cloud(data, 0)  # Fake news

# Get top n words
def get_top_n_words(corpus, n=None): 
    vec = CountVectorizer().fit(corpus) 
    bag_of_words = vec.transform(corpus) 
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()] 
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True) 
    return words_freq[:n] 

common_words = get_top_n_words(data['text'], 20) 
df1 = pd.DataFrame(common_words, columns=['Review', 'count']) 

df1.groupby('Review').sum()['count'].sort_values(ascending=False).plot( 
    kind='bar', 
    figsize=(10, 6), 
    xlabel="Top Words", 
    ylabel="Count", 
    title="Bar Chart of Top Words Frequency"
) 
plt.show()

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size=0.25)

# Vectorize text data
vectorization = TfidfVectorizer() 
x_train = vectorization.fit_transform(x_train) 
x_test = vectorization.transform(x_test)

# Train logistic regression model
model = LogisticRegression() 
model.fit(x_train, y_train) 

# Evaluate logistic regression model
print("Logistic Regression Accuracy:")
print("Training set accuracy:", accuracy_score(y_train, model.predict(x_train))) 
print("Test set accuracy:", accuracy_score(y_test, model.predict(x_test))) 

# Train decision tree classifier
model = DecisionTreeClassifier() 
model.fit(x_train, y_train) 

# Evaluate decision tree classifier
print("\nDecision Tree Accuracy:")
print("Training set accuracy:", accuracy_score(y_train, model.predict(x_train))) 
print("Test set accuracy:", accuracy_score(y_test, model.predict(x_test))) 

# Confusion matrix of Results from Decision Tree classification 
cm = metrics.confusion_matrix(y_test, model.predict(x_test)) 
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True]) 
cm_display.plot() 
plt.show() 
