# Fake News Detection

This project aims to detect fake news using machine learning techniques. It utilizes a dataset containing news articles labeled as either real or fake.

## Dataset

The dataset used for this project can be found [here](https://drive.google.com/file/d/1q5jpI5M1EA9x3YPrLupmiu3gffkmGlHj/view). It consists of news articles along with their corresponding labels indicating whether they are real (class 1) or fake (class 0).

### Data Preparation

Before running the code, make sure to download the dataset and save it as `News.csv` in the project directory. The code automatically reads this CSV file and preprocesses the text data for analysis.

## Setup

To run the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone <repository-url>
   ```
2. Install the required dependencies:

   ```bash
   pip install pandas seaborn matplotlib tqdm nltk wordcloud scikit-learn
   ```
3. Download NLTK resources:

   ```bash
   python -m nltk.downloader punkt stopwords
   ```
4. Run the main script:

   ```bash
   python main.py
   ```

## Usage

Upon running the script, it performs the following tasks:

1. Loads the dataset and preprocesses the text data.
2. Generates word clouds for real and fake news articles.
3. Displays a bar chart of the top words frequency.
4. Splits the data into training and testing sets.
5. Trains logistic regression and decision tree classifiers.
6. Evaluates the performance of the classifiers using accuracy scores and confusion matrices.

## Authors

- [Kairavi Pandya](pandyakairavi@gmail.com)
