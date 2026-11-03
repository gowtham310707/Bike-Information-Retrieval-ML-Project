# Bike Information Retrieval and Machine Learning Project

## Project title
**Bike Information Retrieval and Text Classification Using KNN, Multinomial Naive Bayes and Logistic Regression**

## Aim
Develop a bike-focused text Information Retrieval system and compare three machine-learning algorithms. The project:
1. Represents bike-related documents using TF-IDF.
2. Trains KNN, Multinomial Naive Bayes and Logistic Regression.
3. Evaluates accuracy, weighted precision, weighted recall, weighted F1-score and runtime.
4. Ranks algorithms using measured accuracy.
5. Retrieves the five most relevant bike documents for a user-entered query using cosine similarity.

## Datasets / experiments

### Dataset 1: Included Bike Catalogue Text Corpus
`data/bike_catalog_texts.csv` is included in the ZIP. It contains 45 short illustrative bike-related records in three categories:
- `bicycle`
- `motorcycle`
- `electric_bike`

**Important:** This is a small teaching/demo corpus authored for this project, not a published benchmark dataset. It demonstrates the complete pipeline and can be expanded with genuine descriptions/articles for stronger research results.

### Dataset 2: 20 Newsgroups
The program downloads the standard 20 Newsgroups dataset through scikit-learn and selects:
- `rec.motorcycles`
- `rec.autos`

This provides a larger, real-world discussion-text experiment to compare motorcycle-related posts with automobile-related posts. The first run needs internet access.

## Algorithms
- **K-Nearest Neighbors (KNN):** classifies a document based on the labels of its nearest training documents.
- **Multinomial Naive Bayes:** estimates class probabilities from word features.
- **Logistic Regression:** learns weights for text features and predicts a class.

## Information Retrieval method
- Clean/lowercase text and remove common English stop words.
- Convert documents to TF-IDF feature vectors.
- Convert the user's query using the same vectorizer.
- Calculate cosine similarity between the query and documents.
- Return the top five highest-scoring records.

## Project structure

```text
Bike_IR_ML_Project/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── bike_catalog_texts.csv
└── results/
    └── (generated CSV tables, graphs and confusion matrices)
```

## Windows setup and execution

Install Python 3.10 or later. Open a terminal in this project folder.

Create a virtual environment:
```bash
python -m venv venv
```

Activate it:
```bash
venv\Scripts\activate
```

Install requirements:
```bash
pip install -r requirements.txt
```

Run:
```bash
python main.py
```

The first experiment runs from the included CSV. The second experiment downloads 20 Newsgroups on its first run and therefore needs internet access.

## Output files
The `results/` folder is populated automatically with:
- one CSV comparison table per experiment
- accuracy comparison charts
- confusion matrix images for each algorithm

The terminal prints classification reports and algorithm rankings.

## Evaluation metrics
- **Accuracy:** percentage of all predictions that are correct.
- **Precision:** proportion of predicted class members that are correct.
- **Recall:** proportion of actual class members correctly identified.
- **Weighted F1-score:** weighted harmonic mean of precision and recall.
- **Training time:** time taken to fit the model.
- **Prediction time:** time taken to classify the test set.

## Suggested report structure
1. Abstract
2. Introduction
3. Problem statement
4. Objectives
5. Dataset description
6. Text preprocessing
7. TF-IDF feature extraction
8. Algorithms
9. Information Retrieval and cosine similarity
10. Experimental setup
11. Results and comparison table
12. Accuracy graphs and confusion matrices
13. Discussion and ranking
14. Conclusion
15. References

## Suggested problem statement
Online platforms contain large volumes of information about bicycles, motorcycles and electric bikes. Finding the most relevant information and organizing descriptions into useful categories can be time-consuming. This project applies TF-IDF and machine-learning classifiers to bike-related text, compares three algorithms using standard evaluation metrics, and demonstrates query-based retrieval using cosine similarity.

## Suggested objectives
- To prepare and preprocess bike-related text documents.
- To convert text into numerical TF-IDF vectors.
- To implement KNN, Multinomial Naive Bayes and Logistic Regression.
- To compare algorithms on two text datasets/experiments.
- To rank algorithms using measured accuracy and F1-score.
- To retrieve relevant bike documents for a user query.

## Honest reporting and limitations
Do not insert assumed accuracy values. Run the program and copy the actual generated metrics into your report. The included catalogue corpus is intentionally small and illustrative; results on it should not be presented as a large-scale benchmark. The 20 Newsgroups experiment provides a separate larger discussion-text comparison. The best algorithm may differ by dataset.
