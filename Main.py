import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from sklearn.datasets import fetch_openml
from torch.utils.data import Dataset, DataLoader


# Load the dataset
data = fetch_openml(data_id=46099, as_frame=True)

# Extract features and labels
X = data.data       # contains email text 'data'
y = data.target     # contains 'label' column (either 0 or 1)

nltk.download('stopwords')



# Preprocessing and vectorization using TF-IDF 
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=1000)
X_tfidf = vectorizer.fit_transform(X)


# Convert to PyTorch tensor
email_tensor = torch.tensor(X_tfidf.toarray(), dtype=torch.float32)



