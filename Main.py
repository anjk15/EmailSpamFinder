import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from sklearn.datasets import fetch_openml
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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

class SpamClassifier(nn.Module):
    def __init__(self):
        super(SpamClassifier, self).__init__()
        self.L1 = nn.Linear(1000, 512)
        self.L2 = nn.Linear(512, 128)
        self.L3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = torch.sigmoid(self.L3(x))  
        return x

        

