import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from sklearn.datasets import fetch_openml
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


nltk.download('stopwords')


data = fetch_openml(data_id=46099, as_frame=True, target_column='label')


X = data.data['text_combined']        
y = data.target.astype(int)           

vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=1000)
X_tfidf = vectorizer.fit_transform(X)


X_tensor = torch.tensor(X_tfidf.toarray(), dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1) 

full_dataset = TensorDataset(X_tensor, y_tensor)


train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


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


model = SpamClassifier()
