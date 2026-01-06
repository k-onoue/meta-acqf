import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from bs4 import BeautifulSoup
from scipy.io import loadmat
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# CONFIGURATION
# ==========================================
DATA_RAW_DIR = "./data/raw"
DATA_PROCESSED_DIR = "./data"
SEED = 42

# Reuters ModApte Top 8 Classes
REUTERS_CLASSES = ["earn", "acq", "money-fx", "grain", "crude", "trade", "interest", "ship"]

# WebKB: 4 Classes (Roles) from 4 Universities
WEBKB_CLASSES = ["student", "faculty", "course", "project"]
WEBKB_UNIS = ["cornell", "texas", "washington", "wisconsin"]

# NeurIPS: 13 Years (0-12) used as semantic proxies for topics
NIPS_YEARS = [f"nips{i:02d}" for i in range(13)]

# ==========================================
# ORACLE NETWORK (The Ground Truth Generator)
# ==========================================
class OracleNetwork(nn.Module):
    """4-layer feed-forward classifier used to produce semantic embeddings."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        embeddings = self.net(x)
        logits = self.classifier(embeddings)
        return embeddings, logits


def train_oracle(X_bow, y_labels, epochs=50, lr=1e-3):
    """Train the oracle classifier and return embeddings."""
    print(
        f"   -> Training Oracle on {X_bow.shape[0]} docs, {X_bow.shape[1]} words, {len(np.unique(y_labels))} classes..."
    )

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.FloatTensor(X_bow).to(device)
    y_tensor = torch.LongTensor(y_labels).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = OracleNetwork(input_dim=X_bow.shape[1], num_classes=len(np.unique(y_labels))).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            _, logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/{epochs}: Loss {total_loss/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        embeddings, _ = model(X_tensor)

    return embeddings.cpu()


def clean_and_vectorize(text_data, labels, min_docs=50, min_words=50):
    """Apply paper filtering: remove rare terms and short docs."""
    print("   -> Vectorizing...")
    vectorizer = CountVectorizer(min_df=min_docs, stop_words="english")
    X = vectorizer.fit_transform(text_data)

    doc_lengths = np.array(X.sum(axis=1)).flatten()
    valid_indices = np.where(doc_lengths >= min_words)[0]

    X_filtered = X[valid_indices].toarray()
    labels_filtered = np.array(labels)[valid_indices]

    print(f"   -> Filtered: {len(text_data)} -> {len(valid_indices)} docs. Vocab: {X_filtered.shape[1]}")
    return X_filtered, labels_filtered


# ==========================================
# DATASET PARSERS
# ==========================================

def process_reuters():
    print("\n[1/3] Processing Reuters-21578...")
    raw_path = os.path.join(DATA_RAW_DIR, "reuters21578")
    sgm_files = glob.glob(os.path.join(raw_path, "*.sgm"))

    documents = []
    labels = []

    for file_path in sgm_files:
        try:
            with open(file_path, "rb") as f:
                content = f.read().decode("latin-1")

            soup = BeautifulSoup(content, "html.parser")
            reuters_tags = soup.find_all("reuters")

            for tag in reuters_tags:
                topics_tag = tag.find("topics")
                if not topics_tag:
                    continue

                topics = [d.text for d in topics_tag.find_all("d")]

                if len(topics) == 1 and topics[0] in REUTERS_CLASSES:
                    body = tag.find("body")
                    if body:
                        documents.append(body.text)
                        labels.append(topics[0])

        except Exception as e:
            print(f"Skipping file {file_path} due to error: {e}")

    le = LabelEncoder()
    y = le.fit_transform(labels)

    X, y = clean_and_vectorize(documents, y)
    embs = train_oracle(X, y)

    torch.save(
        {
            "features": torch.FloatTensor(X),
            "embeddings": embs,
            "labels": torch.LongTensor(y),
            "class_names": le.classes_,
        },
        os.path.join(DATA_PROCESSED_DIR, "reuters_processed.pt"),
    )
    print("   -> Reuters Saved.")


def process_webkb():
    print("\n[2/3] Processing WebKB...")
    base_path = os.path.join(DATA_RAW_DIR, "webkb")
    documents = []
    labels = []

    for role in WEBKB_CLASSES:
        role_path = os.path.join(base_path, role)
        if not os.path.exists(role_path):
            continue

        for uni in WEBKB_UNIS:
            uni_path = os.path.join(role_path, uni)
            if not os.path.exists(uni_path):
                continue

            for file_path in glob.glob(os.path.join(uni_path, "*")):
                try:
                    with open(file_path, "r", encoding="latin-1") as f:
                        text = f.read()
                    documents.append(text)
                    labels.append(role)
                except Exception:
                    pass

    le = LabelEncoder()
    y = le.fit_transform(labels)

    X, y = clean_and_vectorize(documents, y)
    embs = train_oracle(X, y)

    torch.save(
        {
            "features": torch.FloatTensor(X),
            "embeddings": embs,
            "labels": torch.LongTensor(y),
            "class_names": le.classes_,
        },
        os.path.join(DATA_PROCESSED_DIR, "webkb_processed.pt"),
    )
    print("   -> WebKB Saved.")


def process_neurips():
    print("\n[3/3] Processing NeurIPS...")
    base_path = os.path.join(DATA_RAW_DIR, "nipstxt")
    documents = []
    labels = []

    for folder in NIPS_YEARS:
        year_path = os.path.join(base_path, folder)
        if not os.path.exists(year_path):
            continue

        for file_path in glob.glob(os.path.join(year_path, "*")):
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    text = f.read()
                documents.append(text)
                labels.append(folder)
            except Exception:
                pass

    le = LabelEncoder()
    y = le.fit_transform(labels)

    X, y = clean_and_vectorize(documents, y)
    embs = train_oracle(X, y)

    torch.save(
        {
            "features": torch.FloatTensor(X),
            "embeddings": embs,
            "labels": torch.LongTensor(y),
            "class_names": le.classes_,
        },
        os.path.join(DATA_PROCESSED_DIR, "neurips_processed.pt"),
    )
    print("   -> NeurIPS Saved.")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    if not os.path.exists(DATA_PROCESSED_DIR):
        os.makedirs(DATA_PROCESSED_DIR)

    process_reuters()
    process_webkb()
    process_neurips()
    print("\nAll datasets processed successfully!")
