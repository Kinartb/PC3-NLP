import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import os

# Configuración de parámetros
VOCAB_SIZE = 8000
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_CLASSES = 5  # 5 clases
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
MAX_LEN = 128  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Dispositivo utilizado: {device}")

# Descargar y cargar el dataset de Amazon
def load_amazon_reviews_dataset():
    import kagglehub
    path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")
    data_path = f"{path}/Reviews.csv"
    df = pd.read_csv(data_path)
    df = df[['Text', 'Score']].dropna()
    df['Score'] = df['Score'].astype(int)
    data = list(zip(df['Text'], df['Score'] - 1))  # Restar 1 para que las etiquetas sean 0, 1, 2, 3, 4
    return data

# Entrenar modelo de subwords
def train_sentencepiece(data, model_prefix="subwordmulti"):
    text_file = "data.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        for text, _ in data:
            f.write(text + "\n")
    spm.SentencePieceTrainer.train(
        input=text_file,
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        character_coverage=0.9995,
        model_type='bpe'
    )
    os.remove(text_file)  # Eliminar archivo temporal

# Dataset para subwords
class SubwordDataset(Dataset):
    def __init__(self, data, sp):
        self.data = []
        self.sp = sp
        for text, label in data:
            tokens = self.sp.encode(text, out_type=int)[:MAX_LEN]
            if len(tokens) > 0:
                self.data.append((tokens, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, label = self.data[idx]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# Dataset para tokenización tradicional
class TraditionalDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = []
        self.vocab = vocab
        self.unk_idx = self.vocab.get("<unk>")
        for text, label in data:
            tokens = [self.vocab.get(word, self.unk_idx) for word in text.split()][:MAX_LEN]
            if len(tokens) > 0:
                self.data.append((tokens, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, label = self.data[idx]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# Construir vocabulario para tokenización tradicional
def build_vocab(data, max_vocab_size=VOCAB_SIZE):
    word_freq = Counter()
    for text, _ in data:
        word_freq.update(text.split())
    most_common = word_freq.most_common(max_vocab_size - 2)
    vocab = {word: idx+2 for idx, (word, _) in enumerate(most_common)}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    return vocab

# DataLoader con padding
def collate_fn(batch):
    batch = [(text, label) for text, label in batch if len(text) > 0]
    if len(batch) == 0:
        return None
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)
    texts_padded = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return texts_padded, labels, lengths

# Modelo RNN
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout=0.5):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        x = self.embedding(x)
        x = self.dropout(x)
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed_input)
        out = self.fc(hidden[-1])
        return out

# Función para calcular precisión
def calculate_accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    return (preds == labels).float().mean().item()

# Entrenar y evaluar el modelo
def train_and_evaluate(model, train_loader, val_loader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        total_samples = 0
        for batch in train_loader:
            if batch is None:
                continue
            texts, labels, lengths = batch
            texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
            optimizer.zero_grad()
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_size = labels.size(0)
            train_loss += loss.item() * batch_size
            train_acc += calculate_accuracy(outputs, labels) * batch_size
            total_samples += batch_size

        train_loss /= total_samples
        train_acc /= total_samples

        model.eval()
        val_loss, val_acc = 0.0, 0.0
        total_val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                texts, labels, lengths = batch
                texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
                outputs = model(texts, lengths)
                loss = criterion(outputs, labels)
                batch_size = labels.size(0)
                val_loss += loss.item() * batch_size
                val_acc += calculate_accuracy(outputs, labels) * batch_size
                total_val_samples += batch_size

        val_loss /= total_val_samples
        val_acc /= total_val_samples

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Perdida de entrenamiento: {train_loss:.4f}, Precision de entrenamiento: {train_acc:.4f}")
        print(f"  Perdida de validacion: {val_loss:.4f}, Precision de validacion: {val_acc:.4f}")

import time

if __name__ == '__main__':
    # Redirigir la salida a txt
    import sys
    sys.stdout = open("multiclase.txt", "w")

    # Cargar datos
    print("Cargando datos...")
    DATA = load_amazon_reviews_dataset()

    # Entrenar modelo de subwords 
    if not os.path.exists("subwordmulti.model"):
        print("Entrenando modelo de subwords...")
        train_sentencepiece(DATA)

    # Cargar modelo de subwords
    sp = spm.SentencePieceProcessor()
    sp.load("subwordmulti.model")

    # Crear vocabulario
    vocab = build_vocab(DATA)

    # Dividir datos en entrenamiento y validación
    print("Dividiendo datos...")
    train_data, val_data = train_test_split(DATA, test_size=0.2, random_state=42)
    train_dataset_subword = SubwordDataset(train_data, sp)
    val_dataset_subword = SubwordDataset(val_data, sp)
    train_dataset_traditional = TraditionalDataset(train_data, vocab)
    val_dataset_traditional = TraditionalDataset(val_data, vocab)

    # DataLoader 
    train_loader_subword = DataLoader(train_dataset_subword, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader_subword = DataLoader(val_dataset_subword, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    train_loader_traditional = DataLoader(train_dataset_traditional, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader_traditional = DataLoader(val_dataset_traditional, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Inicializar modelos
    model_subword = RNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)
    model_traditional = RNN(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_subword = optim.Adam(model_subword.parameters(), lr=LEARNING_RATE)
    optimizer_traditional = optim.Adam(model_traditional.parameters(), lr=LEARNING_RATE)

    # Entrenar y evaluar el modelo con subword
    print("\n=== Modelo Subword ===")
    start_time_subword = time.time()
    train_and_evaluate(model_subword, train_loader_subword, val_loader_subword, optimizer_subword, NUM_EPOCHS)
    end_time_subword = time.time()
    total_time_subword = end_time_subword - start_time_subword
    print(f"Tiempo total de ejecucion para el modelo Subword: {total_time_subword:.2f} segundos")

    # Entrenar y evaluar el modelo tradicional
    print("\n=== Modelo Tradicional ===")
    start_time_traditional = time.time()
    train_and_evaluate(model_traditional, train_loader_traditional, val_loader_traditional, optimizer_traditional, NUM_EPOCHS)
    end_time_traditional = time.time()
    total_time_traditional = end_time_traditional - start_time_traditional
    print(f"Tiempo total de ejecucion para el modelo Tradicional: {total_time_traditional:.2f} segundos")

    # Restaurar la salida en consola
    sys.stdout.close()
    sys.stdout = sys.__stdout__

    # Para saber cuando terminó de redactar el txt
    print("Ya termine")    