import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from PIL import Image

import os
import pandas as pd

def load_pokedex(description_file, image_folder):
    pokedex = pd.read_csv(description_file)
    if 'Type2' in pokedex.columns:
        pokedex.drop('Type2', axis=1, inplace=True)
    pokedex.sort_values(by=['Name'], ascending=True, inplace=True)
    # Budujemy ścieżkę do pliku obrazka po kolumnie Name (np. bulbasaur.png)
    pokedex['Image'] = pokedex['Name'].apply(
        lambda name: os.path.join(image_folder, f"{name.lower()}.png")
    )
    # Zostawiamy tylko te rekordy, które mają istniejący plik
    pokedex = pokedex[pokedex['Image'].apply(os.path.isfile)].reset_index(drop=True)
    return pokedex

class PokedexDataset(Dataset):
    def __init__(self, pokedex_df, transform=None):
        self.data = pokedex_df
        self.transform = transform
        # Mapowanie typów tekstowych na liczby
        self.label_names = sorted(self.data['Type1'].unique())
        self.label2idx = {name: idx for idx, name in enumerate(self.label_names)}
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['Image']
        label_text = self.data.iloc[idx]['Type1']
        label = self.label2idx[label_text]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label



from torchvision import transforms
from torch.utils.data import DataLoader

# Załaduj pokedex
pokedex = load_pokedex("pokemon_dataset/pokemon.csv", "pokemon_dataset/images/images")

# Przygotuj transformacje
transform = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.ToTensor(),
])

# Zrób dataset i dataloader
dataset = PokedexDataset(pokedex, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# Jeśli używasz PokedexDataset
print("Liczba unikalnych etykiet:", dataset.data['Type1'].nunique())
print("Unikalne etykiety:", dataset.data['Type1'].unique())


# Test: wyświetl rozmiar batcha
images, labels = next(iter(dataloader))
print(images.shape, labels[:10])
# Test:
images, labels = next(iter(dataloader))
print(images.shape, labels[:10])
class Encoder(nn.Module):
    def __init__(self, code_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 4, stride=2, padding=1) # 60x60
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1) # 30x30
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2, padding=1) # 15x15
        self.fc = nn.Linear(64 * 15 * 15, code_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        code = self.fc(x)
        return code

class Decoder(nn.Module):
    def __init__(self, code_dim=32):
        super().__init__()
        self.fc = nn.Linear(code_dim, 64 * 15 * 15)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1) # 30x30
        self.deconv2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1) # 60x60
        self.deconv3 = nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1) # 120x120

    def forward(self, code):
        x = F.relu(self.fc(code))
        x = x.view(-1, 64, 15, 15)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # [0, 1] dla obrazów
        return x

class Autoencoder(nn.Module):
    def __init__(self, code_dim=32):
        super().__init__()
        self.encoder = Encoder(code_dim)
        self.decoder = Decoder(code_dim)

    def forward(self, x):
        code = self.encoder(x)
        recon = self.decoder(code)
        return recon


import torch

# Sprawdź kształt danych:
images, labels = next(iter(dataloader))
print(images.shape)  # [batch, 3, 120, 120]
print(labels.shape)  # [batch]

import torch.optim as optim

autoencoder = Autoencoder(code_dim=32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = autoencoder.to(device)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

num_epochs = 10

for epoch in range(num_epochs):
    autoencoder.train()
    epoch_loss = 0
    for batch_images, _ in dataloader:
        batch_images = batch_images.to(device)
        optimizer.zero_grad()
        outputs = autoencoder(batch_images)
        loss = criterion(outputs, batch_images)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_images.size(0)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(dataset):.4f}")

# Po treningu, możesz wyciągać kodowania:
autoencoder.eval()
all_codes = []
all_labels = []
with torch.no_grad():
    for batch_images, batch_labels in dataloader:
        batch_images = batch_images.to(device)
        codes = autoencoder.encoder(batch_images)
        all_codes.append(codes.cpu())
        all_labels.append(batch_labels)
all_codes = torch.cat(all_codes)  # [N, code_dim]
all_labels = torch.cat(all_labels)  # [N]


import matplotlib
matplotlib.use('TkAgg')  # lub 'Qt5Agg' jeśli masz zainstalowane
import matplotlib.pyplot as plt

def show_images(imgs, n=5):
    imgs = imgs[:n].permute(0,2,3,1).numpy()
    for i in range(n):
        plt.subplot(1,n,i+1)
        plt.imshow(imgs[i])
        plt.axis('off')
    plt.show()

show_images(images)


from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
import numpy as np

# Zamiana kodowań i etykiet na numpy
codes_np = all_codes.numpy()   # [N, code_dim]
labels_np = all_labels.numpy() # [N]

# Klasteryzacja KMeans (10 klastrów – odpowiada 10 klasom)
kmeans = KMeans(n_clusters=10, random_state=42)
labels_pred = kmeans.fit_predict(codes_np)

# Porównanie klastrów z prawdziwymi klasami
ari = adjusted_rand_score(labels_np, labels_pred)
print(f"Adjusted Rand Index (ARI): {ari:.4f} (1.0 = idealnie, 0 = losowo)")

# (Opcjonalnie) wyświetl macierz pomyłek: które klastry pokrywają się z klasami
cm = confusion_matrix(labels_np, labels_pred)
plt.figure(figsize=(8,6))
plt.imshow(cm, cmap='Blues')
plt.xlabel('Klaster KMeans')
plt.ylabel('Prawdziwa klasa')
plt.title('Macierz pomyłek klas-klastrów')
plt.colorbar()
plt.show()