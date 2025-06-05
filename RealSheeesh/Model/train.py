import os

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from FeedForwardBlock import FeedForwardBlock
from Encoder import Encoder
from Decoder import Decoder
from AutoEncoder import Autoencoder


def get_model():
    # Parametry
    d_input = 784   # 28x28
    d_model = 128
    d_latent = 32   # rozmiar kodu
    dropout = 0.1

    # ENCODER: 784 -> 128 -> 32
    encoder_layers = nn.ModuleList([
        FeedForwardBlock(d_input, d_model, dropout),   # 784 -> 128
        FeedForwardBlock(d_model, d_latent, dropout)   # 128 -> 32
    ])
    encoder = Encoder(encoder_layers)

    # DECODER: 32 -> 128 -> 784
    decoder_layers = nn.ModuleList([
        FeedForwardBlock(d_latent, d_model, dropout),  # 32 -> 128
        FeedForwardBlock(d_model, d_input, dropout)  # 128 -> 784
    ])
    decoder = Decoder(decoder_layers)

    autoencoder = Autoencoder(encoder, decoder)
    return autoencoder, encoder


def get_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # spłaszczamy 28x28 -> 784
    ])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(mnist_train,
                              batch_size=batch_size,
                              shuffle=True)
    return train_loader


def train_model(autoencoder, train_loader, device, checkpoint_path="checkpoint.pth"):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    autoencoder.to(device)

    start_epoch = 0
    # Jeżeli istnieje plik checkpoint, wczytujemy model, optimizer i numer epoki
    if os.path.isfile(checkpoint_path):
        print(f"Wczytuję checkpoint z: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Zaczynam od epoki {start_epoch}")

    num_epochs = 300
    for epoch in range(start_epoch, num_epochs):
        autoencoder.train()
        running_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)          # kształt (batch_size, 784)
            out = autoencoder(x)
            loss = criterion(out, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss:.6f}")

        # Co epokę zapisujemy checkpoint:
        torch.save({
            'epoch': epoch,
            'model_state_dict': autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)


def get_codings(encoder, train_loader, device):
    X_codes = []
    y_labels = []
    encoder.eval()
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            code = encoder(x).cpu()
            X_codes.append(code)
            y_labels.append(y)
    X_codes = torch.cat(X_codes, dim=0).numpy()
    y_labels = torch.cat(y_labels, dim=0).numpy()
    return X_codes, y_labels


def knn_analysis(X_codes, y_labels):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import adjusted_rand_score

    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_codes, y_labels)
    y_pred = knn.predict(X_codes)
    ari = adjusted_rand_score(y_labels, y_pred)
    print("Adjusted Rand Index (ARI) KNN na kodowaniach vs. prawdziwe etykiety:", ari)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder, encoder = get_model()
    train_loader = get_data(batch_size=128)
    train_model(autoencoder, train_loader, device)

    # Zapisujemy do pliku autoencoder.pth
    torch.save(autoencoder.state_dict(), "autoencoder.pth")
    print("Zapisano model w autoencoder.pth")

    # analiza KNN na kodowaniach
    X_codes, y_labels = get_codings(encoder, train_loader, device)
    knn_analysis(X_codes, y_labels)
