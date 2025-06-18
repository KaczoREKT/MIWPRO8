import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib

from FeedForwardBlock import FeedForwardBlock
from Encoder import Encoder
from Decoder import Decoder
from AutoEncoder import Autoencoder

# --- Krok 1: stwórz model (tak samo jak w train.py) ---
d_input = 784
d_model = 128
d_latent = 32
dropout = 0.1

encoder_layers = torch.nn.ModuleList([
    FeedForwardBlock(d_input, d_model, dropout),
    FeedForwardBlock(d_model, d_latent, dropout)
])
decoder_layers = torch.nn.ModuleList([
    FeedForwardBlock(d_latent, d_model, dropout),
    FeedForwardBlock(d_model, d_input, dropout)
])
encoder = Encoder(encoder_layers)
decoder = Decoder(decoder_layers)
autoencoder = Autoencoder(encoder, decoder)

# --- Krok 2: załaduj wytrenowane wagi (jeśli je zapisałeś) ---
autoencoder.load_state_dict(torch.load('autoencoder.pth', map_location='cpu'))
autoencoder.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder.to(device)

# --- Krok 3: przygotuj dane testowe ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=16, shuffle=True)

# Weź jeden batch testowy
images, _ = next(iter(test_loader))
images = images.to(device)

# --- Krok 4: wykonaj rekonstrukcję ---
with torch.no_grad():
    reconstructions = autoencoder(images)

# --- Krok 5: wyświetl porównanie 8 pierwszych obrazów ---
orig = images.cpu().view(-1, 28, 28)
recon = reconstructions.cpu().view(-1, 28, 28)

fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i in range(8):
    axes[0, i].imshow(orig[i], cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(recon[i], cmap='gray')
    axes[1, i].axis('off')
axes[0, 0].set_title("Oryginały")
axes[1, 0].set_title("Rekonstrukcje")
plt.show()
