from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        code = self.encoder(x)  # kod: (batch, 32)
        out = self.decoder(code)
        return out

