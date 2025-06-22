import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, z_dim=64, label_dim=10, img_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim + label_dim, 128),
            nn.ReLU(),
            nn.Linear(128, img_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim=1)
        return self.model(x)

def generate_images(model, digit, num=5):
    model.eval()
    z_dim = 64
    noise = torch.randn(num, z_dim)
    labels = torch.full((num,), digit, dtype=torch.long)
    one_hot = torch.nn.functional.one_hot(labels, num_classes=10).float()
    with torch.no_grad():
        images = model(noise, one_hot)
    images = images.view(-1, 28, 28).cpu().numpy()
    return images

# UI
st.title("MNIST Digit Generator")
digit = st.selectbox("Select a digit (0–9):", list(range(10)))
if st.button("Generate"):
    G = Generator()
    G.load_state_dict(torch.load("mnist_generator.pth", map_location=torch.device("cpu")))
    images = generate_images(G, digit)

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)
