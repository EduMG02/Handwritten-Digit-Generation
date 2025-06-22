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



st.markdown("---")
st.markdown("Created by: Jorge Eduardo Muñoz Garza · METI Internship Examination · 2025")


st.markdown("""
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jorge-eduardo-munoz-garza-061724304/)
[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/EduMG02)
[![Email](https://img.shields.io/badge/-Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:edu.garza02@gmail.com)
[![Instagram](https://img.shields.io/badge/-Instagram-E4405F?style=flat&logo=instagram&logoColor=white)](https://www.instagram.com/jmunozgarza)
[![Portafolio](https://img.shields.io/badge/-Portafolio-4CAF50?style=flat&logo=vercel&logoColor=white)](https://eduardogarza-portfolio.vercel.app/)
""")