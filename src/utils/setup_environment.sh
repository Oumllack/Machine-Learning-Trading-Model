#!/bin/bash

echo "🚀 Configuration automatique de l'environnement d'analyse boursière"
echo "================================================================"

# Vérifier si conda est installé
if ! command -v conda &> /dev/null; then
    echo "❌ Conda n'est pas installé. Veuillez installer Miniconda ou Anaconda."
    echo "   Téléchargement: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda détecté"

# Créer l'environnement Python 3.11
echo "📦 Création de l'environnement 'bourse' avec Python 3.11..."
conda create -n bourse python=3.11 -y

if [ $? -eq 0 ]; then
    echo "✅ Environnement créé avec succès"
else
    echo "❌ Erreur lors de la création de l'environnement"
    exit 1
fi

# Activer l'environnement
echo "🔄 Activation de l'environnement..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bourse

if [ $? -eq 0 ]; then
    echo "✅ Environnement activé"
else
    echo "❌ Erreur lors de l'activation"
    exit 1
fi

# Installer PyTorch
echo "🤖 Installation de PyTorch..."
pip install torch torchvision torchaudio

# Installer TensorFlow
echo "🧠 Installation de TensorFlow..."
pip install tensorflow

# Installer les autres dépendances
echo "📊 Installation des autres dépendances..."
pip install yfinance pandas numpy matplotlib seaborn scikit-learn ta plotly jupyter notebook

# Vérifier l'installation
echo "🔍 Vérification de l'installation..."
python -c "
import torch
import tensorflow as tf
import yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import ta
import plotly
import jupyter
import notebook
print('✅ Toutes les dépendances sont installées avec succès !')
print(f'PyTorch version: {torch.__version__}')
print(f'TensorFlow version: {tf.__version__}')
print(f'Python version: {pd.__version__}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Installation terminée avec succès !"
    echo "=================================="
    echo "Pour utiliser l'environnement :"
    echo "1. Activer: conda activate bourse"
    echo "2. Lancer: python main.py"
    echo "3. Ou notebook: jupyter notebook analyse_boursiere.ipynb"
    echo ""
    echo "📁 Votre projet est prêt dans: $(pwd)"
else
    echo "❌ Erreur lors de la vérification"
    exit 1
fi 