"""
emotion_encoder.py
==================
Encoder de audio basado en el modelo emocional pre-entrenado de audeering.
Comparte la misma arquitectura entre training (exportar_modelos.py),
auditoria de etiquetas (auditar_etiquetas_ia.py) e inferencia (app.py).

El modelo `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` esta
fine-tuneado en MSP-Podcast para predecir arousal/valence/dominance.
Sus hidden_states (1024 dims) son features especificamente optimizadas
para captura de emocion en voz, no fonemas — mejor base que wav2vec2-base
para clasificacion emocional.

Uso:
    from emotion_encoder import load_encoder, extract_embedding

    proc, model, device = load_encoder()
    emb = extract_embedding(audio_array, sr=16000, processor=proc, model=model, device=device)
    # emb: np.ndarray shape (1024,) float32
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2PreTrainedModel


MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
EMBEDDING_DIM = 1024  # hidden size de wav2vec2-large


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionModel(Wav2Vec2PreTrainedModel):
    # Compatibilidad con transformers >= 4.40 que espera estos atributos
    _tied_weights_keys = []
    all_tied_weights_keys = {}

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = torch.mean(outputs[0], dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


def get_device(prefer_mps=True):
    """Devuelve mps en Mac (si esta y prefer_mps=True), cuda si esta, si no cpu.

    app.py por defecto usa cpu/cuda porque mps con flask debug a veces da problemas
    con torch.no_grad(); training y auditoria si usan mps.
    """
    if prefer_mps and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_encoder(device=None, prefer_mps=True):
    """Carga el processor + EmotionModel y los mueve al device.

    Returns:
        (processor, model, device)
    """
    if device is None:
        device = get_device(prefer_mps=prefer_mps)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = EmotionModel.from_pretrained(MODEL_ID).to(device).eval()
    return processor, model, device


def extract_embedding(y, sr, processor, model, device):
    """Extrae el embedding emocional (1024 dims, mean-pooled) de un audio.

    Args:
        y: np.ndarray con la senal de audio en formato mono
        sr: sample rate (debe ser 16000)
        processor, model, device: del load_encoder()

    Returns:
        np.ndarray shape (1024,) float32 — el hidden_state mean-pooled
        que el modelo usa internamente como representacion emocional.
    """
    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        hidden_states, _logits = model(inputs.input_values.to(device))
    return hidden_states.squeeze().cpu().numpy().astype(np.float32)


def extract_avd(y, sr, processor, model, device):
    """Extrae las 3 dimensiones emocionales [arousal, dominance, valence].

    Util para diagnostico / auditoria. Devuelve tupla de 3 floats en [0, 1] aprox.
    """
    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        _hidden, logits = model(inputs.input_values.to(device))
    arousal, dominance, valence = logits.squeeze().cpu().numpy().tolist()
    return float(arousal), float(dominance), float(valence)
