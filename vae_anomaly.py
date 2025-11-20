"""
Cleaned & minimized version of VAE + AE anomaly detection project.
Fully corrected, no constructor bugs, shorter code, same functionality.

Run:
    python vae_ae_project.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, Model

# ----------------------- CONFIG -----------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

N_TIMESTEPS = 10000
N_FEATURES = 10
ANOMALY_RATIO = 0.01
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HYPERS = [
    {"latent_dim": 2, "depth": 1, "beta": 0.1},
    {"latent_dim": 4, "depth": 1, "beta": 1.0},
    {"latent_dim": 8, "depth": 2, "beta": 1.0},
]

EPOCHS = 20
BATCH = 128
LR = 1e-3

# --------------- DATA GENERATION ----------------------
def generate_synth():
    t = np.arange(N_TIMESTEPS)
    data = np.zeros((N_TIMESTEPS, N_FEATURES), np.float32)

    freqs = np.linspace(0.01, 0.12, N_FEATURES)
    phi = np.random.uniform(0, 2*np.pi, N_FEATURES)

    for f in range(N_FEATURES):
        data[:, f] = (
            np.sin(2*np.pi*freqs[f]*t + phi[f]) +
            0.5*np.sin(2*np.pi*freqs[f]*1.7*t + phi[f]*0.5) +
            0.1*np.random.randn(N_TIMESTEPS)
        )

    data = StandardScaler().fit_transform(data).astype(np.float32)

    labels = np.zeros(N_TIMESTEPS, int)
    n_anom = max(1, int(N_TIMESTEPS * ANOMALY_RATIO))

    idx = np.random.choice(N_TIMESTEPS, n_anom, replace=False)
    labels[idx] = 1

    for i in idx:
        if np.random.rand() < 0.5:
            data[i] += np.random.uniform(5, 12)
        else:
            data[i] = data[i] * np.random.uniform(-3, 3) + np.random.uniform(2, 6)

    return data, labels

# --------------- MODEL: VAE --------------------------
class Sampling(layers.Layer):
    def call(self, inputs):
        mu, log_var = inputs
        eps = tf.random.normal(tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps

def build_vae(dim, latent_dim, depth, beta):
    hidden = [64] * depth

    # Encoder
    inp = layers.Input((dim,))
    x = inp
    for h in hidden:
        x = layers.Dense(h, activation="relu")(x)
    mu = layers.Dense(latent_dim)(x)
    logv = layers.Dense(latent_dim)(x)
    z = Sampling()([mu, logv])
    encoder = Model(inp, [mu, logv, z])

    # Decoder
    z_in = layers.Input((latent_dim,))
    x = z_in
    for h in reversed(hidden):
        x = layers.Dense(h, activation="relu")(x)
    out = layers.Dense(dim)(x)
    decoder = Model(z_in, out)

    class VAE(Model):
        def __init__(self, enc, dec, beta):
            super().__init__()
            self.enc = enc
            self.dec = dec
            self.beta = beta
            self.loss_tracker = tf.keras.metrics.Mean()

        @property
        def metrics(self):
            return [self.loss_tracker]

        def train_step(self, x):
            with tf.GradientTape() as tape:
                mu, logv, z = self.enc(x)
                xr = self.dec(z)
                rec = tf.reduce_mean(tf.reduce_sum((x - xr)**2, -1))
                kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + logv - mu**2 - tf.exp(logv), -1))
                loss = rec + beta * kl
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.loss_tracker.update_state(loss)
            return {"loss": self.loss_tracker.result()}

        def call(self, x):
            return self.dec(self.enc(x)[2])

    vae = VAE(encoder, decoder, beta)
    vae.compile(optimizer=tf.keras.optimizers.Adam(LR))
    return vae, encoder

# --------------- MODEL: AE ---------------------------
def build_ae(dim):
    inp = layers.Input((dim,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(dim)(x)
    ae = Model(inp, out)
    ae.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mse")
    return ae

# ------------------ UTILS ---------------------------
def metric(y, s, thr):
    yp = (s > thr).astype(int)
    try:
        auc = roc_auc_score(y, s)
    except:
        auc = np.nan
    f1 = f1_score(y, yp, zero_division=0)
    p, r, _, _ = precision_recall_fscore_support(y, yp, average="binary", zero_division=0)
    return dict(auc=float(auc), f1=float(f1), precision=float(p), recall=float(r), threshold=float(thr))

# ------------------ EXPERIMENT ----------------------
def run_exp(data, labels, cfg, rid):
    Xtr, Xte, ytr, yte = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)
    Xtr = Xtr[ytr == 0]  # train on normals only

    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    # ----------- VAE -----------
    vae, enc = build_vae(Xtr.shape[1], cfg["latent_dim"], cfg["depth"], cfg["beta"])
    vae.fit(Xtr, epochs=EPOCHS, batch_size=BATCH, verbose=0)

    tr_rec = np.mean((Xtr - vae.predict(Xtr, BATCH))**2, 1)
    te_rec = np.mean((Xte - vae.predict(Xte, BATCH))**2, 1)

    thr_p = np.percentile(tr_rec, 99)
    thr_s = tr_rec.mean() + 3*tr_rec.std()

    m1 = metric(yte, te_rec, thr_p)
    m2 = metric(yte, te_rec, thr_s)

    # ----------- AE -----------
    ae = build_ae(Xtr.shape[1])
    ae.fit(Xtr, Xtr, epochs=EPOCHS, batch_size=BATCH, verbose=0)

    tr2 = np.mean((Xtr - ae.predict(Xtr, BATCH))**2, 1)
    te2 = np.mean((Xte - ae.predict(Xte, BATCH))**2, 1)

    thr2_p = np.percentile(tr2, 99)
    thr2_s = tr2.mean() + 3*tr2.std()

    m3 = metric(yte, te2, thr2_p)
    m4 = metric(yte, te2, thr2_s)

    return dict(config=cfg, vae_p=m1, vae_s=m2, ae_p=m3, ae_s=m4)

# ---------------------- MAIN -------------------------
def main():
    data, labels = generate_synth()
    all_res = []

    for i, cfg in enumerate(HYPERS):
        print(f"Running {cfg}...")
        r = run_exp(data, labels, cfg, f"run{i+1}")
        all_res.append(r)
        print(" VAE  perc99:", r["vae_p"])
        print(" VAE  mean+3:", r["vae_s"])
        print(" AE   perc99:", r["ae_p"])
        print(" AE   mean+3:", r["ae_s"])

    pd.DataFrame(all_res).to_json(os.path.join(OUTPUT_DIR, "results.json"), orient="records")
    print("\nSaved results.json")

if __name__ == "__main__":
    main()
