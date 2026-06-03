# ╔══════════════════════════════════════════════════════════════╗
# ║   PHASE E — PIPELINE TEMPS RÉEL                             ║
# ║   Windows · Webcam → MediaPipe → Transformer → Texte+Audio  ║
# ║   Sign Language Translator · Équipe 8                        ║
# ╚══════════════════════════════════════════════════════════════╝
#
# ÉTAPE 0 — Installer les dépendances (dans ton terminal Windows)
# ──────────────────────────────────────────────────────────────
# pip install opencv-python mediapipe torch gtts pygame numpy
#
# ÉTAPE 1 — Copier best_transformer.pt et label_map.json
#           dans le même dossier que ce fichier
#
# ÉTAPE 2 — Lancer : python demo.py

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import threading
from gtts import gTTS
import pygame
import os
import tempfile
import math

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════

MODEL_PATH     = 'best_transformer.pt'
LABEL_MAP_PATH = 'label_map.json'
N_FRAMES       = 64
N_FEATURES     = 159   # (LH21 + RH21 + Pose11) × 3
N_CLASSES      = 250
DEVICE         = torch.device('cpu')

# Fenêtre — modifiez ces valeurs pour adapter à votre écran
WIN_W          = 1280
WIN_H          = 720
CAM_W          = 854   # largeur zone caméra (le reste = panneau UI)
PANEL_W        = WIN_W - CAM_W  # 426 px

# Landmarks de pose à garder (11 premiers = haut du corps)
POSE_UPPER = list(range(11))
N_LH   = 21
N_RH   = 21
N_POSE = len(POSE_UPPER)  # 11

# ════════════════════════════════════════════════════════════════
# PALETTE DE COULEURS (BGR)
# ════════════════════════════════════════════════════════════════

C_BG_DARK    = (18,  24,  38)    # fond principal
C_BG_PANEL   = (24,  32,  50)    # fond panneau
C_BG_CARD    = (30,  42,  64)    # fond carte
C_TEAL       = (27, 153, 139)    # accent principal
C_TEAL_DIM   = (15,  90,  82)    # accent atténué
C_TEAL_BRIGHT= (60, 210, 195)    # accent lumineux
C_WHITE      = (255,255,255)
C_GRAY_LIGHT = (180,190,210)
C_GRAY_MID   = (100,115,140)
C_GRAY_DIM   = ( 50, 62, 85)
C_GREEN      = ( 40,210,110)
C_ORANGE     = ( 30,155,240)
C_RED        = ( 60, 60,220)
C_GOLD       = ( 30,190,230)
C_NAVY_DEEP  = (12,  18,  30)


# ════════════════════════════════════════════════════════════════
# ARCHITECTURE DU TRANSFORMER (identique à l'entraînement)
# ════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=N_FRAMES, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class SignTransformer(nn.Module):
    def __init__(self, input_size=N_FEATURES, d_model=128,
                 nhead=4, num_layers=4, num_classes=N_CLASSES, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model)
        )
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier  = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = x.mean(dim=1)
        return self.classifier(x)


# ════════════════════════════════════════════════════════════════
# CHARGER LE MODÈLE ET LE LABEL MAP
# ════════════════════════════════════════════════════════════════

print("📂 Chargement du modèle...")
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
model = SignTransformer().to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"✅ Modèle chargé (epoch {ckpt['epoch']} · val_acc={ckpt['val_acc']*100:.1f}%)")

with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)
idx2sign = label_map['index_to_sign']
print(f"✅ {len(idx2sign)} signes chargés")


# ════════════════════════════════════════════════════════════════
# MEDIAPIPE — EXTRACTION DES LANDMARKS
# ════════════════════════════════════════════════════════════════

mp_hands       = mp.solutions.hands
mp_pose        = mp.solutions.pose
mp_drawing     = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS


def extract_landmarks(hand_results, pose_results):
    frame_vec = []
    lh = np.zeros((N_LH, 3), dtype=np.float32)
    rh = np.zeros((N_RH, 3), dtype=np.float32)

    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_lm, handedness in zip(
            hand_results.multi_hand_landmarks,
            hand_results.multi_handedness
        ):
            label  = handedness.classification[0].label
            coords = np.array([[lm.x, lm.y, lm.z]
                               for lm in hand_lm.landmark], dtype=np.float32)
            if label == 'Left':
                lh = coords
            else:
                rh = coords

    frame_vec.append(lh.flatten())
    frame_vec.append(rh.flatten())

    if pose_results.pose_landmarks:
        pose_all = np.array([[lm.x, lm.y, lm.z]
                             for lm in pose_results.pose_landmarks.landmark],
                            dtype=np.float32)
        pose = pose_all[POSE_UPPER]
    else:
        pose = np.zeros((N_POSE, 3), dtype=np.float32)
    frame_vec.append(pose.flatten())

    return np.concatenate(frame_vec)  # (159,)


def normalize_sequence(seq):
    non_zero = seq[:, 63:66].sum(axis=1) != 0
    if non_zero.sum() > 0:
        center      = seq[non_zero, 63:66].mean(axis=0)
        center_full = np.tile(center, N_LH + N_RH + N_POSE)
        seq         = seq - center_full
        std         = seq[non_zero].std() + 1e-8
        seq         = seq / std
    seq = np.clip(seq, -5, 5)
    return seq.astype(np.float32)


def build_tensor(buffer):
    seq = np.array(buffer, dtype=np.float32)
    seq = normalize_sequence(seq)
    T   = len(seq)
    if T >= N_FRAMES:
        seq = seq[:N_FRAMES]
    else:
        pad = np.zeros((N_FRAMES - T, N_FEATURES), dtype=np.float32)
        seq = np.vstack([seq, pad])
    return torch.tensor(seq, dtype=torch.float32).unsqueeze(0)


# ════════════════════════════════════════════════════════════════
# SYNTHÈSE VOCALE (non-bloquante)
# ════════════════════════════════════════════════════════════════

pygame.mixer.init()

def speak(text):
    def _speak():
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tmp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            tts.save(tmp.name)
            tmp.close()
            pygame.mixer.music.load(tmp.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            os.unlink(tmp.name)
        except Exception as e:
            print(f"TTS erreur : {e}")
    threading.Thread(target=_speak, daemon=True).start()


# ════════════════════════════════════════════════════════════════
# PRÉDICTION
# ════════════════════════════════════════════════════════════════

def predict(buffer):
    tensor   = build_tensor(buffer).to(DEVICE)
    pad_mask = (tensor.sum(dim=-1) == 0)

    with torch.no_grad():
        logits = model(tensor, src_key_padding_mask=pad_mask)
        probs  = F.softmax(logits, dim=1)
        top5   = probs.topk(5, dim=1)

    pred_idx  = top5.indices[0][0].item()
    pred_conf = top5.values[0][0].item()
    pred_sign = idx2sign[str(pred_idx)]
    top5_list = [
        (idx2sign[str(top5.indices[0][i].item())], top5.values[0][i].item())
        for i in range(5)
    ]
    return pred_sign, pred_conf, top5_list


# ════════════════════════════════════════════════════════════════
# UTILITAIRES DE DESSIN — INTERFACE FLEXIBLE
# ════════════════════════════════════════════════════════════════

def draw_rect_rounded(img, pt1, pt2, color, radius=12, thickness=-1, alpha=1.0):
    """Rectangle avec coins arrondis, avec support transparence."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2-x1)//2, (y2-y1)//2)
    if thickness == -1:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1+r, y1), (x2-r, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1+r), (x2, y2-r), color, -1)
        cv2.circle(overlay, (x1+r, y1+r), r, color, -1)
        cv2.circle(overlay, (x2-r, y1+r), r, color, -1)
        cv2.circle(overlay, (x1+r, y2-r), r, color, -1)
        cv2.circle(overlay, (x2-r, y2-r), r, color, -1)
        if alpha < 1.0:
            cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
        else:
            img[:] = overlay
    else:
        cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, thickness)
        cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, thickness)
        cv2.circle(img, (x1+r, y1+r), r, color, thickness)
        cv2.circle(img, (x2-r, y1+r), r, color, thickness)
        cv2.circle(img, (x1+r, y2-r), r, color, thickness)
        cv2.circle(img, (x2-r, y2-r), r, color, thickness)


def draw_text_centered(img, text, cx, cy, scale, color, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(img, text, (cx - tw//2, cy + th//2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_progress_bar(img, x, y, w, h, pct, color_fg, color_bg, radius=4):
    """Barre de progression avec coins arrondis."""
    draw_rect_rounded(img, (x, y), (x+w, y+h), color_bg, radius)
    filled = max(0, int(w * min(pct, 1.0)))
    if filled > radius * 2:
        draw_rect_rounded(img, (x, y), (x+filled, y+h), color_fg, radius)


def draw_confidence_ring(img, cx, cy, radius, pct, color, bg_color, thickness=8):
    """Arc de cercle pour afficher la confiance."""
    cv2.circle(img, (cx, cy), radius, bg_color, thickness)
    angle = int(360 * pct)
    if angle > 0:
        for a in range(0, angle, 2):
            rad = math.radians(a - 90)
            x1  = int(cx + (radius) * math.cos(rad))
            y1  = int(cy + (radius) * math.sin(rad))
            cv2.circle(img, (x1, y1), thickness//2, color, -1)


def fit_text(text, max_width, max_scale=2.5, min_scale=0.5, thickness=3):
    """Calcule la scale de fonte pour que le texte tienne dans max_width px."""
    scale = max_scale
    while scale >= min_scale:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if tw <= max_width:
            return scale
        scale -= 0.05
    return min_scale


def put_text_aa(img, text, x, y, scale, color, thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


# ════════════════════════════════════════════════════════════════
# FONCTION DE RENDU DU PANNEAU LATÉRAL — 100% FLEXIBLE
# ════════════════════════════════════════════════════════════════

def draw_panel(canvas, px, pw, ph, state):
    """
    Dessine le panneau latéral entièrement recalculé depuis pw × ph.
    px   : x de départ du panneau dans le canvas
    pw   : largeur du panneau
    ph   : hauteur du panneau
    state: dict contenant toutes les variables UI
    """
    last_sign    = state['last_sign']
    last_conf    = state['last_conf']
    top5_display = state['top5_display']
    is_recording = state['is_recording']
    buffer_len   = state['buffer_len']
    sentence     = state['sentence']
    fps          = state['fps']
    val_acc      = state['val_acc']
    model_epoch  = state['model_epoch']

    PAD = int(pw * 0.04)   # marge intérieure ~4%
    iw  = pw - 2 * PAD    # largeur intérieure disponible

    # ── Fond du panneau ─────────────────────────────────────────
    cv2.rectangle(canvas, (px, 0), (px + pw, ph), C_BG_PANEL, -1)
    # Ligne de séparation gauche
    cv2.line(canvas, (px, 0), (px, ph), C_TEAL_DIM, 2)

    y = 0  # curseur vertical

    # ════════════════════════════════════════
    # SECTION 1 — EN-TÊTE
    # ════════════════════════════════════════
    header_h = int(ph * 0.10)
    cv2.rectangle(canvas, (px, 0), (px+pw, header_h), C_NAVY_DEEP, -1)

    # Logo / icône ASL (cercle teal)
    icon_r  = int(header_h * 0.35)
    icon_cx = px + PAD + icon_r
    icon_cy = header_h // 2
    cv2.circle(canvas, (icon_cx, icon_cy), icon_r, C_TEAL, -1)
    draw_text_centered(canvas, "ASL", icon_cx, icon_cy,
                       icon_r * 0.022, C_WHITE, 2)

    # Titre
    tx = icon_cx + icon_r + PAD
    put_text_aa(canvas, "Sign Language",  tx, icon_cy - 8,
                pw * 0.00085, C_WHITE, 2)
    put_text_aa(canvas, "Translator",     tx, icon_cy + 18,
                pw * 0.00075, C_TEAL_BRIGHT, 1)

    # FPS en haut à droite
    fps_str = f"{fps:.0f} fps"
    (fw,_),_ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    put_text_aa(canvas, fps_str, px+pw-fw-PAD, icon_cy+6, 0.45, C_GRAY_MID, 1)

    y = header_h + PAD

    # ════════════════════════════════════════
    # SECTION 2 — STATUT ENREGISTREMENT
    # ════════════════════════════════════════
    status_h = int(ph * 0.07)
    sx, sy   = px + PAD, y
    draw_rect_rounded(canvas, (sx, sy), (sx+iw, sy+status_h),
                      C_BG_CARD, radius=10)

    if is_recording:
        dot_color = C_GREEN
        dot_label = "  ENREGISTREMENT"
        bar_color = C_GREEN
    else:
        dot_color = C_GRAY_MID
        dot_label = "  EN ATTENTE..."
        bar_color = C_GRAY_DIM

    # Point clignotant
    pulse = int(time.time() * 3) % 2
    if is_recording and pulse:
        cv2.circle(canvas, (sx+18, sy+status_h//2), 7, dot_color, -1)
    else:
        cv2.circle(canvas, (sx+18, sy+status_h//2), 7, dot_color,
                   2 if not is_recording else -1)

    put_text_aa(canvas, dot_label, sx+30, sy+status_h//2+5,
                pw*0.00070, dot_color if is_recording else C_GRAY_LIGHT, 1)

    # Compteur frames à droite
    frames_str = f"{buffer_len}/{N_FRAMES}"
    (ffw,_),_ = cv2.getTextSize(frames_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    put_text_aa(canvas, frames_str, sx+iw-ffw-8, sy+status_h//2+6,
                0.55, C_GRAY_LIGHT, 1)

    y += status_h + int(PAD * 0.5)

    # Barre de progression buffer
    bar_h = 6
    draw_progress_bar(canvas, sx, y, iw, bar_h,
                      buffer_len / N_FRAMES, bar_color, C_GRAY_DIM, 3)
    y += bar_h + PAD

    # ════════════════════════════════════════
    # SECTION 3 — PRÉDICTION PRINCIPALE
    # ════════════════════════════════════════
    pred_h = int(ph * 0.22)
    draw_rect_rounded(canvas, (px+PAD, y), (px+PAD+iw, y+pred_h),
                      C_BG_CARD, radius=12)

    # Titre section
    put_text_aa(canvas, "PREDICTION", px+PAD+12, y+20,
                pw*0.00060, C_TEAL, 1)

    # Ligne de séparation sous le titre
    cv2.line(canvas, (px+PAD+10, y+26), (px+PAD+iw-10, y+26), C_TEAL_DIM, 1)

    if last_sign:
        # Signe en grand — centré, taille adaptive
        sign_upper = last_sign.upper()
        sc = fit_text(sign_upper, iw - 20, max_scale=2.8, min_scale=0.8, thickness=3)
        draw_text_centered(canvas, sign_upper,
                           px + PAD + iw//2,
                           y + int(pred_h * 0.57),
                           sc, C_WHITE, 3)

        # Badge de confiance (couleur selon niveau)
        conf_pct = int(last_conf * 100)
        if conf_pct >= 70:
            badge_c = C_GREEN
        elif conf_pct >= 45:
            badge_c = C_ORANGE
        else:
            badge_c = C_RED

        badge_w, badge_h = 120, 28
        bx = px + PAD + iw//2 - badge_w//2
        by = y + pred_h - badge_h - 10
        draw_rect_rounded(canvas, (bx, by), (bx+badge_w, by+badge_h),
                          badge_c, radius=8, alpha=0.85)
        draw_text_centered(canvas, f"Confiance : {conf_pct}%",
                           bx + badge_w//2, by + badge_h//2,
                           0.42, C_WHITE, 1)
    else:
        draw_text_centered(canvas, "Faites un signe...",
                           px + PAD + iw//2,
                           y + pred_h//2 + 10,
                           0.55, C_GRAY_MID, 1)

    y += pred_h + PAD

    # ════════════════════════════════════════
    # SECTION 4 — TOP 5 CANDIDATS
    # ════════════════════════════════════════
    top5_h = int(ph * 0.23)
    draw_rect_rounded(canvas, (px+PAD, y), (px+PAD+iw, y+top5_h),
                      C_BG_CARD, radius=12)
    put_text_aa(canvas, "TOP 5  CANDIDATS", px+PAD+12, y+20,
                pw*0.00060, C_TEAL, 1)
    cv2.line(canvas, (px+PAD+10, y+26), (px+PAD+iw-10, y+26), C_TEAL_DIM, 1)

    row_h = (top5_h - 36) // 5
    for i, (sign_i, conf_i) in enumerate(top5_display[:5]):
        ry = y + 34 + i * row_h
        # Fond de la ligne (première en évidence)
        if i == 0 and last_sign:
            draw_rect_rounded(canvas,
                              (px+PAD+6, ry-2),
                              (px+PAD+iw-6, ry+row_h-4),
                              C_TEAL_DIM, radius=6, alpha=0.6)

        # Rang
        rank_colors = [C_GOLD, C_GRAY_LIGHT, C_GRAY_MID, C_GRAY_DIM, C_GRAY_DIM]
        put_text_aa(canvas, f"{i+1}",
                    px+PAD+14, ry+row_h-10,
                    0.50, rank_colors[i], 2 if i == 0 else 1)

        # Nom du signe
        sign_color = C_WHITE if i == 0 else C_GRAY_LIGHT
        sign_scale = 0.52 if i == 0 else 0.44
        put_text_aa(canvas, sign_i.capitalize(),
                    px+PAD+34, ry+row_h-10,
                    sign_scale, sign_color, 2 if i == 0 else 1)

        # Barre de confiance
        bar_x  = px + PAD + iw - 115
        bar_y  = ry + row_h//2 - 5
        bar_ww = 80
        bar_hh = 8
        bar_col = C_TEAL if i == 0 else C_GRAY_DIM
        draw_progress_bar(canvas, bar_x, bar_y, bar_ww, bar_hh,
                          conf_i, bar_col, C_GRAY_DIM, 3)

        # Pourcentage
        pct_str = f"{conf_i*100:.1f}%"
        put_text_aa(canvas, pct_str,
                    bar_x + bar_ww + 5, ry+row_h-10,
                    0.38, sign_color, 1)

    y += top5_h + PAD

    # ════════════════════════════════════════
    # SECTION 5 — PHRASE CONSTRUITE
    # ════════════════════════════════════════
    phrase_h = int(ph * 0.14)
    draw_rect_rounded(canvas, (px+PAD, y), (px+PAD+iw, y+phrase_h),
                      C_BG_CARD, radius=12)
    put_text_aa(canvas, "PHRASE", px+PAD+12, y+20, pw*0.00060, C_TEAL, 1)
    cv2.line(canvas, (px+PAD+10, y+26), (px+PAD+iw-10, y+26), C_TEAL_DIM, 1)

    if sentence:
        # Affiche les derniers mots — auto-scroll horizontal
        words     = sentence[-8:]
        word_str  = "  ·  ".join(w.capitalize() for w in words)
        wsc       = fit_text(word_str, iw-16, max_scale=0.75,
                             min_scale=0.30, thickness=2)
        draw_text_centered(canvas, word_str,
                           px+PAD+iw//2,
                           y + phrase_h//2 + 10,
                           wsc, C_WHITE, 2)
        # Compteur de mots
        cnt_str = f"{len(sentence)} mot{'s' if len(sentence)>1 else ''}"
        put_text_aa(canvas, cnt_str, px+PAD+iw-len(cnt_str)*7-6, y+20,
                    0.38, C_GRAY_MID, 1)
    else:
        draw_text_centered(canvas, "—",
                           px+PAD+iw//2, y+phrase_h//2+8,
                           0.8, C_GRAY_DIM, 1)

    y += phrase_h + PAD

    # ════════════════════════════════════════
    # SECTION 6 — INFOS MODÈLE
    # ════════════════════════════════════════
    info_h = int(ph * 0.07)
    draw_rect_rounded(canvas, (px+PAD, y), (px+PAD+iw, y+info_h),
                      C_BG_CARD, radius=8)

    col_w = iw // 3
    infos = [
        (f"Epoch {model_epoch}", "MODÈLE"),
        (f"{val_acc*100:.1f}%",  "VAL ACC"),
        (f"{N_CLASSES}",         "CLASSES"),
    ]
    for j, (val, lbl) in enumerate(infos):
        cx = px + PAD + col_w*j + col_w//2
        cy = y + info_h//2
        put_text_aa(canvas, val, cx-20, cy-2,  0.50, C_TEAL_BRIGHT, 2)
        put_text_aa(canvas, lbl, cx-22, cy+16, 0.32, C_GRAY_MID,    1)
        if j < 2:
            cv2.line(canvas,
                     (px+PAD+col_w*(j+1), y+8),
                     (px+PAD+col_w*(j+1), y+info_h-8),
                     C_GRAY_DIM, 1)

    y += info_h + PAD

    # ════════════════════════════════════════
    # SECTION 7 — RACCOURCIS CLAVIER
    # ════════════════════════════════════════
    kb_h = ph - y - 4
    if kb_h > 30:
        draw_rect_rounded(canvas, (px+PAD, y), (px+PAD+iw, y+kb_h),
                          C_NAVY_DEEP, radius=8)
        shortcuts = [
            ("[ESPACE]", "Prédire maintenant"),
            ("[C]",      "Effacer la phrase"),
            ("[Q]",      "Quitter"),
        ]
        sc_row = kb_h // (len(shortcuts)+1)
        for k, (key_str, desc) in enumerate(shortcuts):
            ky = y + sc_row*(k+1)
            # Badge touche
            (kw,_),_ = cv2.getTextSize(key_str, cv2.FONT_HERSHEY_SIMPLEX,
                                        0.38, 1)
            draw_rect_rounded(canvas, (px+PAD+8, ky-10),
                              (px+PAD+8+kw+8, ky+6),
                              C_GRAY_DIM, radius=4)
            put_text_aa(canvas, key_str, px+PAD+12, ky+4, 0.38, C_TEAL, 1)
            put_text_aa(canvas, desc, px+PAD+8+kw+16, ky+4,
                        0.38, C_GRAY_MID, 1)


# ════════════════════════════════════════════════════════════════
# RENDU DE LA ZONE CAMÉRA — overlays flexibles
# ════════════════════════════════════════════════════════════════

def draw_camera_overlay(canvas, cam_img, cw, ch, is_recording, buffer_len):
    """Dessine la zone caméra avec ses overlays (coin scanning, etc.)."""

    # Redimensionner le flux caméra dans la zone dédiée
    resized = cv2.resize(cam_img, (cw, ch))
    canvas[0:ch, 0:cw] = resized

    # Coins décoratifs
    corner_len = int(min(cw, ch) * 0.06)
    corner_t   = 3
    col = C_TEAL if is_recording else C_GRAY_MID
    pts_corners = [
        # coin haut-gauche
        [(0, corner_len), (0,0), (corner_len, 0)],
        # coin haut-droit
        [(cw-corner_len, 0), (cw, 0), (cw, corner_len)],
        # coin bas-gauche
        [(0, ch-corner_len), (0, ch), (corner_len, ch)],
        # coin bas-droit
        [(cw-corner_len, ch), (cw, ch), (cw, ch-corner_len)],
    ]
    for pts in pts_corners:
        for a, b in zip(pts, pts[1:]):
            cv2.line(canvas, a, b, col, corner_t, cv2.LINE_AA)

    # Scan line animée quand on enregistre
    if is_recording:
        scan_y = int(ch * ((time.time() * 0.3) % 1.0))
        scan_alpha = 0.15
        overlay_scan = canvas[max(0,scan_y-2):scan_y+3, 0:cw].copy()
        cv2.rectangle(canvas, (0, max(0,scan_y-1)), (cw, scan_y+2),
                      C_TEAL, -1)
        cv2.addWeighted(overlay_scan, 1-scan_alpha,
                        canvas[max(0,scan_y-2):scan_y+3, 0:cw],
                        scan_alpha, 0,
                        canvas[max(0,scan_y-2):scan_y+3, 0:cw])

    # Watermark en bas de la zone caméra
    wm  = "Equipe 8 - Phase E"
    put_text_aa(canvas, wm, 10, ch-10, 0.42, C_GRAY_DIM, 1)


# ════════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE — WEBCAM
# ════════════════════════════════════════════════════════════════

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam")
        return

    # Demander une résolution HD à la caméra
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Créer la fenêtre redimensionnable
    cv2.namedWindow('Sign Language Translator - Equipe 8',
                    cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('Sign Language Translator - Equipe 8', WIN_W, WIN_H)

    # État de l'application
    buffer         = []
    last_sign      = ""
    last_conf      = 0.0
    top5_display   = []
    is_recording   = False
    last_pred_time = 0
    sentence       = []
    no_hand_count  = 0
    fps            = 0.0
    fps_timer      = time.time()
    fps_count      = 0

    PRED_INTERVAL  = 0.5
    MAX_NO_HAND    = 20
    MIN_FRAMES     = 15

    # Infos modèle pour l'affichage
    model_epoch = ckpt.get('epoch', '?')
    model_vacc  = ckpt.get('val_acc', 0.0)

    print("\n🎬 Démo lancée !")
    print("   ESPACE : prédire maintenant")
    print("   C      : effacer la phrase")
    print("   Q      : quitter\n")

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as hands_detector, \
    mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as pose_detector:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ── FPS ──────────────────────────────────────────
            fps_count += 1
            if time.time() - fps_timer >= 1.0:
                fps       = fps_count / (time.time() - fps_timer)
                fps_count = 0
                fps_timer = time.time()

            frame = cv2.flip(frame, 1)

            # ── MediaPipe ────────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hand_results = hands_detector.process(rgb)
            pose_results = pose_detector.process(rgb)
            rgb.flags.writeable = True

            # ── Dessiner les landmarks sur la frame caméra ───
            if hand_results.multi_hand_landmarks:
                for hand_lm in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_lm, HAND_CONNECTIONS,
                        mp_draw_styles.get_default_hand_landmarks_style(),
                        mp_draw_styles.get_default_hand_connections_style()
                    )
            mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, POSE_CONNECTIONS
            )

            # ── Logique enregistrement ───────────────────────
            hands_detected = (hand_results.multi_hand_landmarks is not None)

            if hands_detected:
                no_hand_count = 0
                is_recording  = True
                lm = extract_landmarks(hand_results, pose_results)
                buffer.append(lm)
                if len(buffer) > N_FRAMES:
                    buffer = buffer[-N_FRAMES:]
            else:
                no_hand_count += 1
                if no_hand_count >= MAX_NO_HAND and len(buffer) >= MIN_FRAMES:
                    sign, conf, top5_list = predict(buffer)
                    if conf > 0.3:
                        last_sign    = sign
                        last_conf    = conf
                        top5_display = top5_list
                        sentence.append(sign)
                        speak(sign)
                    buffer        = []
                    is_recording  = False
                    no_hand_count = 0
                elif no_hand_count >= MAX_NO_HAND:
                    buffer        = []
                    is_recording  = False
                    no_hand_count = 0

            # ── Prédiction en continu ────────────────────────
            now = time.time()
            if (is_recording and
                len(buffer) >= MIN_FRAMES and
                now - last_pred_time > PRED_INTERVAL):
                sign, conf, top5_list = predict(buffer)
                last_sign      = sign
                last_conf      = conf
                top5_display   = top5_list
                last_pred_time = now

            # ════════════════════════════════════════════════
            # RENDU — CANVAS FLEXIBLE
            # ════════════════════════════════════════════════

            # Récupérer la taille réelle de la fenêtre (flexible)
            rect = cv2.getWindowImageRect(
                'Sign Language Translator - Equipe 8')
            if rect[2] > 100 and rect[3] > 100:
                win_w = rect[2]
                win_h = rect[3]
            else:
                win_w, win_h = WIN_W, WIN_H

            panel_w = max(300, int(win_w * 0.33))  # 33% pour le panneau
            cam_w   = win_w - panel_w
            cam_h   = win_h

            # Canvas vierge
            canvas = np.full((win_h, win_w, 3), C_BG_DARK, dtype=np.uint8)

            # Zone caméra
            draw_camera_overlay(canvas, frame, cam_w, cam_h,
                                is_recording, len(buffer))

            # Panneau latéral droit
            state = {
                'last_sign':    last_sign,
                'last_conf':    last_conf,
                'top5_display': top5_display,
                'is_recording': is_recording,
                'buffer_len':   len(buffer),
                'sentence':     sentence,
                'fps':          fps,
                'val_acc':      model_vacc,
                'model_epoch':  model_epoch,
            }
            draw_panel(canvas, cam_w, panel_w, win_h, state)

            cv2.imshow('Sign Language Translator - Equipe 8', canvas)

            # ── Touches ──────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' ') and len(buffer) >= MIN_FRAMES:
                sign, conf, top5_list = predict(buffer)
                last_sign    = sign
                last_conf    = conf
                top5_display = top5_list
                if conf > 0.2:
                    sentence.append(sign)
                    speak(sign)
                buffer = []
            elif key == ord('c'):
                sentence     = []
                last_sign    = ""
                top5_display = []
                buffer       = []

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("✅ Démo terminée")


if __name__ == '__main__':
    main()