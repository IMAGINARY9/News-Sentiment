"""
Visualization utilities for News-Sentiment prediction explanations.

This module provides functions to visualize model predictions, token importances, and explanations using LIME or similar libraries.
Plots are saved to the /visualizations directory.
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Optional: LIME for transformers
try:
    from lime.lime_text import LimeTextExplainer
except ImportError:
    LimeTextExplainer = None

def _get_bar_colors(importances, pos_color, neg_color):
    return [neg_color if s < 0 else pos_color for s in importances]

def _draw_bar_labels(ax, bars, fontsize):
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (0.004 if width >= 0 else -0.004),
            bar.get_y() + bar.get_height() / 2,
            f'{width:.2f}',
            va='center',
            ha='left' if width >= 0 else 'right',
            fontsize=fontsize,
            color='#333',
            fontweight='normal',
            clip_on=True
        )

def _style_axes(ax, label_fontsize):
    ax.xaxis.grid(True, linestyle=':', linewidth=0.7, color='#bbb', zorder=0, alpha=0.5)
    ax.yaxis.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#bbb')
    ax.tick_params(axis='y', pad=1.0)
    plt.yticks(fontsize=label_fontsize, color='#222')

def _add_prob_annotation(ax, label_names, probs, fontsize, pred_label):
    prob_lines = [f"{n}: {p:.2f}" for n, p in zip(label_names, probs)]
    prob_str = "\n".join(prob_lines)
    # Decide annotation position based on prediction
    pred_name = label_names[pred_label].lower()
    if pred_name == 'positive':
        xy = (0.98, 0.99)
        ha = 'right'
    else:
        xy = (0.05, 0.95)
        ha = 'left'
    ax.annotate(
        f"Probabilities:\n{prob_str}",
        xy=xy, xycoords='axes fraction',
        fontsize=fontsize, color='#222', ha=ha, va='top',
        fontweight='medium',
        bbox=dict(boxstyle='round,pad=0.28', fc='#f8f8f8', ec='#888', lw=1.2, alpha=0.93)
    )

def _finalize_plot(fig, ax):
    # Get current xlim
    xlim = ax.get_xlim()
    x_range = xlim[1] - xlim[0]
    # Add 12.5% margins to the right and left
    ax.set_xlim(xlim[0] - 0.125 * x_range, xlim[1] + 0.125 * x_range)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

def explain_and_plot_transformer(model, tokenizer, text, label_names, save_path):
    """
    Generate a LIME explanation plot for a transformer model prediction.
    """
    if LimeTextExplainer is None:
        raise ImportError("LIME is not installed. Please install lime to use visualization.")

    from matplotlib import cm
    label_fontsize = 13
    header_fontsize = label_fontsize + 2
    pos_color = cm.Greens(0.6)
    neg_color = cm.Reds(0.5)
    explainer = LimeTextExplainer(class_names=label_names)

    def predict_proba(texts):
        import torch
        model.eval()
        encodings = tokenizer(texts, padding=True, truncation=True, max_length=tokenizer.model_max_length, return_tensors='pt')
        with torch.no_grad():
            outputs = model(encodings['input_ids'], encodings['attention_mask'])
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    exp = explainer.explain_instance(text, predict_proba, num_features=20, labels=[0,1,2])
    pred_probs = predict_proba([text])[0]
    pred_label = int(pred_probs.argmax())
    pred_prob = float(pred_probs[pred_label])
    # Get word importances for the predicted label and sort DESC
    word_scores = dict(exp.as_list(label=pred_label))
    sorted_items = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    words = [w for w, _ in sorted_items]
    importances = [word_scores[w] for w in words]
    colors = _get_bar_colors(importances, pos_color, neg_color)
    # Set font sizes
    label_fontsize = 13
    header_fontsize = label_fontsize + 2
    fig, ax = plt.subplots(figsize=(6, min(1.1 + 0.36*len(words), 11)))
    bar_height = 0.38
    bars = ax.barh(range(len(words)), importances, color=colors, edgecolor='none', zorder=3, height=bar_height)
    ax.axvline(0, color='#888', linewidth=1, linestyle='-', zorder=2)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=label_fontsize, fontweight='medium', color='#222')
    ax.set_xlabel('Token Importance', fontsize=label_fontsize, labelpad=6)
    ax.set_title(f"LIME Explanation\nPredicted: {label_names[pred_label]} (p={pred_prob:.2f})", fontsize=header_fontsize, fontweight='bold', pad=8, color='#222')
    _draw_bar_labels(ax, bars, label_fontsize)
    _style_axes(ax, label_fontsize)
    _add_prob_annotation(ax, label_names, pred_probs, label_fontsize, pred_label)
    _finalize_plot(fig, ax)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return save_path

def explain_and_plot_lstm(model, vocab_builder, text, label_names, save_path):
    """
    Visualize token importances for LSTM models (simple gradient-based approach).
    """
    from matplotlib import cm
    label_fontsize = 13
    header_fontsize = label_fontsize + 2
    pos_color = cm.Greens(0.6)
    neg_color = cm.Reds(0.5)
    neu_color = cm.Greys(0.5)
    import torch
    import numpy as np
    model.eval()
    tokens = vocab_builder.preprocess([text])[0]
    input_ids = vocab_builder.encode([tokens], max_len=vocab_builder.max_len)
    input_tensor = torch.tensor(input_ids, dtype=torch.long)
    input_tensor.requires_grad = True
    output = model(input_tensor)
    logits = output if not isinstance(output, dict) else output['logits']
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_label = int(np.argmax(probs))
    pred_prob = float(probs[pred_label])
    score = logits[0, pred_label]
    score.backward()
    grads = input_tensor.grad.abs().squeeze().numpy()
    token_list = tokens[:len(grads)]
    if len(label_names) == 3:
        color_map = {'negative': neg_color, 'neutral': neu_color, 'positive': pos_color}
    else:
        color_map = {n: pos_color for n in label_names}
    color = color_map.get(label_names[pred_label], pos_color)
    token_grads = list(zip(tokens[:len(grads)], grads))
    token_grads.sort(key=lambda x: abs(x[1]), reverse=True)
    sorted_tokens = [t for t, _ in token_grads]
    sorted_grads = [g for _, g in token_grads]
    fig, ax = plt.subplots(figsize=(6, min(1.1 + 0.36*len(sorted_tokens), 11)))
    bar_height = 0.38
    bars = ax.barh(range(len(sorted_tokens)), sorted_grads, tick_label=sorted_tokens, color=color, edgecolor='none', zorder=3, height=bar_height)
    ax.axvline(0, color='#888', linewidth=1, linestyle='-', zorder=2)
    ax.set_xlabel("Gradient Magnitude", fontsize=label_fontsize, labelpad=6)
    ax.set_ylabel("Token", fontsize=label_fontsize)
    ax.set_title(f"LSTM Token Importances\nPredicted: {label_names[pred_label]} (p={pred_prob:.2f})", fontsize=header_fontsize, fontweight='bold', pad=8, color='#222')
    _draw_bar_labels(ax, bars, label_fontsize)
    _style_axes(ax, label_fontsize)
    _add_prob_annotation(ax, label_names, probs, label_fontsize, pred_label)
    _finalize_plot(fig, ax)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return save_path
