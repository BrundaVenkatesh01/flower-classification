# 🌸 Flower Image Classification — CNN Baseline vs Transfer Learning

Classifying flower photos into 5 species (daisy, dandelion, rose, sunflower, tulip) with PyTorch, comparing a from-scratch CNN against ResNet18 transfer learning.

**Final test accuracy: 86.4%** (ResNet18, frozen features)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BrundaVenkatesh01/flower-classification/blob/main/flower_classification.ipynb)

## Dataset

[Flowers Recognition (Kaggle)](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) — 4,317 images across 5 classes, mixed sizes (~320×240), scraped from Flickr/Google/Yandex. The dataset ships unsplit; I created a reproducible 70/15/15 train/val/test split (seed 42): 3,021 / 647 / 649 images.

Classes are mildly imbalanced (dandelion 1,052 vs sunflower 733) — noted but not corrected, since the ratio is well under levels requiring class weighting.

## Approach & Results

| Stage | Model | Val Accuracy | Notes |
|---|---|---|---|
| 1. Baseline | 3-block CNN (from scratch) | ~66% | Overfit after epoch 5 — train loss 0.12 vs val loss 1.68 |
| 2. + Data augmentation | Same CNN | ~67% | Overfitting eliminated (curves stay together), but model capacity became the bottleneck |
| 3. Transfer learning | ResNet18 (ImageNet, frozen) + new 5-class head | ~87% | 78% after a single epoch — only ~2.5K trainable weights |
| **Final** | **ResNet18** | **86.4% test** | Test ≈ val → no leakage, honest estimate |

Key takeaways from the experiments:

- **Overfitting, observed live:** the baseline's train loss collapsed to 0.12 while val loss rose from 0.91 to 1.68 — the model memorized training images. Best checkpoint was epoch 5, not epoch 10.
- **Augmentation fixes memorization, not capacity:** random crops, flips, rotation, and color jitter (train set only) closed the train/val gap but barely moved accuracy — the small CNN had hit its ceiling.
- **Transfer learning dominates on small datasets:** ResNet18's ImageNet features beat 15 epochs of custom training after one epoch, training only the final layer.

## Error Analysis

The confusion matrix shows the model's main weakness is **rose → tulip** (18 of 115 roses misclassified as tulips; the reverse error is rare at 4). Both are cup-shaped and often photographed in similar close-up framing. Dandelion is the strongest class (138/156) — its shape is visually unmistakable.

## Tech Stack

PyTorch · torchvision · scikit-learn · matplotlib · Google Colab (T4 GPU)

## Reproducing

1. Open the notebook in Colab (badge above)
2. Get a Kaggle API token (kaggle.com → Settings → API) and paste it in the first cell
3. Runtime → Change runtime type → T4 GPU
4. Run all cells (~25 min total)

Fixed seed (42) makes the data split and results reproducible.

## Future Work

- Fine-tune ResNet18's last block with a low learning rate (est. +2–4%)
- Early stopping / model checkpointing on best val loss
- Class-weighted loss or targeted augmentation for the rose/tulip confusion
- Deploy as a simple Gradio demo
