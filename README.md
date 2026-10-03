# Pneumonia Detection from Chest X Rays

A PyTorch image-classification project using an ImageNet-pretrained ResNet18, optional CLAHE preprocessing, training augmentation and class-weighted binary cross-entropy.

**Author:** Mert Nusret Sağlam

## Recorded experiment

The ResNet18 experiment completed ten training epochs on the prepared dataset. The final held-out evaluation used **624 images**: 234 NORMAL and 390 PNEUMONIA, with a decision threshold of 0.5.

| Test metric | Result |
|---|---:|
| Accuracy | 84.29% |
| ROC-AUC | 0.963, rounded from the evaluation plot |
| Pneumonia sensitivity | 99.49% |
| Normal specificity | 58.97% |
| Pneumonia precision | 80.17% |

Confusion matrix, rows = actual class and columns = predicted class:

| | Predicted NORMAL | Predicted PNEUMONIA |
|---|---:|---:|
| Actual NORMAL | 138 | 96 |
| Actual PNEUMONIA | 2 | 388 |

High sensitivity comes with substantial false positives among normal images. These are dataset-specific research results, not a clinical validation. The last epoch's validation accuracy of 99.06% is a validation result, not the reported test accuracy.

## Data preparation and checks

Filename groups and exact image hashes were kept separate across splits. A total of 457 images were excluded from the training/validation pool because their filename groups overlapped the test set. Prepared counts were 3,820 training, 955 validation and 624 test images.

Filename grouping does **not** establish verified patient independence. The project audits exact duplicate overlap and supports filename-group checks; stronger independence claims need patient metadata.

Dataset source: [Chest X-Ray Images Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Download it separately from its source and review the dataset terms. Images and trained checkpoints are not included in this repository.

## Quick checks on Windows

Follow [SETUP.md](SETUP.md) for the full dependency, dataset and CLI instructions. Then run:

```powershell
.\.venv\Scripts\python.exe smoke_test.py --clahe
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The random-image smoke test checks the pipeline only; its metrics have no research meaning. The project includes training progress logs, resumable completed-epoch checkpoints and CSV/JSON/PNG evaluation reports. Other implemented architectures have not been benchmarked against this trained ResNet18 experiment.

See [results/summary.json](results/summary.json) for the aggregate result summary. This file records observed results, not the model or image-level predictions.
