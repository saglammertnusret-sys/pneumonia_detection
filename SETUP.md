# Pneumonia detection from chest X-rays

Rebuilt for Mert Nusret Sağlam from the project description in his CV. This is new source code, not a recovery of the lost implementation. It provides a complete path from labeled images to training, held-out evaluation, and a local demonstration.

No trained weights or clinical images are included. The previous project's 91% accuracy and 0.95 AUC are not claimed for this reconstruction. New results must come from the evaluation files produced by a real experiment.

## What is included

- Custom CNN plus ResNet18, VGG16, and InceptionV3 transfer learning options.
- Optional CLAHE, aspect-ratio-preserving resizing, training-only augmentation, and weighted binary loss for class imbalance.
- Dataset checks for damaged images and exact duplicates across splits; optional patient ID checks.
- Validation AUC selects the checkpoint. The test split is used only by the evaluation command.
- Accuracy, ROC AUC, pneumonia recall, normal specificity, confusion matrix, per-image scores, and plots.
- Saved configuration, dataset hashes, package versions, and training history for each run.
- Streamlit image-upload demonstration and a dated backup command.

Research and portfolio use only. This binary classifier cannot establish a clinical diagnosis, detect every disease, or verify that an uploaded image is a chest X-ray. Scores are not calibrated clinical probabilities.

## Windows setup

1. Install **Python 3.11 for Windows** from https://www.python.org/downloads/windows/ with the Python launcher enabled.
2. Extract this ZIP to a normal folder, for example `C:\Projects\pneumonia_detection`. Open Windows Terminal in that folder.
3. This package contains no setup script. Review these commands, then run them individually in PowerShell after any Windows security warning has been identified:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Commands call the environment's Python directly, so you do not need to change PowerShell's execution policy or activate the environment. The default installation uses the CPU. For an NVIDIA GPU, use the compatible Windows/Pip/CUDA command from https://pytorch.org/get-started/locally/ inside the same environment instead of the CPU installation command.

If Windows blocks a file, do not choose Run anyway or disable protection. Record the exact warning and affected file. An unrecognized-app or reputation warning does not tell us the same thing as a named malware detection. For a detected threat, keep the file blocked and inspect Windows Security > Virus & threat protection > Protection history. Removing an installer script does not establish that the remaining package or downloaded dependencies are malware-free.

## Check the installation first

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe smoke_test.py
.\.venv\Scripts\python.exe smoke_test.py --clahe
```

The smoke test creates temporary random images, trains a small CNN for one epoch, reloads the checkpoint, evaluates it, and checks the three transfer model heads. It downloads no model weights. All temporary images and metrics are deleted afterwards. Any scores from this test have no research meaning.

## Get the dataset

If you remember the original dataset, use that version and record its source. A publicly available starting option is the Chest X-Ray Images Pneumonia dataset:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Download it through its source, review its current terms, and extract it yourself. Do not put private patient images in this project. Point `--data` to the directory with this layout (often named `chest_xray`):

```text
data/chest_xray/
  train/NORMAL/        image files
  train/PNEUMONIA/     image files
  val/NORMAL/          image files
  val/PNEUMONIA/       image files
  test/NORMAL/         image files
  test/PNEUMONIA/      image files
```

Both labels must be present in every split. Extra non-image files are ignored. JPEG, PNG, BMP, and TIFF are supported.

```powershell
.\.venv\Scripts\python.exe -m pneumonia.cli audit --data "data\chest_xray" --output audit.json
```

The audit fails if identical image bytes appear in different splits. This does not establish patient independence: differently encoded copies and different views from one patient require metadata checks. If your filenames encode a reliable patient ID, pass `--patient-regex` with that ID in capture group 1 to audit, train, and evaluate. For example, `--patient-regex "^(patient[0-9]+)_"` is appropriate only when every filename actually follows that convention. Do not assume an arbitrary filename number is a patient ID.

A validation split with only a few images is inadequate for stable model selection. Before serious benchmarking, create a larger validation split from the original training pool, keep every patient's images within one split, and leave the test pool untouched. The code preserves the folders you provide; it does not silently redistribute images. Record this split procedure in your experiment notes.

## Train the first model

Start with ResNet18. Its ImageNet weights download on the first training run.

```powershell
.\.venv\Scripts\python.exe -m pneumonia.cli train --data "data\chest_xray" --output "runs\resnet18" --model resnet18 --epochs 10 --batch-size 16 --clahe
```

Training uses automatic CPU/CUDA selection, AdamW, a fixed seed of 42, class-weighted binary cross entropy, and early stopping on validation AUC. Images are converted to grayscale, optionally enhanced with CLAHE, padded to a square, converted to three channels, and normalized. Augmentation uses small affine rotations, translations, and scale changes on training images only. The last incomplete training batch is dropped to prevent one-image batch normalization failures; validation and test images are all included.

Each experiment must have a **new empty output folder**. The command refuses to overwrite earlier runs. Reduce `--batch-size` to `4` if memory is tight. VGG16 and InceptionV3 need more memory and can be slow on a CPU. For a custom CNN trained without any weight download:

```powershell
.\.venv\Scripts\python.exe -m pneumonia.cli train --data "data\chest_xray" --output "runs\cnn" --model cnn --weights none --epochs 15 --batch-size 16 --clahe
```

ResNet18 and VGG16 use 224-pixel inputs; InceptionV3 uses 299-pixel inputs. The checkpoint stores the input size and CLAHE setting so evaluation and the demo use the same preprocessing.

## Evaluate on the held-out test split

```powershell
.\.venv\Scripts\python.exe -m pneumonia.cli evaluate --data "data\chest_xray" --checkpoint "runs\resnet18\best.pt" --output "runs\resnet18\test"
```

Keep `manifest.json` beside `best.pt`: evaluation checks test image hashes against the training and validation manifest stored for that run. If training used patient ID checks, evaluation also uses the recorded regex to check against those patients.

Read `runs\resnet18\test\metrics.json`, `predictions.csv`, and `evaluation.png`. Confusion matrix rows are actual labels and columns are predicted labels, in the order NORMAL, PNEUMONIA. ROC AUC is computed from continuous pneumonia scores. Threshold-based metrics use 0.5 by default. If you choose another threshold, choose it using validation data, then pass `--threshold` when evaluating the test set. Do not tune thresholds or architectures using test results.

## Compare architectures

Run separate training experiments with `--model resnet18`, `--model vgg16`, `--model inception_v3`, and `--model cnn`. Use the same data splits, seed, CLAHE setting, and class balancing for a controlled comparison. Select architecture and settings using validation results first, then conduct the final held-out test evaluation. Training histories and environment files are stored with each checkpoint.

For a useful project report, record dataset source/version, image and patient counts, split method, augmentation, image size, model initialization, epochs, hardware, and all evaluation metrics. Do not compare scores across different splits as though they came from the same experiment.

## Run the local demo

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. Enter the checkpoint path and upload an image. The demo uses the CPU and your trained checkpoint; it does not send the image to an external AI service.

## Back up your work

After code changes, make a backup to a synced folder or external drive outside the project directory:

```powershell
.\.venv\Scripts\python.exe backup.py --destination "E:\ProjectBackups"
.\.venv\Scripts\python.exe backup.py --destination "E:\ProjectBackups" --include-runs
```

The first command backs up source and instructions. The second also includes checkpoints and experiment reports. Dataset images and virtual environments are excluded. Keep an independent copy of your public dataset source/version or download instructions; large public datasets do not need to be in the source ZIP. A second folder on the same disk is not a backup against disk loss.

If you use Git, initialize it and commit your code, then push to your own remote repository. `.gitignore` excludes datasets, checkpoints, and local environments. Git commits that exist only on your computer will not survive losing the computer.

## Troubleshooting

- **`py` not recognized:** install Python with the Windows launcher, then reopen Terminal.
- **Missing train/val/test directories:** point `--data` one directory deeper, to the folder that directly contains these three folders.
- **Not enough memory:** reduce batch size and use CNN or ResNet18.
- **Weight download unavailable:** use `--weights none`. Train a new run; randomly initialized weights are not equivalent to pretrained ones.
- **Run directory already exists:** choose a different output name.
- **One class or duplicate overlap error:** fix the dataset organization before training. Do not remove leakage checks to make a score look better.

## Implementation references

- PyTorch Windows installation: https://pytorch.org/get-started/locally/
- Torchvision models: https://docs.pytorch.org/vision/stable/models.html
- InceptionV3: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.inception_v3.html
- ROC AUC: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
- Streamlit resource caching: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource

See `VALIDATION.md` for what was tested before packaging. Windows commands are provided, but this package was validated in a Linux execution environment.
