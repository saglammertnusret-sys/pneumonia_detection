# Validation record

Checked on 2026-10-02 in a Linux CPU environment.

## Passed

- Six unit tests: class mapping and counts, duplicate overlap rejection, missing class rejection, patient overlap rejection, score-based AUC and recall, and report file generation.
- End-to-end synthetic-data smoke test: audit, CNN training, checkpoint save, safe checkpoint reload, held-out evaluation, CSV/JSON/PNG outputs, and single-image inference using recorded preprocessing.
- Output head forward checks for ResNet18, VGG16, and InceptionV3 without downloaded weights.
- Official pretrained ResNet18 weight download and forward pass.
- Backup checks for both source-only and checkpoint-inclusive archives; dataset files were excluded.
- Python source compilation.

## Environment used

- Python 3.12.14
- PyTorch 2.14.1+cpu
- Torchvision 0.29.1+cpu
- NumPy 2.5.2
- Pillow 12.3.0
- scikit-learn 1.8.0

## Still needs local validation

- The Windows runtime and manual setup commands were not executed here. The original setup script was removed following a reported Windows warning; its exact warning text has not yet been provided.
- OpenCV could not be installed through this environment's network access. The CLAHE branch is implemented but was not executed here. After setup, run `smoke_test.py --clahe`.
- Streamlit could not be installed here. The image-upload app was not run; its shared model-loading and inference code was tested by the smoke test.
- Pretrained VGG16 and InceptionV3 weight downloads were not exercised. Their randomly initialized output heads were checked.
- GPU execution and multi-epoch training on real chest X-rays were not exercised.

Synthetic images only test the software path. No accuracy or AUC from those tests should be used in a CV, research report, or performance claim. Real-data results, patient independence, dataset provenance, and external generalization remain to be established.

## Package update

Removed `setup_windows.cmd` and replaced its instructions with visible manual commands. Reviewed the original script: it changed to the project directory, created a Python virtual environment, installed dependencies through pip, and displayed status messages. No Windows antivirus scan was performed here, and no conclusion about the reported warning can be drawn without its exact text and affected filename.
