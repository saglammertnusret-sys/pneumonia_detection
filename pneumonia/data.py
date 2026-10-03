"""Dataset checks deliberately do not require PyTorch."""
import hashlib
import re
from pathlib import Path
from PIL import Image

CLASSES = ['NORMAL', 'PNEUMONIA']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def scan_dataset(root, patient_regex=None):
    root = Path(root).resolve()
    pattern = re.compile(patient_regex) if patient_regex else None
    records = {split: [] for split in ('train', 'val', 'test')}
    seen_hashes, seen_patients = {}, {}
    for split, rows in records.items():
        for label, class_name in enumerate(CLASSES):
            directory = root / split / class_name
            if not directory.is_dir():
                raise ValueError(f'Missing directory: {directory}. See README.md for the required layout.')
            files = sorted(p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
            if not files:
                raise ValueError(f'No supported images in {directory}')
            for path in files:
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception as error:
                    raise ValueError(f'Unreadable image: {path}') from error
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in seen_hashes and seen_hashes[digest] != split:
                    raise ValueError(f'Identical image occurs in multiple splits: {path}')
                seen_hashes[digest] = split
                patient = None
                if pattern:
                    match = pattern.search(path.name)
                    if not match or not match.groups() or not match.group(1):
                        raise ValueError(f'Patient regex needs a capture group matching {path.name}')
                    patient = match.group(1)
                    if patient in seen_patients and seen_patients[patient] != split:
                        raise ValueError(f'Patient {patient!r} occurs in multiple splits')
                    seen_patients[patient] = split
                rows.append({'path': path.relative_to(root).as_posix(), 'label': label,
                             'sha256': digest, 'patient': patient})
    return records

def counts(records):
    return {split: {name: sum(row['label'] == i for row in rows)
                    for i, name in enumerate(CLASSES)} for split, rows in records.items()}
