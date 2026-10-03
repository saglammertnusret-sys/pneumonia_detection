"""Make a dated local ZIP, optionally including run files; never include patient data."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True, help='External drive or synced folder')
    parser.add_argument('--include-runs', action='store_true', help='Include checkpoints and experiment reports')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    destination = Path(args.destination).resolve()
    if destination == root or root in destination.parents:
        parser.error('Choose a destination outside the project directory.')
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"pneumonia-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.zip"
    excluded = {'.venv', '__pycache__', '.git', 'data', 'backups', '.streamlit'}
    if not args.include_runs:
        excluded.add('runs')
    with ZipFile(path, 'x', ZIP_DEFLATED) as archive:
        for file in sorted(root.rglob('*')):
            relative = file.relative_to(root)
            if file.is_file() and not any(part in excluded for part in relative.parts) and file.suffix != '.pyc':
                archive.write(file, Path(root.name) / relative)
    print(f'Backup saved: {path}')

if __name__ == '__main__':
    main()
