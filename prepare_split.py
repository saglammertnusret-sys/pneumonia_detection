r"""Create a 20% validation split without changing the original dataset.

Run from the project folder:
  .venv\Scripts\python.exe prepare_split.py --source data/chest_xray \
      --output data/chest_xray_grouped --exclude-test-overlap

Images are copied; the original test set is copied with its labels unchanged.
Related filenames and byte-identical images stay together. Filename groups are
proxies, not verified patient identities. Source metadata is needed to establish
true patient independence. Groups whose filename IDs occur in the test set are
excluded from the training/validation pool only when explicitly requested.

Method reference:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
"""
import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from pneumonia.data import CLASSES, counts, scan_dataset

# The first capture group is compatible with pneumonia.cli --patient-regex.
GROUP_REGEX = r'^((?:NORMAL2-)?IM-[0-9]+|person[0-9]+)(?:-|_)'


def filename_group(relative_path):
    name = Path(relative_path).name
    match = re.match(GROUP_REGEX, name)
    if not match:
        raise ValueError(f'Unrecognized filename: {name}. No split created; share this filename for review.')
    return match.group(1)


def prepare(source, output, seed=42, exclude_test_overlap=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f'Output already exists: {output}. Choose a new folder; existing files will not be overwritten.')
    if output == source or source in output.parents:
        raise ValueError('Output must be outside the original dataset folder.')
    records = scan_dataset(source)
    for rows in records.values():
        for row in rows:
            row['filename_group'] = filename_group(row['path'])
    pool = records['train'] + records['val']

    # Union filename groups connected by identical image bytes, so duplicate
    # images within the original training split cannot leak into the new val set.
    parent = {row['filename_group']: row['filename_group'] for row in pool}
    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    seen_hash = {}
    for row in pool:
        group = row['filename_group']
        if row['sha256'] in seen_hash:
            union(group, seen_hash[row['sha256']])
        else:
            seen_hash[row['sha256']] = group
    test_groups = {row['filename_group'] for row in records['test']}
    overlapping = {find(group) for group in test_groups if group in parent}
    excluded = [row for row in pool if find(row['filename_group']) in overlapping]
    if excluded and not exclude_test_overlap:
        raise ValueError(
            f'{len(excluded)} pool images belong to filename groups overlapping the test set. '
            'Use --exclude-test-overlap to set those images aside while keeping test unchanged.')
    pool = [row for row in pool if find(row['filename_group']) not in overlapping]
    component_labels = {}
    for row in pool:
        component = find(row['filename_group'])
        component_labels.setdefault(component, set()).add(row['label'])
    if any(len(labels) != 1 for labels in component_labels.values()):
        raise ValueError('A filename or duplicate-image group contains conflicting labels. Review the data.')
    for label, name in enumerate(CLASSES):
        number = sum(labels == {label} for labels in component_labels.values())
        if number < 5:
            raise ValueError(f'Need at least five independent filename groups for {name}; found {number}.')

    labels = np.array([row['label'] for row in pool])
    groups = np.array([find(row['filename_group']) for row in pool])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates = list(splitter.split(np.zeros(len(pool)), labels, groups))
    # Select only by split sizes/class balance, never by model/test performance.
    def split_cost(candidate):
        _, validation = candidate
        return sum(abs(np.sum(labels[validation] == label) / np.sum(labels == label) - .2)
                   for label in (0, 1))
    selected = min(range(len(candidates)), key=lambda i: (split_cost(candidates[i]), i))
    training, validation = candidates[selected]
    planned = {'train': [pool[i] for i in training], 'val': [pool[i] for i in validation],
               'test': records['test']}
    if any(not any(row['label'] == label for row in rows)
           for rows in planned.values() for label in (0, 1)):
        raise ValueError('A planned split has only one class. No split created.')

    report = {
        'source': str(source), 'output': str(output), 'seed': seed,
        'target_validation_fraction': .2, 'selected_fold': selected,
        'actual_validation_fraction': len(validation) / len(pool),
        'filename_group_regex': GROUP_REGEX,
        'verified_patient_metadata': False,
        'grouping_limit': 'Filename IDs are proxies. IM and NORMAL2-IM namespaces are kept distinct; '
                          'unrecorded identities or alternate filenames require source metadata.',
        'original_counts': counts(records), 'prepared_counts': counts(planned),
        'excluded_pool_counts': {name: sum(row['label'] == label for row in excluded)
                                 for label, name in enumerate(CLASSES)},
        'excluded_pool_images': excluded,
        'test_image_bytes_and_labels_unchanged': True,
        'manifest': [],
    }
    print('Original counts:', json.dumps(report['original_counts']), flush=True)
    print('Prepared counts:', json.dumps(report['prepared_counts']), flush=True)
    print('Excluded from train/val due to test-group overlap:', len(excluded), flush=True)
    print('Copying images to a separate folder. Original files stay in place.', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.split-building-', dir=output.parent))
    try:
        for split in planned:
            for name in CLASSES:
                (staging / split / name).mkdir(parents=True)
        for split, rows in planned.items():
            for row in rows:
                parts = Path(row['path']).parts
                original_split, class_name = parts[0], parts[1]
                tail = Path(*parts[2:])
                if split != 'test':
                    tail = Path(f'from_{original_split}') / tail
                relative = Path(split) / class_name / tail
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / row['path'], target)
                report['manifest'].append({
                    'source_path': row['path'], 'output_path': relative.as_posix(),
                    'label': row['label'], 'sha256': row['sha256'],
                    'filename_group': row['filename_group'],
                })
        checked = scan_dataset(staging, GROUP_REGEX)
        assert counts(checked) == report['prepared_counts']
        expected_hashes = {row['output_path']: row['sha256'] for row in report['manifest']}
        assert all(row['sha256'] == expected_hashes[row['path']]
                   for rows in checked.values() for row in rows)
        (staging / 'split_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f'SPLIT READY: {output}', flush=True)
    print('Report: split_report.json. Filename groups and exact image bytes do not cross splits.', flush=True)
    print('Filename grouping does not establish verified patient independence.', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description='Copy a grouped 80/20 train/val split; preserve the test set.')
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exclude-test-overlap', action='store_true',
                        help='Exclude pool filename groups found in test, and record all excluded images')
    args = parser.parse_args()
    try:
        prepare(args.source, args.output, args.seed, args.exclude_test_overlap)
    except (ValueError, FileNotFoundError, OSError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
