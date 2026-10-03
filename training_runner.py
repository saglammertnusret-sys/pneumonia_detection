# Standalone entry point: put this file beside smoke_test.py in the project root.
import sys
from pathlib import Path
if not __package__:
    if not (Path(__file__).resolve().parent / "pneumonia" / "data.py").is_file():
        raise SystemExit("Place training_runner.py in the project folder beside smoke_test.py, then run it again.")
    __package__ = "pneumonia"

import argparse
import json
import sys
import time
from pathlib import Path
from .data import scan_dataset, counts

def _write_json(path, value):
    import os
    temporary = path.with_name(path.name + '.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def _save_checkpoint(path, checkpoint):
    import os
    import torch
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def _progress(label, step, total, started, every, loss=None):
    if step == 1 or step % every == 0 or step == total:
        elapsed = time.monotonic() - started
        remaining = elapsed / step * (total - step)
        extra = f'; loss={loss:.4f}' if loss is not None else ''
        print(f'{label}: batch {step}/{total} ({100 * step / total:.1f}%); '
              f'elapsed={elapsed:.0f}s; ETA={remaining:.0f}s{extra}', flush=True)

def _predict_progress(model, loader, device, label, every):
    import torch
    paths, labels, scores = [], [], []
    model.eval()
    started = time.monotonic()
    print(f'{label}: starting {len(loader)} batches', flush=True)
    with torch.inference_mode():
        for step, (images, targets, names) in enumerate(loader, 1):
            probabilities = model(images.to(device)).flatten().sigmoid().cpu().tolist()
            paths.extend(names)
            labels.extend(targets.tolist())
            scores.extend(probabilities)
            _progress(label, step, len(loader), started, every)
    return paths, labels, scores

def _rng_state(generator):
    import random
    import numpy as np
    import torch
    state = np.random.get_state()
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'generator': generator.get_state(),
            'numpy': {'name': state[0], 'keys': state[1].tolist(), 'position': int(state[2]),
                      'has_gauss': int(state[3]), 'cached_gaussian': float(state[4])},
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def _restore_rng(state, generator):
    import random
    import numpy as np
    import torch
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    generator.set_state(state['generator'].cpu())
    n = state['numpy']
    np.random.set_state((n['name'], np.array(n['keys'], dtype=np.uint32), n['position'],
                         n['has_gauss'], n['cached_gaussian']))
    if state.get('cuda') and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])

def audit(args):
    records = scan_dataset(args.data, args.patient_regex)
    report = {'counts': counts(records), 'patient_overlap_checked': bool(args.patient_regex),
              'exact_duplicate_overlap_checked': True}
    print(json.dumps(report, indent=2))
    if len(records['val']) < 50:
        print('Validation set is small; model selection may be unstable.', file=sys.stderr)
    if args.output:
        Path(args.output).write_text(json.dumps({'report': report, 'manifest': records}, indent=2),
                                     encoding='utf-8')

def train(args):
    import platform
    import subprocess
    from datetime import datetime, timezone
    import torch
    import torchvision
    from torch import nn
    from torch.utils.data import DataLoader
    from .models import build_model, image_size
    from .pipeline import XRayDataset, make_transform, seed_everything, resolve_device
    from .metrics import summarize
    if args.epochs < 1 or args.batch_size < 2 or args.learning_rate <= 0 or args.patience < 1 or args.log_every < 1:
        raise ValueError('Use epochs >= 1, batch-size >= 2, learning-rate > 0, patience >= 1, and log-every >= 1.')
    output = Path(args.output)
    checkpoint, resume_path = None, None
    if args.resume:
        for name in ('last.pt', 'best.pt'):
            candidate = output / name
            if candidate.is_file():
                resume_path = candidate
                break
        if resume_path is None:
            raise ValueError('No completed-epoch checkpoint found. Keep this folder and choose a new output directory.')
        print(f'Loading checkpoint: {resume_path}', flush=True)
        checkpoint = torch.load(resume_path, map_location='cpu', weights_only=True)
        previous = checkpoint['config']
        for field in ('model', 'clahe', 'balance', 'seed', 'patient_regex', 'batch_size', 'learning_rate'):
            if previous.get(field) != getattr(args, field):
                raise ValueError(f'Resume setting differs for {field}. Use the original value: {previous.get(field)!r}')
        if args.epochs <= checkpoint['epoch']:
            print(f"Already completed epoch {checkpoint['epoch']}. Increase --epochs to continue.", flush=True)
            return
    elif output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} is not empty. Add --resume to continue, or choose a new output directory.')
    print('Checking dataset images, hashes, and filename groups...', flush=True)
    records = scan_dataset(args.data, args.patient_regex)
    if checkpoint:
        manifest_path = output / 'manifest.json'
        if not manifest_path.is_file():
            raise ValueError('Missing original manifest.json. Keep all run files together.')
        original = json.loads(manifest_path.read_text(encoding='utf-8'))
        if records != original:
            raise ValueError('Dataset or grouping differs from the saved manifest. Start a separate run.')
    print('Dataset checks passed.', flush=True)
    seed_everything(args.seed)
    if args.model == 'cnn':
        args.weights = 'none'
    device = resolve_device(args.device)
    size = image_size(args.model)
    config = vars(args).copy()
    config.pop('handler')
    config.update(image_size=size, class_names=['NORMAL', 'PNEUMONIA'], positive_class='PNEUMONIA')
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / 'config.json', config)
    if not checkpoint:
        _write_json(output / 'manifest.json', records)
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
                   'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                   'device': str(device)}
    _write_json(output / 'environment.json', environment)
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
    if freeze.returncode == 0:
        (output / 'requirements-frozen.txt').write_text(freeze.stdout, encoding='utf-8')
    root = Path(args.data).resolve()
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(XRayDataset(root, records[split], make_transform(size, args.clahe, split == 'train')),
                          batch_size=args.batch_size, shuffle=split == 'train', num_workers=0,
                          drop_last=split == 'train', generator=generator)
        for split in ('train', 'val')}
    if len(loaders['train']) == 0:
        raise ValueError('Training set is smaller than batch size. Reduce --batch-size.')
    print('Preparing model...', flush=True)
    model = build_model(args.model, args.weights == 'imagenet' and not checkpoint).to(device)
    train_counts = counts(records)['train']
    ratio = train_counts['NORMAL'] / train_counts['PNEUMONIA'] if args.balance else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history, best_auc, stale, first_epoch = [], -1.0, 0, 1
    if checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        first_epoch = checkpoint['epoch'] + 1
        best_auc = checkpoint.get('best_auc', checkpoint['validation_metrics']['roc_auc'])
        stale = checkpoint.get('stale', 0)
        if 'history' in checkpoint:
            history = checkpoint['history']
        elif (output / 'history.json').is_file():
            history = [row for row in json.loads((output / 'history.json').read_text(encoding='utf-8'))
                       if row['epoch'] <= checkpoint['epoch']]
        full_state = 'optimizer_state' in checkpoint and 'rng_state' in checkpoint
        if full_state:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            _restore_rng(checkpoint['rng_state'], generator)
            mode = 'completed_epoch_with_optimizer_and_rng'
            print(f'Resuming from completed epoch {checkpoint["epoch"]}, including optimizer state.', flush=True)
            if stale >= args.patience:
                print('Early stopping was already reached. Evaluate best.pt, or explicitly increase --patience.', flush=True)
                return
        else:
            mode = 'legacy_model_weights_with_fresh_optimizer'
            print(f'Continuing legacy model weights from epoch {checkpoint["epoch"]}. '
                  'Old checkpoint has no optimizer/RNG state; optimizer starts fresh.', flush=True)
        events_path = output / 'resume_events.json'
        events = json.loads(events_path.read_text(encoding='utf-8')) if events_path.is_file() else []
        events.append({'checkpoint': str(resume_path), 'from_epoch': checkpoint['epoch'],
                       'mode': mode, 'environment': environment, 'time_utc':
                       datetime.now(timezone.utc).isoformat()})
        _write_json(events_path, events)
        del checkpoint
    print(f'Device: {device}; image size: {size}; class counts: {counts(records)}', flush=True)
    if len(records['val']) < 50:
        print('WARNING: small validation set. Read the evaluation guidance in README.md.', flush=True)
    for epoch in range(first_epoch, args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs}: training {len(loaders["train"])} batches', flush=True)
        model.train()
        loss_total, processed = 0.0, 0
        started = time.monotonic()
        for step, (images, labels, _) in enumerate(loaders['train'], 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            result = model(images)
            if hasattr(result, 'logits'):
                loss = criterion(result.logits.flatten(), labels)
                if result.aux_logits is not None:
                    loss = loss + .4 * criterion(result.aux_logits.flatten(), labels)
            else:
                loss = criterion(result.flatten(), labels)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss. Check inputs and reduce the learning rate.')
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(labels)
            processed += len(labels)
            _progress(f'Epoch {epoch}/{args.epochs} train', step, len(loaders['train']),
                      started, args.log_every, loss_total / processed)
        _, labels, scores = _predict_progress(model, loaders['val'], device,
                                              f'Epoch {epoch}/{args.epochs} validation', args.log_every)
        metrics = summarize(labels, scores)
        row = {'epoch': epoch, 'train_loss': loss_total / processed,
               'val_accuracy': metrics['accuracy'], 'val_roc_auc': metrics['roc_auc']}
        history.append(row)
        if metrics['roc_auc'] > best_auc:
            best_auc, stale = metrics['roc_auc'], 0
            _save_checkpoint(output / 'best.pt', {'model_state': model.state_dict(), 'config': config,
                              'epoch': epoch, 'validation_metrics': metrics, 'environment': environment})
        else:
            stale += 1
        _save_checkpoint(output / 'last.pt', {
            'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
            'config': config, 'epoch': epoch, 'validation_metrics': metrics,
            'environment': environment, 'history': history, 'best_auc': best_auc, 'stale': stale,
            'rng_state': _rng_state(generator),
        })
        _write_json(output / 'history.json', history)
        print(json.dumps(row), flush=True)
        print(f'Completed-epoch checkpoint saved: {output / "last.pt"}', flush=True)
        if stale >= args.patience:
            print('Early stopping: no validation AUC improvement.', flush=True)
            break
    print(f"Saved {output / 'best.pt'}. Evaluate once on held-out test images.", flush=True)

def evaluate(args):
    from torch.utils.data import DataLoader
    from .pipeline import (load_checkpoint, resolve_device, XRayDataset, make_transform)
    from .metrics import save_evaluation
    if not 0 < args.threshold < 1:
        raise ValueError('Threshold must be between 0 and 1.')
    if args.batch_size < 1 or args.log_every < 1:
        raise ValueError('Batch-size and log-every must be positive.')
    records = scan_dataset(args.data, args.patient_regex)
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint['config']
    # Reject test images that were part of the recorded training/validation splits.
    manifest_path = Path(args.checkpoint).parent / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError('Missing manifest.json beside checkpoint. Keep run files together.')
    original = json.loads(manifest_path.read_text(encoding='utf-8'))
    used = {row['sha256'] for split in ('train', 'val') for row in original[split]}
    if any(row['sha256'] in used for row in records['test']):
        raise ValueError('Test data overlaps the training or validation data recorded for this model.')
    if config.get('patient_regex'):
        checked = scan_dataset(args.data, config['patient_regex'])
        used_patients = {row['patient'] for split in ('train', 'val') for row in original[split]}
        if any(row['patient'] in used_patients for row in checked['test']):
            raise ValueError('Test patients overlap patients used to train or select this model.')
    loader = DataLoader(XRayDataset(Path(args.data).resolve(), records['test'],
                        make_transform(config['image_size'], config['clahe'])),
                        batch_size=args.batch_size, num_workers=0)
    paths, labels, scores = _predict_progress(model, loader, device, 'Test evaluation', args.log_every)
    directory = Path(args.output)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Evaluation output is not empty; choose a new output directory.')
    metrics = save_evaluation(directory, paths, labels, scores, args.threshold)
    (directory / 'test_manifest.json').write_text(json.dumps(records['test'], indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))

def main():
    parser = argparse.ArgumentParser(description='Chest X-ray classification research pipeline')
    sub = parser.add_subparsers(dest='command', required=True)
    for name, handler in [('audit', audit), ('train', train), ('evaluate', evaluate)]:
        p = sub.add_parser(name)
        p.add_argument('--data', required=True, help='Directory containing train, val, and test')
        p.add_argument('--patient-regex', help='Optional filename regex with patient ID in capture group 1')
        p.set_defaults(handler=handler)
        if name == 'audit':
            p.add_argument('--output', help='Optional audit JSON file')
        else:
            p.add_argument('--output', required=True)
            p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
            p.add_argument('--batch-size', type=int, default=16)
            p.add_argument('--log-every', type=int, default=25, help='Print progress every N batches')
        if name == 'train':
            p.add_argument('--model', choices=['cnn', 'resnet18', 'vgg16', 'inception_v3'], default='resnet18')
            p.add_argument('--weights', choices=['imagenet', 'none'], default='imagenet')
            p.add_argument('--epochs', type=int, default=10)
            p.add_argument('--learning-rate', type=float, default=1e-4)
            p.add_argument('--patience', type=int, default=3)
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--clahe', action='store_true')
            p.add_argument('--balance', action=argparse.BooleanOptionalAction, default=True)
            p.add_argument('--resume', action='store_true', help='Continue this output folder from last.pt or legacy best.pt')
        if name == 'evaluate':
            p.add_argument('--checkpoint', required=True)
            p.add_argument('--threshold', type=float, default=.5)
    args = parser.parse_args()
    try:
        args.handler(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f'Error: {error}\n')
    except KeyboardInterrupt:
        if args.command == 'train':
            parser.exit(130, '\nStopped by Ctrl+C. Any completed checkpoints remain intact. '
                        'Use the same command with --resume to continue from a saved epoch.\n')
        parser.exit(130, '\nStopped by Ctrl+C.\n')

if __name__ == '__main__':
    main()
