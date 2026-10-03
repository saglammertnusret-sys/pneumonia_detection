"""Exercise training/evaluation on random images, without downloading weights."""
import json
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from pneumonia.models import build_model
from pneumonia.pipeline import load_checkpoint, make_transform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clahe', action='store_true', help='Also exercise OpenCV CLAHE')
    args = parser.parse_args()
    torch.set_num_threads(2)
    project = Path(__file__).resolve().parent
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory(prefix='pneumonia-smoke-') as directory:
        root = Path(directory)
        for split in ('train', 'val', 'test'):
            for label in ('NORMAL', 'PNEUMONIA'):
                target = root / 'data' / split / label
                target.mkdir(parents=True)
                for i in range(4):
                    Image.fromarray(rng.integers(0, 256, (48, 64), dtype=np.uint8)).save(target / f'{i}.png')
        common = [sys.executable, '-m', 'pneumonia.cli']
        environment = os.environ.copy()
        environment.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        def run(*args):
            subprocess.run(common + list(args), cwd=project, env=environment, check=True)
        run('audit', '--data', str(root / 'data'))
        run('train', '--data', str(root / 'data'), '--output', str(root / 'run'),
            '--model', 'cnn', '--weights', 'none', '--epochs', '1', '--batch-size', '4',
            '--device', 'cpu', *(['--clahe'] if args.clahe else []))
        run('evaluate', '--data', str(root / 'data'), '--checkpoint', str(root / 'run/best.pt'),
            '--output', str(root / 'evaluation'), '--device', 'cpu')
        metrics = json.loads((root / 'evaluation/metrics.json').read_text())
        assert metrics['samples'] == 8
        assert (root / 'evaluation/evaluation.png').is_file()
        model, checkpoint = load_checkpoint(root / 'run/best.pt', torch.device('cpu'))
        image = Image.open(root / 'data/test/NORMAL/0.png')
        tensor = make_transform(checkpoint['config']['image_size'], checkpoint['config']['clahe'])(image)
        with torch.inference_mode():
            assert model(tensor.unsqueeze(0)).shape == (1, 1)
        # Exercise transfer model output heads without pretrained weight downloads.
        for name, size in [('resnet18', 224), ('vgg16', 224), ('inception_v3', 299)]:
            network = build_model(name).eval()
            with torch.inference_mode():
                assert network(torch.zeros(1, 3, size, size)).shape == (1, 1)
            del network
    print('\nSMOKE TEST PASSED. Random-image metrics have no scientific meaning.')

if __name__ == '__main__':
    main()
