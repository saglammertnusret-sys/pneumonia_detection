import random
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

class Preprocess:
    def __init__(self, size, clahe=False):
        self.size, self.clahe = size, clahe

    def __call__(self, image):
        image = ImageOps.exif_transpose(image).convert('L')
        if self.clahe:
            import cv2
            array = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(np.array(image))
            image = Image.fromarray(array)
        # Preserve aspect ratio and keep the full image rather than cropping lungs.
        return ImageOps.pad(image.convert('RGB'), (self.size, self.size), color=(0, 0, 0))

def make_transform(size, clahe=False, training=False):
    operations = [Preprocess(size, clahe)]
    if training:
        operations.append(transforms.RandomAffine(degrees=7, translate=(.03, .03), scale=(.95, 1.05)))
    operations.extend([transforms.ToTensor(),
                       transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    return transforms.Compose(operations)

class XRayDataset(Dataset):
    def __init__(self, root, rows, transform):
        self.root, self.rows, self.transform = root, rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['path']) as image:
            tensor = self.transform(image)
        return tensor, torch.tensor(row['label'], dtype=torch.float32), row['path']

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def resolve_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable. Install a compatible PyTorch CUDA build, or use --device cpu.')
    return torch.device(name)

def load_checkpoint(path, device):
    from .models import build_model
    # Checkpoints contain tensors and plain metadata, not serialized Python models.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model(checkpoint['config']['model'])
    model.load_state_dict(checkpoint['model_state'])
    model.to(device).eval()
    return model, checkpoint

@torch.inference_mode()
def predict_loader(model, loader, device):
    paths, labels, scores = [], [], []
    model.eval()
    for images, targets, names in loader:
        probabilities = model(images.to(device)).flatten().sigmoid().cpu().tolist()
        paths.extend(names)
        labels.extend(targets.tolist())
        scores.extend(probabilities)
    return paths, labels, scores
