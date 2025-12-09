import torch
from torchvision.utils import save_image
import os

# --- Load model safely ---
def load_model(model, path, device):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        print(f"✅ Model loaded from {path}")
    else:
        print(f"⚠️ Model path not found: {path}")
    return model


# --- Save image tensor to file ---
def save_image_tensor(tensor, filename):
    tensor = (tensor + 1) / 2  # De-normalize (-1,1) → (0,1)
    save_image(tensor, filename)
    print(f"💾 Saved output image → {filename}")


# --- Normalize age values (works for scalar or batch) ---
def normalize_age(age, max_age=100):
    """
    Normalize scalar, list, or tensor of ages to [0, 1].
    Handles batch tensors properly.
    """
    if isinstance(age, torch.Tensor):
        return age.float().view(-1, 1) / max_age
    elif isinstance(age, (list, tuple)):
        return torch.tensor(age, dtype=torch.float32).view(-1, 1) / max_age
    else:
        return torch.tensor([[age / max_age]], dtype=torch.float32)


# --- De-normalize age ---
def denormalize_age(norm_age, max_age=100):
    return norm_age * max_age


# --- Random age generator for training ---
def sample_random_age(batch_size=16, min_age=20, max_age=80):
    random_ages = torch.FloatTensor(batch_size).uniform_(min_age, max_age)
    normalized_ages = (random_ages - min_age) / (max_age - min_age)
    return normalized_ages.view(-1, 1)
