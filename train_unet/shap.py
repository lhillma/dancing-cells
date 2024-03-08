import torch
import numpy as np
import shap

from pathlib import Path
from torch.utils.data import DataLoader
from monai.data.utils import no_collation
from train_unet.models import ResUNet
from monai_pipeline.monai_dataset import get_dataset



def get_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
    """Load the model with specified weights."""
    model = ResUNet(n_classes=1, in_channels=2).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()  
    return model

def model_wrapper(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """Wrap the model function for compatibility with SHAP."""
    with torch.no_grad():
        return model(x).detach().cpu().numpy()

def get_shap_explainer(model: torch.nn.Module, background: torch.Tensor, device: torch.device) -> shap.DeepExplainer:
    """Initialize and return the SHAP DeepExplainer."""

    model.eval()
    model.to(device)
    wrapped_model = lambda x: model_wrapper(model, x.to(device))
    explainer = shap.DeepExplainer(wrapped_model, background.to(device))
    return explainer


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = Path("path/to/your/model/weights.pth")

    model = get_model(weights_path, device)

    dataset_path = Path(r"G:\Hackathon\hires_lofrac_short")
    
    model = get_model(weights_path, device)
    min_step = 3000
    n_workers = 4    

    keys = ["image", "label", "nucleus", "cluster_id"]

    
    background_data = get_dataset(
        dataset_path,
        keys,
        "valid",
        min_step,
        distance_transform=False,
        cache=False,
    )

    # Initialize SHAP explainer
    # : probably test first with only one of the images  and then all of them
    explainer = get_shap_explainer(model, background_data, device)
    
    
    # Compute SHAP values
    shap_values = explainer.shap_values(background_data)

    #shap.initjs()
    
    shap.image_plot(shap_values, -background_data.cpu().numpy())

    

if __name__ == "__main__":
    main()
    