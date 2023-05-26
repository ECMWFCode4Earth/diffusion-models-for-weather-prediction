from dm_zoo.dff.PixelDiffusion import PixelDiffusion
from pathlib import Path
import numpy as np

model_load_dir = Path(
    "/data/compoundx/ml_models/diffusion_models/lightning_logs/version_1/checkpoints/"
)
model_ckpt = [x for x in model_load_dir.iterdir()][0]
restored_model = PixelDiffusion.load_from_checkpoint(model_ckpt).to(
    "cuda"
)  # to gpu makes it so that the GPU gets used for constructing the images!
B = 16
out = restored_model(batch_size=B, shape=(32, 64), verbose=True)

model_write_dir = Path(str(model_load_dir).replace("ml_models", "ml_models_output"))
model_write_dir.mkdir(parents=True, exist_ok=True)
model_write_dir = model_write_dir / "sim_jon.npy"

np.save(model_write_dir, np.squeeze(out.cpu().numpy()))
