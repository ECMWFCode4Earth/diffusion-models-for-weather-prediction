from dm_zoo.dff.PixelDiffusion import PixelDiffusionConditional
from pathlib import Path
import numpy as np

from WD.datasets_old import Conditional_Dataset

model_load_dir = Path(
    "/data/compoundx/ml_models/diffusion_models/lightning_logs/version_13/checkpoints/"
)

ds = Conditional_Dataset(
    "/data/compoundx/WeatherBench/pytorch_datasets/first_attempt_conditional_test.pt"
)
model_ckpt = [x for x in model_load_dir.iterdir()][0]
restored_model = PixelDiffusionConditional.load_from_checkpoint(
    model_ckpt, generated_channels=1, condition_channels=1
).to(
    "cuda"
)  # to cuda makes it so that the GPU gets used for constructing the images!#


B = 16
# sample first batch
inputs, targets, _ = ds[:B]
out = restored_model(ds[:B], verbose=True)

model_write_dir = Path(str(model_load_dir).replace("ml_models", "ml_models_output"))
model_write_dir.mkdir(parents=True, exist_ok=True)
model_write_dir = model_write_dir / "sim.npy"

print(np.squeeze(inputs.numpy()).shape)
print(np.squeeze(out.cpu().numpy()).shape)

np.save(
    model_write_dir,
    (
        np.squeeze(inputs.numpy()),
        np.squeeze(targets.numpy()),
        np.squeeze(out.cpu().numpy()),
    ),
)  # save conditions, true targets and predictions
