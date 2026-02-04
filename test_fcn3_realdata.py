import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import h5py
import numpy as np
from makani.models.model_package import LocalPackage, load_model_package
from datetime import datetime, timezone

# Load model
package = LocalPackage('/workspace/makani/data/models/fourcastnet3_v0.1.0')
print('Loading model...')
model = load_model_package(package, pretrained=True, device='cuda')
print(f'Model loaded. Expects {model.params.N_in_channels} input channels')

# Get model's expected channel names
model_channels = model.params.channel_names
print(f'Model channels ({len(model_channels)}): {model_channels[:5]}...')

# Load real data
print('\nLoading ERA5 data...')
with h5py.File('/workspace/makani/data/era5_test/2020.h5', 'r') as f:
    data_channels = [c.decode() if isinstance(c, bytes) else c for c in f['channel'][:]]
    fields = f['fields'][0]  # First timestep
    print(f'Data shape: {fields.shape}')
    print(f'Timestamp: {f["timestamp"][0]}')

# Build channel index mapping
channel_indices = [data_channels.index(mc) for mc in model_channels]
print(f'Selecting {len(channel_indices)} channels')

# Select and prepare input
x = fields[channel_indices]
x = torch.from_numpy(x).float().unsqueeze(0).cuda()
print(f'Input tensor: {x.shape}, {x.dtype}')

# Normalize (like the notebook does)
x = (x - model.in_bias.cuda()) / model.in_scale.cuda()

# Timestamp for zenith angle
time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
print(f'Running inference for {time}...')

# Forward pass with bfloat16 (like notebook)
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = model(x, time, normalized_data=True)

# Denormalize output
y = y * model.out_scale.cuda() + model.out_bias.cuda()

peak_mem = torch.cuda.max_memory_allocated() / 1024**3
print(f'Output shape: {y.shape}')
print(f'Peak GPU memory: {peak_mem:.2f} GB')
print('\nFCN3 inference on real ERA5 data successful!')
