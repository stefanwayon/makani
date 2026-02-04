import torch
from makani.models.model_package import LocalPackage, load_model_package
import datetime
from datetime import timezone

# Load model from package
package = LocalPackage('/opt/makani/data/models/fourcastnet3_v0.1.0')
print('Loading model from package...')
model = load_model_package(package, pretrained=True, device='cuda')
print('Model loaded!')

# Get config info
params = model.params
print(f'Input shape: {params.img_shape_x}x{params.img_shape_y}')
print(f'N_in_channels: {params.N_in_channels}')
print(f'N_out_channels: {params.N_out_channels}')
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

# Create dummy input
# Note: batch size > 1 requires sequential processing due to zenith angle computation
batch_size = 1
x = torch.randn(batch_size, params.N_in_channels, params.img_shape_x, params.img_shape_y, device='cuda')
print(f'Input tensor shape: {x.shape}')

# Forward pass with timezone-aware datetime
time = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    y = model(x, time)

peak_mem = torch.cuda.max_memory_allocated() / 1024**3
print(f'Output shape: {y.shape}')
print(f'Peak memory: {peak_mem:.2f} GB')
print('FCN3 forward pass successful!')
