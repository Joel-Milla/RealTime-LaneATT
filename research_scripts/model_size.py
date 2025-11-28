import torch

checkpoint = torch.load('/home/joel/Documents/research/RealTime-LaneATT/checkpoints/laneatt_100.pt', map_location='cpu')
model_dict = checkpoint.get('model', checkpoint)
total_params = sum(p.numel() for p in model_dict.values() if isinstance(p, torch.Tensor))
print(f"Total parameters: {total_params:,}")