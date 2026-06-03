import torch


@torch.compile
def fwht_512(x: torch.Tensor) -> torch.Tensor:
    """Computes the 512-row Fast Walsh-Hadamard Transform along axis 0.
    
    This implementation uses 9 unrolled, functional tensor reshaping-and-stacking stages
    suitable for torch.compile graph capture on NVIDIA GPUs.
    
    Args:
        x: A torch tensor of shape (512, d) or (512,).
        
    Returns:
        The transformed torch tensor, normalized by 1 / sqrt(512).
    """
    n = 512
    is_1d = x.ndim == 1
    if is_1d:
        x = x[:, None]
        
    d = x.shape[1]
    
    # 9 unrolled stages for radix-2 512-row FWHT
    # Stage 0: h = 1
    x = x.reshape(256, 2, 1, d)
    x = torch.stack([x[:, 0, :, :] + x[:, 1, :, :], x[:, 0, :, :] - x[:, 1, :, :]], dim=1).reshape(n, d)
    
    # Stage 1: h = 2
    x = x.reshape(128, 2, 2, d)
    x = torch.stack([x[:, 0, :, :] + x[:, 1, :, :], x[:, 0, :, :] - x[:, 1, :, :]], dim=1).reshape(n, d)
    
    # Stage 2: h = 4
    x = x.reshape(64, 2, 4, d)
    x = torch.stack([x[:, 0, :, :] + x[:, 1, :, :], x[:, 0, :, :] - x[:, 1, :, :]], dim=1).reshape(n, d)
    
    # Stage 3: h = 8
    x = x.reshape(32, 2, 8, d)
    x = torch.stack([x[:, 0, :, :] + x[:, 1, :, :], x[:, 0, :, :] - x[:, 1, :, :]], dim=1).reshape(n, d)
    
    # Stage 4: h = 16
    x = x.reshape(16, 2, 16, d)
    x = torch.stack([x[:, 0, :, :] + x[:, 1, :, :], x[:, 0, :, :] - x[:, 1, :, :]], dim=1).reshape(n, d)
    
    # Stage 5: h = 32
    x = x.reshape(8, 2, 32, d)
    x = torch.stack([x[:, 0, :, :] + x[:, 1, :, :], x[:, 0, :, :] - x[:, 1, :, :]], dim=1).reshape(n, d)
    
    # Stage 6: h = 64
    x = x.reshape(4, 2, 64, d)
    x = torch.stack([x[:, 0, :, :] + x[:, 1, :, :], x[:, 0, :, :] - x[:, 1, :, :]], dim=1).reshape(n, d)
    
    # Stage 7: h = 128
    x = x.reshape(2, 2, 128, d)
    x = torch.stack([x[:, 0, :, :] + x[:, 1, :, :], x[:, 0, :, :] - x[:, 1, :, :]], dim=1).reshape(n, d)
    
    # Stage 8: h = 256
    x = x.reshape(1, 2, 256, d)
    x = torch.stack([x[:, 0, :, :] + x[:, 1, :, :], x[:, 0, :, :] - x[:, 1, :, :]], dim=1).reshape(n, d)
    
    # Normalize by 1 / sqrt(512)
    x = x / 22.627416997969522  # sqrt(512) = 22.627416997969522
    
    if is_1d:
        x = x.squeeze(1)
    return x
