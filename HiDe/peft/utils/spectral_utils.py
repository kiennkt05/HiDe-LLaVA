import torch
import torch.fft

def apply_spectral_norm(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    # doc theo chieu feature (dim = -1)
    fft_x = torch.fft.fft(x.float(), dim=-1)
    amp = torch.abs(fft_x)
    phase = torch.angle(fft_x)
    
    # Min-max normalization
    amp_min = amp.min(dim=-1, keepdim=True)[0]
    amp_max = amp.max(dim=-1, keepdim=True)[0]
    amp_norm = (amp - amp_min) / (amp_max - amp_min + eps)
    
    fft_updated = torch.polar(amp_norm, phase)
    x_clean = torch.fft.ifft(fft_updated, dim=-1).real
    
    return x_clean.to(x.dtype)