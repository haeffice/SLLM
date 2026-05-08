import torch


def infer(waveform: torch.Tensor, sample_rate: int) -> dict:
    if waveform.dim() == 2:
        mono = waveform.mean(dim=0)
    else:
        mono = waveform

    num_samples = int(mono.shape[-1])
    duration = num_samples / sample_rate
    rms = float(mono.pow(2).mean().sqrt())

    return {
        "sample_rate": sample_rate,
        "num_samples": num_samples,
        "duration_seconds": round(duration, 3),
        "rms": round(rms, 5),
        "transcript": "[mock] audio received",
    }
