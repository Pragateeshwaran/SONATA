import os
import numpy as np
import torch
import torch.nn.functional as F
from subprocess import run, CalledProcessError
 
SAMPLE_RATE = 16000  # Samples per second
N_FFT = 400          # Number of FFT points
HOP_LENGTH = 160     # Number of samples between frames
CHUNK_LENGTH = 30    # Length of audio chunk in seconds
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # Total number of samples in a chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH      # Number of frames in the spectrogram
device: torch.device = 'cpu' if torch.cuda.is_available() else 'cpu'
# Function to load audio from a file
def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Load an audio file and resample it to the specified sample rate.

    Parameters:
    - file: Path to the audio file.
    - sr: Sample rate to resample the audio.

    Returns:
    - A NumPy array containing the audio waveform.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",      # Mono audio
        "-ar", str(sr),  # Sample rate
        "-"
    ]
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0

# Function to pad or trim audio data
def pad_or_trim(array, length: int = N_SAMPLES):
    """
    Pad or trim the audio array to the specified length.

    Parameters:
    - array: The audio data as a NumPy array or PyTorch tensor.
    - length: The desired length of the audio array.

    Returns:
    - The audio array padded or trimmed to the specified length.
    """
    if isinstance(array, torch.Tensor):
        if array.size(0) > length:
            array = array[:length]
        elif array.size(0) < length:
            array = F.pad(array, (0, length - array.size(0)))
    else:
        if array.size > length:
            array = array[:length]
        elif array.size < length:
            array = np.pad(array, (0, length - array.size))
    
    return array
stft= None
window= None
# Function to load Mel filterbanks
def mel_filters(n_mels: int) -> torch.Tensor:
    """
    Load Mel filterbank matrix for converting FFT to Mel spectrogram.

    Parameters:
    - n_mels: Number of Mel frequency bins (80 or 128).

    Returns:
    - A PyTorch tensor containing the Mel filterbank matrix.
    """
    filters_path = os.path.join(r'src', "assets", "mel_filters.npz")
    with np.load(filters_path) as f:
        mel_filter = f[f"mel_{n_mels}"]
    return torch.from_numpy(mel_filter)

# Function to compute the log-Mel spectrogram
def log_mel_spectrogram(audio, n_mels: int = 80, padding: int = 0) -> torch.Tensor:
    """
    Compute the log-Mel spectrogram of the audio signal.

    Parameters:
    - audio: The audio data as a file path, NumPy array, or PyTorch tensor.
    - n_mels: Number of Mel-frequency bins (80 or 128).
    - padding: Number of zeros to pad to the end of the audio.
    - device: The device to perform computations on (CPU or GPU).

    Returns:
    - A PyTorch tensor containing the log-Mel spectrogram.
    """
    if isinstance(audio, str):
        audio = load_audio(audio)
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
        
    audio = audio.to(device)

    if padding > 0:
        audio = F.pad(audio, (0, padding))

    # Compute Short-Time Fourier Transform (STFT)
    window = torch.hann_window(N_FFT, device=device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft.abs() ** 2
    # print(window.shape)
    # plt.figure(figsize=(20, 5))
    print(stft.shape)
    # plt.imshow(window.reshape(20, 20))
    # Compute Mel spectrogram
    filters = mel_filters(n_mels)
    # plt.imshow(filters)
    mel_spec = torch.matmul(filters, magnitudes)
    # plt.show()

    # Compute log-Mel spectrogram
    log_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
    log_spec = (log_spec - log_spec.max() + 8.0) / 4.0

    return log_spec

