import math
from tinygrad import Tensor
from examples.audio_helpers import mel, rfft_matrices, hann_window

RATE = 16000
SEGMENT_SECONDS = 30
SAMPLES_PER_SEGMENT = RATE * SEGMENT_SECONDS  # 480000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
FRAMES_PER_SEGMENT = SAMPLES_PER_SEGMENT // HOP_LENGTH  # 3000

class MelSpec:
  # waveform [1, 480000] -> log-mel [1, 80, 3000], Whisper-normalized.
  # Matches librosa.stft+mel within ~1e-3. rDFT as a matmul with Hann folded into the weights.
  def __init__(self):
    cos_mat, sin_mat = rfft_matrices(N_FFT)
    hann = hann_window(N_FFT).reshape(-1, 1)
    self.cos_mat = (cos_mat * hann).realize()
    self.sin_mat = (sin_mat * hann).realize()
    self.mel_filter = mel(sr=RATE, n_fft=N_FFT, n_mels=N_MELS).realize()

  def __call__(self, wav:Tensor) -> Tensor:
    x = wav.pad(((0, 0), (N_FFT//2, N_FFT//2)), mode="reflect")  # librosa center=True
    frames = x.unfold(1, N_FFT, HOP_LENGTH).shrink(((0, wav.shape[0]), (0, FRAMES_PER_SEGMENT), (0, N_FFT)))
    mag2 = (frames @ self.cos_mat).square() + (frames @ self.sin_mat).square()
    spec = self.mel_filter @ mag2.permute(0, 2, 1)
    log = spec.maximum(1e-10).log() * (1.0 / math.log(10.0))
    log = log.maximum(log.max(axis=(1, 2), keepdim=True) - 8.0)
    return (log + 4.0) / 4.0
