from pathlib import Path

class lib_librosa:
    from librosa.filters import mel
    from librosa import stft
    from librosa import load

class lib_numpy:
    from numpy import ndarray, float32, abs, log10, maximum, matmul

class lib_matplotlib:
    from matplotlib import figure, axes, image, colors, pyplot


class Cl_mel:
    """
    Class for computing mel-spectrograms from audio signals.

    Matrix conventions used throughout:
    - Waveform: (n_samples,)
    - STFT complex matrix: (n_fft/2 + 1, n_frames)
    - Power spectrogram: (n_fft/2 + 1, n_frames)
    - Mel filter bank: (n_mel, n_fft/2 + 1)
    - Mel spectrogram: (n_mel, n_frames)
    """

    def __init__(
        self,
        i_n_sample_rate: int = 16000,
        i_n_fft: int = 400,
        i_n_hop_length: int = 160,
        i_n_mel: int = 128,
        i_n_noise_floor : float = 8.0
    ) -> None:
        """
        Initialize mel-spectrogram processing parameters.

        Parameters:
            i_n_sample_rate (int):
                Sampling rate [Hz].

            i_n_fft (int):
                FFT size.
                Frequency bins = (i_n_fft // 2) + 1.

            i_n_hop_length (int):
                Hop size between frames [samples].
                Number of frames ≈ n_samples / hop_length.

            i_n_mel (int, optional):
                Number of mel bands.

        Internal matrices:
            g_aan_mel_weights:
                Shape: (i_n_mel, i_n_fft//2 + 1)
        """
        self.g_n_sample_rate = i_n_sample_rate
        self.g_n_fft = i_n_fft
        self.g_n_hop_length = i_n_hop_length
        self.g_n_mel = i_n_mel
        self.g_n_noise_floor = i_n_noise_floor

        self.g_aan_mel_weights = self._load_mel_weights()

        self.g_n_duration_s = 0

        return

    def load_audio_file_as_wav(
        self,
        i_s_path_audio: str | Path
    ) -> lib_numpy.ndarray:
        """
        Load an audio file and convert it to a mono waveform.

        Parameters:
            i_s_path_audio (str | Path):
                Path to audio file.

        Returns:
            numpy.ndarray:
                Shape: (n_samples,)
                dtype: float32
        """
        an_wav, _ = lib_librosa.load(i_s_path_audio, sr=self.g_n_sample_rate, mono=True)

        self.g_n_duration_s = len(an_wav) / self.g_n_sample_rate

        return an_wav.astype(lib_numpy.float32)

    def compute_mel_spectrogram_from_audio_file(
        self,
        i_s_path_audio: str | Path
    ):
        """
        Compute mel spectrogram directly from an audio file.

        Processing pipeline:
            file → waveform → STFT → power → mel → log → normalized

        Returns:
            numpy.ndarray:
                Mel spectrogram
                Shape: (n_mel, n_frames)
        """
        an_wav = self.load_audio_file_as_wav(i_s_path_audio)
        aan_mel = self.compute_mel_from_wav(an_wav)

        return aan_mel

    def compute_mel_from_wav(
        self,
        i_an_wav: lib_numpy.ndarray
    ):
        """
        Compute mel spectrogram from waveform.

        Parameters:
            i_an_wav (numpy.ndarray):
                Shape: (n_samples,)

        Processing steps and dimensions:

        1. STFT:
            aan_transform:
                Shape: (n_fft//2 + 1, n_frames)

        2. Power spectrum:
            aan_mag = |STFT|^2
                Shape: (n_fft//2 + 1, n_frames)

        3. Mel projection:
            g_aan_mel_weights:
                Shape: (n_mel, n_fft//2 + 1)

            aan_mel_audio = W @ power
                Shape: (n_mel, n_frames)

        4. Log scaling:
            Same shape: (n_mel, n_frames)

        5. Output:
            Normalized mel spectrogram
                Shape: (n_mel, n_frames)
                dtype: float32
        """
        aan_transform = lib_librosa.stft(
            i_an_wav,
            n_fft=self.g_n_fft,
            hop_length=self.g_n_hop_length
        )

        aan_mag = lib_numpy.abs(aan_transform) ** 2

        aan_mel_audio = lib_numpy.matmul(self.g_aan_mel_weights, aan_mag)

        aan_mel_audio_log = lib_numpy.log10(lib_numpy.maximum(aan_mel_audio, 1e-10))

        aan_mel_audio_log = lib_numpy.maximum(
            aan_mel_audio_log,
            aan_mel_audio_log.max() - self.g_n_noise_floor
        )

        aan_mel_audio_log = ((aan_mel_audio_log + 4.0) / 4.0).astype(lib_numpy.float32)

        return aan_mel_audio_log

    @staticmethod
    def save_mel_as_image(i_aan_mel: lib_numpy.ndarray, i_s_path: Path) -> None:
        """
        Save mel spectrogram as image.

        Parameters:
            i_aan_mel (numpy.ndarray):
                Mel spectrogram
                Shape: (n_mel, n_frames)

            i_s_path (Path):
                Output image path

        Rendering details:
            - Transposed before plotting:
                (n_frames, n_mel)
            - X-axis: time (frames)
            - Y-axis: mel bins
        """
        cmap_custom = lib_matplotlib.colors.LinearSegmentedColormap.from_list(
            "RedZeroBlue",
            [
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0)
            ],
            N=256,
        )

        fig: lib_matplotlib.figure.Figure = lib_matplotlib.pyplot.figure(figsize=(10, 4))
        ax: lib_matplotlib.axes.Axes = fig.add_subplot(111)

        im: lib_matplotlib.image.AxesImage = ax.imshow(
            i_aan_mel.T,
            aspect="auto",
            origin="lower",
            cmap=cmap_custom,
        )

        ax.set_xlabel("Time (frames)")
        ax.set_ylabel("Mel bins")

        fig.colorbar(im, ax=ax)
        lib_matplotlib.pyplot.tight_layout()
        lib_matplotlib.pyplot.savefig(i_s_path)

    def _load_mel_weights(
        self
    ) -> lib_numpy.ndarray:
        """
        Generate mel filter bank.

        Returns:
            numpy.ndarray:
                Mel filter bank matrix
                Shape: (n_mel, n_fft//2 + 1)
        """
        fn_mel = lib_librosa.mel(
            sr=self.g_n_sample_rate,
            n_fft=self.g_n_fft,
            n_mels=self.g_n_mel
        ).astype(lib_numpy.float32)

        return fn_mel