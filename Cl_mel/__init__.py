

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
    def __init__(
        self,
        i_n_sample_rate : int,
        i_n_fft : int,
        i_n_hop_length : int,
        i_n_mel : int  = 128
    ) -> None:
        
        self.g_n_sample_rate = i_n_sample_rate
        self.g_n_fft = i_n_fft
        self.g_n_hop_length = i_n_hop_length
        self.g_n_mel = i_n_mel
        self.g_aan_mel_weights = self._load_mel_weights()

        self.g_n_duration_s = 0

        return
    
    
    def load_audio_file_as_wav(
        self,
        i_s_path_audio : str | Path
    ) -> lib_numpy.ndarray:
        an_wav, _ = lib_librosa.load(i_s_path_audio, sr=self.g_n_sample_rate, mono=True)
        self.g_n_duration_s = len(an_wav) / self.g_n_sample_rate
        return an_wav.astype(lib_numpy.float32)


    def compute_mel_spectrogram_from_audio_file(
        self,
        i_s_path_audio : str | Path
    ):
        an_wav = self.load_audio_file_as_wav( i_s_path_audio )
        aan_mel = self.compute_mel_from_wav( an_wav )

        return aan_mel

    
    
    def compute_mel_from_wav(
        self,
        i_an_wav : lib_numpy.ndarray
    ):
        aan_transform = lib_librosa.stft(
            i_an_wav,
            n_fft=self.g_n_fft,
            hop_length=self.g_n_hop_length
        )
        aan_mag = lib_numpy.abs(aan_transform) ** 2
        aan_mel_audio = lib_numpy.matmul( self.g_aan_mel_weights, aan_mag )
        aan_mel_audio_log = lib_numpy.log10(lib_numpy.maximum(aan_mel_audio, 1e-10))
        aan_mel_audio_log = lib_numpy.maximum(aan_mel_audio_log, aan_mel_audio_log.max() - 8.0)
        return ((aan_mel_audio_log + 4.0) / 4.0).astype(lib_numpy.float32)


    @staticmethod
    def save_mel_as_image(i_aan_mel: lib_numpy.ndarray, i_s_path: Path) -> None:
        """
        Convert a mel‑spectrogram to a 2‑D RGB image and write it to disk.
        
        The mapping is defined such that the lowest non‑zero value is shown in red,
        zero intensity becomes black, and the highest value becomes blue. Values
        are linearly interpolated between these extremes.

        Parameters:
            i_mel (numpy.ndarray): 2‑D array of mel values (float32).
            i_path (Path): Destination file path (.png or .jpg).

        Returns:
            None: The image is written directly to the given path.
        """
        # Ensure the destination directory exists
        #i_s_path.parent.mkdir(parents=True, exist_ok=True)

        # Clip / normalize so that all values are in [0, 1]
        #mel_normalized = (i_mel - i_mel.min()) / (i_mel.max() - i_mel.min())

        # Build a custom colormap: min → red, 0 → black, max → blue
        

        cmap_custom = lib_matplotlib.colors.LinearSegmentedColormap.from_list(
            "RedZeroBlue",
            [
                (1.0, 0.0, 0.0),   # Red at the very low end (min)
                (0.0, 0.0, 0.0),   # Black at zero
                (0.0, 0.0, 1.0)    # Blue at the high end (max)
            ],
            N=256,
        )

        fig: lib_matplotlib.figure.Figure = lib_matplotlib.pyplot.figure(figsize=(10, 4))
        ax: lib_matplotlib.axes.Axes = fig.add_subplot(111)
        im: lib_matplotlib.image.AxesImage = ax.imshow(
            i_aan_mel.T,          # Transpose so time is horizontal
            aspect="auto",
            origin="lower",
            cmap=cmap_custom,
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel frequency")
        fig.colorbar(im, ax=ax)
        lib_matplotlib.pyplot.tight_layout()
        lib_matplotlib.pyplot.savefig(i_s_path)


    def _load_mel_weights(
        self
    ) -> lib_numpy.ndarray:

        fn_mel = lib_librosa.mel(
            sr=self.g_n_sample_rate,
            n_fft=self.g_n_fft,
            n_mels=self.g_n_mel
        ).astype(lib_numpy.float32)

        return fn_mel



