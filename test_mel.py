#python test_mel.py "D:\Data\Project\Project-LLM\Audio Samples\wav\SAMPLE-british-man-7s.wav"

from time import time

from pathlib import Path

from argparse import ArgumentParser

from Cl_mel import Cl_mel

C_N_SAMPLE_RATE = 16000
C_N_FFT = 400
C_NHOP_LENGTH = 160
C_N_MELS = 128

def main():
    cl_argument_parser = ArgumentParser()
    cl_argument_parser.add_argument("audio", nargs="+")
    a_args = cl_argument_parser.parse_args()
    total_start = time()

    s_audio_path = Path(a_args.audio[0])
    print(f"Audio source {s_audio_path}")
    
    #Import the mel utility class
    cl_mel = Cl_mel( C_N_SAMPLE_RATE, C_N_FFT, C_NHOP_LENGTH, C_N_MELS )

    #an_wav = cl_mel.load_audio(s_audio_path)

    aan_mel = cl_mel.compute_mel_spectrogram_from_audio_file( s_audio_path )
    
    #aan_mel = cl_mel.compute_mel( an_wav )

    cl_mel.save_mel_as_image( aan_mel, "mel.jpg")


if __name__ == "__main__":
    main()