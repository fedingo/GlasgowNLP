import librosa
import warnings
warnings.simplefilter("ignore")


def get_mfcc(path, start=0.0, duration = None):
    x, sr = librosa.load(path, sr=None, offset=start, duration=duration)

    return librosa.feature.mfcc(x, sr=sr)