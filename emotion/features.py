import numpy as np
import librosa


def extract_acoustic_features(
    audio,
    sr=16000
):

    audio = audio.astype(
        np.float32
    )

    features = {}

    features["duration"] = (
        len(audio) / sr
    )

    rms = librosa.feature.rms(
        y=audio
    )[0]

    features["rms_mean"] = float(
        np.mean(rms)
    )

    features["rms_std"] = float(
        np.std(rms)
    )

    features["rms_max"] = float(
        np.max(rms)
    )

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )[0]

    features["zcr_mean"] = float(
        np.mean(zcr)
    )

    features["zcr_std"] = float(
        np.std(zcr)
    )

    centroid = (
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )[0]
    )

    features["spectral_centroid_mean"] = float(
        np.mean(centroid)
    )

    features["spectral_centroid_std"] = float(
        np.std(centroid)
    )

    bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr
        )[0]
    )

    features["spectral_bandwidth_mean"] = float(
        np.mean(bandwidth)
    )

    f0, _, _ = librosa.pyin(

        audio,

        fmin=librosa.note_to_hz("C2"),

        fmax=librosa.note_to_hz("C7"),

        sr=sr
    )

    valid_f0 = f0[
        ~np.isnan(f0)
    ]

    if len(valid_f0) > 0:

        features["pitch_mean"] = float(
            np.mean(valid_f0)
        )

        features["pitch_std"] = float(
            np.std(valid_f0)
        )

        features["pitch_min"] = float(
            np.min(valid_f0)
        )

        features["pitch_max"] = float(
            np.max(valid_f0)
        )

        features["pitch_range"] = float(
            np.max(valid_f0)
            -
            np.min(valid_f0)
        )

    else:

        features["pitch_mean"] = 0.0
        features["pitch_std"] = 0.0
        features["pitch_min"] = 0.0
        features["pitch_max"] = 0.0
        features["pitch_range"] = 0.0

    return features