from math import gcd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt
import soundfile as sf
import speech_recognition as srec
from scipy.signal import butter, resample_poly, sosfiltfilt
from skimage.restoration import (
    denoise_bilateral,
    denoise_invariant,
    denoise_tv_chambolle,
    denoise_wavelet,
)


SAMPLE_RATE = 44100
SAMPLE_WIDTH = 2
DTYPE = np.int16

BASE_DIR = Path(__file__).resolve().parent
SOUNDS_DIR = BASE_DIR / "Sounds"
PLOTS_DIR = BASE_DIR / "Plots"

NAME_ORIGINAL_WAV = SOUNDS_DIR / f"Sound_{SAMPLE_RATE}[Hz]_{SAMPLE_WIDTH}[byte].wav"
NAME_ORIGINAL_RAW = SOUNDS_DIR / f"Sound_{SAMPLE_RATE}[Hz]_{SAMPLE_WIDTH}[byte].raw"
NAME_RESAMPLED_WAV = SOUNDS_DIR / "Sound_4000[Hz]_2[byte].wav"
NAME_RESAMPLED_RAW = SOUNDS_DIR / "Sound_4000[Hz]_2[byte].raw"
NAME_FILTERED_WAV = SOUNDS_DIR / "Filtered_4000[Hz]_2[byte].wav"
NAME_FILTERED_RAW = SOUNDS_DIR / "Filtered_4000[Hz]_2[byte].raw"

TARGET_RATE = 4000
FILTER_CUTOFF = 4000
FILTER_ORDER = 6
MICROPHONE_INDEX = 1


def ensure_directories():
    SOUNDS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)


def to_mono(data):
    if len(data.shape) > 1:
        return data[:, 0]
    return data


def save_recording(audio):
    wav_data = audio.get_wav_data(
        convert_rate=SAMPLE_RATE,
        convert_width=SAMPLE_WIDTH,
    )
    raw_data = audio.get_raw_data(
        convert_rate=SAMPLE_RATE,
        convert_width=SAMPLE_WIDTH,
    )

    with open(NAME_ORIGINAL_WAV, "wb") as f:
        f.write(wav_data)

    with open(NAME_ORIGINAL_RAW, "wb") as f:
        f.write(raw_data)


def resample_wav():
    data, fs_original = sf.read(NAME_ORIGINAL_WAV)
    data = to_mono(data)

    g = gcd(fs_original, TARGET_RATE)
    up = TARGET_RATE // g
    down = fs_original // g
    data_resampled = resample_poly(data, up, down)

    sf.write(NAME_RESAMPLED_WAV, data_resampled, TARGET_RATE)


def resample_raw():
    with open(NAME_ORIGINAL_RAW, "rb") as f:
        raw_bytes = f.read()

    signal = np.frombuffer(raw_bytes, dtype=DTYPE)
    signal_float = signal.astype(np.float32) / 32768.0

    g = gcd(SAMPLE_RATE, TARGET_RATE)
    up = TARGET_RATE // g
    down = SAMPLE_RATE // g
    resampled = resample_poly(signal_float, up, down)
    resampled_int16 = np.int16(np.clip(resampled, -1.0, 1.0) * 32767)

    with open(NAME_RESAMPLED_RAW, "wb") as f:
        f.write(resampled_int16.tobytes())


def filter_wav():
    data, fs_original = sf.read(NAME_ORIGINAL_WAV)
    data = to_mono(data)

    sos = butter(
        FILTER_ORDER,
        FILTER_CUTOFF,
        btype="low",
        fs=fs_original,
        output="sos",
    )
    filtered = sosfiltfilt(sos, data)

    sf.write(NAME_FILTERED_WAV, filtered, fs_original)


def filter_raw():
    with open(NAME_ORIGINAL_RAW, "rb") as f:
        raw_bytes = f.read()

    signal = np.frombuffer(raw_bytes, dtype=DTYPE)
    signal_float = signal.astype(np.float32) / 32768.0

    sos = butter(
        FILTER_ORDER,
        FILTER_CUTOFF,
        btype="low",
        fs=SAMPLE_RATE,
        output="sos",
    )
    filtered_raw = sosfiltfilt(sos, signal_float)
    filtered_int16 = np.int16(np.clip(filtered_raw, -1.0, 1.0) * 32767)

    with open(NAME_FILTERED_RAW, "wb") as f:
        f.write(filtered_int16.tobytes())


def plot_signals():
    original, fs_original = sf.read(NAME_ORIGINAL_WAV)
    resampled, fs_resampled = sf.read(NAME_RESAMPLED_WAV)
    filtered, fs_filtered = sf.read(NAME_FILTERED_WAV)

    original = to_mono(original)
    resampled = to_mono(resampled)
    filtered = to_mono(filtered)

    time_original = np.arange(len(original)) / fs_original
    time_resampled = np.arange(len(resampled)) / fs_resampled
    time_filtered = np.arange(len(filtered)) / fs_filtered

    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(time_original, original, label="Original")
    plt.title("Original signal")
    plt.xlabel("Time, s")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_resampled, resampled, label="Resampled")
    plt.title("Resampled signal")
    plt.xlabel("Time, s")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_filtered, filtered, label="Filtered")
    plt.title("Filtered signal")
    plt.xlabel("Time, s")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "Practical_2_comparison.png", dpi=150)
    plt.close()


def process_recording():
    ensure_directories()
    resample_wav()
    resample_raw()
    filter_wav()
    filter_raw()
    plot_signals()


def sound_recoder(rec, mic):
    with mic as source:
        print("Speak now...")
        audio = rec.listen(source)

    save_recording(audio)
    process_recording()


def wavelet_denoiser(signal, level, mode, wavelet):
    """
    Denoise a 1D signal using Discrete Wavelet Transform thresholding.

    Args:
        signal (np.ndarray): The input signal.
        wavelet (str): The name of the mother wavelet.
        level (int): The level of decomposition.
        mode (str): Thresholding mode, 'soft' or 'hard'.

    Returns:
        np.ndarray: The denoised signal.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(signal.size))
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]
    ]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    return denoised_signal[: len(signal)]


def invarince_denoiser(image, **kwargs):
    return denoise_wavelet(image, sigma=0.5, wavelet="db4", mode="soft")


def save_filtered_plot(time, data, filtered, title, output_name):
    plt.figure(figsize=(10, 6))
    plt.plot(time, data, "b-", label="Original Clean Signal")
    plt.plot(time, filtered, "g-", linewidth=2, label=title)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / output_name, dpi=150)
    plt.close()


def sound_filter():
    ensure_directories()

    if not NAME_ORIGINAL_WAV.exists():
        raise FileNotFoundError(
            f"Input file was not found: {NAME_ORIGINAL_WAV}. "
            "Run practical work 2 first or copy the recorded WAV file into Sounds."
        )

    if not NAME_ORIGINAL_RAW.exists():
        data_for_raw, _ = sf.read(NAME_ORIGINAL_WAV)
        data_for_raw = to_mono(data_for_raw)
        data_for_raw = np.int16(np.clip(data_for_raw, -1.0, 1.0) * 32767)
        with open(NAME_ORIGINAL_RAW, "wb") as f:
            f.write(data_for_raw.tobytes())

    process_recording()

    data, fs_original = sf.read(NAME_ORIGINAL_WAV)
    data = to_mono(data)
    time = np.arange(len(data)) / fs_original
    data_2d = data.reshape(1, -1)

    invariance = denoise_invariant(
        data_2d,
        denoise_function=invarince_denoiser,
    ).flatten()
    total_variation = denoise_tv_chambolle(
        data_2d,
        weight=0.1,
        channel_axis=None,
    ).flatten()
    bilateral = denoise_bilateral(
        data_2d,
        sigma_color=0.05,
        sigma_spatial=15,
        channel_axis=None,
    ).flatten()
    wavelet = wavelet_denoiser(data, level=5, mode="soft", wavelet="db4")

    sf.write(SOUNDS_DIR / "Filtered_Invariance.wav", invariance, SAMPLE_RATE)
    sf.write(SOUNDS_DIR / "Filtered_Total_Variation.wav", total_variation, SAMPLE_RATE)
    sf.write(SOUNDS_DIR / "Filtered_Bilateral.wav", bilateral, SAMPLE_RATE)
    sf.write(SOUNDS_DIR / "Filtered_Wavelet.wav", wavelet, SAMPLE_RATE)

    save_filtered_plot(time, data, invariance, "J-Invariance", "J_Invariance.png")
    save_filtered_plot(time, data, total_variation, "Total Variation", "Total_Variation.png")
    save_filtered_plot(time, data, bilateral, "Bilateral", "Bilateral.png")
    save_filtered_plot(time, data, wavelet, "Wavelet", "Wavelet.png")


if __name__ == "__main__":
    sound_filter()
