import speech_recognition as srec
import soundfile as sf
from math import gcd

from scipy.signal import resample_poly, butter, sosfiltfilt
import numpy as np
import matplotlib.pyplot as plt


SAMPLE_RATE = 44100
SAMPLE_WIDTH = 2
DTYPE = np.int16

NAME_ORIGINAL_WAV = f"./Sounds/Sound_{SAMPLE_RATE}[Hz]_{SAMPLE_WIDTH}[byte].wav"
NAME_ORIGINAL_RAW = f"./Sounds/Sound_{SAMPLE_RATE}[Hz]_{SAMPLE_WIDTH}[byte].raw"
NAME_RESAMPLED_WAV = "./Sounds/Sound_4000[Hz]_2[byte].wav"
NAME_RESAMPLED_RAW = "./Sounds/Sound_4000[Hz]_2[byte].raw"
NAME_FILTERED_WAV = "./Sounds/Filtered_4000[Hz]_2[byte].wav"
NAME_FILTERED_RAW = "./Sounds/Filtered_4000[Hz]_2[byte].raw"

TARGET_RATE = 4000
FILTER_CUTOFF = 4000
FILTER_ORDER = 6
MICROPHONE_INDEX = 1


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
    if len(data.shape) > 1:
        data = data[:, 0]

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
    if len(data.shape) > 1:
        data = data[:, 0]

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

    if len(original.shape) > 1:
        original = original[:, 0]
    if len(resampled.shape) > 1:
        resampled = resampled[:, 0]
    if len(filtered.shape) > 1:
        filtered = filtered[:, 0]

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
    plt.show()


def process_recording():
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


if __name__ == "__main__":
    recognizer = srec.Recognizer()
    microphone = srec.Microphone(
        device_index=MICROPHONE_INDEX,
        sample_rate=SAMPLE_RATE,
    )

    sound_recoder(recognizer, microphone)
