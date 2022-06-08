import os
import sys
import contextlib
import torch
import argparse
from pathlib import Path

import pyaudio
import wave
import torch
import torchaudio
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC


# Config for microphone
CHUNK = 1024 
FORMAT = pyaudio.paInt16
CHANNELS = 1 
RATE = 16000


@contextlib.contextmanager
def ignore_stderr():
    """
    Suppress ALSA messages from the console
    Source: https://stackoverflow.com/a/70467199/5707560
    """

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)


def run(args):
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(args.model_id)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_id)
    model.to('cpu')

    while True:
        output_filename = f"/tmp/recording.wav"

        with ignore_stderr():
            p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        # Recording audio
        print("* recording")
        frames = []
        for _ in range(0, int(RATE / CHUNK * args.record_seconds)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save a recording into a WAV file
        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        print("* recognizing")

        wav_file_path = str(Path(output_filename).absolute())
        waveform, _ = torchaudio.load(wav_file_path)
        sp = waveform.squeeze().numpy()

        input_values = processor(sp,
                                 sample_rate=16000,
                                 sampling_rate=16000,
                                 return_tensors="pt").input_values

        with torch.no_grad():
            logits = model(input_values).logits

        prediction = processor.batch_decode(logits.numpy()).text

        print()
        print('\t', prediction[0])
        print()

        print("* done recognizing")

        print('Listen to you again...')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id", type=str, required=True, help="Model identifier. Should be loadable with 🤗 Transformers"
    )
    parser.add_argument(
        "--record_seconds", type=int, default=10, required=False, help="How long to record from microphone"
    )

    args = parser.parse_args()

    run(args)
