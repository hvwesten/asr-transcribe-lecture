from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import shlex
import subprocess
import sys
import wave
import json

from deepspeech import Model, version
from timeit import default_timer as timer

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

from deepspeech.client import convert_samplerate, metadata_to_string


def speech_recognition_cpu(model_file, scorer_file, audio_file ):
    audio_file = str(audio_file)
    print(f"Loading model from file {model_file}")
    model_load_start = timer()

    # Creating a model instance and loading model
    ds = Model(model_file)

    model_load_end = timer() - model_load_start
    print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)

    # TODO args.beam_width: ?
    desired_sample_rate = ds.sampleRate()

    if scorer_file:
        print(f"Loading scorer from files {scorer_file}")
        scorer_load_start = timer()
        ds.enableExternalScorer(scorer_file)
        scorer_load_end = timer() - scorer_load_start
        print(f"Loaded scorer in {scorer_load_end:.3}s.")

        # TODO args.lm_alpha and args.lm_beta ?

    # TODO args.hot_words?

    fin = wave.open(audio_file, 'rb')
    fs_orig = fin.getframerate()
    if fs_orig != desired_sample_rate:
        print(f"Warning: original sample rate ({fs_orig}) is different than {desired_sample_rate}hz. Resampling might produce erratic speech recognition.")
        fs_new, audio = convert_samplerate(audio_file, desired_sample_rate)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    audio_length = fin.getnframes() * (1/fs_orig)
    fin.close()

    # Performing inference
    print("Running inference.")
    inference_start = timer()

    inference_end = timer() - inference_start
    print(f"Inference took {inference_end:0.3f}s for {audio_length:0.3f}s audio file.")

    result = ds.stt(audio)
    return result



def speech_recognition_gpu(model_file, scorer_file, audio_file ):
    pass