import os
from flask import Flask, render_template, request

#general imports
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib as mpl
import jams

import IPython.display as ipd
from ipywidgets import interactive_output 
from ipywidgets import IntSlider, FloatSlider, fixed, Checkbox
from ipywidgets import VBox, Label
import soundfile as sf

# from werkzeug import secure_filename

# signal processing parameters
n_bins = 72                              # notes, essentially
mag_exp = 1                              # magnitude exponent for scaling if needed
pre_post_max = 30                        # pre post max
sr = 44100                               #typical sampling rate using
overlap = 0.5                            # overlaps per hop in percentage
hop_length = 512                        # samples in between  frames
cqt_threshold = -60                     # muting threshold

app = Flask(__name__)

@app.route("/")
def index():
    return "Home Page"
    
@app.route("/page")
def upload_page():
    return render_template("upload.html")


@app.route("/uploader", methods=["POST"])
def upload_file():
    f = request.files['file']

    # save file
    filename = f.filename
    file_path = os.path.join("files", filename)
    print(file_path)
    f.save(file_path)

    # Process this audio file
    result_path = process_file(file_path, filename)



    return f"<audio controls><source src=\"{result_path}\" type=\"audio/wav\"</audio>"

CQTdB = None

def process_file(audio_path, filename):
    global sr    
    global CQTdB
    
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    CQTdB = calc_cqt(y, sr=sr)

    onset_sec, onset_boundaries, onset_env, onset_frames =  calc_onset(CQTdB)

    #here I create a sinewave to imitate the sound with fluctations for the pitch
    sinewave = np.concatenate([
        estimate_pitch_and_sine(CQTdB, onset_boundaries, i, sr=sr) 
        for i in range(len(onset_boundaries)-1)
    ])

    print(sinewave.shape)

    # save this file to wav
    result_path = os.path.join("static", filename)
    sf.write(result_path, sinewave, sr)

    return result_path


#cleans cqt by 'removing' noise below threshold value
def cqt_clean(cqt,thres=cqt_threshold):
    cqt_new=np.copy(cqt)
    cqt_new[cqt_new<thres]=-120
    return cqt_new
    
    #this is the actual constant Q transforamtion that pulls out individual pitches for the exponential frequencies
def calc_cqt(y,sr=sr,hop_length=hop_length, n_bins=n_bins, mag_exp=mag_exp):
    global CQTdB

    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, fmin=None, n_bins=n_bins))
    C_mag = librosa.magphase(C)[0]**mag_exp
    CQTdB = librosa.core.amplitude_to_db(C_mag ,ref=np.max)
    CQTdB = cqt_clean(CQTdB,cqt_threshold)
    return CQTdB

    #This estimates the pitch from the freqency
def estimate_pitch(interval, threshold = cqt_threshold):
    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=librosa.note_to_hz('C1'),
                            bins_per_octave=12)
    if interval.max()<threshold:
        return [None, np.mean((np.amax(interval,axis=0)))]
    else:
        f0 = int(np.mean((np.argmax(interval,axis=0))))
    return [freqs[f0], np.mean((np.amax(interval,axis=0)))]

# Rescaling aplitude between 0 and 1
def rescale(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def generate_sine(f0_info, sr, interval):
    global CQTdB

    f0=f0_info[0]
    A=rescale(f0_info[1], CQTdB.min(), CQTdB.max(), 0, 1)
    n = np.arange(librosa.frames_to_samples(interval, hop_length=hop_length))
    if f0==None:
        f0=0
    sine_wave = A*np.sin(2*np.pi*f0*n/float(sr))
    
    return sine_wave

# Generate notes from Pitch estimate
def estimate_pitch_and_sine(CQTdB, onset_boundaries, i, sr=sr):
    n0 = onset_boundaries[i]
    n1 = onset_boundaries[i+1]
    interval = np.mean(CQTdB[:,n0:n1],axis=1)
    f0_info = estimate_pitch(interval,threshold=cqt_threshold)
    return generate_sine(f0_info, sr, n1-n0)

# This is the calculation of the onset envelope from the cqt
def calc_onset_env(cqt):
    return librosa.onset.onset_strength(S=cqt, sr=sr, aggregate=np.mean, hop_length=hop_length)

# Onset from Onset Envelope
def calc_onset(cqt, pre_post_max=pre_post_max):
    onset_env=calc_onset_env(cqt)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env
                                              ,sr=sr
                                              , units='frames'
                                              , hop_length=hop_length
                                              , pre_max=pre_post_max
                                              ,post_max=pre_post_max)
    onset_boundaries = np.concatenate([[0], onset_frames, [cqt.shape[1]]])
    onset_times = librosa.frames_to_time(onset_boundaries, sr=sr, hop_length=hop_length)
    return [onset_times, onset_boundaries, onset_env, onset_frames]

# def audio_approx(CQTdB = CQTdB, onset_boundaries = onset_boundaries,sr=sr):
#     x = np.concatenate([
#     estimate_pitch_and_sine(CQTdB, onset_boundaries, i, sr=sr) 
#     for i in range(len(onset_boundaries)-1)
#     ])
#     return x


if __name__ == '__main__':
    app.run(host="0.0.0.0")