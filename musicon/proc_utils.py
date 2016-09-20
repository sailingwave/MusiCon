#import youtube_dl
import os
import subprocess as sp
import pafy
import numpy as np
import pandas as pd
import scipy.io.wavfile as wv
from scipy.signal import hann
from numpy.fft import rfft,rfftfreq
from scipy import log10
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from flask import current_app


def audio_process(url,out_audio_name):
    # parameter setting
    FFMPEG_BIN = os.environ['FFMPEG_BIN']
    SITE_ROOT = current_app.root_path

    out_audio_url = os.path.join(SITE_ROOT,"static/workdir",out_audio_name+".wav")

    piece_len = 5  # duration of a piece in seconds being processed at a time
    rate = 44100    #sample rate of audio, must be divisible by 2*freq_bin_size
    freq_bin_size = 10    #for bin fft freqs
    n_freq_bins = int(rate/freq_bin_size/2)    #number of freq bins
    n_channel = 2    #2 channels, just use the 1st channel for classification

    piece_size = rate * piece_len    #data size of each piece

    #load ML models
    model_url = os.path.join(SITE_ROOT, "ml_model/logit.pkl")
    model_fit = joblib.load(model_url)

    pca_url = os.path.join(SITE_ROOT, "ml_model/pca.pkl")
    pca_fit = joblib.load(pca_url)


    #stream video
    video = pafy.new(url)
    bestaudio = video.getbestaudio()

    command = [FFMPEG_BIN,
               '-i', bestaudio.url,
               '-f', 's16le',
               '-acodec', 'pcm_s16le',
               '-ar', str(rate),
               '-ac', str(n_channel),
               '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10 ** 8)

    raw_audio = b'00'  # initial

    # initialize
    out_audio = np.empty((0,2))
    #init = True    #if first round of ananlysis

    while raw_audio != b'':
        raw_audio = pipe.stdout.read(piece_size*n_channel*2)    #*2: int16 is two byte

        clip_2c = np.fromstring(raw_audio, dtype="int16")
        clip_2c = clip_2c.reshape((len(clip_2c) / 2, 2))
        clip = clip_2c[:,0]    #[0]: the 1st channel

        curr_size = len(clip)
        window = hann(curr_size)    #apply window function before fft
        clip_w = window * clip

        mags_w = abs(rfft(clip_w))

        if(np.all(mags_w==0)):   #blank piece
            continue

        mags_w = 20 * log10(mags_w)  # to db
        mag_mean = np.mean(mags_w)

        if(curr_size != piece_size):    #last iter
            break
            freqs = rfftfreq(curr_size, d=1 / rate)
            freq_bins = np.linspace(0,rate/2,n_freq_bins+1)+1e-4    #10Hz/bin, added 1e-4 for exact values, e.g. 0Hz, 10Hz...; +1:two ends
            freq_grp = np.digitize(freqs,freq_bins)

            mags_data = pd.DataFrame(data=np.stack((mags_w/mag_mean,freq_grp)).transpose(),
                                     columns=('mag','grp'))

            mags_data = mags_data.groupby('grp').mean().iloc[1:,].transpose()

        else:
            #hard code the binning rule, assuming
            mags_data = np.mean(mags_w[1:,].reshape((n_freq_bins,-1)),axis=1)-mag_mean    #1:, exclude the DC

        mags_data = mags_data.reshape(1,-1)

        #construct ML variables
        # project to PCA
        data_pca = pca_fit.transform(mags_data)

        is_music = model_fit.predict(data_pca).astype('bool')

        if(is_music):
            out_audio = np.concatenate((out_audio,clip_2c))

        #clip_last = clip

    ##TO DO: apply a window function before joining the clips

    # if(audio.ndim == 1):
    #     out_audio = audio[audio_idx]
    # else:
    #     out_audio = audio[audio_idx,:]

    wv.write(out_audio_url,rate=rate,data=out_audio)