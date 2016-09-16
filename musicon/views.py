from flask import render_template,current_app
from musicon import app
from flask import request
import youtube_dl
import os
import scipy.io.wavfile as wv
from scipy.signal import hann
from scipy.fftpack import rfft,rfftfreq
from scipy import log10
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression


#import pandas as pd


@app.route('/')
@app.route('/index')
@app.route('/input')
def input():
    return render_template("input.html")


@app.route('/output')
def output():
    video_url = request.args.get('video_url')
    audio_format = 'wav'
    in_audio_name = 'audio'    #name for downloaed audio
    out_audio_name = 'out_audio'    #name for the processed audio

    #download video
    SITE_ROOT = current_app.root_path
    down_path = os.path.join(SITE_ROOT,"static/workdir")

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': audio_format,
            'preferredquality': '192',
        }],
        'outtmpl': down_path+'/'+in_audio_name+'.%(ext)s',
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        #info_video = ydl.extract_info(video_url, download=False)
        #video_title = info_video.get('title', None)
        ydl.download([video_url])


    #parameter setting
    piece_dur = 5  # duration of the piece in seconds
    n_max_freqs = 100

    model_url = os.path.join(SITE_ROOT, "ml_model/logit.pkl")
    model_fit = joblib.load(model_url)

    audio_url = os.path.join(down_path,in_audio_name+"."+audio_format)
    rate, audio = wv.read(audio_url)    #load audio

    if(audio.ndim == 1):
        in_audio = audio
    else:
        in_audio = np.mean(audio,axis=1)    #mean of the channels from the two channels


    # initialize
    audio_stats = np.empty((0,2))    #record stats

    audio_len = len(in_audio)
    piece_len = rate * piece_dur
    window = hann(piece_len)

    start = 0
    end = piece_len
    i = 0

    while start < audio_len:
        if(end) > audio_len:
            end = audio_len
            window = hann(end-start)

        clip = in_audio[start:end]

        clip_w = window * clip

        mags_w = abs(rfft(clip_w))
        freqs = rfftfreq(len(mags_w), d=1 / rate)

        mags_w = 20 * log10(mags_w)  # to db

        max_idx = np.argsort(mags_w)[::-1][:n_max_freqs]
        max_freqs = freqs[max_idx]
        max_mags = mags_w[max_idx]

        freq_wmean = np.sum(max_mags * max_freqs) / np.sum(max_mags)
        freq_sd = np.std(max_freqs)

        audio_stats = np.concatenate((audio_stats,np.array([[freq_wmean,freq_sd]])))


        start += piece_len
        end += piece_len
        i += 1

    is_music = model_fit.predict(audio_stats).astype('bool')

    print(is_music)
    #output audio
    audio_idx = np.arange((i-1)*piece_len).reshape((-1,piece_len))[is_music[0:-1],].ravel()    #all the piece_len part

    if(is_music[-1]):
        audio_idx = np.append(audio_idx,np.arange(i*piece_len,audio_len))    #the ending part that is shorter than piece_len

    print(audio_idx)

    ##TO DO: apply a window function before joining the clips

    if(audio.ndim == 1):
        out_audio = audio[audio_idx]
    else:
        out_audio = audio[audio_idx,:]

    out_audio_url = os.path.join(down_path,out_audio_name+"."+audio_format)

    wv.write(out_audio_url,rate=rate,data=out_audio)

    #os.remove(audio_url)    #remove downloaded original audio

    return render_template("output.html",
                           audio_name = 'workdir/'+out_audio_name+'.'+audio_format)
