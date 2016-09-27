import os
import subprocess as sp
import pafy
import numpy as np
import pandas as pd
from scipy.signal import hann
from numpy.fft import rfft,rfftfreq
from scipy import log10
from sklearn.externals import joblib
from sklearn.decomposition import PCA
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from flask import current_app
import re


def audio_process(url):
    '''Find the music/non-music part in a youtube video'''
    # parameter setting
    FFMPEG_BIN = os.environ['FFMPEG_BIN']
    #SITE_ROOT = current_app.root_path
    #SITE_ROOT = os.path.abspath('./musicon/')
    SITE_ROOT = './musicon/'

    piece_len = 2  # duration of a piece in seconds being processed at a time
    rate = 44100    #sample rate of audio, must be divisible by 2*freq_bin_size
    freq_bin_size = 10    #for bin fft freqs
    n_freq_bins = int(rate/freq_bin_size/2)    #number of freq bins

    piece_size = rate * piece_len    #data size of each piece

    #load ML models
    #model_url = os.path.join(SITE_ROOT, "ml_model/rf.pkl")
    model_url = os.path.join(SITE_ROOT, "ml_model/logit.pkl")
    model_fit = joblib.load(model_url)

    pca_url = os.path.join(SITE_ROOT, "ml_model/pca.pkl")
    pca_fit = joblib.load(pca_url)


    #stream video
    video = pafy.new(url)
    video_len = video.length
    video_title = video.title

    yield ('event: start\ndata: {"video_title":"%s","video_len":"%s"}\n\n' % (video_title,video_len))  # start of stream

    bestaudio = video.getbestaudio()

    command = [FFMPEG_BIN,
               '-i', bestaudio.url,
               '-f', 's16le',
               '-acodec', 'pcm_s16le',
               '-ar', str(rate),
               '-ac', '2',
               '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10 ** 8)

    #construct the embedding url
    video_id = youtube_ulr_conv(url)

    # initialize
    n_consec_music = 0    #when consec music>=2, classify as music start
    n_consec_nonmusic = 0
    n_music_end_consec_piece = 5    #how many consec nonmusic pieces needed to determine an end
    is_music_started = False    #has this piece of music started
    music_min_len = 10    #the minimum length of piece to output


    now = 0    #time of current processing (seconds)
    start = 0    #time of current music start (seconds)
    end = 0    #time of current music end (seconds)
    emb_url = ''    #url for embedding videos


    while True:
        print(now)

        raw_audio = pipe.stdout.read(piece_size*2*2)    #*2: int16 is two byte

        if(raw_audio == b''):    #end
            break

        clip_2c = np.fromstring(raw_audio, dtype="int16")
        clip_2c = clip_2c.reshape((len(clip_2c) / 2, 2))

        if(np.all(clip_2c==0)):    #all silence
            now = now + piece_len
            continue

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
            #hard code the binning rule, assuming data size is divisible by bin size
            mags_data = np.mean(mags_w[1:,].reshape((n_freq_bins,-1)),axis=1)-mag_mean    #1:, exclude the DC

        mags_sub = mags_data[0:500]
        mags_data = mags_data.reshape(1,-1)

        #construct ML variables
        # project to PCA
        data_pca = pca_fit.transform(mags_data)

        data_pred = data_pca
        # data_pred = np.hstack((data_pca, np.array([[mag_mean]]), np.array([cat_mag(mags_sub)]),
        #                             np.array([[spec_rolloff(mags_sub)]])))


        is_music = model_fit.predict(data_pred).astype('bool')
        is_music_prob = model_fit.predict_proba(data_pred)

        yield ('event: processing\ndata: {"progress":"%s","is_music_prob":"%s"}\n\n' % (now,is_music_prob))  #on processing

        #need 2 consecutive pieces to be music/non-music to segment

        if(not is_music_started):
            if(is_music):
                if(n_consec_music == 1):
                    start = now - piece_len
                    is_music_started = True
                    n_consec_music = 0
                else:
                    n_consec_music += 1
            else:
                n_consec_music = 0
        else:
            if(not is_music):
                if(n_consec_nonmusic==n_music_end_consec_piece-1):
                    end = now - piece_len*(n_music_end_consec_piece-1)
                    print(start, end)
                    is_music_started = False
                    if(end-start>music_min_len):
                        emb_url = video_id + "?start=" + str(start) + "&end=" + str(end)
                        yield(server_sent_event(emb_url))
                else:
                    n_consec_nonmusic += 1
            else:
                n_consec_nonmusic = 0
        #when music is not started, determine when to start using n_consec_music;
        #when music has started, determine when to stop using n_consec_nonmusic.

        print(str(is_music) + "," + str(is_music_prob))
        print(n_consec_music, n_consec_nonmusic, is_music_started)

        #before next iter
        now = now+piece_len
        print("=====")

    #check the last piece
    if(start>end):
        emb_url = video_id + "?start=" + str(start)
        yield(server_sent_event(emb_url))

    yield("event: end\ndata: {}\n\n")    #end of stream


def cat_mag(vect):    #vect length of 500
    return(np.hstack((np.mean(vect[0:8]),np.mean(vect[8:300]),np.mean(vect[300:]))))


def spec_rolloff(vect,k=0.85):
    spectralSum = np.sum(vect)
    sr_t = np.where(np.cumsum(vect) >= k * spectralSum)[0][0]
    return(sr_t)


def youtube_ulr_conv(in_url):
    '''Find the name of the video in the url for setting the start and end time'''

    search = re.search("watch\?v=(.*)\&*",in_url)

    if(search != None):
        video_urlname = search.group(1)
    else:
        video_urlname = None

    return(video_urlname)


def server_sent_event(url_name):
    return('event: video\ndata: {"video_url":"https://www.youtube.com/embed/%s"}\n\n' % url_name)    #has to conform to this format for EventSource in js to run
