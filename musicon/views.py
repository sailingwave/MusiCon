from flask import render_template
from musicon import app,proc_utils
from flask import request
#import os

@app.route('/')
@app.route('/index')
@app.route('/input')
def input():
    return render_template("input.html")


@app.route('/output')
def output():
    video_url = request.args.get('video_url')
    out_audio_name = 'out_audio'    #name for the processed audio

    proc_utils.audio_process(video_url,out_audio_name)

    #os.remove(audio_url)    #remove downloaded original audio

    return render_template("output.html",
                           audio_name = 'workdir/'+out_audio_name+'.wav')
