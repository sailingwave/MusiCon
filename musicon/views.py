from flask import render_template
from musicon import app,proc_utils
from flask import request
#import os


@app.route('/index')
@app.route('/input')
@app.route('/')
def input():
    return render_template("input.html")


@app.route('/output')
def output():
    video_url = request.args.get('video_url')

    links = proc_utils.audio_process(video_url)

    #os.remove(audio_url)    #remove downloaded original audio

    return render_template("output.html",
                           links = links)
