from flask import render_template, Response
from musicon import app,proc_utils
from flask import request, send_file
import time
#import os


@app.route('/index')
@app.route('/')
def input():
    return render_template("index.html")


@app.route('/output/')
def output():
    video_url = request.args.get('video_url')

    #links = proc_utils.audio_process(video_url)

    # return render_template("output.html",links = links)

    def event():  # for testing
        links = ['https://www.youtube.com/embed/i_QdlmDToVM',
                 'https://www.youtube.com/embed/CQY3KUR3VzM',
                 'https://www.youtube.com/embed/hzKGo0q4T7c']

        for l in links:
            print('hi')
            yield l
            time.sleep(2)

    #return Response(stream_template('output.html', links = proc_utils.audio_process(video_url)))
    return Response(event(), mimetype="text/event-stream")


