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

    def event():  # for testing
        links = ['i_QdlmDToVM',
                 'CQY3KUR3VzM',
                 'hzKGo0q4T7c']

        for l in links:
            yield server_sent_event(l)
            time.sleep(2)
        else:
            yield "event: end\ndata: {}\n\n"


    return Response(event(), mimetype="text/event-stream")


def server_sent_event(url_name):
    return('event: video\ndata: {"video_url":"https://www.youtube.com/embed/%s"}\n\n' % url_name)    #has to conform to this format for EventSource in js to run