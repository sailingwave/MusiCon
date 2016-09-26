from flask import Response,render_template
from musicon import app,proc_utils
import urllib
import time
#import os


@app.route('/index')
@app.route('/')
def input():
    return render_template("index.html")


@app.route('/output/')
@app.route('/output/<path:video_url>')
def output(video_url=None):
    if not video_url: # test
        video_url = 'https://www.youtube.com/watch?v=mrU6e6kvFR8'
    else:
        video_url = urllib.parse.unquote_plus(video_url)

    def event():  # for testing
        links = ['i_QdlmDToVM',
                 'CQY3KUR3VzM',
                 'hzKGo0q4T7c']

        for l in links:
            yield server_sent_event(l)
            time.sleep(2)
        else:
            yield "event: end\ndata: {}\n\n"

    #return Response(proc_utils.audio_process(video_url), mimetype="text/event-stream")
    return Response(event(), mimetype="text/event-stream")


def server_sent_event(url_name):
    return('event: video\ndata: {"video_url":"https://www.youtube.com/embed/%s"}\n\n' % url_name)    #has to conform to this format for EventSource in js to run
