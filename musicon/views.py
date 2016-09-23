from flask import render_template, Response
from musicon import app,proc_utils
from flask import request
import time
#import os


@app.route('/index')
@app.route('/input')
@app.route('/')
def input():
    return render_template("input.html")


@app.route('/output/')
def output():
    video_url = request.args.get('video_url')

    #links = proc_utils.audio_process(video_url)

    # return render_template("output.html",links = links)

    return Response(stream_template('output.html', links = proc_utils.audio_process(video_url)))
    #return Response(event(), mimetype="text/event-stream")


# def event():    #for testing
#     links = ['https://www.youtube.com/embed/i_QdlmDToVM',
#              'https://www.youtube.com/embed/CQY3KUR3VzM',
#              'https://www.youtube.com/embed/hzKGo0q4T7c']
#
#     for l in links:
#         yield l
#         time.sleep(5)


def stream_template(template_name, **context):
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    #rv.enable_buffering(5)
    return rv