# -*- coding: UTF-8 -*-
import sys
import os
from imp import reload

reload(sys)
from flask import Flask, make_response, request
from io import BytesIO
from flask_compress import Compress
from MoeGoeCLI import do_model

app = Flask(__name__)
# json 乱码
app.config['JSON_AS_ASCII'] = False
Compress(app)

model = os.environ.get("MODEL_PATH")
config = os.environ.get("CONFIG_PATH")

@app.route("/tts", methods=["post", "get"])
def tts():
    global vits_length_scale
    txt = request.args.get("txt")
    vits_length_scale = 1.0/request.args.get("speed", 1, type=float)
    bs = BytesIO()
    do_model(
        model,
        config,
        2,
        txt,
        bs,
        "t"
    )
    res = make_response(bs.getvalue())
    res.headers['Content-Type'] = 'audio/wave'
    res.headers['Content-Disposition'] = 'attachment;filename=out.wav'
    return res


# if __name__ == '__main__':    app.run(host='0.0.0.0', port=int(os.environ.get("MOE_TTS_PORT")), threaded=True)
if __name__ == '__main__':    app.run(host='0.0.0.0', port=8080, threaded=True)
