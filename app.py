import os
import uuid
import json
import queue
import threading
from flask import Flask, request, jsonify, render_template, send_from_directory, Response, stream_with_context
from werkzeug.utils import secure_filename

from nst_core import run_style_transfer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'outputs')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# job_id → queue of SSE events
_job_queues: dict[str, queue.Queue] = {}
_job_lock = threading.Lock()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _run_job(job_id, content_path, style_path, num_steps, style_weight, content_weight):
    q = _job_queues[job_id]

    def progress(step, total, s_loss, c_loss):
        pct = int(step / total * 100)
        q.put(json.dumps({
            'type': 'progress',
            'step': step,
            'total': total,
            'pct': pct,
            'style_loss': round(s_loss, 4),
            'content_loss': round(c_loss, 4),
        }))

    try:
        output_img = run_style_transfer(
            content_path, style_path,
            num_steps=num_steps,
            style_weight=style_weight,
            content_weight=content_weight,
            progress_callback=progress,
        )
        out_filename = f'{job_id}.png'
        out_path = os.path.join(OUTPUT_FOLDER, out_filename)
        output_img.save(out_path)
        q.put(json.dumps({'type': 'done', 'filename': out_filename}))
    except Exception as e:
        q.put(json.dumps({'type': 'error', 'message': str(e)}))
    finally:
        q.put(None)  # sentinel


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'content' not in request.files or 'style' not in request.files:
        return jsonify({'error': 'Both content and style images are required.'}), 400

    content_file = request.files['content']
    style_file   = request.files['style']

    for f in (content_file, style_file):
        if f.filename == '' or not allowed_file(f.filename):
            return jsonify({'error': f'Invalid file: {f.filename}'}), 400

    try:
        num_steps      = int(request.form.get('num_steps', 300))
        style_weight   = float(request.form.get('style_weight', 30_000_000))
        content_weight = float(request.form.get('content_weight', 3))
    except ValueError:
        return jsonify({'error': 'Invalid parameter values.'}), 400

    num_steps    = max(50,  min(num_steps,    1000))
    style_weight = max(1e4, min(style_weight, 1e9))
    content_weight = max(1, min(content_weight, 100))

    job_id = str(uuid.uuid4())
    c_ext = secure_filename(content_file.filename).rsplit('.', 1)[1].lower()
    s_ext = secure_filename(style_file.filename).rsplit('.', 1)[1].lower()
    content_path = os.path.join(UPLOAD_FOLDER, f'{job_id}_content.{c_ext}')
    style_path   = os.path.join(UPLOAD_FOLDER, f'{job_id}_style.{s_ext}')
    content_file.save(content_path)
    style_file.save(style_path)

    q = queue.Queue()
    with _job_lock:
        _job_queues[job_id] = q

    t = threading.Thread(
        target=_run_job,
        args=(job_id, content_path, style_path, num_steps, style_weight, content_weight),
        daemon=True,
    )
    t.start()

    return jsonify({'job_id': job_id})


@app.route('/progress/<job_id>')
def progress(job_id):
    with _job_lock:
        if job_id not in _job_queues:
            return jsonify({'error': 'Unknown job'}), 404
        q = _job_queues[job_id]

    def generate():
        while True:
            msg = q.get()
            if msg is None:
                break
            yield f'data: {msg}\n\n'
        with _job_lock:
            _job_queues.pop(job_id, None)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


@app.route('/outputs/<filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
