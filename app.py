import os
import uuid
import json
import time
import queue
import threading
from flask import Flask, request, jsonify, render_template, send_from_directory, Response, stream_with_context
from werkzeug.utils import secure_filename

from nst_core import run_style_transfer
from fast_nst import stylize as fast_stylize

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

UPLOAD_FOLDER  = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUT_FOLDER  = os.path.join(os.path.dirname(__file__), 'outputs')
MODELS_FOLDER  = os.path.join(os.path.dirname(__file__), 'models')
SAMPLES_FOLDER = os.path.join(os.path.dirname(__file__), 'samples')
META_FOLDER    = os.path.join(OUTPUT_FOLDER, 'meta')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

for d in (UPLOAD_FOLDER, OUTPUT_FOLDER, MODELS_FOLDER, SAMPLES_FOLDER, META_FOLDER):
    os.makedirs(d, exist_ok=True)

# job_id → queue of SSE events
_job_queues: dict[str, queue.Queue] = {}
_job_lock = threading.Lock()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_meta(filename, mode, style_name=None):
    meta = {'mode': mode, 'timestamp': time.time(), 'style_name': style_name}
    meta_path = os.path.join(META_FOLDER, filename.replace('.png', '.json'))
    with open(meta_path, 'w') as f:
        json.dump(meta, f)


def _run_job(job_id, content_path, style_path, num_steps, style_weight, content_weight):
    q = _job_queues[job_id]

    def progress(step, total, s_loss, c_loss):
        pct = int(step / total * 100)
        q.put(json.dumps({
            'type': 'progress',
            'step': step, 'total': total, 'pct': pct,
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
        save_meta(out_filename, 'classic')
        q.put(json.dumps({'type': 'done', 'filename': out_filename}))
    except Exception as e:
        q.put(json.dumps({'type': 'error', 'message': str(e)}))
    finally:
        q.put(None)


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

    num_steps      = max(50,  min(num_steps,    1000))
    style_weight   = max(1e4, min(style_weight, 1e9))
    content_weight = max(1,   min(content_weight, 100))

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

    threading.Thread(
        target=_run_job,
        args=(job_id, content_path, style_path, num_steps, style_weight, content_weight),
        daemon=True,
    ).start()

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
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/outputs/<filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route('/models', methods=['GET'])
def list_models():
    models = []
    for f in sorted(os.listdir(MODELS_FOLDER)):
        if f.endswith('.pth') and '_ep' not in f and '_resume' not in f:
            name = f[:-4]
            # Check for a sample thumbnail
            sample_dir = os.path.join(SAMPLES_FOLDER, name)
            thumbnail = None
            if os.path.isdir(sample_dir):
                jpgs = sorted([x for x in os.listdir(sample_dir) if x.endswith('.jpg')])
                if jpgs:
                    thumbnail = f'/sample-img/{name}/{jpgs[-1]}'
            models.append({'name': name, 'filename': f, 'thumbnail': thumbnail})
    return jsonify({'models': models})


@app.route('/fast-stylize', methods=['POST'])
def fast_stylize_route():
    if 'content' not in request.files or 'model' not in request.form:
        return jsonify({'error': 'content image and model name are required'}), 400

    content_file = request.files['content']
    model_name   = request.form.get('model', '').strip()

    if not content_file.filename or not allowed_file(content_file.filename):
        return jsonify({'error': 'Invalid content image'}), 400

    model_path = os.path.join(MODELS_FOLDER, model_name + '.pth')
    if not os.path.isfile(model_path):
        return jsonify({'error': f'Model "{model_name}" not found'}), 404

    job_id   = str(uuid.uuid4())
    c_ext    = secure_filename(content_file.filename).rsplit('.', 1)[1].lower()
    c_path   = os.path.join(UPLOAD_FOLDER, f'{job_id}_content.{c_ext}')
    out_name = f'{job_id}.png'
    out_path = os.path.join(OUTPUT_FOLDER, out_name)
    content_file.save(c_path)

    try:
        result = fast_stylize(model_path, c_path)
        result.save(out_path)
        save_meta(out_name, 'fast', style_name=model_name)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'filename': out_name})


@app.route('/gallery')
def gallery():
    items = []
    try:
        files = sorted(
            [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith('.png')],
            key=lambda f: os.path.getmtime(os.path.join(OUTPUT_FOLDER, f)),
            reverse=True,
        )
        for f in files[:60]:
            meta_path = os.path.join(META_FOLDER, f.replace('.png', '.json'))
            meta = {}
            if os.path.isfile(meta_path):
                with open(meta_path) as fp:
                    meta = json.load(fp)
            items.append({
                'filename': f,
                'url': f'/outputs/{f}',
                'timestamp': meta.get('timestamp', 0),
                'mode': meta.get('mode', 'classic'),
                'style_name': meta.get('style_name'),
            })
    except Exception:
        pass
    return jsonify({'items': items})


@app.route('/samples/<model_name>')
def list_samples(model_name):
    model_samples_dir = os.path.join(SAMPLES_FOLDER, secure_filename(model_name))
    if not os.path.isdir(model_samples_dir):
        return jsonify({'samples': []})
    files = sorted([f for f in os.listdir(model_samples_dir) if f.endswith('.jpg')])
    samples = []
    for f in files:
        try:
            step = int(f.replace('step_', '').replace('.jpg', ''))
        except ValueError:
            step = 0
        samples.append({'filename': f, 'url': f'/sample-img/{model_name}/{f}', 'step': step})
    return jsonify({'samples': samples})


@app.route('/sample-img/<model_name>/<filename>')
def serve_sample_img(model_name, filename):
    model_samples_dir = os.path.join(SAMPLES_FOLDER, secure_filename(model_name))
    return send_from_directory(model_samples_dir, filename)


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
