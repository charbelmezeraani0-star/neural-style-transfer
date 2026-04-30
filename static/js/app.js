/* ── Animated background ────────────────────────────────────── */
(function initBackground() {
  const canvas = document.getElementById('bg-canvas');
  const ctx = canvas.getContext('2d');
  let W, H, orbs;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function makeOrb() {
    return {
      x:   Math.random() * W,
      y:   Math.random() * H,
      r:   120 + Math.random() * 200,
      dx:  (Math.random() - .5) * .35,
      dy:  (Math.random() - .5) * .35,
      hue: Math.random() < .5 ? 270 : 320,   // purple or pink
      alpha: .06 + Math.random() * .06,
    };
  }

  function init() {
    resize();
    orbs = Array.from({ length: 6 }, makeOrb);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    for (const o of orbs) {
      const g = ctx.createRadialGradient(o.x, o.y, 0, o.x, o.y, o.r);
      g.addColorStop(0, `hsla(${o.hue},80%,55%,${o.alpha})`);
      g.addColorStop(1, 'transparent');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(o.x, o.y, o.r, 0, Math.PI * 2);
      ctx.fill();

      o.x += o.dx;
      o.y += o.dy;
      if (o.x < -o.r || o.x > W + o.r) o.dx *= -1;
      if (o.y < -o.r || o.y > H + o.r) o.dy *= -1;
    }
    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  init();
  draw();
})();


/* ── State ──────────────────────────────────────────────────── */
const state = {
  contentFile: null,
  styleFile:   null,
  contentDataURL: null,
  styleDataURL:   null,
};


/* ── DOM refs ───────────────────────────────────────────────── */
const contentZone    = document.getElementById('content-zone');
const styleZone      = document.getElementById('style-zone');
const contentInput   = document.getElementById('content-input');
const styleInput     = document.getElementById('style-input');
const contentPreview = document.getElementById('content-preview');
const stylePreview   = document.getElementById('style-preview');
const contentImg     = document.getElementById('content-img');
const styleImg       = document.getElementById('style-img');

const runBtn         = document.getElementById('run-btn');
const runBtnText     = document.querySelector('.run-btn-text');
const runBtnSpinner  = document.querySelector('.run-btn-spinner');

const settingsToggle = document.getElementById('settings-toggle');
const settingsBody   = document.getElementById('settings-body');

const stepsSlider    = document.getElementById('steps-slider');
const styleSlider    = document.getElementById('style-slider');
const contentSlider  = document.getElementById('content-slider');
const stepsVal       = document.getElementById('steps-val');
const styleVal       = document.getElementById('style-val');
const contentVal     = document.getElementById('content-val');

const progressPanel  = document.getElementById('progress-panel');
const progressLabel  = document.getElementById('progress-label');
const progressPct    = document.getElementById('progress-pct');
const progressFill   = document.getElementById('progress-fill');
const styleLossVal   = document.getElementById('style-loss-val');
const contentLossVal = document.getElementById('content-loss-val');

const resultPanel    = document.getElementById('result-panel');
const resultImg      = document.getElementById('result-img');
const resultContentThumb = document.getElementById('result-content-thumb');
const resultStyleThumb   = document.getElementById('result-style-thumb');
const downloadBtn    = document.getElementById('download-btn');

const errorPanel     = document.getElementById('error-panel');
const errorMsg       = document.getElementById('error-msg');

const againBtn       = document.getElementById('again-btn');
const errorAgainBtn  = document.getElementById('error-again-btn');


/* ── Settings accordion ─────────────────────────────────────── */
settingsToggle.addEventListener('click', () => {
  const open = settingsBody.classList.toggle('open');
  settingsToggle.classList.toggle('open', open);
});


/* ── Slider labels ──────────────────────────────────────────── */
const styleWeightMap = [1e4, 5e5, 1e6, 5e6, 1e7, 2e7, 3e7, 5e7, 7.5e7, 1e8];
const styleWeightLabels = ['10K','500K','1M','5M','10M','20M','30M','50M','75M','100M'];

stepsSlider.addEventListener('input', () => {
  stepsVal.textContent = stepsSlider.value;
});
styleSlider.addEventListener('input', () => {
  const idx = parseInt(styleSlider.value) - 1;
  styleVal.textContent = styleWeightLabels[idx];
});
contentSlider.addEventListener('input', () => {
  contentVal.textContent = contentSlider.value;
});


/* ── Drop zone wiring ───────────────────────────────────────── */
function wireDropZone(zone, input, type) {
  zone.addEventListener('click', (e) => {
    if (e.target.closest('.remove-btn')) return;
    input.click();
  });

  input.addEventListener('change', () => {
    if (input.files[0]) setFile(type, input.files[0]);
  });

  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) setFile(type, file);
  });
}

wireDropZone(contentZone, contentInput, 'content');
wireDropZone(styleZone,   styleInput,   'style');


/* ── Remove buttons ─────────────────────────────────────────── */
document.querySelectorAll('.remove-btn').forEach(btn => {
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile(btn.dataset.for);
  });
});


/* ── File handling ──────────────────────────────────────────── */
function setFile(type, file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const dataURL = e.target.result;
    if (type === 'content') {
      state.contentFile    = file;
      state.contentDataURL = dataURL;
      contentImg.src = dataURL;
      contentZone.classList.add('has-image');
    } else {
      state.styleFile    = file;
      state.styleDataURL = dataURL;
      styleImg.src = dataURL;
      styleZone.classList.add('has-image');
    }
    updateRunBtn();
  };
  reader.readAsDataURL(file);
}

function clearFile(type) {
  if (type === 'content') {
    state.contentFile = state.contentDataURL = null;
    contentInput.value = '';
    contentImg.src = '';
    contentZone.classList.remove('has-image');
  } else {
    state.styleFile = state.styleDataURL = null;
    styleInput.value = '';
    styleImg.src = '';
    styleZone.classList.remove('has-image');
  }
  updateRunBtn();
}

function updateRunBtn() {
  runBtn.disabled = !(state.contentFile && state.styleFile);
}


/* ── Run ────────────────────────────────────────────────────── */
runBtn.addEventListener('click', async () => {
  if (!state.contentFile || !state.styleFile) return;

  // Build form data
  const fd = new FormData();
  fd.append('content', state.contentFile);
  fd.append('style',   state.styleFile);
  fd.append('num_steps',      stepsSlider.value);

  const swIdx = parseInt(styleSlider.value) - 1;
  fd.append('style_weight',   styleWeightMap[swIdx]);
  fd.append('content_weight', contentSlider.value);

  // UI: loading state
  setLoading(true);
  showPanel('progress');
  progressFill.style.width = '0%';
  progressPct.textContent  = '0%';
  progressLabel.textContent = 'Sending images…';
  styleLossVal.textContent = contentLossVal.textContent = '—';

  let jobId;
  try {
    const res = await fetch('/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Upload failed');
    jobId = data.job_id;
  } catch (err) {
    setLoading(false);
    showError(err.message);
    return;
  }

  // SSE progress
  const evtSource = new EventSource(`/progress/${jobId}`);

  evtSource.onmessage = (e) => {
    const msg = JSON.parse(e.data);

    if (msg.type === 'progress') {
      const pct = msg.pct;
      progressFill.style.width = pct + '%';
      progressPct.textContent  = pct + '%';
      progressLabel.textContent = `Step ${msg.step} / ${msg.total}`;
      styleLossVal.textContent   = msg.style_loss.toFixed(2);
      contentLossVal.textContent = msg.content_loss.toFixed(4);

    } else if (msg.type === 'done') {
      evtSource.close();
      setLoading(false);

      const outputUrl = `/outputs/${msg.filename}`;
      resultImg.src = outputUrl;
      resultContentThumb.src = state.contentDataURL;
      resultStyleThumb.src   = state.styleDataURL;
      downloadBtn.href = outputUrl;
      showPanel('result');

    } else if (msg.type === 'error') {
      evtSource.close();
      setLoading(false);
      showError(msg.message);
    }
  };

  evtSource.onerror = () => {
    evtSource.close();
    setLoading(false);
    showError('Connection to server lost.');
  };
});


/* ── UI helpers ─────────────────────────────────────────────── */
function setLoading(on) {
  runBtn.disabled = on;
  runBtnText.hidden  = on;
  runBtnSpinner.hidden = !on;
}

function showPanel(which) {
  progressPanel.hidden = which !== 'progress';
  resultPanel.hidden   = which !== 'result';
  errorPanel.hidden    = which !== 'error';
}

function showError(msg) {
  errorMsg.textContent = msg;
  showPanel('error');
}

function resetUI() {
  showPanel('none');
  setLoading(false);
  updateRunBtn();
}

againBtn.addEventListener('click', resetUI);
errorAgainBtn.addEventListener('click', resetUI);
