/* ── Animated background ─────────────────────────────────────── */
(function () {
  const canvas = document.getElementById('bg-canvas');
  const ctx = canvas.getContext('2d');
  let W, H, orbs;

  function resize() { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }

  function makeOrb() {
    return {
      x: Math.random() * W, y: Math.random() * H,
      r: 100 + Math.random() * 220,
      dx: (Math.random() - .5) * .3, dy: (Math.random() - .5) * .3,
      hue: Math.random() < .5 ? 270 : 320,
      alpha: .045 + Math.random() * .045,
    };
  }

  function init() { resize(); orbs = Array.from({ length: 7 }, makeOrb); }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    for (const o of orbs) {
      const g = ctx.createRadialGradient(o.x, o.y, 0, o.x, o.y, o.r);
      g.addColorStop(0, `hsla(${o.hue},75%,55%,${o.alpha})`);
      g.addColorStop(1, 'transparent');
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(o.x, o.y, o.r, 0, Math.PI * 2); ctx.fill();
      o.x += o.dx; o.y += o.dy;
      if (o.x < -o.r || o.x > W + o.r) o.dx *= -1;
      if (o.y < -o.r || o.y > H + o.r) o.dy *= -1;
    }
    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  init(); draw();
})();


/* ── Toast ───────────────────────────────────────────────────── */
function toast(message, type = 'success') {
  const container = document.getElementById('toast-container');
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  const icon = type === 'success'
    ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>'
    : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>';
  el.innerHTML = `${icon}<span>${message}</span>`;
  container.appendChild(el);
  setTimeout(() => {
    el.style.animation = 'toastOut .3s ease forwards';
    el.addEventListener('animationend', () => el.remove());
  }, 3200);
}


/* ── State ───────────────────────────────────────────────────── */
const state = {
  mode: 'classic',
  contentFile: null, styleFile: null,
  contentDataURL: null, styleDataURL: null,
  fastContentFile: null, fastContentDataURL: null,
  selectedModel: null,
  currentOutputUrl: null,
};


/* ── DOM refs ────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);

const modeTabs          = document.querySelectorAll('.mode-tab');
const classicPanel      = $('classic-panel');
const fastPanel         = $('fast-panel');
const modeBadge         = $('mode-badge');

const contentZone       = $('content-zone');
const styleZone         = $('style-zone');
const contentInput      = $('content-input');
const styleInput        = $('style-input');
const contentImg        = $('content-img');
const styleImg          = $('style-img');

const fastContentZone   = $('fast-content-zone');
const fastContentInput  = $('fast-content-input');
const fastContentImg    = $('fast-content-img');
const modelList         = $('model-list');
const noModels          = $('no-models');

const samplesViewer     = $('samples-viewer');
const samplesStrip      = $('samples-strip');
const samplesClose      = $('samples-close');

const runBtn            = $('run-btn');
const runBtnText        = document.querySelector('.run-btn-text');
const runBtnSpinner     = document.querySelector('.run-btn-spinner');

const settingsToggle    = $('settings-toggle');
const settingsBody      = $('settings-body');
const stepsSlider       = $('steps-slider');
const styleSlider       = $('style-slider');
const contentSlider     = $('content-slider');
const stepsVal          = $('steps-val');
const styleVal          = $('style-val');
const contentVal        = $('content-val');

const progressPanel     = $('progress-panel');
const progressLabel     = $('progress-label');
const progressPct       = $('progress-pct');
const progressFill      = $('progress-fill');
const styleLossVal      = $('style-loss-val');
const contentLossVal    = $('content-loss-val');

const resultPanel       = $('result-panel');
const resultImg         = $('result-img');
const resultContentThumb = $('result-content-thumb');
const resultStyleThumb  = $('result-style-thumb');
const styleRef          = $('style-ref');
const downloadBtn       = $('download-btn');
const fullscreenBtn     = $('fullscreen-btn');
const againBtn          = $('again-btn');

const compareRange      = $('compare-range');
const compareAfter      = document.querySelector('.compare-after');
const compareHandle     = $('compare-handle');

const fsViewer          = $('fs-viewer');
const fsImg             = $('fs-img');
const fsDownload        = $('fs-download');
const fsClose           = $('fs-close');

const gallerySection    = $('gallery-section');
const galleryGrid       = $('gallery-grid');
const galleryRefresh    = $('gallery-refresh');

const errorPanel        = $('error-panel');
const errorMsg          = $('error-msg');
const errorAgainBtn     = $('error-again-btn');


/* ── Comparison slider ───────────────────────────────────────── */
function setCompare(val) {
  compareAfter.style.clipPath = `inset(0 ${100 - val}% 0 0)`;
  compareHandle.style.left    = val + '%';
}
compareRange.addEventListener('input', () => setCompare(compareRange.value));
setCompare(50);


/* ── Fullscreen viewer ───────────────────────────────────────── */
fullscreenBtn.addEventListener('click', () => {
  if (!state.currentOutputUrl) return;
  fsImg.src = state.currentOutputUrl;
  fsDownload.href = state.currentOutputUrl;
  fsViewer.hidden = false;
  document.body.style.overflow = 'hidden';
});

fsClose.addEventListener('click', closeFs);
fsViewer.addEventListener('click', e => { if (e.target === fsViewer) closeFs(); });

function closeFs() {
  fsViewer.hidden = true;
  document.body.style.overflow = '';
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') closeFs(); });


/* ── Mode switching ──────────────────────────────────────────── */
const modeMeta = {
  classic: { badge: 'any style · ~5 min' },
  fast:    { badge: 'trained model · instant' },
};

modeTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    state.mode = tab.dataset.mode;
    modeTabs.forEach(t => t.classList.toggle('active', t === tab));
    classicPanel.hidden = state.mode !== 'classic';
    fastPanel.hidden    = state.mode !== 'fast';
    modeBadge.textContent = modeMeta[state.mode].badge;
    showPanel('none');
    updateRunBtn();
  });
});


/* ── Settings ────────────────────────────────────────────────── */
settingsToggle.addEventListener('click', () => {
  const open = settingsBody.classList.toggle('open');
  settingsToggle.classList.toggle('open', open);
});


/* ── Sliders ─────────────────────────────────────────────────── */
const styleWeightMap    = [1e4, 5e5, 1e6, 5e6, 1e7, 2e7, 3e7, 5e7, 7.5e7, 1e8];
const styleWeightLabels = ['10K','500K','1M','5M','10M','20M','30M','50M','75M','100M'];

stepsSlider.addEventListener('input',   () => stepsVal.textContent   = stepsSlider.value);
styleSlider.addEventListener('input',   () => styleVal.textContent   = styleWeightLabels[+styleSlider.value - 1]);
contentSlider.addEventListener('input', () => contentVal.textContent = contentSlider.value);


/* ── Drop zones — Classic ────────────────────────────────────── */
wireDropZone(contentZone, contentInput, 'content');
wireDropZone(styleZone,   styleInput,   'style');

document.querySelectorAll('.remove-btn').forEach(btn => {
  btn.addEventListener('click', e => { e.stopPropagation(); clearFile(btn.dataset.for); });
});

function wireDropZone(zone, input, type) {
  zone.addEventListener('click', e => { if (!e.target.closest('.remove-btn')) input.click(); });
  input.addEventListener('change', () => { if (input.files[0]) setFile(type, input.files[0]); });
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f?.type.startsWith('image/')) setFile(type, f);
  });
}

function setFile(type, file) {
  readAsDataURL(file, dataURL => {
    if (type === 'content') {
      state.contentFile = file; state.contentDataURL = dataURL;
      contentImg.src = dataURL; contentZone.classList.add('has-image');
    } else {
      state.styleFile = file; state.styleDataURL = dataURL;
      styleImg.src = dataURL; styleZone.classList.add('has-image');
    }
    updateRunBtn();
  });
}

function clearFile(type) {
  if (type === 'content') {
    state.contentFile = state.contentDataURL = null;
    contentInput.value = ''; contentImg.src = '';
    contentZone.classList.remove('has-image');
  } else if (type === 'style') {
    state.styleFile = state.styleDataURL = null;
    styleInput.value = ''; styleImg.src = '';
    styleZone.classList.remove('has-image');
  } else if (type === 'fast-content') {
    state.fastContentFile = state.fastContentDataURL = null;
    fastContentInput.value = ''; fastContentImg.src = '';
    fastContentZone.classList.remove('has-image');
  }
  updateRunBtn();
}


/* ── Drop zone — Fast ────────────────────────────────────────── */
fastContentZone.addEventListener('click', e => { if (!e.target.closest('.remove-btn')) fastContentInput.click(); });
fastContentInput.addEventListener('change', () => { if (fastContentInput.files[0]) setFastContent(fastContentInput.files[0]); });
fastContentZone.addEventListener('dragover', e => { e.preventDefault(); fastContentZone.classList.add('drag-over'); });
fastContentZone.addEventListener('dragleave', () => fastContentZone.classList.remove('drag-over'));
fastContentZone.addEventListener('drop', e => {
  e.preventDefault(); fastContentZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f?.type.startsWith('image/')) setFastContent(f);
});

function setFastContent(file) {
  readAsDataURL(file, dataURL => {
    state.fastContentFile = file; state.fastContentDataURL = dataURL;
    fastContentImg.src = dataURL; fastContentZone.classList.add('has-image');
    updateRunBtn();
  });
}


/* ── Model list ──────────────────────────────────────────────── */
async function loadModels() {
  try {
    const res  = await fetch('/models');
    const data = await res.json();
    renderModelList(data.models || []);
  } catch (_) {
    renderModelList([]);
  }
}

function renderModelList(models) {
  modelList.innerHTML = '';
  if (models.length === 0) { noModels.hidden = false; return; }
  noModels.hidden = true;

  models.forEach(m => {
    const btn = document.createElement('button');
    btn.className = 'model-chip';
    btn.dataset.name = m.name;

    const thumb = m.thumbnail
      ? `<img class="model-chip-thumb" src="${m.thumbnail}" alt="${m.name}" />`
      : `<svg class="model-chip-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>`;

    btn.innerHTML = `
      ${thumb}
      <div class="model-chip-info">
        <span class="model-chip-name">${m.name}</span>
        <span class="model-chip-sub">Fast NST model</span>
      </div>`;

    btn.addEventListener('click', () => selectModel(m.name));
    modelList.appendChild(btn);
  });
}

function selectModel(name) {
  state.selectedModel = name;
  document.querySelectorAll('.model-chip').forEach(c => {
    c.classList.toggle('selected', c.dataset.name === name);
  });
  updateRunBtn();
  loadSamples(name);
}

loadModels();


/* ── Training samples ────────────────────────────────────────── */
async function loadSamples(modelName) {
  try {
    const res  = await fetch(`/samples/${encodeURIComponent(modelName)}`);
    const data = await res.json();
    if (!data.samples || data.samples.length === 0) { samplesViewer.hidden = true; return; }
    samplesStrip.innerHTML = '';
    data.samples.forEach(s => {
      const img = document.createElement('img');
      img.className = 'sample-thumb';
      img.src  = s.url;
      img.alt  = `Step ${s.step}`;
      img.title = `Step ${s.step}`;
      img.addEventListener('click', () => {
        fsImg.src = s.url; fsDownload.href = s.url;
        fsViewer.hidden = false; document.body.style.overflow = 'hidden';
      });
      samplesStrip.appendChild(img);
    });
    samplesViewer.hidden = false;
  } catch (_) {
    samplesViewer.hidden = true;
  }
}

samplesClose.addEventListener('click', () => { samplesViewer.hidden = true; });


/* ── Run button state ────────────────────────────────────────── */
function updateRunBtn() {
  if (state.mode === 'classic') {
    runBtn.disabled = !(state.contentFile && state.styleFile);
  } else {
    runBtn.disabled = !(state.fastContentFile && state.selectedModel);
  }
}


/* ── Run ─────────────────────────────────────────────────────── */
runBtn.addEventListener('click', () => {
  if (state.mode === 'classic') runClassic();
  else runFast();
});


async function runClassic() {
  const fd = new FormData();
  fd.append('content',        state.contentFile);
  fd.append('style',          state.styleFile);
  fd.append('num_steps',      stepsSlider.value);
  fd.append('style_weight',   styleWeightMap[+styleSlider.value - 1]);
  fd.append('content_weight', contentSlider.value);

  setLoading(true);
  showPanel('progress');
  progressFill.style.width  = '0%';
  progressPct.textContent   = '0%';
  progressLabel.textContent = 'Sending images…';
  styleLossVal.textContent  = contentLossVal.textContent = '—';

  let jobId;
  try {
    const res  = await fetch('/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Upload failed');
    jobId = data.job_id;
  } catch (err) {
    setLoading(false); showError(err.message); return;
  }

  const evtSource = new EventSource(`/progress/${jobId}`);
  evtSource.onmessage = e => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'progress') {
      progressFill.style.width  = msg.pct + '%';
      progressPct.textContent   = msg.pct + '%';
      progressLabel.textContent = `Step ${msg.step} / ${msg.total}`;
      styleLossVal.textContent   = msg.style_loss.toFixed(2);
      contentLossVal.textContent = msg.content_loss.toFixed(4);
    } else if (msg.type === 'done') {
      evtSource.close(); setLoading(false);
      showResult(`/outputs/${msg.filename}`, state.contentDataURL, state.styleDataURL);
      toast('Stylization complete!');
      loadGallery();
    } else if (msg.type === 'error') {
      evtSource.close(); setLoading(false); showError(msg.message);
      toast(msg.message, 'error');
    }
  };
  evtSource.onerror = () => {
    evtSource.close(); setLoading(false); showError('Connection lost.');
  };
}


async function runFast() {
  const fd = new FormData();
  fd.append('content', state.fastContentFile);
  fd.append('model',   state.selectedModel);

  setLoading(true);
  try {
    const res  = await fetch('/fast-stylize', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Stylization failed');
    setLoading(false);
    showResult(`/outputs/${data.filename}`, state.fastContentDataURL, null);
    toast('Stylization complete!');
    loadGallery();
  } catch (err) {
    setLoading(false); showError(err.message);
    toast(err.message, 'error');
  }
}


/* ── Gallery ─────────────────────────────────────────────────── */
async function loadGallery() {
  try {
    const res  = await fetch('/gallery');
    const data = await res.json();
    const items = data.items || [];
    if (items.length === 0) { gallerySection.hidden = true; return; }

    galleryGrid.innerHTML = '';
    items.forEach(item => {
      const div = document.createElement('div');
      div.className = 'gallery-item';

      const badge = item.style_name
        ? `<span class="gallery-item-badge fast">${item.style_name}</span>`
        : `<span class="gallery-item-badge ${item.mode}">${item.mode}</span>`;

      div.innerHTML = `<img src="${item.url}" alt="result" loading="lazy" />${badge}`;
      div.addEventListener('click', () => {
        fsImg.src = item.url; fsDownload.href = item.url;
        fsViewer.hidden = false; document.body.style.overflow = 'hidden';
      });
      galleryGrid.appendChild(div);
    });

    gallerySection.hidden = false;
  } catch (_) {
    gallerySection.hidden = true;
  }
}

galleryRefresh.addEventListener('click', () => {
  loadGallery();
  toast('Gallery refreshed');
});

loadGallery();


/* ── UI helpers ──────────────────────────────────────────────── */
function setLoading(on) {
  runBtn.disabled      = on;
  runBtnText.hidden    = on;
  runBtnSpinner.hidden = !on;
}

function showPanel(which) {
  progressPanel.hidden = which !== 'progress';
  resultPanel.hidden   = which !== 'result';
  errorPanel.hidden    = which !== 'error';
}

function showResult(outputUrl, contentDataURL, styleDataURL) {
  state.currentOutputUrl = outputUrl;

  // Comparison slider images
  resultImg.src          = outputUrl;
  resultContentThumb.src = contentDataURL || outputUrl;
  downloadBtn.href       = outputUrl;

  // Reset slider
  compareRange.value = 50;
  setCompare(50);

  // Style reference thumbnail (classic only)
  if (styleDataURL) {
    resultStyleThumb.src  = styleDataURL;
    styleRef.hidden       = false;
  } else {
    styleRef.hidden = true;
  }

  showPanel('result');

  // Scroll result into view smoothly
  setTimeout(() => resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
}

function showError(msg) {
  errorMsg.textContent = msg;
  showPanel('error');
}

function resetUI() {
  showPanel('none');
  setLoading(false);
  updateRunBtn();
  state.currentOutputUrl = null;
  if (state.mode === 'fast') loadModels();
}

againBtn.addEventListener('click', resetUI);
errorAgainBtn.addEventListener('click', resetUI);


/* ── Utility ─────────────────────────────────────────────────── */
function readAsDataURL(file, cb) {
  const r = new FileReader();
  r.onload = e => cb(e.target.result);
  r.readAsDataURL(file);
}
