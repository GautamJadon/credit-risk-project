'use strict';

// const API_BASE = 'http://localhost:8000/api/v1';
const API_BASE = 'https://credit-risk-project-qaya.onrender.com/api/v1';

// ── Live DTI / LTI preview ─────────────────────────────────────────────────
function updateDTI() {
  const income   = parseFloat(document.getElementById('income').value);
  const loan     = parseFloat(document.getElementById('loan_amount').value);
  const annuity  = parseFloat(document.getElementById('annuity').value);
  const preview  = document.getElementById('dtiPreview');
  if (income > 0 && loan > 0 && annuity > 0) {
    const dti = annuity / (income / 12);
    const lti = loan / income;
    document.getElementById('dtiVal').textContent = dti.toFixed(2);
    document.getElementById('ltiVal').textContent = lti.toFixed(2);
    document.getElementById('dtiVal').style.color =
      dti > 0.4 ? '#ef4444' : dti > 0.2 ? '#f59e0b' : '#22c55e';
    preview.style.display = 'block';
  } else {
    preview.style.display = 'none';
  }
}
['income','loan_amount','annuity'].forEach(id =>
  document.getElementById(id).addEventListener('input', updateDTI)
);

// ── Live credit score bars ─────────────────────────────────────────────────
['1','2','3'].forEach(n => {
  document.getElementById(`ext_source_${n}`).addEventListener('input', function() {
    const v = Math.min(Math.max(parseFloat(this.value)||0, 0), 1);
    document.getElementById(`sb${n}`).style.width = (v * 100) + '%';
    const bar = document.getElementById(`sb${n}`);
    bar.style.background = v > 0.65 ? '#22c55e' : v > 0.4 ? '#f59e0b' : '#ef4444';
  });
});

// ── Loading animation ──────────────────────────────────────────────────────
function showLoading() {
  document.getElementById('loadingOverlay').style.display = 'flex';
  const steps = ['ls1','ls2','ls3','ls4'];
  steps.forEach(id => {
    const el = document.getElementById(id);
    el.classList.remove('active','done');
    el.querySelector('i').className = 'bi bi-circle';
  });
  document.getElementById('ls1').classList.add('active');
  document.getElementById('ls1').querySelector('i').className = 'bi bi-arrow-right-circle-fill';
  let cur = 0;
  return setInterval(() => {
    if (cur < steps.length - 1) {
      document.getElementById(steps[cur]).classList.remove('active');
      document.getElementById(steps[cur]).classList.add('done');
      document.getElementById(steps[cur]).querySelector('i').className = 'bi bi-check-circle-fill';
      cur++;
      document.getElementById(steps[cur]).classList.add('active');
      document.getElementById(steps[cur]).querySelector('i').className = 'bi bi-arrow-right-circle-fill';
    }
  }, 400);
}
function hideLoading(timer) {
  clearInterval(timer);
  ['ls1','ls2','ls3','ls4'].forEach(id => {
    document.getElementById(id).classList.add('done');
    document.getElementById(id).querySelector('i').className = 'bi bi-check-circle-fill';
  });
  setTimeout(() => {
    document.getElementById('loadingOverlay').style.display = 'none';
  }, 300);
}

// ── Form submit ────────────────────────────────────────────────────────────
document.getElementById('loanForm').addEventListener('submit', async function(e) {
  e.preventDefault();
  const timer = showLoading();

  const payload = {
    age:            parseInt(document.getElementById('age').value),
    gender:         document.getElementById('gender').value,
    education_type: document.getElementById('education_type').value,
    income_type:    document.getElementById('income_type').value,
    family_status:  document.getElementById('family_status').value,
    housing_type:   document.getElementById('housing_type').value,
    income:         parseFloat(document.getElementById('income').value),
    loan_amount:    parseFloat(document.getElementById('loan_amount').value),
    annuity:        parseFloat(document.getElementById('annuity').value),
    employment_yrs: parseFloat(document.getElementById('employment_yrs').value) || 0,
    ext_source_1:   parseFloat(document.getElementById('ext_source_1').value) || 0.5,
    ext_source_2:   parseFloat(document.getElementById('ext_source_2').value) || 0.5,
    ext_source_3:   parseFloat(document.getElementById('ext_source_3').value) || 0.5,
    family_members: parseInt(document.getElementById('family_members').value) || 2,
  };

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Prediction failed');
    }
    const data = await res.json();
    hideLoading(timer);
    displayResults(data);
  } catch (err) {
    hideLoading(timer);
    alert(`⚠️ Error: ${err.message}\n\nMake sure the FastAPI backend is running:\n  cd backend && uvicorn app.main:app --reload`);
  }
});

// ── Display results ────────────────────────────────────────────────────────
function displayResults(data) {
  document.getElementById('placeholderCard').style.display = 'none';
  document.getElementById('resultSection').style.display   = 'block';
  document.getElementById('resultSection').scrollIntoView({ behavior: 'smooth', block: 'start' });

  const pct   = data.risk_percentage;
  const color = data.risk_color;

  // Badge
  const badge = document.getElementById('riskBadge');
  badge.textContent = data.risk_category;
  badge.style.background = color + '22';
  badge.style.color      = color;
  badge.style.border     = `1.5px solid ${color}55`;

  // Probability
  const probEl = document.getElementById('riskProb');
  probEl.textContent = pct + '%';
  probEl.style.color = color;

  // Recommendation & description
  document.getElementById('riskRec').textContent  = data.recommendation;
  document.getElementById('riskDesc').textContent = data.description;

  // Model tag
  document.getElementById('modelTag').textContent =
    `${data.model_name}  ·  AUC ${data.model_auc}`;

  // Gauge
  drawGauge(data.risk_probability, color);

  // SHAP factors
  renderSHAP(data.top_factors);

  // Model comparison (fetch from /model-info)
  fetch(`${API_BASE}/model-info`)
    .then(r => r.json())
    .then(m => renderModelComparison(m.comparison, data.model_name))
    .catch(() => {});
}

// ── Gauge canvas ───────────────────────────────────────────────────────────
// function drawGauge(prob, color) {
//   const canvas = document.getElementById('gaugeCanvas');
//   const ctx    = canvas.getContext('2d');
//   const W = canvas.width, H = canvas.height;
//   ctx.clearRect(0, 0, W, H);

//   const cx = W / 2, cy = H - 20, r = 80;
//   const startA = Math.PI, endA = 2 * Math.PI;
//   const valueA = startA + prob * Math.PI;

//   // Track segments (low / medium / high)
//   // const segs = [
//   //   [Math.PI,       Math.PI * 1.33, '#16a34a33'],
//   //   [Math.PI * 1.33, Math.PI * 1.67, '#d9770633'],
//   //   [Math.PI * 1.67, Math.PI * 2,   '#dc262633'],
//   // ];

//   const segs = [
//     [Math.PI,        Math.PI * 1.15, '#16a34a33'],  // LOW:    0–15%
//     [Math.PI * 1.15, Math.PI * 1.50, '#d9770633'],  // MEDIUM: 15–50%  
//     [Math.PI * 1.50, Math.PI * 2,    '#dc262633'],  // HIGH:   50–100%
//   ];

//   segs.forEach(([s, e, c]) => {
//     ctx.beginPath(); ctx.arc(cx, cy, r, s, e);
//     ctx.strokeStyle = c; ctx.lineWidth = 20; ctx.stroke();
//   });

//   // Value arc
//   ctx.beginPath(); ctx.arc(cx, cy, r, startA, valueA);
//   ctx.strokeStyle = color; ctx.lineWidth = 20;
//   ctx.lineCap = 'round'; ctx.stroke();

//   // Needle
//   const angle = startA + prob * Math.PI;
//   const nx = cx + (r - 22) * Math.cos(angle);
//   const ny = cy + (r - 22) * Math.sin(angle);
//   ctx.beginPath(); ctx.moveTo(cx, cy);
//   ctx.lineTo(nx, ny);
//   ctx.strokeStyle = '#fff'; ctx.lineWidth = 2.5;
//   ctx.lineCap = 'round'; ctx.stroke();
//   ctx.beginPath(); ctx.arc(cx, cy, 5, 0, 2*Math.PI);
//   ctx.fillStyle = '#fff'; ctx.fill();

//   // Labels
//   ctx.font = '10px system-ui'; ctx.fillStyle = '#94a3b8'; ctx.textAlign = 'center';
//   ctx.fillText('Low', cx - 74, cy + 14);
//   ctx.fillText('Mid', cx, cy - 84);
//   ctx.fillText('High', cx + 74, cy + 14);
// }

function drawGauge(prob, color) {
  const canvas = document.getElementById('gaugeCanvas');
  const ctx    = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2, cy = H - 20, r = 80;

  // // ── Background segments ──────────────────────────────
  // const segs = [
  //   [Math.PI,        Math.PI * 1.15, '#16a34a55'],  // LOW    0–15%
  //   [Math.PI * 1.15, Math.PI * 1.50, '#d9770655'],  // MEDIUM 15–50%
  //   [Math.PI * 1.50, Math.PI * 2,    '#dc262655'],  // HIGH   50–100%
  // ];
  // segs.forEach(([s, e, c]) => {
  //   ctx.beginPath();
  //   ctx.arc(cx, cy, r, s, e);
  //   ctx.strokeStyle = c;
  //   ctx.lineWidth = 20;
  //   ctx.stroke();
  // });

  // ── Background track (dark) ──────────────────────────
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, Math.PI * 2);
  ctx.strokeStyle = '#ffffff11';
  ctx.lineWidth = 22;
  ctx.stroke();

  // ── Coloured segments ────────────────────────────────
  const segs = [
      [Math.PI,        Math.PI * 1.15, '#16a34a77'],  // LOW    0–15%
      [Math.PI * 1.15, Math.PI * 1.60, '#d9770699'],  // MEDIUM 15–60%
      [Math.PI * 1.60, Math.PI * 2,    '#dc262699'],  // HIGH   60–100%
  ];
  segs.forEach(([s, e, c]) => {
      ctx.beginPath();
      ctx.arc(cx, cy, r, s, e);
      ctx.strokeStyle = c;
      ctx.lineWidth = 16;
      ctx.lineCap = 'butt';
      ctx.stroke();
  });

  // ── Needle only (no value arc) ────────────────────────
  const angle = Math.PI + prob * Math.PI;
  const nx = cx + (r - 10) * Math.cos(angle);
  const ny = cy + (r - 10) * Math.sin(angle);
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(nx, ny);
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2.5;
  ctx.lineCap = 'round';
  ctx.stroke();

  // ── Centre dot ───────────────────────────────────────
  ctx.beginPath();
  ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
  ctx.fillStyle = '#fff';
  ctx.fill();

  // ── Labels ───────────────────────────────────────────
  ctx.font = '10px system-ui';
  ctx.fillStyle = '#94a3b8';
  ctx.textAlign = 'center';
  ctx.fillText('Low',  cx - 74, cy + 14);
  ctx.fillText('Mid',  cx,      cy - 84);
  ctx.fillText('High', cx + 74, cy + 14);
}

// ── SHAP factors ───────────────────────────────────────────────────────────
function renderSHAP(factors) {
  const container = document.getElementById('shapFactors');
  container.innerHTML = '';
  const maxAbs = Math.max(...factors.map(f => Math.abs(f.shap_value)), 0.001);

  factors.slice(0, 10).forEach(f => {
    const barPct  = (Math.abs(f.shap_value) / maxAbs * 100).toFixed(1);
    const isInc   = f.direction === 'increases_risk';
    const color   = isInc ? '#ef4444' : '#22c55e';
    const arrow   = isInc ? '▲' : '▼';
    const dirText = isInc ? 'increases risk' : 'decreases risk';

    const div = document.createElement('div');
    div.className = 'shap-factor-row';
    div.innerHTML = `
      <div class="shap-factor-top">
        <span class="shap-factor-name">${f.label}</span>
        <div class="d-flex align-items-center gap-2">
          <span class="shap-factor-val">val=${f.value.toFixed(3)}</span>
          <span class="shap-factor-shap" style="color:${color}">${arrow} ${Math.abs(f.shap_value).toFixed(4)}</span>
        </div>
      </div>
      <div class="shap-bar-track">
        <div class="shap-bar-fill" style="width:${barPct}%;background:${color}77;"></div>
      </div>
      <div class="shap-factor-dir">${dirText}</div>`;
    container.appendChild(div);
  });
}

// ── Model comparison ───────────────────────────────────────────────────────
function renderModelComparison(comparison, bestName) {
  const container = document.getElementById('modelComparison');
  container.innerHTML = '';
  const colors = { 'Logistic Regression':'#3b82f6','Decision Tree':'#f59e0b','Random Forest':'#22c55e','XGBoost':'#f43f5e' };

  const sorted = Object.entries(comparison).sort((a,b) => b[1].auc - a[1].auc);
  const maxAUC = sorted[0][1].auc;

  sorted.forEach(([name, stats]) => {
    const isBest = name === bestName;
    const color  = colors[name] || '#94a3b8';
    const pct    = ((stats.auc / maxAUC) * 100).toFixed(1);

    const row = document.createElement('div');
    row.className = 'model-comp-row';
    row.innerHTML = `
      <div class="model-comp-name">
        ${name}${isBest ? '<span class="model-best-tag">BEST</span>' : ''}
      </div>
      <div class="model-comp-bar-wrap">
        <div class="model-comp-bar" style="width:${pct}%;background:${color}99;"></div>
      </div>
      <div class="model-comp-auc" style="color:${color}">${stats.auc.toFixed(4)}</div>`;
    container.appendChild(row);
  });
}

// ── Reset ──────────────────────────────────────────────────────────────────
function resetForm() {
  document.getElementById('loanForm').reset();
  document.getElementById('fm_val').textContent = '2';
  document.getElementById('dtiPreview').style.display = 'none';
  ['1','2','3'].forEach(n => {
    document.getElementById(`sb${n}`).style.width = '65%';
    document.getElementById(`sb${n}`).style.background = '#3b82f6';
  });
  document.getElementById('placeholderCard').style.display = 'flex';
  document.getElementById('resultSection').style.display   = 'none';
}
