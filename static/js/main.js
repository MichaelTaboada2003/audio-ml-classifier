// Global variables
let mediaRecorder = null;
let audioChunks = [];
let recordInterval = null;
let recordDuration = 0;
let isRecording = false;
let datasetLoaded = false;
let currentPlayingBtn = null;
const globalAudio = new Audio();
const appConfig = window.APP_CONFIG || { activeClasses: [], classStyles: {} };
const classStyles = appConfig.classStyles || {};
const experimentHistory = appConfig.experimentHistory || { experiments: [], default_experiment_id: null };

// DOM Elements
const tabButtons = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const recordBtn = document.getElementById('record-btn');
const recordTimer = document.getElementById('record-timer');
const recordingStatus = document.getElementById('recording-status');
const waveformAnim = document.getElementById('waveform-anim');
const recordInstruction = document.getElementById('record-instruction');
const loader = document.getElementById('loader');

// Results elements
const emptyResults = document.getElementById('empty-results');
const activeResults = document.getElementById('active-results');
const predictedEmotionBadge = document.getElementById('predicted-emotion-badge');
const predictionPercentage = document.getElementById('prediction-percentage');
const progressCircle = document.getElementById('progress-bar-circle');
const probBars = document.getElementById('prob-bars');

// Initialize SVG circular progress bar circumference
const progressCircumference = 2 * Math.PI * 64; // r=64 -> 402.12
if (progressCircle) {
    progressCircle.style.strokeDasharray = `${progressCircumference}`;
    progressCircle.style.strokeDashoffset = `${progressCircumference}`;
}
const metricDuration = document.getElementById('metric-duration');
const metricPitch = document.getElementById('metric-pitch');
const metricEnergy = document.getElementById('metric-energy');
const metricZcr = document.getElementById('metric-zcr');
const experimentOptions = document.querySelectorAll('.experiment-option');

function getExperimentById(experimentId) {
    return (experimentHistory.experiments || []).find((exp) => exp.id === experimentId);
}

function formatPercent(value) {
    return `${(value * 100).toFixed(1)}%`;
}

// =========================================================================
// TOMA DE DECISIONES - Modulo completo
// =========================================================================
const decisionsData = appConfig.decisionsData || null;

const decisionState = {
    scenarioId: '3clases',
    modelKey: 'svm_lineal',
    threshold: 0.50,
    targetClass: 'Enojo',
    costs: { TP: 25, FP: 4, FN: 80, TN: 0 },
    business: { volume: 10000, prevalence: 0.18, inferenceCost: 0.02, fixedCost: 1200 }
};

function fmtUSD(value) {
    const sign = value < 0 ? '-' : '';
    const abs = Math.abs(value);
    if (abs >= 1000) return `${sign}USD ${(abs / 1000).toFixed(abs >= 10000 ? 1 : 2)}k`;
    return `${sign}USD ${abs.toFixed(0)}`;
}

function fmtNum(value) {
    return value.toLocaleString('es-CO');
}

// Recalcula la matriz de confusion binaria (target vs resto) para un umbral dado
function computeBinaryCM(scenario, modelKey, threshold) {
    const model = scenario.models[modelKey];
    if (!model) return null;
    const classesOrder = scenario.classes;
    const targetIdx = classesOrder.indexOf(decisionState.targetClass);
    if (targetIdx < 0) return null;

    let TP = 0, FP = 0, FN = 0, TN = 0;
    model.loo_probs.forEach((probs, i) => {
        const trueLabel = model.loo_true[i];
        const isPositive = trueLabel === decisionState.targetClass;
        const scoreTarget = probs[targetIdx];
        const predPositive = scoreTarget >= threshold;
        if (isPositive && predPositive) TP++;
        else if (isPositive && !predPositive) FN++;
        else if (!isPositive && predPositive) FP++;
        else TN++;
    });

    const tpr = TP + FN > 0 ? TP / (TP + FN) : 0;
    const fpr = FP + TN > 0 ? FP / (FP + TN) : 0;
    const precision = TP + FP > 0 ? TP / (TP + FP) : 0;
    const f1 = (precision + tpr) > 0 ? 2 * precision * tpr / (precision + tpr) : 0;
    const balanced = (tpr + (TN + FP > 0 ? TN / (TN + FP) : 0)) / 2;

    return { TP, FP, FN, TN, tpr, fpr, precision, recall: tpr, f1, balanced };
}

// Valor neto esperado mensual dadas tasas TPR/FPR y costos
function computeMonthlyValue(cm, costs, biz) {
    const { TP, FP, FN, TN } = cm;
    const total = TP + FP + FN + TN;
    if (total === 0) return null;
    // Tasas condicionales
    const pos = TP + FN;
    const neg = FP + TN;
    const tpr = pos > 0 ? TP / pos : 0;
    const fnr = pos > 0 ? FN / pos : 0;
    const fpr = neg > 0 ? FP / neg : 0;
    const tnr = neg > 0 ? TN / neg : 0;

    // Llamadas esperadas al mes
    const volPositive = biz.volume * biz.prevalence;
    const volNegative = biz.volume * (1 - biz.prevalence);
    const eTP = volPositive * tpr;
    const eFN = volPositive * fnr;
    const eFP = volNegative * fpr;
    const eTN = volNegative * tnr;

    const benefit = eTP * costs.TP;
    const costFP = eFP * costs.FP;
    const costFN = eFN * costs.FN;
    const costTN = eTN * costs.TN;
    const costInfer = biz.volume * biz.inferenceCost;
    const costFixed = biz.fixedCost;

    const net = benefit - costFP - costFN - costTN - costInfer - costFixed;

    return {
        net,
        breakdown: {
            benefitTP: benefit,
            costFP,
            costFN,
            costTN,
            costInfer,
            costFixed
        },
        expected: { eTP, eFP, eFN, eTN }
    };
}

// Busca el umbral que maximiza el VPN esperado para un escenario+modelo
function findOptimalThreshold(scenario, modelKey, costs, biz) {
    let bestThr = 0.5;
    let bestVPN = -Infinity;
    for (let t = 0.05; t <= 0.95; t += 0.01) {
        const cm = computeBinaryCM(scenario, modelKey, t);
        if (!cm) continue;
        const v = computeMonthlyValue(cm, costs, biz);
        if (v && v.net > bestVPN) {
            bestVPN = v.net;
            bestThr = t;
        }
    }
    return { threshold: parseFloat(bestThr.toFixed(2)), vpn: bestVPN };
}

// Prevalencia minima para que el VPN sea >= 0 (break-even)
function findBreakEvenPrevalence(scenario, modelKey, costs, biz, threshold) {
    const cm = computeBinaryCM(scenario, modelKey, threshold);
    if (!cm) return null;
    const pos = cm.TP + cm.FN;
    const neg = cm.FP + cm.TN;
    const tpr = pos > 0 ? cm.TP / pos : 0;
    const fnr = pos > 0 ? cm.FN / pos : 0;
    const fpr = neg > 0 ? cm.FP / neg : 0;
    const tnr = neg > 0 ? cm.TN / neg : 0;
    // VPN(p) = V * [ p*(tpr*TP - fnr*FN) + (1-p)*(- fpr*FP - tnr*TN) ] - V*infer - fixed = 0
    const a = (tpr * costs.TP - fnr * costs.FN);
    const b = (- fpr * costs.FP - tnr * costs.TN);
    // (a - b) * p + b - infer - fixed/V = 0
    const denom = (a - b) * biz.volume;
    if (Math.abs(denom) < 1e-9) return null;
    const p = (biz.volume * (biz.inferenceCost - b) + biz.fixedCost) / denom;
    if (p < 0) return 0;
    if (p > 1) return null;
    return p;
}

// ============== Render: matriz de confusion en vivo ==============
function renderConfusionMatrix(cm) {
    const wrap = document.getElementById('confusion-matrix');
    const stats = document.getElementById('confusion-stats');
    if (!wrap) return;

    const cellHTML = (label, value, klass, sub) => `
        <div class="cm-cell ${klass}">
            <span class="cm-cell-label">${label}</span>
            <span class="cm-cell-value">${value}</span>
            ${sub ? `<span class="cm-cell-sub">${sub}</span>` : ''}
        </div>
    `;

    wrap.innerHTML = `
        <div class="cm-headers">
            <div class="cm-axis-y">Real</div>
            <div class="cm-axis-x">Predicción</div>
            <div class="cm-col-headers">
                <span class="cm-pos">Enojo</span>
                <span class="cm-neg">No-Enojo</span>
            </div>
            <div class="cm-row-headers">
                <span class="cm-pos">Enojo</span>
                <span class="cm-neg">No-Enojo</span>
            </div>
            <div class="cm-cells">
                ${cellHTML('TP', cm.TP, 'cm-tp', 'Escalación correcta')}
                ${cellHTML('FN', cm.FN, 'cm-fn', 'Enojo no detectado')}
                ${cellHTML('FP', cm.FP, 'cm-fp', 'Falsa alarma')}
                ${cellHTML('TN', cm.TN, 'cm-tn', 'No-escalación correcta')}
            </div>
        </div>
    `;

    if (stats) {
        stats.innerHTML = `
            <div class="cm-stat"><span>Recall (TPR)</span><strong>${formatPercent(cm.recall)}</strong></div>
            <div class="cm-stat"><span>Precision</span><strong>${formatPercent(cm.precision)}</strong></div>
            <div class="cm-stat"><span>F1</span><strong>${(cm.f1 * 100).toFixed(1)}%</strong></div>
            <div class="cm-stat"><span>FPR</span><strong>${formatPercent(cm.fpr)}</strong></div>
        `;
    }
}

// ============== Render: curva ROC ==============
function renderROC(scenario, modelKey, currentThr, optimalThr) {
    const svg = document.getElementById('roc-curve');
    if (!svg) return;
    const model = scenario.models[modelKey];
    if (!model || !model.roc) {
        svg.innerHTML = '<text x="160" y="120" fill="#94a3b8" text-anchor="middle" font-size="12">Curva ROC no disponible</text>';
        return;
    }
    const W = 320, H = 240, PAD_L = 40, PAD_B = 30, PAD_T = 12, PAD_R = 12;
    const innerW = W - PAD_L - PAD_R;
    const innerH = H - PAD_B - PAD_T;
    const xs = model.roc.fpr.map(v => PAD_L + v * innerW);
    const ys = model.roc.tpr.map(v => H - PAD_B - v * innerH);
    const path = xs.map((x, i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(' ');

    // AUC aproximada (trapecio)
    let auc = 0;
    for (let i = 1; i < model.roc.fpr.length; i++) {
        auc += (model.roc.fpr[i] - model.roc.fpr[i - 1]) * (model.roc.tpr[i] + model.roc.tpr[i - 1]) / 2;
    }
    auc = Math.abs(auc);

    // Punto actual y optimo
    const cmA = computeBinaryCM(scenario, modelKey, currentThr);
    const cmO = computeBinaryCM(scenario, modelKey, optimalThr);
    const ptA = cmA ? [PAD_L + cmA.fpr * innerW, H - PAD_B - cmA.tpr * innerH] : null;
    const ptO = cmO ? [PAD_L + cmO.fpr * innerW, H - PAD_B - cmO.tpr * innerH] : null;

    let gridLines = '';
    for (let g = 0; g <= 4; g++) {
        const x = PAD_L + (g / 4) * innerW;
        const y = H - PAD_B - (g / 4) * innerH;
        gridLines += `<line x1="${x}" y1="${PAD_T}" x2="${x}" y2="${H - PAD_B}" stroke="rgba(255,255,255,0.05)"/>`;
        gridLines += `<line x1="${PAD_L}" y1="${y}" x2="${W - PAD_R}" y2="${y}" stroke="rgba(255,255,255,0.05)"/>`;
    }

    svg.innerHTML = `
        ${gridLines}
        <line x1="${PAD_L}" y1="${H - PAD_B}" x2="${W - PAD_R}" y2="${PAD_T}" stroke="rgba(255,255,255,0.18)" stroke-dasharray="3 3"/>
        <path d="${path}" fill="none" stroke="url(#roc-grad)" stroke-width="2.4"/>
        <defs>
            <linearGradient id="roc-grad" x1="0" y1="1" x2="1" y2="0">
                <stop offset="0" stop-color="#6366f1"/>
                <stop offset="1" stop-color="#22d3ee"/>
            </linearGradient>
        </defs>
        ${ptA ? `<circle cx="${ptA[0]}" cy="${ptA[1]}" r="6" fill="#22c55e" stroke="#0f172a" stroke-width="2"/>` : ''}
        ${ptO ? `<text x="${ptO[0]}" y="${ptO[1] - 8}" fill="#fbbf24" text-anchor="middle" font-size="14">★</text>` : ''}
        <text x="${PAD_L}" y="${H - 6}" fill="#94a3b8" font-size="10">0</text>
        <text x="${W - PAD_R}" y="${H - 6}" fill="#94a3b8" font-size="10" text-anchor="end">FPR=1</text>
        <text x="6" y="${H - PAD_B}" fill="#94a3b8" font-size="10">0</text>
        <text x="6" y="${PAD_T + 8}" fill="#94a3b8" font-size="10">TPR=1</text>
        <text x="${W - PAD_R - 4}" y="${PAD_T + 12}" fill="#cbd5e1" font-size="11" text-anchor="end">AUC=${auc.toFixed(2)}</text>
    `;
}

// ============== Render: VPN y desglose ==============
function renderVPN(cm, value) {
    const el = document.getElementById('vpn-value');
    const baseline = document.getElementById('vpn-vs-baseline');
    const breakdown = document.getElementById('vpn-breakdown');
    if (!el || !value) return;

    el.textContent = fmtUSD(value.net);
    el.style.color = value.net >= 0 ? '#22c55e' : '#ef4444';

    if (baseline) {
        baseline.textContent = value.net >= 0
            ? `+${fmtUSD(value.net)} vs no desplegar (status quo = USD 0)`
            : `${fmtUSD(value.net)} vs no desplegar — perderías dinero`;
    }

    if (breakdown) {
        const b = value.breakdown;
        breakdown.innerHTML = `
            <div class="vpn-row vpn-pos"><span>+ Beneficio TP (${cm.TP} casos LOO)</span><strong>+${fmtUSD(b.benefitTP)}</strong></div>
            <div class="vpn-row vpn-neg"><span>− Costo FN (${cm.FN} casos LOO)</span><strong>−${fmtUSD(b.costFN)}</strong></div>
            <div class="vpn-row vpn-neg"><span>− Costo FP (${cm.FP} casos LOO)</span><strong>−${fmtUSD(b.costFP)}</strong></div>
            <div class="vpn-row vpn-neg"><span>− Costo inferencia</span><strong>−${fmtUSD(b.costInfer)}</strong></div>
            <div class="vpn-row vpn-neg"><span>− Costo fijo mensual</span><strong>−${fmtUSD(b.costFixed)}</strong></div>
        `;
    }
}

// ============== Render: evidencia (Fase 2) ==============
function renderEvidence() {
    const wrap = document.getElementById('evidence-grid');
    if (!wrap || !decisionsData) return;

    const cards = Object.values(decisionsData.scenarios).map(scenario => {
        const classCounts = scenario.classes
            .map(c => `<span class="evidence-chip"><i>${c}</i> ${scenario.class_counts[c]}</span>`)
            .join('');
        // mejor modelo del escenario
        let bestKey = null;
        let bestBal = -1;
        Object.entries(scenario.models).forEach(([k, m]) => {
            if (m.balanced_accuracy > bestBal) { bestBal = m.balanced_accuracy; bestKey = k; }
        });
        const best = scenario.models[bestKey];
        const perClass = scenario.classes.map(c => {
            const pc = best.per_class[c] || {};
            return `<tr>
                <td>${c}</td>
                <td>${formatPercent(pc.precision || 0)}</td>
                <td>${formatPercent(pc.recall || 0)}</td>
                <td>${((pc.f1 || 0) * 100).toFixed(1)}%</td>
            </tr>`;
        }).join('');

        return `
            <article class="evidence-card">
                <header class="evidence-card-header">
                    <span class="decision-problem-kicker">Escenario ${scenario.id}</span>
                    <h3>${scenario.name}</h3>
                    <p>${scenario.subtitle}</p>
                </header>
                <div class="evidence-chips">${classCounts}</div>
                <div class="evidence-best">
                    <span>Mejor modelo</span>
                    <strong>${best.label}</strong>
                    <span class="evidence-balacc">${formatPercent(best.balanced_accuracy)} BalAcc</span>
                </div>
                <table class="evidence-table">
                    <thead><tr><th>Clase</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
                    <tbody>${perClass}</tbody>
                </table>
            </article>
        `;
    }).join('');
    wrap.innerHTML = cards;
}

// ============== Render: tornado ==============
function renderTornado() {
    const wrap = document.getElementById('tornado-chart');
    if (!wrap || !decisionsData) return;
    const scenario = decisionsData.scenarios[decisionState.scenarioId];
    if (!scenario) return;

    const params = [
        { key: 'cost_FN', label: 'Costo FN', base: decisionState.costs.FN, low: 0.7, high: 1.3 },
        { key: 'cost_FP', label: 'Costo FP', base: decisionState.costs.FP, low: 0.7, high: 1.3 },
        { key: 'value_TP', label: 'Valor TP', base: decisionState.costs.TP, low: 0.7, high: 1.3 },
        { key: 'prevalence', label: 'Prevalencia', base: decisionState.business.prevalence, low: 0.7, high: 1.3 },
        { key: 'volume', label: 'Volumen', base: decisionState.business.volume, low: 0.7, high: 1.3 },
        { key: 'inferenceCost', label: 'Costo inferencia', base: decisionState.business.inferenceCost, low: 0.7, high: 1.3 }
    ];

    const cm = computeBinaryCM(scenario, decisionState.modelKey, decisionState.threshold);
    const baseValue = computeMonthlyValue(cm, decisionState.costs, decisionState.business);
    const baseVPN = baseValue ? baseValue.net : 0;

    const results = params.map(p => {
        // low
        const costsLow = { ...decisionState.costs };
        const bizLow = { ...decisionState.business };
        const costsHigh = { ...decisionState.costs };
        const bizHigh = { ...decisionState.business };
        const apply = (target, factor) => {
            if (p.key === 'cost_FN') target.FN = decisionState.costs.FN * factor;
            else if (p.key === 'cost_FP') target.FP = decisionState.costs.FP * factor;
            else if (p.key === 'value_TP') target.TP = decisionState.costs.TP * factor;
        };
        apply(costsLow, p.low); apply(costsHigh, p.high);
        if (p.key === 'prevalence') { bizLow.prevalence = decisionState.business.prevalence * p.low; bizHigh.prevalence = decisionState.business.prevalence * p.high; }
        if (p.key === 'volume') { bizLow.volume = decisionState.business.volume * p.low; bizHigh.volume = decisionState.business.volume * p.high; }
        if (p.key === 'inferenceCost') { bizLow.inferenceCost = decisionState.business.inferenceCost * p.low; bizHigh.inferenceCost = decisionState.business.inferenceCost * p.high; }

        const vLow = computeMonthlyValue(cm, costsLow, bizLow).net;
        const vHigh = computeMonthlyValue(cm, costsHigh, bizHigh).net;
        const lo = Math.min(vLow, vHigh);
        const hi = Math.max(vLow, vHigh);
        return { ...p, lo, hi, range: hi - lo };
    }).sort((a, b) => b.range - a.range);

    const maxRange = Math.max(...results.map(r => Math.abs(r.lo - baseVPN)), ...results.map(r => Math.abs(r.hi - baseVPN)));

    wrap.innerHTML = results.map(r => {
        const totalRange = maxRange * 2 || 1;
        const lowFromBase = (r.lo - baseVPN) / totalRange;
        const highFromBase = (r.hi - baseVPN) / totalRange;
        const center = 50;
        const startPct = center + lowFromBase * 50;
        const widthPct = (highFromBase - lowFromBase) * 50;
        return `
            <div class="tornado-row">
                <span class="tornado-label">${r.label}</span>
                <div class="tornado-bar-wrap">
                    <div class="tornado-zero"></div>
                    <div class="tornado-bar" style="left:${startPct.toFixed(2)}%;width:${widthPct.toFixed(2)}%;"></div>
                </div>
                <span class="tornado-range">${fmtUSD(r.lo)} → ${fmtUSD(r.hi)}</span>
            </div>
        `;
    }).join('') + `<div class="tornado-base">VPN base: <strong>${fmtUSD(baseVPN)}</strong> · variación ±30% por parámetro</div>`;
}

// ============== Monte Carlo ==============
function runMonteCarlo(N = 2000) {
    const scenario = decisionsData.scenarios[decisionState.scenarioId];
    if (!scenario) return null;
    const cm = computeBinaryCM(scenario, decisionState.modelKey, decisionState.threshold);
    const samples = [];
    let positive = 0;

    const tri = (mode, low, high) => {
        // Distribucion triangular sencilla
        const u = Math.random();
        const f = (mode - low) / (high - low);
        if (u < f) return low + Math.sqrt(u * (high - low) * (mode - low));
        return high - Math.sqrt((1 - u) * (high - low) * (high - mode));
    };

    for (let i = 0; i < N; i++) {
        const sampleCosts = {
            TP: tri(decisionState.costs.TP, decisionState.costs.TP * 0.7, decisionState.costs.TP * 1.3),
            FP: tri(decisionState.costs.FP, decisionState.costs.FP * 0.7, decisionState.costs.FP * 1.5),
            FN: tri(decisionState.costs.FN, decisionState.costs.FN * 0.7, decisionState.costs.FN * 1.4),
            TN: 0
        };
        const sampleBiz = {
            volume: tri(decisionState.business.volume, decisionState.business.volume * 0.7, decisionState.business.volume * 1.3),
            prevalence: Math.min(0.95, Math.max(0.01, tri(decisionState.business.prevalence, decisionState.business.prevalence * 0.6, decisionState.business.prevalence * 1.6))),
            inferenceCost: tri(decisionState.business.inferenceCost, decisionState.business.inferenceCost * 0.5, decisionState.business.inferenceCost * 1.8),
            fixedCost: decisionState.business.fixedCost
        };
        const v = computeMonthlyValue(cm, sampleCosts, sampleBiz);
        if (v) {
            samples.push(v.net);
            if (v.net > 0) positive++;
        }
    }
    samples.sort((a, b) => a - b);
    return {
        samples,
        mean: samples.reduce((s, v) => s + v, 0) / samples.length,
        median: samples[Math.floor(samples.length / 2)],
        p5: samples[Math.floor(samples.length * 0.05)],
        p95: samples[Math.floor(samples.length * 0.95)],
        probPositive: positive / samples.length
    };
}

function renderMonteCarlo() {
    const svg = document.getElementById('montecarlo-hist');
    const stats = document.getElementById('mc-stats');
    if (!svg) return;
    const mc = runMonteCarlo();
    if (!mc || mc.samples.length === 0) return;

    const W = 360, H = 200, PAD = 28;
    const innerW = W - PAD * 2;
    const innerH = H - PAD * 2;
    const min = mc.samples[0];
    const max = mc.samples[mc.samples.length - 1];
    const bins = 30;
    const binSize = (max - min) / bins || 1;
    const counts = new Array(bins).fill(0);
    mc.samples.forEach(v => {
        const idx = Math.min(bins - 1, Math.floor((v - min) / binSize));
        counts[idx]++;
    });
    const maxCount = Math.max(...counts);

    const xZero = PAD + ((0 - min) / (max - min)) * innerW;
    const bars = counts.map((c, i) => {
        const x = PAD + (i / bins) * innerW;
        const h = (c / maxCount) * innerH;
        const y = H - PAD - h;
        const cx = min + (i + 0.5) * binSize;
        const fill = cx >= 0 ? '#22c55e' : '#ef4444';
        return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(innerW / bins - 1).toFixed(1)}" height="${h.toFixed(1)}" fill="${fill}" fill-opacity="0.6"/>`;
    }).join('');

    let zeroLine = '';
    if (xZero >= PAD && xZero <= PAD + innerW) {
        zeroLine = `<line x1="${xZero}" y1="${PAD}" x2="${xZero}" y2="${H - PAD}" stroke="#facc15" stroke-dasharray="4 3" stroke-width="1.5"/>
                    <text x="${xZero}" y="${PAD - 4}" fill="#facc15" font-size="10" text-anchor="middle">VPN=0</text>`;
    }

    svg.innerHTML = `
        ${bars}
        ${zeroLine}
        <text x="${PAD}" y="${H - 6}" fill="#94a3b8" font-size="10">${fmtUSD(min)}</text>
        <text x="${W - PAD}" y="${H - 6}" fill="#94a3b8" font-size="10" text-anchor="end">${fmtUSD(max)}</text>
    `;

    if (stats) {
        const sign = mc.probPositive >= 0.7 ? 'mc-pos' : mc.probPositive >= 0.5 ? 'mc-mid' : 'mc-neg';
        stats.innerHTML = `
            <div class="mc-stat ${sign}"><span>P(VPN > 0)</span><strong>${(mc.probPositive * 100).toFixed(1)}%</strong></div>
            <div class="mc-stat"><span>Media</span><strong>${fmtUSD(mc.mean)}</strong></div>
            <div class="mc-stat"><span>Mediana</span><strong>${fmtUSD(mc.median)}</strong></div>
            <div class="mc-stat"><span>IC 90%</span><strong>${fmtUSD(mc.p5)} … ${fmtUSD(mc.p95)}</strong></div>
        `;
    }
    return mc;
}

// ============== Tabla de break-even ==============
function renderBreakeven() {
    const table = document.getElementById('breakeven-table');
    if (!table || !decisionsData) return;
    const tbody = table.querySelector('tbody');
    const rows = [];

    Object.values(decisionsData.scenarios).forEach(scenario => {
        Object.entries(scenario.models).forEach(([modelKey, model]) => {
            const opt = findOptimalThreshold(scenario, modelKey, decisionState.costs, decisionState.business);
            const breakeven = findBreakEvenPrevalence(scenario, modelKey, decisionState.costs, decisionState.business, opt.threshold);
            const verdict = opt.vpn > 0
                ? (opt.vpn > 5000 ? '<span class="verdict-go">GO fuerte</span>' : '<span class="verdict-go">GO</span>')
                : (opt.vpn > -1000 ? '<span class="verdict-mid">Marginal</span>' : '<span class="verdict-no">NO-GO</span>');
            rows.push({
                scenarioId: scenario.id,
                scenarioName: scenario.name,
                modelKey,
                modelLabel: model.label,
                threshold: opt.threshold,
                balacc: model.balanced_accuracy,
                vpn: opt.vpn,
                breakeven,
                verdict
            });
        });
    });

    rows.sort((a, b) => b.vpn - a.vpn);
    const winner = rows[0];

    tbody.innerHTML = rows.map((r, i) => `
        <tr class="${i === 0 ? 'breakeven-winner' : ''}">
            <td>${r.scenarioName}</td>
            <td>${r.modelLabel}</td>
            <td>${r.threshold.toFixed(2)}</td>
            <td>${formatPercent(r.balacc)}</td>
            <td class="${r.vpn >= 0 ? 'cell-pos' : 'cell-neg'}">${fmtUSD(r.vpn)}</td>
            <td>${r.breakeven === null ? '—' : `${(r.breakeven * 100).toFixed(1)}%`}</td>
            <td>${r.verdict}</td>
        </tr>
    `).join('');

    return { winner, all: rows };
}

// ============== Recomendacion Final ==============
function renderFinalRecommendation(breakevenResult, mcResult) {
    if (!breakevenResult) return;
    const winner = breakevenResult.winner;
    const runnerUp = breakevenResult.all.find(r => r.scenarioId !== winner.scenarioId);
    const title = document.getElementById('final-rec-title');
    const subtitle = document.getElementById('final-rec-subtitle');
    const badge = document.getElementById('final-rec-badge');
    const vpn = document.getElementById('final-rec-vpn');
    const summary = document.getElementById('final-rec-summary');
    const why = document.getElementById('final-rec-why');
    const tradeoff = document.getElementById('final-rec-tradeoff');
    const conditions = document.getElementById('final-rec-conditions');
    const next = document.getElementById('final-rec-next');

    let verdict = 'NO-GO';
    let verdictClass = 'badge-no';
    if (mcResult && mcResult.probPositive >= 0.75 && winner.vpn > 1000) { verdict = 'GO'; verdictClass = 'badge-go'; }
    else if (mcResult && mcResult.probPositive >= 0.55) { verdict = 'GO condicional'; verdictClass = 'badge-mid'; }

    if (badge) { badge.textContent = verdict; badge.className = `final-rec-badge ${verdictClass}`; }
    if (vpn) vpn.textContent = fmtUSD(winner.vpn);
    if (title) title.textContent = `Recomendación: ${verdict} con ${winner.modelLabel} (${winner.scenarioName})`;
    if (subtitle) subtitle.textContent = `Umbral óptimo ${winner.threshold.toFixed(2)} · BalAcc ${formatPercent(winner.balacc)} · Volumen ${fmtNum(decisionState.business.volume)} llamadas/mes · Prevalencia ${(decisionState.business.prevalence * 100).toFixed(1)}%`;

    if (summary) {
        summary.textContent = winner.vpn > 0
            ? `Con la matriz de costos actual y ${fmtNum(decisionState.business.volume)} llamadas/mes, el escenario ${winner.scenarioName} con ${winner.modelLabel} maximiza el VPN a ${fmtUSD(winner.vpn)} usando un umbral operativo de ${winner.threshold.toFixed(2)}. La simulación Monte Carlo indica que ${(mcResult ? mcResult.probPositive * 100 : 0).toFixed(0)}% de los escenarios futuros mantienen un VPN positivo.`
            : `Con los costos actuales ningún escenario tiene VPN positivo. Antes de desplegar, hay que renegociar el costo de inferencia, reducir el costo fijo mensual o ampliar el dataset para mejorar la BalAcc.`;
    }
    if (why) {
        const margin = runnerUp ? winner.vpn - runnerUp.vpn : winner.vpn;
        why.textContent = `${winner.modelLabel} en el escenario ${winner.scenarioName} supera al runner-up por ${fmtUSD(margin)} mensuales gracias a su balance de precisión/recall en Enojo y a que opera mejor cerca del umbral ${winner.threshold.toFixed(2)} donde la asimetría de costos (FN ${(decisionState.costs.FN / decisionState.costs.FP).toFixed(1)}× FP) lo favorece.`;
    }
    if (tradeoff) {
        tradeoff.textContent = winner.scenarioId === '2clases'
            ? 'El escenario binario sacrifica la cobertura de Feliz para ganar separabilidad. Si el call center necesita medir satisfacción positiva en el futuro, este enfoque obligaría a un re-entrenamiento.'
            : 'El escenario de 3 emociones gana cobertura analítica (mide enojo, tristeza y felicidad) pero introduce confusión Enojo↔Feliz por activación acústica similar. Las llamadas felices podrían ser escaladas erróneamente.';
    }
    if (conditions) {
        const minPrev = winner.breakeven === null ? null : (winner.breakeven * 100).toFixed(1);
        conditions.textContent = minPrev
            ? `Esta recomendación se mantiene mientras: prevalencia de enojo ≥ ${minPrev}%, costo de inferencia ≤ ${fmtUSD(decisionState.business.inferenceCost * 1.5)}/llamada y el equipo opere en el umbral ${winner.threshold.toFixed(2)}. Si baja el costo del FN o sube el costo del FP, recalcular.`
            : `No existe una prevalencia de enojo que rentabilice este modelo con los costos actuales. La decisión depende fuertemente de los inputs económicos.`;
    }
    if (next) {
        next.textContent = winner.vpn > 0
            ? `Piloto de 30 días con muestra de 500-1 000 llamadas etiquetadas, midiendo TPR/FPR reales y comparándolos con los del laboratorio (${formatPercent(winner.balacc)} BalAcc). Si el VPN observado se mantiene dentro del IC 90% del Monte Carlo (${fmtUSD(mcResult ? mcResult.p5 : 0)} → ${fmtUSD(mcResult ? mcResult.p95 : 0)}), escalar a producción.`
            : `Recolectar más audios de Enojo (especialmente con recolectores diversos), explorar ensembles, o reducir el costo fijo mensual de operación. No desplegar hasta que el VPN esperado sea positivo.`;
    }
}

// ============== Renderizado principal ==============
function refreshDecisionDashboard() {
    if (!decisionsData) return;
    const scenario = decisionsData.scenarios[decisionState.scenarioId];
    if (!scenario) return;

    const optimal = findOptimalThreshold(scenario, decisionState.modelKey, decisionState.costs, decisionState.business);
    const cm = computeBinaryCM(scenario, decisionState.modelKey, decisionState.threshold);
    if (!cm) return;
    const value = computeMonthlyValue(cm, decisionState.costs, decisionState.business);

    renderConfusionMatrix(cm);
    renderROC(scenario, decisionState.modelKey, decisionState.threshold, optimal.threshold);
    renderVPN(cm, value);
    renderTornado();
    const mc = renderMonteCarlo();
    const breakeven = renderBreakeven();
    renderFinalRecommendation(breakeven, mc);

    // Costo ratio hint
    const ratio = decisionState.costs.FP > 0 ? decisionState.costs.FN / decisionState.costs.FP : 0;
    const ratioEl = document.getElementById('cost-ratio');
    const hintEl = document.getElementById('cost-ratio-hint');
    if (ratioEl) ratioEl.textContent = `${ratio.toFixed(1)}x`;
    if (hintEl) {
        if (ratio > 5) hintEl.textContent = `Omitir enojo es ${ratio.toFixed(1)}× más caro que escalar de más → favorece umbrales bajos (más recall).`;
        else if (ratio > 1.5) hintEl.textContent = `Asimetría moderada: el modelo debe priorizar recall ligeramente sobre precisión.`;
        else if (ratio > 0.8) hintEl.textContent = `Costos casi simétricos: el umbral óptimo está cerca de 0.5.`;
        else hintEl.textContent = `Falsa alarma es más cara que omitir → favorece umbrales altos (más precisión).`;
    }
}

function initDecisionsModule() {
    if (!decisionsData) {
        console.warn('decisionsData no disponible. Genera outputs/decisions_data.json.');
        return;
    }

    // Custom dropdown para seleccionar modelo
    const dropdown = document.getElementById('sim-model-dropdown');
    const dropdownTrigger = dropdown ? dropdown.querySelector('.sim-dropdown-trigger') : null;
    const dropdownMenu = dropdown ? dropdown.querySelector('.sim-dropdown-menu') : null;
    const dropdownCurrentLabel = dropdown ? dropdown.querySelector('.sim-dropdown-current-label') : null;
    const dropdownCurrentMeta = dropdown ? dropdown.querySelector('.sim-dropdown-current-meta') : null;

    const closeDropdown = () => {
        if (!dropdown) return;
        dropdown.classList.remove('open');
        if (dropdownTrigger) dropdownTrigger.setAttribute('aria-expanded', 'false');
    };

    const setDropdownValue = (modelKey, models) => {
        if (!dropdown) return;
        const meta = models[modelKey];
        if (!meta) return;
        dropdown.setAttribute('data-value', modelKey);
        if (dropdownCurrentLabel) dropdownCurrentLabel.textContent = meta.label;
        if (dropdownCurrentMeta) dropdownCurrentMeta.textContent = `${formatPercent(meta.balanced_accuracy)} BalAcc`;
        if (dropdownMenu) {
            dropdownMenu.querySelectorAll('li').forEach(li => {
                li.classList.toggle('selected', li.getAttribute('data-value') === modelKey);
            });
        }
    };

    const refreshModelOptions = () => {
        const scenario = decisionsData.scenarios[decisionState.scenarioId];
        if (!scenario || !dropdownMenu) return;
        const entries = Object.entries(scenario.models)
            .sort((a, b) => b[1].balanced_accuracy - a[1].balanced_accuracy);
        dropdownMenu.innerHTML = entries.map(([k, m]) => `
            <li role="option" data-value="${k}" class="${k === decisionState.modelKey ? 'selected' : ''}">
                <div class="sim-dropdown-opt-main">
                    <span class="sim-dropdown-opt-name">${m.label}</span>
                    <span class="sim-dropdown-opt-acc">${formatPercent(m.balanced_accuracy)} BalAcc</span>
                </div>
            </li>
        `).join('');
        // Asegura que el modelKey existe en el nuevo escenario
        if (!scenario.models[decisionState.modelKey]) {
            decisionState.modelKey = entries[0][0];
        }
        setDropdownValue(decisionState.modelKey, scenario.models);

        // Bind click en cada opcion
        dropdownMenu.querySelectorAll('li').forEach(li => {
            li.addEventListener('click', () => {
                const k = li.getAttribute('data-value');
                if (!k) return;
                decisionState.modelKey = k;
                setDropdownValue(k, scenario.models);
                closeDropdown();
                refreshDecisionDashboard();
            });
        });
    };

    if (dropdownTrigger) {
        dropdownTrigger.addEventListener('click', (e) => {
            e.stopPropagation();
            const isOpen = dropdown.classList.toggle('open');
            dropdownTrigger.setAttribute('aria-expanded', String(isOpen));
        });
    }
    document.addEventListener('click', (e) => {
        if (dropdown && !dropdown.contains(e.target)) {
            closeDropdown();
        }
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeDropdown();
    });

    refreshModelOptions();

    // Listeners de escenario
    document.querySelectorAll('#sim-scenario-toggle .sim-seg-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#sim-scenario-toggle .sim-seg-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            decisionState.scenarioId = btn.getAttribute('data-scenario');
            refreshModelOptions();
            refreshDecisionDashboard();
        });
    });

    // Sliders simulator
    const bindSlider = (id, displayId, setter, fmt) => {
        const el = document.getElementById(id);
        const disp = document.getElementById(displayId);
        if (!el) return;
        el.addEventListener('input', (e) => {
            setter(Number(e.target.value));
            if (disp) disp.textContent = fmt(Number(e.target.value));
            refreshDecisionDashboard();
        });
    };

    bindSlider('sim-threshold', 'sim-threshold-value', v => decisionState.threshold = v, v => v.toFixed(2));
    bindSlider('sim-volume', 'sim-volume-value', v => decisionState.business.volume = v, v => fmtNum(v));
    bindSlider('sim-prevalence', 'sim-prevalence-value', v => decisionState.business.prevalence = v, v => `${(v * 100).toFixed(1)}%`);
    bindSlider('sim-inference', 'sim-inference-value', v => decisionState.business.inferenceCost = v, v => `USD ${v.toFixed(3)}`);

    // Inputs de costos
    const bindCost = (id, key) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('input', () => {
            decisionState.costs[key] = Number(el.value) || 0;
            refreshDecisionDashboard();
        });
    };
    bindCost('cost-tp', 'TP');
    bindCost('cost-fp', 'FP');
    bindCost('cost-fn', 'FN');
    bindCost('cost-tn', 'TN');

    renderEvidence();
    refreshDecisionDashboard();
}

function renderAnalyticsExperiment(experimentId) {
    const experiment = getExperimentById(experimentId)
        || getExperimentById(experimentHistory.default_experiment_id)
        || (experimentHistory.experiments || [])[0];

    if (!experiment) return;

    experimentOptions.forEach((option) => {
        option.classList.toggle('active', option.getAttribute('data-experiment-id') === experiment.id);
    });

    const summaryClasses = document.getElementById('summary-classes');
    const summaryDataset = document.getElementById('summary-dataset');
    const summaryChance = document.getElementById('summary-chance');
    const summaryBestModel = document.getElementById('summary-best-model');
    const summaryBestBalAcc = document.getElementById('summary-best-balacc');

    if (summaryClasses) summaryClasses.textContent = experiment.classes.join(' / ');
    if (summaryDataset) summaryDataset.textContent = `${experiment.dataset_size} audios`;
    if (summaryChance) summaryChance.textContent = formatPercent(experiment.chance_accuracy || 0);

    const bestMetric = (experiment.metrics || {})[experiment.best_model] || {};
    if (summaryBestModel) summaryBestModel.textContent = bestMetric.label || experiment.best_model || '-';
    if (summaryBestBalAcc) summaryBestBalAcc.textContent = formatPercent(experiment.best_balanced_accuracy || 0);

    (experiment.cards || []).forEach((card, index) => {
        const cardNum = index + 1;
        const titleEl = document.getElementById(`analytics-card-${cardNum}-title`);
        const descEl = document.getElementById(`analytics-card-${cardNum}-description`);
        const imgEl = document.getElementById(`analytics-card-${cardNum}-image`);
        const obsEl = document.getElementById(`analytics-card-${cardNum}-observation`);

        if (titleEl) titleEl.textContent = `${cardNum}. ${card.title}`;
        if (descEl) descEl.textContent = card.description;
        if (imgEl) {
            imgEl.src = card.image;
            imgEl.alt = card.title;
        }
        if (obsEl) obsEl.innerHTML = `<strong>Observación:</strong> ${card.observation}`;
    });
}

function slugifyEmotion(name) {
    return (name || '')
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '')
        .toLowerCase()
        .replace(/\s+/g, '-');
}

function getClassStyle(name) {
    return classStyles[name] || {
        slug: slugifyEmotion(name),
        label: name,
        accent: '#a855f7',
        accent_dark: '#7e22ce'
    };
}

function getRingVisual(name) {
    const slug = getClassStyle(name).slug;
    if (slug === 'enojo') {
        return { stroke: 'url(#gradient-enojo)', glow: 'drop-shadow(0 0 10px rgba(239, 68, 68, 0.5))' };
    }
    if (slug === 'tristeza') {
        return { stroke: 'url(#gradient-tristeza)', glow: 'drop-shadow(0 0 10px rgba(6, 182, 212, 0.5))' };
    }
    if (slug === 'feliz') {
        return { stroke: 'url(#gradient-feliz)', glow: 'drop-shadow(0 0 10px rgba(245, 158, 11, 0.45))' };
    }
    return { stroke: 'url(#gradient-default)', glow: 'none' };
}

function renderProbabilityBars(probabilities) {
    if (!probBars) return;

    const orderedClasses = [
        ...appConfig.activeClasses.filter(name => Object.prototype.hasOwnProperty.call(probabilities, name)),
        ...Object.keys(probabilities).filter(name => !appConfig.activeClasses.includes(name))
    ];

    probBars.innerHTML = orderedClasses.map((name) => {
        const style = getClassStyle(name);
        const pct = Math.round((probabilities[name] || 0) * 100);
        return `
            <div class="prob-item">
                <div class="prob-meta">
                    <span>${style.label}</span>
                    <span>${pct}%</span>
                </div>
                <div class="progress-track">
                    <div
                        class="progress-fill"
                        style="width: ${pct}%; --emotion-color: ${style.accent}; --emotion-color-dark: ${style.accent_dark};"
                    ></div>
                </div>
            </div>
        `;
    }).join('');
}

// 1. Tab Navigation
tabButtons.forEach(button => {
    button.addEventListener('click', () => {
        const targetTab = button.getAttribute('data-tab');
        
        // Update active class on buttons
        tabButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
        
        // Show/hide content panes
        tabPanes.forEach(pane => {
            if (pane.id === targetTab) {
                pane.classList.add('active');
            } else {
                pane.classList.remove('active');
            }
        });
        
        // Auto-load dataset if clicking on dataset explorer
        if (targetTab === 'dataset-explorer' && !datasetLoaded) {
            loadDataset();
        }
        if (targetTab === 'decisions') {
            refreshDecisionDashboard();
        }
        if (targetTab === 'analytics') {
            renderAnalyticsExperiment(
                document.querySelector('.experiment-option.active')?.getAttribute('data-experiment-id')
                || experimentHistory.default_experiment_id
            );
        }
    });
});

// 2. Drag & Drop File Upload
if (dropZone) {
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
}

if (fileInput) {
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

function handleFileUpload(file) {
    if (!file.type.startsWith('audio/')) {
        alert('Por favor, selecciona un archivo de audio válido.');
        return;
    }
    
    const selectedModel = document.getElementById('model-selector').value;
    const formData = new FormData();
    formData.append('audio', file);
    formData.append('model', selectedModel);
    
    showLoader(true);
    
    fetch('/api/predict_upload', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        showLoader(false);
        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            displayResults(data);
        }
    })
    .catch(err => {
        showLoader(false);
        alert('Ocurrió un error al procesar el audio: ' + err.message);
    });
}

// 3. Audio Recording
if (recordBtn) {
    recordBtn.addEventListener('click', toggleRecording);
}

function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

function startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Tu navegador no soporta grabación de audio por micrófono.');
        return;
    }

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            audioChunks = [];
            isRecording = true;
            recordBtn.classList.add('recording');
            recordBtn.innerHTML = '<i class="fa-solid fa-square"></i>';
            recordingStatus.style.visibility = 'visible';
            waveformAnim.style.visibility = 'visible';
            recordInstruction.textContent = 'Haz clic para detener';
            
            // Timer logic
            recordDuration = 0;
            recordTimer.textContent = '00:00';
            recordInterval = setInterval(() => {
                recordDuration++;
                const mins = String(Math.floor(recordDuration / 60)).padStart(2, '0');
                const secs = String(recordDuration % 60).padStart(2, '0');
                recordTimer.textContent = `${mins}:${secs}`;
                
                // Limit recording to 10 seconds
                if (recordDuration >= 10) {
                    stopRecording();
                }
            }, 1000);

            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.addEventListener('dataavailable', event => {
                audioChunks.push(event.data);
            });

            mediaRecorder.addEventListener('stop', () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                sendBlob(audioBlob);
                
                // Stop microphone stream tracks
                stream.getTracks().forEach(track => track.stop());
            });

            mediaRecorder.start();
        })
        .catch(err => {
            alert('Permiso de micrófono denegado o no disponible: ' + err.message);
        });
}

function stopRecording() {
    if (!isRecording) return;
    
    isRecording = false;
    clearInterval(recordInterval);
    recordBtn.classList.remove('recording');
    recordBtn.innerHTML = '<i class="fa-solid fa-microphone"></i>';
    recordingStatus.style.visibility = 'hidden';
    waveformAnim.style.visibility = 'hidden';
    recordInstruction.textContent = 'Haz clic para empezar a grabar (máx. 10s)';
    
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
}

function sendBlob(blob) {
    const selectedModel = document.getElementById('model-selector').value;
    const formData = new FormData();
    formData.append('audio', blob, 'recording.wav');
    formData.append('model', selectedModel);
    
    showLoader(true);
    
    fetch('/api/predict_upload', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        showLoader(false);
        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            displayResults(data);
        }
    })
    .catch(err => {
        showLoader(false);
        alert('Ocurrió un error al enviar la grabación: ' + err.message);
    });
}

// 4. UI Loader & Results Rendering
function showLoader(show) {
    loader.style.display = show ? 'flex' : 'none';
}

function displayResults(data) {
    emptyResults.style.display = 'none';
    activeResults.style.display = 'block';
    
    const pred = data.prediction;
    const confidence = Math.round(data.probabilities[pred] * 100);
    const predStyle = getClassStyle(pred);
    
    predictedEmotionBadge.textContent = pred;
    predictedEmotionBadge.className = `badge ${predStyle.slug}`;
    
    predictionPercentage.textContent = `${confidence}%`;
    
    // Set dynamic SVG progress ring and glow
    if (progressCircle) {
        const offset = progressCircumference - (confidence / 100) * progressCircumference;
        const ringVisual = getRingVisual(pred);
        progressCircle.setAttribute('stroke', ringVisual.stroke);
        progressCircle.style.filter = ringVisual.glow;
        progressCircle.style.strokeDashoffset = offset;
    }
    
    renderProbabilityBars(data.probabilities);
    
    // Set acoustic metrics
    metricDuration.textContent = `${data.metrics.duration.toFixed(1)}s`;
    metricPitch.textContent = data.metrics.pitch_mean_hz > 0 ? `${data.metrics.pitch_mean_hz} Hz` : 'N/A';
    metricEnergy.textContent = `${data.metrics.energy_db} dB`;
    metricZcr.textContent = data.metrics.zcr;
}

// 5. Dataset Explorer loading & interactions
function loadDataset() {
    const tbody = document.getElementById('dataset-tbody');
    
    fetch('/api/dataset')
        .then(res => res.json())
        .then(data => {
            datasetLoaded = true;
            tbody.innerHTML = '';
            
            data.forEach(item => {
                const tr = document.createElement('tr');
                tr.className = `audio-row class-${item.slug || slugifyEmotion(item.clase)}`;
                
                tr.innerHTML = `
                    <td><strong>${item.archivo}</strong></td>
                    <td><span class="recolector-tag">${item.recolector}</span></td>
                    <td><span class="badge ${item.slug || slugifyEmotion(item.clase)}">${item.clase}</span></td>
                    <td><code>${item.score.toFixed(4)}</code></td>
                    <td class="audio-cell">
                        <button class="custom-audio-btn" data-url="${item.url}">
                            <i class="fa-solid fa-play"></i>
                        </button>
                    </td>
                    <td style="text-align: right;">
                        <button class="btn btn-sm btn-secondary test-model-btn" data-class="${item.clase}" data-file="${item.archivo}">
                            <i class="fa-solid fa-vial"></i> Probar Modelo
                        </button>
                    </td>
                `;
                tbody.appendChild(tr);
            });
            
            setupDatasetInteractions();
        })
        .catch(err => {
            tbody.innerHTML = `
                <tr>
                    <td colspan="6" style="text-align: center; color: #ef4444; padding: 2rem;">
                        <i class="fa-solid fa-triangle-exclamation" style="font-size: 2rem; margin-bottom: 0.5rem;"></i>
                        <p>Error al cargar el dataset: ${err.message}</p>
                    </td>
                </tr>
            `;
        });
}

function setupDatasetInteractions() {
    // 1. Filter buttons logic
    const filterBtns = document.querySelectorAll('.btn-filter');
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const filterValue = btn.getAttribute('data-filter');
            const rows = document.querySelectorAll('.audio-row');
            
            rows.forEach(row => {
                if (filterValue === 'all') {
                    row.classList.remove('hidden');
                } else if (row.classList.contains(`class-${slugifyEmotion(filterValue)}`)) {
                    row.classList.remove('hidden');
                } else {
                    row.classList.add('hidden');
                }
            });
        });
    });

    // 2. Audio Play/Pause handlers
    const audioBtns = document.querySelectorAll('.custom-audio-btn');
    audioBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const url = btn.getAttribute('data-url');
            
            if (currentPlayingBtn === btn) {
                // Toggle play/pause for same audio
                if (globalAudio.paused) {
                    globalAudio.play();
                    btn.innerHTML = '<i class="fa-solid fa-pause"></i>';
                    btn.classList.add('playing');
                } else {
                    globalAudio.pause();
                    btn.innerHTML = '<i class="fa-solid fa-play"></i>';
                    btn.classList.remove('playing');
                }
            } else {
                // Pause currently playing audio if any
                if (currentPlayingBtn) {
                    currentPlayingBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
                    currentPlayingBtn.classList.remove('playing');
                }
                
                // Play new audio
                globalAudio.src = url;
                globalAudio.play();
                btn.innerHTML = '<i class="fa-solid fa-pause"></i>';
                btn.classList.add('playing');
                currentPlayingBtn = btn;
            }
        });
    });

    // Clean button icon when audio reaches the end
    globalAudio.addEventListener('ended', () => {
        if (currentPlayingBtn) {
            currentPlayingBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
            currentPlayingBtn.classList.remove('playing');
            currentPlayingBtn = null;
        }
    });

    // 3. Test Model buttons logic
    const testBtns = document.querySelectorAll('.test-model-btn');
    testBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const clase = btn.getAttribute('data-class');
            const archivo = btn.getAttribute('data-file');
            const selectedModel = document.getElementById('model-selector').value;
            
            // Switch tab to live-predictor first
            document.querySelector('.tab-btn[data-tab="live-predictor"]').click();
            
            // Call API
            showLoader(true);
            
            fetch('/api/predict_dataset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ clase, archivo, model: selectedModel })
            })
            .then(res => res.json())
            .then(data => {
                showLoader(false);
                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    displayResults(data);
                }
            })
            .catch(err => {
                showLoader(false);
                alert('Ocurrió un error al consultar el clasificador: ' + err.message);
            });
        });
    });
}

// 6. Interactive Model Grid Click Handlers
const modelOptions = document.querySelectorAll('.model-option');
const hiddenModelInput = document.getElementById('model-selector');

if (modelOptions.length > 0 && hiddenModelInput) {
    modelOptions.forEach(opt => {
        opt.addEventListener('click', () => {
            // Remove active class from all options
            modelOptions.forEach(o => o.classList.remove('active'));
            
            // Add active class to clicked option
            opt.classList.add('active');
            
            // Update hidden input value
            hiddenModelInput.value = opt.getAttribute('data-model');
        });
    });
}

if (experimentOptions.length > 0) {
    experimentOptions.forEach((option) => {
        option.addEventListener('click', () => {
            renderAnalyticsExperiment(option.getAttribute('data-experiment-id'));
        });
    });
    renderAnalyticsExperiment(
        document.querySelector('.experiment-option.active')?.getAttribute('data-experiment-id')
        || experimentHistory.default_experiment_id
    );
}

// Inicializa modulo de decisiones (Toma de Decisiones)
initDecisionsModule();

renderDecisionScenarios();
