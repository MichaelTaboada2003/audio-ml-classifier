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
const decisionModeButtons = document.querySelectorAll('.decision-mode-btn');
const sadnessWeightSlider = document.getElementById('sadness-weight-slider');

let decisionMode = 'conservative';
let sadnessWeight = 1;

function getExperimentById(experimentId) {
    return (experimentHistory.experiments || []).find((exp) => exp.id === experimentId);
}

function formatPercent(value) {
    return `${(value * 100).toFixed(1)}%`;
}

function getScenarioDecisionScore(experiment, mode, sadnessPriority) {
    const coverageScore = experiment.classes.length / 3;
    const accuracyScore = experiment.best_balanced_accuracy || 0;
    const complexityPenalty = (experiment.classes.length - 2) * 0.08;
    const sadnessFactor = experiment.classes.includes('Tristeza') ? 1 : 0.75;

    const profiles = {
        conservative: { acc: 0.7, cov: 0.15, sad: 0.15, penalty: 1.1 },
        balanced: { acc: 0.5, cov: 0.3, sad: 0.2, penalty: 1.0 },
        coverage: { acc: 0.35, cov: 0.5, sad: 0.15, penalty: 0.8 }
    };

    const profile = profiles[mode] || profiles.balanced;
    return (
        accuracyScore * profile.acc
        + coverageScore * profile.cov
        + sadnessFactor * (sadnessPriority / 3) * profile.sad
        - complexityPenalty * profile.penalty
    );
}

function getDecisionExperiments() {
    return (experimentHistory.experiments || []).filter(exp =>
        exp.id === '2clases_10x10' || exp.id === '3clases_10x10x10'
    );
}

function renderDecisionScenarios() {
    const container = document.getElementById('decision-scenario-grid');
    if (!container) return;

    const scenarios = getDecisionExperiments();
    const ranked = scenarios
        .map((exp) => ({
            ...exp,
            decisionScore: getScenarioDecisionScore(exp, decisionMode, sadnessWeight)
        }))
        .sort((a, b) => b.decisionScore - a.decisionScore);

    container.innerHTML = ranked.map((exp, index) => {
        const bestMetric = (exp.metrics || {})[exp.best_model] || {};
        return `
            <article class="decision-scenario-card ${index === 0 ? 'recommended' : ''}">
                <div class="decision-scenario-header">
                    <div>
                        <span class="decision-scenario-kicker">${index === 0 ? 'Recomendado ahora' : 'Alternativa'}</span>
                        <h3>${exp.name}</h3>
                    </div>
                    <span class="decision-score-chip">${(exp.decisionScore * 100).toFixed(1)}</span>
                </div>
                <p class="decision-scenario-subtitle">${exp.subtitle}</p>
                <div class="decision-scenario-metrics">
                    <div><span>Clases</span><strong>${exp.classes.join(' / ')}</strong></div>
                    <div><span>Dataset</span><strong>${exp.dataset_size} audios</strong></div>
                    <div><span>Mejor modelo</span><strong>${bestMetric.label || exp.best_model}</strong></div>
                    <div><span>BalAcc</span><strong>${formatPercent(exp.best_balanced_accuracy || 0)}</strong></div>
                </div>
                <p class="decision-scenario-note">${
                    exp.classes.length === 2
                        ? 'Escenario más simple y robusto para despliegue temprano.'
                        : 'Escenario con mayor cobertura emocional, pero más propenso a confusiones entre clases activas.'
                }</p>
            </article>
        `;
    }).join('');

    renderDecisionRecommendation(ranked);
}

function renderDecisionRecommendation(rankedScenarios) {
    if (!rankedScenarios || rankedScenarios.length === 0) return;

    const winner = rankedScenarios[0];
    const runnerUp = rankedScenarios[1];
    const bestMetric = (winner.metrics || {})[winner.best_model] || {};

    const badge = document.getElementById('decision-badge');
    const title = document.getElementById('decision-title');
    const summary = document.getElementById('decision-summary');
    const why = document.getElementById('decision-why');
    const tradeoff = document.getElementById('decision-tradeoff');
    const nextStep = document.getElementById('decision-next-step');
    const interpretation = document.getElementById('sensitivity-interpretation');
    const impact = document.getElementById('sensitivity-impact');
    const weightValue = document.getElementById('sadness-weight-value');

    if (weightValue) weightValue.textContent = `${sadnessWeight.toFixed(2)}x`;

    const modeLabels = {
        conservative: 'modo de máxima confiabilidad',
        balanced: 'modo balanceado',
        coverage: 'modo de mayor cobertura'
    };

    const sadnessMsg = sadnessWeight < 1.5
        ? 'Con peso bajo, priorizamos rendimiento global y simplicidad operativa.'
        : sadnessWeight < 2.5
            ? 'Con peso medio, la detección estable de Tristeza empieza a influir más en la decisión.'
            : 'Con peso alto, la decisión penaliza más los escenarios donde Tristeza pueda degradarse o confundirse.';

    const impactMsg = winner.classes.length === 2
        ? 'El escenario de 2 emociones conserva ventaja cuando importa más desplegar con menos incertidumbre y menor complejidad.'
        : 'El escenario de 3 emociones gana cuando la cobertura emocional y el valor analítico adicional justifican aceptar más riesgo de error.';

    if (interpretation) interpretation.textContent = sadnessMsg;
    if (impact) impact.textContent = impactMsg;

    if (badge) badge.textContent = `Recomendación para ${modeLabels[decisionMode]}`;
    if (title) title.textContent = `${winner.name} · ${winner.subtitle}`;
    if (summary) {
        summary.textContent = `${winner.name} obtiene la mejor puntuación de decisión en este contexto, combinando ${formatPercent(winner.best_balanced_accuracy || 0)} de balanced accuracy con ${winner.classes.length} clases de cobertura.`;
    }
    if (why) {
        why.textContent = `${bestMetric.label || winner.best_model} lidera este escenario y su relación entre precisión, cobertura y complejidad es superior a ${runnerUp ? runnerUp.name.toLowerCase() : 'las demás alternativas'} bajo el criterio actual.`;
    }
    if (tradeoff) {
        tradeoff.textContent = winner.classes.length === 2
            ? 'Pierdes cobertura emocional frente al escenario de 3 emociones, pero ganas estabilidad y una decisión de despliegue más conservadora.'
            : 'Ganas cobertura emocional y valor narrativo para el dashboard, pero aceptas una caída frente al escenario binario y más confusión entre clases activas.';
    }
    if (nextStep) {
        nextStep.textContent = winner.classes.length === 2
            ? 'Usar 2 emociones para una fase operativa inicial y recolectar más audios de Feliz o clases ambiguas antes de ampliar el sistema.'
            : 'Mantener 3 emociones como escenario analítico ampliado y priorizar nuevas grabaciones o recolección dirigida para reducir la confusión entre Enojo y Feliz.';
    }
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
            renderDecisionScenarios();
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

if (decisionModeButtons.length > 0) {
    decisionModeButtons.forEach((button) => {
        button.addEventListener('click', () => {
            decisionModeButtons.forEach((btn) => btn.classList.remove('active'));
            button.classList.add('active');
            decisionMode = button.getAttribute('data-mode') || 'balanced';
            renderDecisionScenarios();
        });
    });
}

if (sadnessWeightSlider) {
    sadnessWeightSlider.addEventListener('input', (event) => {
        sadnessWeight = Number(event.target.value || 1);
        renderDecisionScenarios();
    });
    sadnessWeight = Number(sadnessWeightSlider.value || 1);
}

renderDecisionScenarios();
