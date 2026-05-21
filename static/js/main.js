// Global variables
let mediaRecorder = null;
let audioChunks = [];
let recordInterval = null;
let recordDuration = 0;
let isRecording = false;
let datasetLoaded = false;
let currentPlayingBtn = null;
const globalAudio = new Audio();

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
const probEnojoVal = document.getElementById('prob-enojo-val');

// Initialize SVG circular progress bar circumference
const progressCircumference = 2 * Math.PI * 64; // r=64 -> 402.12
if (progressCircle) {
    progressCircle.style.strokeDasharray = `${progressCircumference}`;
    progressCircle.style.strokeDashoffset = `${progressCircumference}`;
}
const probEnojoBar = document.getElementById('prob-enojo-bar');
const probTristezaVal = document.getElementById('prob-tristeza-val');
const probTristezaBar = document.getElementById('prob-tristeza-bar');
const metricDuration = document.getElementById('metric-duration');
const metricPitch = document.getElementById('metric-pitch');
const metricEnergy = document.getElementById('metric-energy');
const metricZcr = document.getElementById('metric-zcr');

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
    
    predictedEmotionBadge.textContent = pred;
    predictedEmotionBadge.className = 'badge ' + pred.toLowerCase();
    
    predictionPercentage.textContent = `${confidence}%`;
    
    // Set dynamic SVG progress ring and glow
    if (progressCircle) {
        const offset = progressCircumference - (confidence / 100) * progressCircumference;
        if (pred === 'Enojo') {
            progressCircle.setAttribute('stroke', 'url(#gradient-enojo)');
            progressCircle.style.filter = 'drop-shadow(0 0 10px rgba(239, 68, 68, 0.5))';
        } else if (pred === 'Tristeza') {
            progressCircle.setAttribute('stroke', 'url(#gradient-tristeza)');
            progressCircle.style.filter = 'drop-shadow(0 0 10px rgba(6, 182, 212, 0.5))';
        } else {
            progressCircle.setAttribute('stroke', 'url(#gradient-default)');
            progressCircle.style.filter = 'none';
        }
        progressCircle.style.strokeDashoffset = offset;
    }
    
    // Set probability bars
    const enojoProb = Math.round(data.probabilities['Enojo'] * 100);
    const tristezaProb = Math.round(data.probabilities['Tristeza'] * 100);
    
    probEnojoVal.textContent = `${enojoProb}%`;
    probEnojoBar.style.width = `${enojoProb}%`;
    
    probTristezaVal.textContent = `${tristezaProb}%`;
    probTristezaBar.style.width = `${tristezaProb}%`;
    
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
                tr.className = `audio-row class-${item.clase}`;
                
                tr.innerHTML = `
                    <td><strong>${item.archivo}</strong></td>
                    <td><span class="recolector-tag">${item.recolector}</span></td>
                    <td><span class="badge ${item.clase.toLowerCase()}">${item.clase}</span></td>
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
                } else if (row.classList.contains(`class-${filterValue}`)) {
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
