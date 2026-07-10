// static/js/report.js
// Handles the "Diagnosis Report" modal: showing prediction summary,
// generating a PDF via POST /report, and downloading it via GET /download-report/:id.

document.addEventListener('DOMContentLoaded', () => {

    // --- Element refs (must match your HTML ids) ---
    const reportModalOverlay = document.getElementById('report-modal-overlay');
    const reportModal = document.getElementById('report-modal');
    const generatePdfBtn = document.getElementById('generate-pdf-btn');
    const downloadPdfBtn = document.getElementById('download-pdf-btn');
    const reportPrediction = document.getElementById('report-prediction');
    const reportConfidence = document.getElementById('report-confidence');
    const reportAbnormal = document.getElementById('report-abnormal');

    // Module-level state: the last prediction data received from /predict
    let currentPredictionData = null;

    // --- FIX: these two functions were referenced (openModal / closeModal)
    // but never defined anywhere in the original file. That caused a
    // ReferenceError the moment the script tried to build window.ReportModule,
    // which meant window.ReportModule was NEVER assigned, which in turn meant:
    //   - the report modal never opened after a successful /predict call
    //   - the Generate PDF / Download PDF click listeners never got attached
    //   - any downstream calls into ReportModule silently no-op'd
    function openModal() {
        reportModal.classList.add('visible');
        reportModalOverlay.classList.add('visible');
    }

    function closeModal() {
        reportModal.classList.remove('visible');
        reportModalOverlay.classList.remove('visible');
    }

    /** Called by index.html's renderResults() after a successful /predict response */
    function setPredictionData(data) {
        currentPredictionData = data;

        // Reset report action buttons whenever a new prediction comes in
        downloadPdfBtn.classList.add('hidden');
        downloadPdfBtn.removeAttribute('data-report-id');
        generatePdfBtn.classList.remove('hidden');
        generatePdfBtn.disabled = false;
        generatePdfBtn.textContent = 'Generate PDF';

        if (!data) {
            closeModal();
            return;
        }

        reportPrediction.textContent = data.prediction ?? '--';

        let confStr = data.confidence;
        if (!isNaN(parseFloat(data.confidence))) {
            confStr = (parseFloat(data.confidence) * 100).toFixed(1) + '%';
        }
        reportConfidence.textContent = confStr ?? '--';
        reportAbnormal.textContent = data.abnormal_region_count ?? 0;

        // Prediction + confidence are now populated -> show the modal
        openModal();
    }

    /** Clears stored prediction data and closes the modal (call on new upload) */
    function clearPredictionData() {
        currentPredictionData = null;
        closeModal();
        downloadPdfBtn.classList.add('hidden');
        downloadPdfBtn.removeAttribute('data-report-id');
        generatePdfBtn.classList.remove('hidden');
    }

    /** Small helper to surface errors — reuses #error-alert if present, else alert() */
    function showError(message) {
        const errorAlert = document.getElementById('error-alert');
        const errorAlertText = document.getElementById('error-alert-text');
        if (errorAlert && errorAlertText) {
            errorAlertText.textContent = message;
            errorAlert.classList.remove('hidden');
            clearTimeout(showError._t);
            showError._t = setTimeout(() => errorAlert.classList.add('hidden'), 6000);
        } else {
            alert(message);
        }
    }

    /** POST /report with the current prediction data, get back a report_id */
    async function generateReport() {
        if (!currentPredictionData) {
            showError('No analysis data available. Please analyze a scan first.');
            return;
        }

        if (!currentPredictionData.heatmap_url) {
            showError('This scan has no Grad-CAM heatmap, so a report cannot be generated.');
            return;
        }

        generatePdfBtn.disabled = true;
        const originalText = generatePdfBtn.textContent;
        generatePdfBtn.textContent = 'Generating...';

        try {
            const response = await fetch('/report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(currentPredictionData)
            });

            if (!response.ok) {
                let detail = 'Failed to generate report';
                try {
                    const errBody = await response.json();
                    detail = errBody?.detail || detail;
                } catch (_) { /* ignore parse errors */ }
                showError(detail);
                return;
            }

            const reportData = await response.json();

            if (!reportData.report_id) {
                showError('Report generated but no report_id was returned.');
                return;
            }

            downloadPdfBtn.dataset.reportId = reportData.report_id;
            downloadPdfBtn.classList.remove('hidden');
            generatePdfBtn.classList.add('hidden');
            showError('✅ Report generated successfully!');

        } catch (error) {
            showError('Error generating report: ' + error.message);
            console.error('generateReport error:', error);
        } finally {
            generatePdfBtn.disabled = false;
            generatePdfBtn.textContent = originalText;
        }
    }

    /** Opens the generated PDF in a new tab via GET /download-report/:id */
    function downloadReport() {
        const reportId = downloadPdfBtn.dataset.reportId;
        if (!reportId) {
            showError('No report available to download yet.');
            return;
        }
        window.open(`/download-report/${encodeURIComponent(reportId)}`, '_blank');
    }

    // --- Wire up exactly ONE listener per element (avoids duplicate-fire bugs) ---
    if (generatePdfBtn) {
        generatePdfBtn.addEventListener('click', generateReport);
    }
    if (downloadPdfBtn) {
        downloadPdfBtn.addEventListener('click', downloadReport);
    }
    if (reportModalOverlay) {
        reportModalOverlay.addEventListener('click', closeModal);
    }
    // Also allow the header "X" close button to work, if present
    const reportCloseBtn = document.getElementById('report-close-btn');
    if (reportCloseBtn) {
        reportCloseBtn.addEventListener('click', closeModal);
    }

    // Expose a small public API for the rest of your app to call into
    window.ReportModule = {
        setPredictionData,
        clearPredictionData,
        generateReport,
        downloadReport,
        openModal,
        closeModal
    };
});