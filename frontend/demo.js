/* ── KYCGuard demo page logic ─────────────────────────────── */

(() => {
  "use strict";

  const fileInput = document.getElementById("file-input");
  const filename = document.getElementById("filename");
  const checkBtn = document.getElementById("check-btn");
  const spinner = document.getElementById("spinner");
  const error = document.getElementById("error");
  const result = document.getElementById("result");
  const signalsEl = document.getElementById("signals");
  const finalEl = document.getElementById("final-decision");
  const ocrText = document.getElementById("ocr-text");
  const explanationBox = document.getElementById("explanation-box");
  const ocrWarning = document.getElementById("ocr-warning");
  const ocrSourceBadge = document.getElementById("ocr-source-badge");

  // Webcam elements
  const camStage = document.getElementById("cam-stage");
  const camVideo = document.getElementById("cam-video");
  const camCanvas = document.getElementById("cam-canvas");
  const camStill = document.getElementById("cam-still");
  const camLoading = document.getElementById("cam-loading");
  const camError = document.getElementById("cam-error");
  const camCaptureBtn = document.getElementById("cam-capture");
  const camRetakeBtn = document.getElementById("cam-retake");

  let activeFile = null;
  let activeZone = null;
  let camStream = null;
  let camStillUrl = null;

  function statusClass(s) {
    if (!s && s !== 0) return "neutral";
    if (s < 0.3) return "low";
    if (s < 0.6) return "medium";
    return "high";
  }

  function decisionBadge(d) {
    if (d === "APPROVE") return "approve";
    if (d === "REVIEW_REQUIRED") return "review";
    return "suspicious";
  }

  function highlightZone(zone) {
    document.querySelectorAll(".upload-area").forEach(z => z.classList.remove("has-file"));
    if (zone) zone.classList.add("has-file");
  }

  function onFileSelected(file, zone) {
    activeFile = file;
    activeZone = zone;
    filename.textContent = file.name;
    checkBtn.disabled = false;
    highlightZone(zone);
  }

  // ── Webcam capture flow ────────────────────────────────────
  function stopWebcam() {
    if (camStream) {
      camStream.getTracks().forEach(track => track.stop());
      camStream = null;
    }
    if (camVideo.srcObject) camVideo.srcObject = null;
  }

  function setCameraUI(mode) {
    camVideo.hidden = mode !== "live";
    camStill.hidden = mode !== "captured";
    camCaptureBtn.hidden = mode !== "live";
    camRetakeBtn.hidden = mode !== "captured";
    camLoading.hidden = mode !== "loading";
    camError.hidden = mode !== "error";
    camStage.classList.toggle("has-capture", mode === "captured");
  }

  async function startWebcam() {
    if (camStream) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraUI("error");
      return;
    }
    setCameraUI("loading");
    try {
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
      } catch (err) {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
      }
      if (camStream) {   // tab switched away while requesting
        stream.getTracks().forEach(track => track.stop());
        return;
      }
      camStream = stream;
      camVideo.srcObject = stream;
      camVideo.play().catch(() => {});
      setCameraUI("live");
    } catch (err) {
      stopWebcam();
      setCameraUI("error");
    }
  }

  function releaseStill() {
    if (camStillUrl) {
      URL.revokeObjectURL(camStillUrl);
      camStillUrl = null;
    }
    camStill.removeAttribute("src");
  }

  camCaptureBtn.addEventListener("click", () => {
    if (!camStream) return;
    const w = camVideo.videoWidth || 1280;
    const h = camVideo.videoHeight || 720;
    camCanvas.width = w;
    camCanvas.height = h;
    camCanvas.getContext("2d").drawImage(camVideo, 0, 0, w, h);
    camCanvas.toBlob(blob => {
      if (!blob) {
        setCameraUI("error");
        return;
      }
      const file = new File([blob], "webcam-capture.jpg", { type: "image/jpeg", lastModified: Date.now() });
      stopWebcam();
      releaseStill();
      camStillUrl = URL.createObjectURL(file);
      camStill.src = camStillUrl;
      setCameraUI("captured");
      onFileSelected(file, null);
    }, "image/jpeg", 0.92);
  });

  camRetakeBtn.addEventListener("click", () => {
    activeFile = null;
    filename.textContent = "";
    checkBtn.disabled = true;
    highlightZone(null);
    releaseStill();
    setCameraUI("live");
    startWebcam();
  });

  // Be a good citizen: kill the stream when leaving/navigating away
  window.addEventListener("pagehide", stopWebcam);
  window.addEventListener("beforeunload", stopWebcam);

  // Tab switching
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const wasCamera = document.querySelector(".tab-panel.active").id === "panel-camera";
      document.querySelectorAll(".tab-btn").forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      const panel = document.getElementById("panel-" + btn.dataset.tab);
      requestAnimationFrame(() => panel.classList.add("active"));
      activeFile = null;
      activeZone = null;
      filename.textContent = "";
      checkBtn.disabled = true;
      highlightZone(null);

      if (btn.dataset.tab === "camera") {
        startWebcam();
      } else if (wasCamera) {
        stopWebcam();
        releaseStill();
        setCameraUI("live");
      }
    });
  });

  // File input
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) onFileSelected(fileInput.files[0], document.getElementById("drop-zone"));
  });

  // Drag and drop
  const dropZone = document.getElementById("drop-zone");
  if (dropZone) {
    ["dragenter", "dragover"].forEach(evt => {
      dropZone.addEventListener(evt, e => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--accent)";
        dropZone.style.background = "rgba(56, 189, 248, 0.06)";
      });
    });
    ["dragleave", "drop"].forEach(evt => {
      dropZone.addEventListener(evt, e => {
        e.preventDefault();
        dropZone.style.borderColor = "";
        dropZone.style.background = "";
      });
    });
    dropZone.addEventListener("drop", e => {
      e.preventDefault();
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        fileInput.files = files;
        onFileSelected(files[0], dropZone);
      }
    });
  }

  // Check button
  checkBtn.addEventListener("click", async () => {
    const file = activeFile;
    if (!file) return;

    const docType = document.querySelector('input[name="doc_type"]:checked').value;

    checkBtn.classList.add("loading");
    spinner.style.display = "flex";
    result.style.display = "none";
    result.classList.remove("visible");
    error.style.display = "none";
    ocrWarning.style.display = "none";
    checkBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", file);
    formData.append("doc_type", docType);

    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `Server error (${res.status})`);
      }
      const data = await res.json();

      const sigs = [
        { label: "Visual Forensics", score: data.visual_forensics.risk_score, status: data.visual_forensics.status, checks: [] },
        { label: "OCR / Text Consistency", score: data.semantic_validation.risk_score, status: data.semantic_validation.risk_score === null ? "N/A" : (data.semantic_validation.risk_score === 0 ? "LOW_RISK" : "HIGH_RISK"), checks: data.semantic_validation.failed_checks },
        { label: "Document Structure", score: data.structural_validation.risk_score, status: data.structural_validation.risk_score === 0 ? "VALID" : "INVALID", checks: data.structural_validation.failed_checks },
        { label: "Layout Consistency", score: data.layout_validation.risk_score, status: data.layout_validation.risk_score === 0 ? "LOW_RISK" : "HIGH_RISK", checks: data.layout_validation.failed_checks },
      ];

      signalsEl.innerHTML = sigs.map(s => {
        const sc = statusClass(s.score);
        const displayScore = s.score !== null && s.score !== undefined ? (s.score * 100).toFixed(0) + "%" : "—";
        const checksHtml = (s.checks && s.checks.length)
          ? `<div class="checks">${s.checks.map(c => "• " + c).join("<br>")}</div>`
          : "";
        let statusClassExtra = "";
        if (s.status === "LOW_RISK") statusClassExtra = "status-low";
        else if (s.status === "MEDIUM_RISK" || s.status === "REVIEW_REQUIRED") statusClassExtra = "status-medium";
        else if (s.status === "HIGH_RISK" || s.status === "SUSPICIOUS" || s.status === "INVALID") statusClassExtra = "status-high";
        else if (s.status === "VALID") statusClassExtra = "status-valid";
        return `
          <div class="signal-card ${statusClassExtra}">
            <div class="signal-label">${s.label}</div>
            <div class="signal-status"><span class="badge ${sc}">${s.status}</span></div>
            <div class="signal-score">Score: ${displayScore}</div>
            ${checksHtml}
          </div>
        `;
      }).join("");

      const fd = data.final_decision;
      const reasons = [];
      if (data.semantic_validation.failed_checks.length) reasons.push(...data.semantic_validation.failed_checks);
      if (data.structural_validation.failed_checks.length) reasons.push(...data.structural_validation.failed_checks);
      if (data.layout_validation.failed_checks.length) reasons.push(...data.layout_validation.failed_checks);
      if (data.visual_forensics.status !== "LOW_RISK") reasons.push("Visual forensics detected potential manipulation");

      const riskPercent = (fd.risk_score * 100).toFixed(1);
      const riskBarClass = statusClass(fd.risk_score);

      finalEl.innerHTML = `
        <div class="fd-label">Final Risk Assessment</div>
        <div class="fd-score"><span class="badge ${statusClass(fd.risk_score)}">${fd.category}</span></div>
        <div class="fd-meta">Risk score: ${riskPercent}%</div>
        <div style="margin-top:0.6rem"><span class="badge ${decisionBadge(fd.decision)}">${fd.decision.replace("_", " ")}</span></div>
        <div class="risk-bar-wrap">
          <div class="risk-bar ${riskBarClass}" id="risk-bar" style="width: 0%;"></div>
        </div>
        ${reasons.length ? `<div class="fd-reasons"><strong>Reasons flagged:</strong><ul>${reasons.map(r => `<li>${r}</li>`).join("")}</ul></div>` : ""}
      `;

      const src = (data.ocr && data.ocr.source) ? data.ocr.source : "easyocr";
      ocrSourceBadge.textContent = src === "mistral_fallback" ? "Mistral OCR (fallback)" : "EasyOCR";
      ocrSourceBadge.className = "badge " + (src === "mistral_fallback" ? "source" : "neutral");

      ocrText.textContent = data.ocr.extracted_text_sample || "(no text extracted)";

      if (data.ocr.quality_warning) {
        ocrWarning.innerHTML = `<i data-lucide="triangle-alert" class="warn-icon" aria-hidden="true"></i><span>${data.ocr.quality_warning}</span>`;
        if (window.lucide) lucide.createIcons();
        ocrWarning.style.display = "flex";
      } else {
        ocrWarning.style.display = "none";
      }

      if (data.explanation) {
        explanationBox.textContent = data.explanation;
        explanationBox.style.display = "block";
      } else {
        explanationBox.style.display = "none";
      }

      // Reset and replay card animations
      document.querySelectorAll(".signal-card, .final-decision, .ocr-section, .explanation-box").forEach(el => {
        el.style.animation = "none";
        el.offsetHeight;
        el.style.animation = "";
      });

      result.style.display = "block";
      requestAnimationFrame(() => result.classList.add("visible"));

      // Animate risk bar after a brief delay so the transition is visible
      setTimeout(() => {
        const bar = document.getElementById("risk-bar");
        if (bar) bar.style.width = riskPercent + "%";
      }, 100);

    } catch (e) {
      error.textContent = "Error: " + e.message;
      error.style.display = "block";
    } finally {
      spinner.style.display = "none";
      checkBtn.classList.remove("loading");
      checkBtn.disabled = false;
    }
  });

  // ── Intersection Observer for scroll-triggered reveals ────
  const observerOptions = { threshold: 0.15, rootMargin: "0px 0px -40px 0px" };
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  document.querySelectorAll(".reveal").forEach(el => observer.observe(el));

  // Render Lucide icons (also re-invoked after dynamic content is injected)
  if (window.lucide) lucide.createIcons();
})();