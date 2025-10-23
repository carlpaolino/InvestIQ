const healthStatusEl = document.getElementById("health-status");
const tickerInput = document.getElementById("ticker-input");
const refreshButton = document.getElementById("refresh-button");
const marketSummaryEl = document.getElementById("market-summary");
const ocrFileInput = document.getElementById("ocr-file");
const ocrOutputEl = document.getElementById("ocr-output");
const insightButton = document.getElementById("insight-button");
const insightOutputEl = document.getElementById("insight-output");

const renderError = (target, error) => {
  target.innerHTML = `<span style="color:#ff3b30;">${error}</span>`;
};

const renderMarketSummary = (summary) => {
  marketSummaryEl.innerHTML = `
    <div><strong>${summary.symbol}</strong> @ $${summary.price.toFixed(2)}</div>
    <div>Change: ${summary.change >= 0 ? "+" : ""}${summary.change.toFixed(
      2,
    )} (${summary.change_percent >= 0 ? "+" : ""}${summary.change_percent.toFixed(
    2,
  )}%)</div>
    <div>Range: $${summary.day_low.toFixed(2)} – $${summary.day_high.toFixed(2)}</div>
    <div>Volume: ${summary.volume ? summary.volume.toLocaleString() : "n/a"}</div>
    <div>Last update: ${new Date(summary.timestamp).toLocaleString()}</div>
  `;
};

const renderInsight = (insight) => {
  insightOutputEl.innerHTML = `
    <div class="sentiment" data-sentiment="${insight.sentiment}">
      ${insight.sentiment.toUpperCase()}
    </div>
    <div><strong>${insight.headline}</strong></div>
    <div>${insight.rationale}</div>
    <div style="margin-top:8px;font-size:12px;">Generated: ${new Date(
      insight.timestamp,
    ).toLocaleTimeString()}</div>
  `;
};

const checkHealth = async () => {
  try {
    const response = await window.investLens.getHealth();
    healthStatusEl.textContent = `Backend online • ${window.investLens.backendUrl}`;
    healthStatusEl.style.color = "#4cd964";
    return response;
  } catch (error) {
    healthStatusEl.textContent = `Backend offline: ${error.message}`;
    healthStatusEl.style.color = "#ff3b30";
    throw error;
  }
};

const refreshMarketSummary = async () => {
  const ticker = tickerInput.value.trim().toUpperCase();
  if (!ticker) {
    renderError(marketSummaryEl, "Enter a ticker symbol to fetch data.");
    return;
  }

  refreshButton.disabled = true;
  renderError(marketSummaryEl, "Loading latest market data…");

  try {
    const summary = await window.investLens.getMarketSummary(ticker);
    renderMarketSummary(summary);
    return summary;
  } catch (error) {
    renderError(marketSummaryEl, error.message);
    throw error;
  } finally {
    refreshButton.disabled = false;
  }
};

const handleInsightGeneration = async () => {
  const ticker = tickerInput.value.trim().toUpperCase();
  const ocrText = ocrOutputEl.value.trim() || undefined;
  if (!ticker) {
    renderError(insightOutputEl, "Enter a ticker first.");
    return;
  }

  insightButton.disabled = true;
  renderError(insightOutputEl, "Generating insight…");

  try {
    const insight = await window.investLens.generateInsight({
      ticker,
      ocr_text: ocrText,
    });
    renderInsight(insight);
  } catch (error) {
    renderError(insightOutputEl, error.message);
  } finally {
    insightButton.disabled = false;
  }
};

const handleOcrUpload = async (event) => {
  const [file] = event.target.files || [];
  if (!file) return;

  insightOutputEl.innerHTML =
    "<span style='color:#7f8c9b;'>Using latest OCR result for insights once generated.</span>";
  ocrOutputEl.value = "Extracting text…";

  try {
    const result = await window.investLens.runOcr(file);
    ocrOutputEl.value = result.text || "";
    if (result.confidence !== undefined && result.confidence !== null) {
      ocrOutputEl.value = `${ocrOutputEl.value}\n\nConfidence: ${result.confidence}%`;
    }
  } catch (error) {
    ocrOutputEl.value = "";
    renderError(marketSummaryEl, "OCR failed. Check backend logs.");
    renderError(insightOutputEl, error.message);
  }
};

document.addEventListener("DOMContentLoaded", async () => {
  try {
    await checkHealth();
    await refreshMarketSummary();
  } catch (error) {
    console.error(error);
  }
});

refreshButton.addEventListener("click", refreshMarketSummary);
ocrFileInput.addEventListener("change", handleOcrUpload);
insightButton.addEventListener("click", handleInsightGeneration);
