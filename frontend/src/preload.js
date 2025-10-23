const { contextBridge } = require("electron");

const BACKEND_URL =
  process.env.BACKEND_URL || process.env.ELECTRON_BACKEND_URL || "http://127.0.0.1:8000";

const request = async (path, options = {}) => {
  const response = await fetch(`${BACKEND_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with ${response.status}`);
  }

  return response.json();
};

contextBridge.exposeInMainWorld("investLens", {
  backendUrl: BACKEND_URL,
  getHealth: () => request("/health"),
  getMarketSummary: (ticker) =>
    request(`/market/summary?ticker=${encodeURIComponent(ticker)}`),
  generateInsight: (payload) =>
    request("/insights/generate", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  runOcr: async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${BACKEND_URL}/ocr/extract`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `Request failed with ${response.status}`);
    }

    return response.json();
  },
});

