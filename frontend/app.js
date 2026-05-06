const form = document.getElementById("prediction-form");
const resultPanel = document.querySelector(".result-panel");
const riskProbability = document.getElementById("risk_probability");
const intensity = document.getElementById("intensity");
const riskLabel = document.getElementById("risk_label");
const confidence = document.getElementById("confidence");
const message = document.getElementById("message");
const responseState = document.getElementById("response-state");
const predictButton = document.getElementById("predict-button");
const selectedModel = document.getElementById("selected-model");
const backendStatus = document.getElementById("backend-status");
const statusDot = document.getElementById("status-dot");
const auditText = document.getElementById("audit-text");
const modelTypeSelect = document.getElementById("model_type");

const apiBaseUrl = "http://127.0.0.1:8000";

// Keep these IDs exactly aligned with backend.schemas.PredictionRequest.
const predictionFieldIds = [
    "rainfall",
    "temperature",
    "humidity",
    "wind_speed",
    "ndvi",
    "elevation",
    "latitude",
    "longitude",
    "pressure_mean",
    "solar_radiation_mean",
    "evapotranspiration_total",
    "cloud_cover_mean",
    "dewpoint_mean",
    "wind_direction_mean",
];

function formatPercent(value) {
    const numericValue = Number(value);
    if (Number.isNaN(numericValue)) {
        return "--";
    }
    return `${Math.round(numericValue * 100)}%`;
}

function setRiskTheme(label) {
    resultPanel.classList.remove("high", "moderate");

    if (label.toLowerCase().includes("high")) {
        resultPanel.classList.add("high");
    }

    if (label.toLowerCase().includes("moderate")) {
        resultPanel.classList.add("moderate");
    }
}

function buildAuditHash(payload, data) {
    const raw = `${JSON.stringify(payload)}-${data.risk_probability}-${Date.now()}`;
    let hash = 0;

    for (let index = 0; index < raw.length; index += 1) {
        hash = (hash << 5) - hash + raw.charCodeAt(index);
        hash |= 0;
    }

    return `0x${Math.abs(hash).toString(16).padStart(8, "0")}...${Date.now().toString(16).slice(-6)}`;
}

function collectPayload() {
    return predictionFieldIds.reduce((payload, fieldId) => {
        payload[fieldId] = parseFloat(document.getElementById(fieldId).value);
        return payload;
    }, {});
}

async function checkBackend() {
    try {
        const response = await fetch(`${apiBaseUrl}/health`);
        if (!response.ok) {
            throw new Error("Health check failed");
        }

        backendStatus.textContent = "Backend online";
        statusDot.classList.add("online");
        statusDot.classList.remove("offline");
    } catch (error) {
        backendStatus.textContent = "Backend offline";
        statusDot.classList.add("offline");
        statusDot.classList.remove("online");
    }
}

modelTypeSelect.addEventListener("change", () => {
    selectedModel.textContent = modelTypeSelect.options[modelTypeSelect.selectedIndex].text;
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!form.checkValidity()) {
        form.reportValidity();
        return;
    }

    const modelType = modelTypeSelect.value;
    const modelLabel = modelTypeSelect.options[modelTypeSelect.selectedIndex].text;
    const payload = collectPayload();

    predictButton.disabled = true;
    predictButton.textContent = "Running Model...";
    responseState.textContent = "Processing";
    selectedModel.textContent = modelLabel;
    message.textContent = "Sending readings to the model service and preparing the audit record.";

    try {
        const response = await fetch(`${apiBaseUrl}/predict/${modelType}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Prediction failed");
        }

        const probability = Number(data.risk_probability);
        const angle = Math.round(probability * 360);
        const auditHash = buildAuditHash(payload, data);

        resultPanel.style.setProperty("--meter-angle", `${angle}deg`);
        riskProbability.textContent = formatPercent(data.risk_probability);
        intensity.textContent = data.intensity;
        riskLabel.textContent = data.risk_label;
        confidence.textContent = formatPercent(data.confidence);
        message.textContent = data.message;
        responseState.textContent = "Complete";
        auditText.textContent = `Latest ${modelLabel.toLowerCase()} prediction logged as demo transaction ${auditHash}. This represents the hash that would be written to the blockchain audit contract.`;
        setRiskTheme(data.risk_label);
    } catch (error) {
        responseState.textContent = "Failed";
        message.textContent = error.message;
    } finally {
        predictButton.disabled = false;
        predictButton.textContent = "Run Prediction";
    }
});

checkBackend();
