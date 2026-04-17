# Frontend — AdaptRoute Web UI

React + Vite frontend for the AdaptRoute system. Beautiful, responsive interface for querying the pipeline and understanding the architecture.

## Quick Start

```bash
npm install
npm run dev
```

Open `http://localhost:5173`

## Features

- **Home Page** — Project overview and key statistics
- **Architecture Page** — Interactive pipeline visualization with hardware flexibility info
- **Evaluation Page** — Performance benchmarks and quality metrics
- **Firewall Page** — Prompt injection detection demo
- **Deployment** — Easily switch between local and cloud endpoints

## Environment Variables

Create `.env.local`:

```env
VITE_WORKER_URL=http://localhost:5000
```

For cloud deployment (e.g., ngrok from Colab):

```env
VITE_WORKER_URL=https://<your-ngrok-url>
```

## Hardware Compatibility Note

The backend can run on diverse hardware:

- **GPU:** NVIDIA T4, RTX 3070+, A100, etc.
- **Apple Silicon:** M1/M2/M3/M4 with NPU acceleration
- **CPU:** Fallback mode for resource-constrained devices

See the **Architecture** page in-app for detailed hardware specs.
