# AdaptRoute — How to Run

AdaptRoute is a task-aware routing inference pipeline. You can run it locally or access the deployed version.

## Live Demo
The project is deployed and accessible at: **[adaptroute.vercel.app](https://adaptroute.vercel.app)**

---

## Local Development Setup

### 1. Prerequisites
- **Python 3.10+**
- **Node.js 18+**
- **CUDA-enabled GPU** (Recommended for inference, though CPU is supported)

### 2. Backend Setup (app.py)
The backend is a FastAPI server that handles model routing and inference.

1.  **Navigate to the Backend directory:**
    ```bash
    cd Backend
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up Environment Variables (Optional):**
    Create a `.env` file in the `Backend` folder if you need to set your Hugging Face token:
    ```env
    HF_TOKEN=your_huggingface_token_here
    ```
4.  **Run the server:**
    ```bash
    python app.py
    ```
    The server will start on `http://localhost:7180` by default.

### 3. Frontend Setup
The frontend is a Vite-powered React application.

1.  **Navigate to the Frontend directory:**
    ```bash
    cd Frontend
    ```
2.  **Install dependencies:**
    ```bash
    npm install
    ```
3.  **Configure API URL:**
    Create a `.env.local` file in the `Frontend` folder and point it to your local backend:
    ```env
    VITE_WORKER_URL=http://localhost:7180
    ```
4.  **Start the development server:**
    ```bash
    npm run dev
    ```
    The application will be available at `http://localhost:5173`.

---

## Project Structure
- **/Backend**: FastAPI server (app.py) and inference logic (pipeline.py).
- **/Frontend**: React application built with Vite and Tailwind CSS.
- **/Adapters**: Directory where LoRA adapters are downloaded (created automatically on first run).
- **/Datasets**: Training data and evaluation logs.
