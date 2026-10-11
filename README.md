# Placement Predictor

A FastAPI service that predicts whether a student will be placed, using a trained scikit-learn model, plus a single-file web UI to use it.

## Project structure

```
project/
├── api.py                      # FastAPI app
├── placement_model.joblib      # trained model
├── columns.joblib              # list of training feature columns
├── placement_predictor.html    # web UI (HTML + CSS + JS in one file)
├── requirements.txt
└── README.md
```

All files must be in the same folder. `api.py` loads the model files by relative path, so run the server from that folder.

## Requirements

- Python 3.9 or newer
- The packages in `requirements.txt`
- A scikit-learn version that matches the one used to train the model (a mismatch can cause warnings or wrong predictions)

## Setup

1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Mac/Linux
   source venv/bin/activate
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Enable CORS

The web UI calls the API from a browser, so `api.py` must allow cross-origin requests. Add this after `app = FastAPI()`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

`allow_origins=["*"]` is fine for local development. Restrict it to your frontend's domain before deploying.

## Serve the UI from the API (recommended)

Add this to the bottom of `api.py` so the page is served by the same server:

```python
from fastapi.responses import FileResponse
from pathlib import Path

@app.get("/ui")
def ui():
    return FileResponse(Path(__file__).parent / "placement_predictor.html")
```

## Run the server

```bash
uvicorn api:app --reload
```

The server starts at `http://127.0.0.1:8000`.

## Usage

| Address | Purpose |
|---|---|
| http://127.0.0.1:8000/ | Health check. Returns `{"message": "API is running!"}` |
| http://127.0.0.1:8000/ui | Web UI (after adding the `/ui` route) |
| http://127.0.0.1:8000/docs | Interactive API documentation (Swagger UI) |

### Prediction endpoint

`POST /predict` with a JSON body:

```json
{
  "gender": "male",
  "study_hours": 4,
  "attendance": 85,
  "sleep_hours": 7,
  "internet_usage": 2,
  "assignments_completed": 90,
  "previous_score": 70,
  "extracurricular": "yes",
  "exam_score": 75,
  "branch": "CSE"
}
```

Response:

```json
{ "prediction": "Placed" }
```

or `{ "prediction": "Not Placed" }`.

### Input notes

- `gender`: `male` or `m` maps to 1; anything else maps to 0.
- `extracurricular`: `yes` maps to 1; anything else maps to 0.
- `branch`: case-insensitive and trimmed. It must match a branch seen during training. Unknown branches are encoded as all zeros.

### Example with curl

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"male","study_hours":4,"attendance":85,"sleep_hours":7,"internet_usage":2,"assignments_completed":90,"previous_score":70,"extracurricular":"yes","exam_score":75,"branch":"CSE"}'
```

## Troubleshooting

**`Error loading ASGI app. Could not import module "api"`**
Run `python -c "import api"` in the project folder to see the underlying error. Common causes:
- Missing package: run `pip install -r requirements.txt`.
- Missing `columns.joblib` or `placement_model.joblib`: copy them into the project folder.
- Wrong folder: run uvicorn from the folder that contains `api.py`.
- File saved as `api.py.txt` (Windows hides extensions by default).

**Web page says "Failed to load page"**
- Open it through `http://127.0.0.1:8000/ui` rather than double-clicking the file.
- Check that the file is named `placement_predictor.html`, not `.html.txt`.

**UI shows "API unreachable"**
- Confirm the server is running and http://127.0.0.1:8000/ opens.
- Confirm the Base URL in the UI matches the server address.
- Confirm CORS is enabled in `api.py`.

**`InconsistentVersionWarning` from scikit-learn**
Install the scikit-learn version used during training: `pip install scikit-learn==<version>`.

**`422 Unprocessable Entity`**
A field is missing or has the wrong type. The UI shows which field; with curl, check the JSON body against the example above.
