# Soccer Video Analysis

Version 0.1 detects people and sports balls in a constant-frame-rate soccer video
and writes an annotated MP4.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run detection

The project uses the constant-frame-rate input `data/soccervideo_cfr.mp4` by
default. For a variable-frame-rate source, convert a local copy first:

```bash
ffmpeg -y -i data/soccervideo.mp4 \
  -vf "fps=30" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -an \
  data/soccervideo_cfr.mp4
```

Then run:

```bash
python src/detect.py
```

The first run downloads `yolo11n.pt`. The annotated video is written to
`outputs/detected_cfr.mp4`.

Current pretrained labels:

- `person` is shown as a green `player` box.
- `sports ball` is shown as a blue `ball` box.
- A red `referee` color is reserved for a future custom three-class model;
  the pretrained model cannot reliably separate referees from players.

The local input video, generated output, and model weights are intentionally
ignored by Git. The repository stores the code and the reproducible command.
