# Soccer Video Analysis

Version 0.2 detects people and sports balls in a constant-frame-rate soccer
video, filters likely off-field detections, and writes an annotated MP4.

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

The detector uses separate confidence thresholds for players and balls. It
also checks the grass support around each detection anchor: the bottom-center
of a player box, or the center of a ball box. This reduces false positives in
the stands and around advertising boards without adding tracking yet.

Optional tuning flags:

```bash
python src/detect.py \
  --player-conf 0.5 \
  --ball-conf 0.3 \
  --min-field-green-ratio 0.25
```

Current pretrained labels:

- `person` is shown as a green `player` box.
- `sports ball` is shown as a blue `ball` box.
- A red `referee` color is reserved for a future custom three-class model;
  the pretrained model cannot reliably separate referees from players.

The local input video, generated output, and model weights are intentionally
ignored by Git. The repository stores the code and the reproducible command.
