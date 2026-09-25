# Soccer Video Analysis

Version 0.3 tracks people and detects sports balls in a constant-frame-rate
soccer video, filters likely off-field detections, and writes an annotated MP4.

## Version history

Each milestone has its own permanent README so that later improvements do not
erase the reasoning, parameters, results, and limitations of earlier versions.

| Version | Status | Milestone | Documentation |
| --- | --- | --- | --- |
| V0.1 | Complete | Pretrained YOLO player and ball detection | [V0.1 README](docs/versions/v0.1.md) |
| V0.2 | Complete | Field filtering and separate confidence thresholds | [V0.2 README](docs/versions/v0.2.md) |
| V0.3 | Complete | ByteTrack IDs for people and independent ball detection | [V0.3 README](docs/versions/v0.3.md) |
| V0.4 | Planned | Team and referee classification with temporal voting | [V0.4 README](docs/versions/v0.4.md) |

The complete milestone index is available in
[docs/versions/README.md](docs/versions/README.md). The root README describes
the latest implemented version only.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run V0.3

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

The first run downloads `yolo11n.pt`. The tracked video is written to
`outputs/v03_player_tracking.mp4`.

The pipeline uses separate confidence thresholds for players and balls. People
are sent to ByteTrack while balls use an independent detection pass, so balls
never receive a track ID. Grass support filtering is applied before annotations
are drawn. Two inference passes are run per frame; this costs more processing
time than V0.2. The player display threshold stays at 0.5. ByteTrack also sees
weaker person detections to help associate people across frames.

Optional tuning flags:

```bash
python src/detect.py \
  --player-conf 0.5 \
  --ball-conf 0.18 \
  --min-field-green-ratio 0.25
```

Current pretrained labels:

- `person` is shown as a green `player #ID` box when ByteTrack has assigned an
  ID.
- `sports ball` is shown as a blue `ball` box.
- A red `referee` color is reserved for a future custom three-class model;
  the pretrained model cannot reliably separate referees from players.

## V0.2 comparison

The local comparison uses the same input, frame numbers, and 30 FPS timeline:

| File | Version | Player confidence | Ball confidence |
| --- | --- | --- | --- |
| `outputs/v02_player_detection.mp4` | V0.2 baseline, no IDs | 0.5 | 0.3 |
| `outputs/v03_player_tracking.mp4` | V0.3, people with IDs | 0.5 | 0.18 |
| `outputs/comparison_v02_v03.mp4` | V0.2 left, V0.3 right | | |

The existing V0.2 video is preserved. If it needs to be regenerated, its code
is available at commit `009ac8a`:

```bash
git show 009ac8a:src/detect.py > /tmp/soccer_detect_v02.py
python /tmp/soccer_detect_v02.py \
  --source data/soccervideo_cfr.mp4 \
  --output outputs/v02_player_detection.mp4 \
  --player-conf 0.5 --ball-conf 0.3
```

To rebuild the side-by-side video without the caption band, run after both
versions finish:

```bash
ffmpeg -n \
  -i outputs/v02_player_detection.mp4 -i outputs/v03_player_tracking.mp4 \
  -filter_complex "[0:v]setpts=PTS-STARTPTS[left];[1:v]setpts=PTS-STARTPTS[right];[left][right]hstack=inputs=2[v]" \
  -map "[v]" -an -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  outputs/comparison_v02_v03.mp4
```

`-n` protects any existing comparison; choose another output filename to keep
both. The comparison changes both people tracking and the ball threshold, so it
is a version comparison rather than a controlled tracking accuracy benchmark.

## Limitations and tests

Track IDs are temporary associations, not jersey numbers or player identities.
They are intended for continuous shots; occlusions, fast camera movement, cuts,
and replays can cause switches or reuse. Cross-shot identity recovery and shot
boundary resets are not implemented. Missing IDs fall back to a plain player
label. The grass filter is a heuristic and can still accept some sideline people
or reject valid detections.

Lowering ball confidence to 0.18 retains more candidates, including potential
false positives such as pitch markings. It does not fill missing detections or
guarantee continuous ball tracking. More boxes alone do not prove better accuracy.

Local spot checks on `soccervideo_cfr.mp4`: many visible people keep their IDs
between 20.0 and 20.5 seconds, but some switch. At 46 seconds the 0.18 version
accepts a penalty-spot false positive scored 0.26 that V0.2 suppresses. At 80
seconds the visible ball is still missed. These checks are examples, not a
ground-truth accuracy evaluation.

```bash
python -m unittest discover -s tests -v
```

The local input video, generated output, and model weights are intentionally
ignored by Git. The repository stores the code and the reproducible command.
