# Soccer Video Analysis

Version 0.4 tracks people, classifies their match roles from kit colors, detects
sports balls, filters likely off-field detections, and writes an annotated MP4
plus per-detection CSV data.

## Version history

Each milestone has its own permanent README so that later improvements do not
erase the reasoning, parameters, results, and limitations of earlier versions.

| Version | Status | Milestone | Documentation |
| --- | --- | --- | --- |
| V0.1 | Complete | Pretrained YOLO player and ball detection | [V0.1 README](docs/versions/v0.1.md) |
| V0.2 | Complete | Field filtering and separate confidence thresholds | [V0.2 README](docs/versions/v0.2.md) |
| V0.3 | Complete | ByteTrack IDs for people and independent ball detection | [V0.3 README](docs/versions/v0.3.md) |
| V0.4 | Complete | Team, goalkeeper, and referee classification | [V0.4 README](docs/versions/v0.4.md) |

The complete milestone index is available in
[docs/versions/README.md](docs/versions/README.md). The root README describes
the latest implemented version only.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run V0.4

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

The first run downloads `yolo11n.pt`. V0.4 writes:

- `outputs/v04_team_classification.mp4`
- `outputs/v04_detections.csv`

The pipeline uses separate confidence thresholds for players and balls. People
are sent to ByteTrack while balls use an independent detection pass, so balls
never receive a track ID. After grass filtering, the central upper-body region
of each person box is classified using match-specific HSV kit colors. A rolling
vote by track ID prevents one noisy frame from immediately changing an
established role. Two inference passes are run per frame. The player display
threshold stays at 0.5, while ByteTrack sees weaker person detections to help
associate people across frames.

Optional tuning flags:

```bash
python src/detect.py \
  --player-conf 0.5 \
  --ball-conf 0.18 \
  --min-field-green-ratio 0.25 \
  --team-a-color red \
  --team-b-color white \
  --referee-color black \
  --team-a-goalkeeper-color none \
  --team-b-goalkeeper-color blue
```

Current labels:

- Red kits are shown as red `Team A #ID` boxes.
- White kits are shown as cyan `Team B #ID` boxes.
- Blue kits are shown as magenta `Team B GK #ID` boxes for this match.
- `Team A GK` is supported but remains disabled until its kit color is
  confirmed in the source video.
- Black referee kits are shown as yellow `Referee #ID` boxes.
- Ambiguous or unconfigured kits use gray `Unknown #ID` boxes.
- `sports ball` is shown as a blue `ball` box.

The CSV contains zero-based frame numbers, timestamps, track IDs, roles,
bounding boxes, and YOLO detection confidence. Role classification is based on
kit color and is not a custom-trained referee detector.

## Earlier comparisons

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
boundary resets are not implemented. People without an ID use the current
frame's kit classification without an `#ID` suffix. The grass filter is a
heuristic and can still accept some sideline people or reject valid detections.

Lowering ball confidence to 0.18 retains more candidates, including potential
false positives such as pitch markings. It does not fill missing detections or
guarantee continuous ball tracking. More boxes alone do not prove better accuracy.

Kit-color classification is match-specific. Distant players, shadows,
occlusion, close-ups, and goalkeeper kits that are not configured can produce
`Unknown` or an incorrect role. Temporal voting reduces frame-to-frame flicker
but cannot fix a ByteTrack ID switch.

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
