# Soccer Video Analysis

Version 0.6 adds a restricted, manually calibrated pitch mapping experiment on
an 11.9-second continuous shot. It reuses the V0.5 people and ball detections,
then maps only objects inside the right penalty area to a synchronized 2D view.
This is a local demonstration, not automatic full-match camera calibration.

## Version history

Each milestone has its own permanent README so that later improvements do not
erase the reasoning, parameters, results, and limitations of earlier versions.

| Version | Status | Milestone | Documentation |
| --- | --- | --- | --- |
| V0.1 | Complete | Pretrained YOLO player and ball detection | [V0.1 README](docs/versions/v0.1.md) |
| V0.2 | Complete | Field filtering and separate confidence thresholds | [V0.2 README](docs/versions/v0.2.md) |
| V0.3 | Complete | ByteTrack IDs for people and independent ball detection | [V0.3 README](docs/versions/v0.3.md) |
| V0.4 | Complete | Team, goalkeeper, and referee classification | [V0.4 README](docs/versions/v0.4.md) |
| V0.5 | Complete | Ball association and short-gap prediction | [V0.5 README](docs/versions/v0.5.md) |
| V0.6 | Implemented, local experiment | Keyframe homography and restricted 2D pitch mapping | [V0.6 README](docs/versions/v0.6.md) |

The complete milestone index is available in
[docs/versions/README.md](docs/versions/README.md). The root README describes
the latest implemented version only.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run V0.6

Install FFmpeg as well as the Python requirements. The selected interval is
**74.7 seconds to 86.6 seconds**, using the original video's timeline (not the
broadcast match clock). Extract the CFR source's frames 2241 through 2597:

```bash
ffmpeg -n -i data/soccervideo_cfr.mp4 \
  -vf "trim=start_frame=2241:end_frame=2598,setpts=PTS-STARTPTS" \
  -an -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  data/v06_homography_clip.mp4
```

The local clip has already been extracted. With the full-source V0.5 CSV
available, generate the V0.6 output and inspect the calibration:

```bash
python src/map_pitch.py
python src/calibrate.py --review
```

Outputs:

- `outputs/v06_pitch_mapping.mp4`: detected objects and synchronized 2D map.
- `outputs/v06_pitch_mapping.csv`: original detections plus mapping coordinates,
  validity, original frame/time, and observed/predicted ball state.
- `outputs/v06_pitch_mapping.json`: input hashes, counts, calibration fit
  residuals, and limitations.
- `outputs/v06_calibration_review.jpg`: projected lines at keyframes and
  intermediate frames for visual inspection.

The map uses 13 manual keyframes and interpolated image-space reference points
to compensate for camera movement. People use the bottom-center of their boxes;
balls use box centers. Objects outside the configured penalty-area polygon are
kept in the CSV with empty pitch coordinates, not silently clamped into the map.
Pitch dimensions are nominally 105 x 68 m; airborne balls are ground-plane
projections, not real ground positions. See the
[V0.6 experiment record](docs/versions/v0.6.md) for results and limitations.

Output files are protected by default. Regenerate explicitly with
`python src/map_pitch.py --overwrite`, or choose a new `--output` filename.
V0.5 code and previous videos are unchanged.

## Run The Detection Stage (V0.5)

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

The first run downloads `yolo11n.pt`. V0.5 writes:

- `outputs/v05_ball_tracking.mp4`
- `outputs/v05_detections.csv`

The pipeline uses separate confidence thresholds for players and balls. People
are sent to ByteTrack while balls use an independent detection pass, so balls
never receive a track ID. After grass filtering, the central upper-body region
of each person box is classified using match-specific HSV kit colors. A rolling
vote by track ID prevents one noisy frame from immediately changing an
established role. Two inference passes are run per frame. The player display
threshold stays at 0.5, while ByteTrack sees weaker person detections to help
associate people across frames. Ball candidates are passed to a separate
single-ball tracker. It requires two nearby detections to confirm a track,
associates later candidates with a predicted image-space position, and fills up
to eight missing frames with a constant-velocity estimate.

Optional tuning flags:

```bash
python src/detect.py \
  --player-conf 0.5 \
  --ball-conf 0.18 \
  --ball-confirm-hits 2 \
  --ball-max-missing 8 \
  --ball-max-distance 120 \
  --ball-max-box-side 48 \
  --ball-max-aspect-ratio 1.8 \
  --ball-min-texture-std 25 \
  --ball-min-saturation 180 \
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
- A detected sports ball uses a solid blue `ball` box.
- A short-gap estimate uses a dashed blue `ball predicted` box.

The CSV contains zero-based frame numbers, timestamps, track IDs, roles,
bounding boxes, YOLO detection confidence, and `ball_state`. Predicted ball rows
leave confidence empty because no YOLO detection exists on that frame. Role
classification is based on kit color and is not a custom-trained referee
detector.

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
false positives such as pitch markings. V0.5 combines box shape, local texture,
color saturation, player proximity, and temporal association to select at most
one result and bridge short gaps. These are match-specific heuristics, so a
visually similar pitch marking can still establish a false track.
Constant-velocity predictions do not represent new YOLO detections and can
drift during fast kicks or camera movement. Shot-boundary reset is not
implemented yet.

Kit-color classification is match-specific. Distant players, shadows,
occlusion, close-ups, and goalkeeper kits that are not configured can produce
`Unknown` or an incorrect role. Temporal voting reduces frame-to-frame flicker
but cannot fix a ByteTrack ID switch.

Local spot checks on `soccervideo_cfr.mp4`: many visible people keep their IDs
between 20.0 and 20.5 seconds, but some switch. V0.5 suppresses the previously
documented penalty-spot false positive around 46 seconds and bridges a short
ball-detection gap around 80 seconds. These checks are examples, not a
ground-truth accuracy evaluation.

```bash
python -m unittest discover -s tests -v
```

The local input video, generated output, and model weights are intentionally
ignored by Git. The repository stores the code and the reproducible command.
