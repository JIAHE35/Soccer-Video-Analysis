# Version History

This directory preserves the development story of Soccer Video Analysis. Each
version document is a snapshot of one milestone rather than a replacement for
the previous README.

| Version | Status | Main improvement | Git reference |
| --- | --- | --- | --- |
| [V0.1](v0.1.md) | Complete | Baseline YOLO detection | `3b6567b` |
| [V0.2](v0.2.md) | Complete | Grass-field filtering and tuned thresholds | `155f1d3`, `009ac8a` |
| [V0.3](v0.3.md) | Complete | ByteTrack IDs for people | `7e6d346` |
| [V0.4](v0.4.md) | Complete | Team, goalkeeper, and referee classification | V0.4 implementation commit |
| [V0.5](v0.5.md) | Complete | Single-ball association and short-gap prediction | V0.5 implementation commit |
| [V0.6](v0.6.md) | Implemented, local experiment | Manual keyframe homography for a restricted penalty-area clip | V0.6 implementation commit |
| [V0.7](v0.7.md) | Implemented, local experiment | Local track fragments, team position heatmaps, and data-quality statistics | V0.7 implementation commit |

## Development path

```text
V0.1  Detect people and sports balls with pretrained YOLO
  |
V0.2  Remove likely off-field detections
  |
V0.3  Assign temporary track IDs to people
  |
V0.4  Classify tracked people by team or referee role
  |
V0.5  Select one ball track and bridge short detection gaps
  |
V0.6  Map a selected continuous shot to a restricted 2D pitch region
  |
V0.7  Visualize local fragments and position distributions; report data quality
  |
Future  Validate geometry and improve identity continuity before metric analytics
```

Generated videos, input footage, and model weights remain local and are ignored
by Git. The repository records code, configuration, tests, and reproducible
commands.
