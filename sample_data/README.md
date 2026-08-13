# Sample data files

Real index files, truncated to five rows, with media paths replaced by
`/path/to/...` placeholders and clip identifiers replaced by synthetic ones.
Everything structural — extensions, filename conventions, label formats, column
counts — is exactly what MJEPA was trained and evaluated with. Point the paths at
your own media and the format will parse.

Several of these formats are **space-delimited and headerless**, which is easy to
get wrong, so copy the structure exactly.

## Pre-training

### Paired audio-video — [`audioset_2m_train_audio_video_paths.csv`](audioset_2m_train_audio_video_paths.csv)

Read by `src/datasets/audio_video_dataset.py` for the paired AudioSet-2M stream.

- Space-delimited, **no header row**
- Column 0: audio file path
- Column 1: video file path

```
/path/to/audioset/unbalanced_train_segments/audio/clip00001_0_10000.flac /path/to/audioset/unbalanced_train_segments/video_256px/clip00001_0_10000.mp4
```

Audio is FLAC and video is MP4 in our setup, but any format your `torchaudio` /
`decord` builds can decode will work. The `_<start_ms>_<end_ms>` filename suffix
is only our naming convention for 10-second segments — nothing parses it.

> **Column order matters.** Column 0 is audio, column 1 is video. Swapping them
> raises no error — it silently trains with the modalities transposed.

Wire it up under `data.datasets` in a training config:

```yaml
data:
  datasets:
  - /path/to/audioset_2m_train_audio_video_paths.csv
```

### Video-only — [`k400_train_paths.csv`](k400_train_paths.csv), [`ssv2_train_paths.csv`](ssv2_train_paths.csv)

Read by `src/datasets/video_dataset.py` for the video-only stream of the
VideoMix2M mixture, and by the K400/SSv2 evals. One format covers all of
Kinetics-710, Kinetics-400, SSv2, and HowTo100M.

- Space-delimited, **no header row**
- Column 0: video file path
- Column 1: integer class label

Two examples, showing that neither the container nor the directory layout is
prescribed:

```
/path/to/kinetics400/train/weaving_basket/clip00001_000468_000478.mp4 0
/path/to/ssv2/videos/108961.webm 166
```

Labels are ignored during self-supervised pre-training, but the column must still
be present. The K400 and SSv2 evals do use them.

> Because the delimiter is a space, **paths containing spaces will not parse.**
> The loader retries with `::` as the delimiter if the space-delimited parse
> raises, so use `::` between fields if your paths contain spaces.

## Evaluation

### AudioSet-20K audio-video — [`audioset20k_audio_video_label_paths.csv`](audioset20k_audio_video_label_paths.csv)

Read by `src/datasets/audioset_eval_dataset.py` for the `audioset_audio_video`
dataset type (`as20k-video.yaml`, `as20k-audio-video.yaml`).

- Space-delimited, **no header row**
- Column 0: audio file path
- Column 1: video file path
- Columns 2+: one or more label ids, variable count per row

```
.../clip00001_30000_40000.flac .../clip00001_30000_40000.mp4 /m/09x0r /t/dd00088
.../clip00003_0_10000.flac     .../clip00003_0_10000.mp4     /m/03fwl /m/04rlf /m/09x0r
```

The label ids above are AudioSet mids, which use both `/m/` and `/t/` prefixes.
The loader treats them as opaque strings, so any label vocabulary works as long as
it is used consistently across your train and val CSVs.

> **Two things to watch.**
>
> 1. `num_classes` is derived from the set of label ids actually present in the
>    CSV, **not** from the `num_classes: 527` field in the config. If your train
>    and val CSVs do not contain the same set of ids, label indices will not line
>    up between them and the resulting metrics are meaningless. Full AudioSet
>    splits contain all 527, so this only bites on subsets — including the
>    five-row sample here.
> 2. An `ontology.json` is looked up **as a sibling of the CSV** (see below). If
>    it is missing the eval still runs, logs a warning, and reports numeric
>    indices instead of class names.

### Ontology (not shipped)

No ontology file is included here. Supply your own and place it next to your eval
CSVs as `ontology.json`.

For AudioSet, download the official 527-class ontology from the
[AudioSet ontology repository](https://github.com/audioset/ontology). If you are
evaluating on a different dataset, any ontology works — the loader only reads two
fields per entry:

```json
[
  {"id": "/m/09x0r", "name": "Speech"},
  {"id": "/m/04rlf", "name": "Music"}
]
```

`id` must match the label ids used in your CSV; `name` is used only for
human-readable logging. Extra fields are ignored, so the official AudioSet file
can be used as-is.

## Datasets that use their upstream public format

No samples are shipped for these — use the files distributed with the datasets
themselves, unmodified.

| Dataset | Files | Format |
|---|---|---|
| AudioSet (audio-only eval) | `balanced_train_segments.csv`, `eval_segments.csv` | Official AudioSet CSV. Comma-delimited with `# ` comment lines, columns `YTID, start_seconds, end_seconds, "mid,mid,..."`. Needs a sibling `ontology.json`. |
| ESC-50 | `esc_{train,eval}_data_{fold}.json`, `esc_class_labels_indices.csv` | AST-style JSON with a top-level `data` key, 5 folds. Labels CSV is `index,mid,display_name`. |
| FSD50K | `dev.csv`, `eval.csv`, `vocabulary.csv` | Official FSD50K ground truth. `dev.csv` is `fname,labels,mids,split` with `split` in `{train,val}`; `vocabulary.csv` is `index,label_name,mid`. |

For the audio-only AudioSet eval the config points at a directory of audio files
(`train_audio_dir` / `val_audio_dir`) alongside the official CSV, and filenames
are resolved from the `YTID` column.
