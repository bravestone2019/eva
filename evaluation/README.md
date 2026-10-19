# EVA affect evaluation

This folder contains the manifest format for evaluating EVA's voice and text affect estimates. The example CSV intentionally contains only a header; add only examples you have explicit permission to use.

## FLEURS baseline snapshot

The local FLEURS-small test sets were scored on 11 October 2026 with the same Unicode NFC, lowercase, and basic punctuation normalization:

| Language | Model | Test examples | WER |
| --- | --- | ---: | ---: |
| English | Whisper-small | 647 | 7.47% |
| English | EVA default, Whisper large-v3-turbo | 647 | 5.43% |
| Hindi | Whisper-small | 418 | 67.38% |
| Hindi | Whisper-small fine-tuned for one epoch on FLEURS train | 418 | 35.70% |
| Hindi | EVA default, Whisper large-v3-turbo | 418 | 33.03% |

The fine-tuned Hindi model improved over Whisper-small, but did not beat EVA's current large-v3-turbo model on this test split. Keep the current runtime model configured by default; do not replace it with `checkpoints/eva-whisper-hi` based on this run. The fine-tuned checkpoint is retained for further validation only. FLEURS is read speech and does not establish performance on spontaneous, noisy, accented, or code-switched microphone speech.

Reports are saved under `outputs/fleurs-en-baseline/`, `outputs/fleurs-hi-baseline/`, `outputs/fleurs-hi-finetuned/`, and `outputs/fleurs-current-whisper/`. `scripts/evaluate_whisper_fleurs.py` reproduces the current OpenAI Whisper runtime-model comparison with the locally cached checkpoint.

## Prepare a consented set

1. Copy `consented_manifest.example.csv` to a private location.
2. Add one row per voice turn. Use a pseudonymous `speaker_group_id`; never put a person's name or contact details in this file.
3. Keep recordings outside Git. Set `audio_path` to a local path relative to the CSV or an absolute local path.
4. Label the speaker's self-reported emotion when possible. For voice evaluation, use `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, or `surprised`. Do not infer the label from EVA's prediction.
5. Use `dev` for threshold calibration and `test` for the held-out result. Keep all turns from the same speaker in one split.
6. Mark every row `consent=true`. The evaluator refuses rows without explicit consent.

`accent_group`, `noise_level`, and `language_group` are optional aggregate labels for robustness checks; use broad values such as `english`, `hindi`, or `hinglish` and avoid identifying details. The evaluator prints metrics and pseudonymous group IDs, never transcripts or audio.

## Run the evaluation

From the EVA repository root:

```powershell
python scripts/evaluate_affect.py --manifest C:\private\eva-affect\manifest.csv --output C:\private\eva-affect\report.json
```

The report includes accuracy, macro F1, coverage (how often EVA abstains), accuracy on non-abstained examples, per-class precision/recall/F1, a Brier score, and breakdowns by speaker group, accent group, noise level, and language group. It checks that no speaker appears in multiple splits, selects a confidence threshold from `dev` when supplied, and reports `test` separately. If there is no `dev` or `test` split, the report warns that the result is exploratory, not held-out validation.

After reviewing the held-out report, set `EVA_AFFECT_CONFIDENCE_THRESHOLD` to the selected dev threshold in EVA's environment to apply it. Do not tune against the test split.

## Voice emotion baseline and fine-tuning

The expanded voice classifier uses the union of the acted-corpus labels: `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, and `surprised`. RAVDESS contains all eight; CREMA-D contains `angry`, `disgust`, `fearful`, `happy`, `neutral`, and `sad`. A pretrained four-class SUPERB checkpoint can only predict `angry`, `happy`, `sad`, and `neutral`; EVA maps its outputs into the same canonical label names. The expanded classes become recognizable only after training and deploying an eight-class checkpoint. Confusion remains a text-only cue.

Keep the audio and manifest outside the repository. Copy `voice_training_manifest.example.csv` beside a private `audio/` folder and replace every sample row. Provide one row per clip with `train`, `dev`, or `test`; use a pseudonymous `speaker_group_id`, the transcript, an audio path relative to the CSV or an absolute path, one of the eight expected voice labels, and explicit `consent=true`. A speaker must appear in only one split. The script normalizes common corpus aliases (`ANG`, `FEA`, `DIS`, `HAP`, and similar) to the canonical names. For natural speech, record the speaker's own label; do not equate an acted emotion label with a person's natural emotional state.

First evaluate the pretrained baseline on your held-out clips. This downloads the base classifier if it is not cached, but does not train or save model weights:

```powershell
python scripts/train_voice_affect.py --manifest C:\private\eva-affect\manifest.csv --eval-only --output-dir C:\private\eva-affect\baseline
```

Only if the baseline shows a useful, repeatable gap, fine-tune on the `train` split and select the checkpoint using `dev`:

```powershell
python scripts/train_voice_affect.py --manifest C:\private\eva-affect\manifest.csv --output-dir checkpoints\eva-voice-affect
```

The default run uses a batch size of 1, gradient accumulation, four epochs, a conservative learning rate, and freezes the convolutional feature encoder. GPU VRAM determines whether training fits; 16 GB of system RAM alone does not guarantee it. If it runs out of memory, keep the batch size at 1, reduce clip duration consistently, or use a machine with more VRAM. Do not increase epochs or tune against `test-report.json`.

The training run replaces the pretrained four-class classifier head with an eight-class head and saves the canonical emotion labels in the checkpoint. To try it in EVA, point the API to the checkpoint before starting the backend:

```powershell
$env:EVA_AFFECT_AUDIO_MODEL = (Resolve-Path .\checkpoints\eva-voice-affect-ravdess-cremad-8class).Path
python -m uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
```

The starter manifest is an annotation format, not a dataset. The voice training script does not download datasets. For local RAVDESS and CREMA-D downloads, prepare a manifest with the helper script. It keeps the eight classes used by the expanded voice classifier, splits by actor separately within each corpus, and records dataset/license provenance without falsely marking the rows as individually consented:

```powershell
python scripts/prepare_acted_voice_manifest.py --ravdess-dir .\RAVDESS --cremad-dir .\crema-d-mirror\AudioWAV --output C:\eva-affect\ravdess-cremad-manifest.csv
```

This keeps RAVDESS audio-only speech (`03-01`) and includes all eight RAVDESS emotions. CREMA-D contributes its six available labels. The generated manifest uses `permission_basis` and includes `source_dataset`/`source_license`; review both dataset terms before training or sharing a resulting checkpoint. Keep acted-speech results separate from natural user speech results. Keep downloaded datasets and checkpoints local; `data/fleurs/`, `data/go_emotions/`, and `checkpoints/` are ignored by Git.

Run the pretrained baseline on this generated manifest, then fine-tune only after reviewing the baseline:

```powershell
python scripts/train_voice_affect.py --manifest C:\eva-affect\ravdess-crema-manifest.csv --eval-only --output-dir C:\eva-affect\ravdess-crema-baseline
python scripts/train_voice_affect.py --manifest C:\eva-affect\ravdess-crema-manifest.csv --output-dir checkpoints\eva-voice-affect-ravdess-cremad-8class
```

The RAVDESS corpus is CC BY-NC-SA 4.0; the CREMA-D database and contents have separate ODbL 1.0 and DbCL 1.0 terms. A RAVDESS-trained model may have use/distribution restrictions, so review the applicable licenses for your intended use before deploying or distributing it.

## Natural-speech collection and training

For natural-speech training, use recordings from people who explicitly agree to this use. Store recordings outside the repository in one pseudonymous folder per speaker. Do not use names or contact details as folder names. The helper creates a CSV template, assigns whole speakers to train/dev/test, starts with consent set to false, and never uploads audio:

```powershell
python scripts/prepare_natural_voice_manifest.py --audio-dir C:\eva-affect\natural-audio --output C:\eva-affect\natural-voice-manifest.csv
```

Expected folder layout:

```text
C:\eva-affect\natural-audio\speaker_001\turn_001.wav
C:\eva-affect\natural-audio\speaker_001\turn_002.wav
C:\eva-affect\natural-audio\speaker_002\turn_001.wav
```

The folder names must be pseudonymous and each must represent one speaker. A flat folder of clips cannot support speaker-held-out evaluation because the helper cannot know which clips came from the same person. The helper now adds blank `language_group`, `accent_group`, and `noise_level` fields so you can record broad categories without adding identifying details.

After the speaker gives informed permission, edit the CSV and set `consent=true` for those clips. Ask the speaker what they felt while saying each utterance and enter that in `self_reported_emotion` using the eight voice labels listed above. `perceived_emotion` is optional and records how a listener thought the voice sounded; it is reported separately and is never used as the training target. Include only labels the speaker actually reports. Keep each speaker entirely in the generated split. The trainer refuses blank labels, missing consent, blank speaker IDs, speaker split leakage, or training data missing one of the eight classes.

Train the audio model against self-reported labels:

```powershell
python scripts/train_voice_affect.py --manifest C:\eva-affect\natural-voice-manifest.csv --output-dir checkpoints\eva-natural-voice-affect --epochs 4 --batch-size 1 --gradient-accumulation 4
```

The report includes agreement between self-reported and optional perceived labels on test clips, but the model learns only from self-reports. Use `dev` to select the checkpoint and keep `test` untouched until final evaluation. The same speaker-separated split is essential; splitting clips at random can let the same person's voice appear in both train and test. Results remain estimates of expressed emotion, not facts about a person's inner state.

For an additional EVA end-to-end evaluation, `scripts/evaluate_affect.py` accepts `self_reported_emotion` in place of `expected_emotion`. Set `EVA_AFFECT_AUDIO_MODEL` to the checkpoint before running it so EVA evaluates with the model you trained. The evaluator uses the transcript too when one is provided; this checks EVA's fused audio/text behavior, whereas `train_voice_affect.py` evaluates the audio classifier alone.

### Evaluate speech recognition on natural speech

FLEURS is read speech, so it does not measure the microphone conditions, accents, or code-switching your users actually have. Prepare a consented CSV with `split`, `speaker_group_id`, `audio_path`, `transcript`, `language_group`, and `consent` columns. Keep all clips from a speaker in one split, and use only `test` for the final score. The natural-voice manifest helper creates these fields. Run the recognizer configured for EVA (including the selected backend/model and language setting):

```powershell
python scripts/evaluate_natural_asr.py --manifest C:\private\eva-affect\manifest.csv --output C:\private\eva-affect\asr-test.json
```

The report contains WER overall and by `language_group`; it does not include transcripts, predictions, or audio paths. Use broad groups such as `english`, `hindi`, and `hinglish`. The Hindi/Hinglish groups need human-checked transcripts and enough held-out speakers before they can support a useful claim. Never move a speaker across splits to increase sample counts.

## FLEURS speech-recognition training

Install the optional training dependencies in EVA's Python environment:

```powershell
python -m pip install -r requirements-training.txt
```

Download English and Hindi FLEURS splits (this can use significant disk space and network bandwidth):

```powershell
python scripts/download_fleurs.py --language both
```

For an initial small experiment, preserve the official validation and test splits and sample only training data:

```powershell
python scripts/prepare_fleurs.py --input-dir data/fleurs/english --output-dir data/fleurs/english-small --train-limit 1500
python scripts/train_whisper.py --dataset-dir data/fleurs/english-small --language english --output-dir checkpoints/eva-whisper-en --model openai/whisper-small --epochs 3 --batch-size 1 --gradient-accumulation 8
```

For Hindi, substitute `data/fleurs/hindi` and `--language hindi`. Start with a small sample and check GPU memory before increasing it. `--eval-only` scores a pretrained checkpoint on the held-out test split without training. The report records WER; select settings/checkpoints on validation data and only inspect test results after selection.

To run EVA with the saved Hugging Face Whisper checkpoint, set these variables before starting the API:

```powershell
$env:EVA_WHISPER_BACKEND = "transformers"
$env:EVA_WHISPER_MODEL = (Resolve-Path .\checkpoints\eva-whisper-en).Path
$env:EVA_WHISPER_LANGUAGE = "english"
python -m uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
```

Use `hindi` for the Hindi checkpoint. `auto` asks Whisper to detect the language when the selected multilingual checkpoint supports it. The default backend remains the OpenAI Whisper package; this setting opts into Transformers.

## GoEmotions text-affect fine-tuning

GoEmotions allows multiple emotion labels per text, so this trainer preserves the multi-label task and reports micro/macro F1 rather than forcing each example into one class. The script selects checkpoints by validation macro-F1. It can read the local download directly with `--dataset-dir`:

```powershell
python scripts/train_text_emotion.py --dataset-dir data/go_emotions --eval-only --eval-split validation --model SamLowe/roberta-base-go_emotions --output-dir outputs/go-emotions-baseline
python scripts/train_text_emotion.py --dataset-dir data/go_emotions --output-dir checkpoints/eva-text-affect-go-emotions --model SamLowe/roberta-base-go_emotions --train-limit 0 --validation-limit 0 --test-limit 0 --batch-size 8
```

Use `--train-limit 0 --validation-limit 0 --test-limit 0` to use each full downloaded split. Reduce `--batch-size` if memory is insufficient. To use the saved text classifier, set this before launching the API:

```powershell
$env:EVA_AFFECT_TEXT_MODEL = (Resolve-Path .\checkpoints\eva-text-affect-go-emotions).Path
```

This model is trained on English Reddit comments. Its benchmark score does not establish performance on EVA users, Hindi, or other domains; evaluate with consented in-domain examples before making product claims.

## HinEmo Hindi/code-mixed text-affect pilot

HinEmo uses a separate four-class single-label task (`joy`, `anger`, `disgust`, `sadness`) from its `gpt_emotion` column. The labels are machine-generated, and the dataset's language-identification metadata is multilingual, so this run measures agreement with that noisy target—not verified emotion accuracy or Hindi-only accuracy. Keep it separate from GoEmotions; the labels, languages, and multi-label/single-label tasks differ. The pilot saves a standalone XLM-R checkpoint and a per-class test report:

```powershell
python scripts/train_hinemo_emotion.py --dataset-dir data/hinemo --output-dir checkpoints/eva-hinemo-pilot --model FacebookAI/xlm-roberta-base --batch-size 4 --gradient-accumulation 4
```

Do not set this checkpoint as EVA's production text classifier without adapting and evaluating the live text scorer: it currently expects GoEmotions-style multi-label probabilities and does not treat `disgust` as a response-style category. Build a Hindi/Hinglish held-out set with human-reviewed labels before using the pilot to claim or tune live accuracy.

For a Hindi/Hinglish benchmark that matches EVA's current live text path, annotate consented utterances with the five labels `frustrated`, `confused`, `sad`, `excited`, or `neutral`. Use the blank `text_affect_review.example.csv` as a format guide. Have two annotators label independently, resolve disagreements before evaluation, use `conversation_group_id` to keep related turns in one split, and record broad language groups (`hindi`, `hinglish`, `english`). Do not copy `gpt_emotion` into `human_emotion`; it is only a noisy pilot label. Run the live model on the reviewed set with:

```powershell
python scripts/evaluate_text_affect.py --manifest C:\private\eva-affect\human-text-review.csv --output C:\private\eva-affect\human-text-report.json
```

This evaluates the configured EVA text-affect model, calibrates abstention on `dev`, and reports test metrics by language group without writing reviewed text or predictions into the report. Use only examples whose collection/use permission and review consent are clear.

## Scope of this plan

The repository has separate workflows for consented voice-affect data, FLEURS Whisper ASR, GoEmotions text affect, and the HinEmo text pilot. OASST1 is conversational assistant data; this repository does not fine-tune an LLM on it, and it has no emotion targets. Dataset/license review, label definitions, consent and privacy decisions, suitable hardware, and separate validation remain necessary. Whisper training here is ASR; it does not train emotion recognition from speech.

## Collect reply feedback

EVA's thumbs-up/down feedback is stored in this browser only. It excludes the conversation text and voice recording. A **Tone felt wrong** rating is a user-centered signal for possible emotion misinterpretation; it is not proof of a model error. Select **Export reply ratings** in the sidebar to save the JSON file, then summarize it with:

```powershell
python scripts/evaluate_affect.py --feedback C:\private\eva-affect\eva-affect-feedback.json --output C:\private\eva-affect\feedback-report.json
```

Feedback comparisons between adapted and standard/uncertain replies are observational. They can highlight problems, but they do not establish that adaptation caused the difference. A controlled paired study should compare both styles for the same consented prompts and have raters judge the replies without knowing which style produced each one.

### Run a blinded paired reply-style study

Create a private CSV from real, explicitly consented scenarios and the two candidate replies. Start from `evaluation/reply_style_study.example.csv`, replace the placeholder, and set `consent=true` only after permission covers this evaluation. Use the same prompt for both versions; keep the responses comparable in factual content. Run:

```powershell
python scripts/prepare_reply_style_study.py --input C:\private\eva-affect\reply-scenarios.csv --ratings C:\private\eva-affect\blind-ratings.csv --key C:\private\eva-affect\condition-key.csv
```

Give each rater a copy of `blind-ratings.csv`; keep the key private until ratings are complete. Raters enter a pseudonymous `rater_id`, then score helpfulness, tone fit, and unwarranted assumptions from 1 to 5 (for assumptions, 1 means few and 5 means many). Collect one rating for each response in every pair, without telling raters which response was adapted. Combine the completed copies under the same header and analyze the CSV:

```powershell
python scripts/analyze_reply_style_study.py --ratings C:\private\eva-affect\blind-ratings.csv --key C:\private\eva-affect\condition-key.csv --output C:\private\eva-affect\reply-style-report.json
```

The report summarizes paired preferences overall and by expected emotion. It is descriptive, not a significance test; a small convenience sample cannot prove general benefit. Include multiple raters and more held-out scenarios before making a product claim. Do not upload private prompts or responses to the repository.
