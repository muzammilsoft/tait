# Publishing a TAIT model to Hugging Face

This is the small update that lets a phone-trained TAIT model be shared
as a link: train on the phone, publish to Hugging Face, and anyone can
download the same `.npz` file, or open it in KAIT, the Android runner
for TAIT models (a model index in the style of F-Droid: name, link, and
model card, no hosting clone).

Core TAIT still needs only NumPy. Publishing is the only step that
needs the official Hugging Face client:

```bash
pip install "tait[publish]"
```

## 1. Save your token safely (once)

Never paste a token into chat, a script, or a command line: it would
stay in history. Type it hidden, once, with:

```bash
tait hf login
```

You will be asked for:

1. A Hugging Face token, created at huggingface.co/settings/tokens.
   Use a **fine-grained** token, write access to **your own
   repositories only**, and revoke it any time from the same page.
2. Optionally, a default repo, e.g. `muzammilsoft/my-tait-model`.

The token is stored in `~/.config/tait/huggingface_token` with mode
0600, and is never printed, logged, or put in a model card.

Check the setup without ever showing the token:

```bash
tait hf status
tait hf logout   # deletes the stored token
```

To change the default repo later:

```bash
tait hf config username/model-name
```

## 2. Publish a trained model

Publish a model you already trained (no retraining needed, the `.npz`
format is unchanged):

```bash
tait publish --model my_model.npz --repo username/my-model
```

Or publish right after training:

```bash
tait train --data examples/datasets/demo.jsonl --output my_model.npz --publish --publish-repo username/my-model
```

Add `--private` for a private repo. Add `--dry-run` to generate and
inspect the model card without uploading anything:

```bash
tait publish --model my_model.npz --repo username/my-model --dry-run
```

What gets uploaded:

- The `.npz` model file itself.
- `README.md`, the model card, generated from real metadata: shapes,
  vocab, file size and hash, training data name, examples, epochs,
  final and validation loss (when a `<model>.meta.json` sidecar from
  `tait train` is present), how to run it, and an honest limitations
  note that tiny models may hallucinate.
- `tait-meta.json`, the same training metadata, for tools like KAIT.

If publishing fails, the local model is still saved and the error is
reported; training is never lost because of a failed upload.

## 3. Use the published model

```bash
# Download the .npz file from the model page, then:
tait chat --model my_model.npz
```

Or paste the Hugging Face link into KAIT, which reads the same `.npz`
file on Android.

## Security notes

- The token file is `~/.config/tait/huggingface_token` (0600). Delete
  it with `tait hf logout`, and revoke the token on Hugging Face if
  you ever suspect it leaked.
- Training metadata stores the dataset **name** only, never a phone
  path, so a published card cannot leak your local directories.
- A published model page is public unless you used `--private`.
  Anyone with the link can download the model, which is exactly the
  point of sharing it for the KAIT index.
