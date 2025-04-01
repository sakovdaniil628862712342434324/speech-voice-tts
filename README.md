# Speech Voice TTS
Speech Voice is a text-to-speech (TTS) model based on a GPT-2 architecture combined with the SNAC neural audio codec that can produce natural-sounding speech and is capable of generating diverse audio outputs based on the input text.

## Model Description
"Speech Voice TTS" utilizes a Transformer-based language model (GPT-2) to autoregressively predict a sequence of tokens representing both text and audio information. The audio component is encoded into discrete tokens using the "Multi-Scale Neural Audio Codec (SNAC)" model (`hubertsiuzdak/snac_24khz`).

The model is trained to predict the combined sequence of text tokens followed by SNAC audio tokens, separated by a special `SPACER` token. During inference, the model generates this combined sequence, and the audio tokens are then decoded back into a waveform using the SNAC decoder.

## Features
*   Generates speech from input text.
*   Uses GPT-2 for sequence modeling and SNAC for audio representation.
*   Supports training from scratch and continuation from checkpoints.
*   Includes scripts for easy training and inference.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sakovdaniil628862712342434324/speech-voice-tts.git
    cd speech-voice-tts
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Log in to Hugging Face Hub (Optional but Recommended):**
    This might be needed to download the SNAC model or if you plan to push your trained models.
    ```bash
    huggingface-cli login
    ```

5.  **Log in to Weights & Biases (Optional):**
    Required if you want to log training progress using WandB.
    ```bash
    wandb login
    ```
    You can also provide your WandB API key via the `--wandb_key` argument or the `WANDB_API_KEY` environment variable during training.

## Data Preparation

The training script (`train.py`) expects a preprocessed dataset saved in the Hugging Face `datasets` format using `save_to_disk`.

1.  **Data Sources:** The original model was trained on a combination of datasets like LibriTTS, LibriSpeech, HifiTTS, and Globe TTS.
2.  **Preprocessing Steps:**
    *   Ensure all audio files are **mono** and resampled to **24kHz** (required by SNAC).
    *   Normalize the corresponding transcripts (e.g., lowercase, remove punctuation if desired). The dataset should have columns like `audio` (containing the audio data dictionary with 'array' and 'sampling_rate') and `text_normalized` (containing the transcript).
    *   Filter out audio samples longer than a certain duration (e.g., 10 seconds) to fit within the model's context length (`n_ctx=1024`) and manage memory usage. The `tts_snac_base.py` script in the original description shows an example using `librosa.get_duration`.
3.  **Save the Dataset:** Use `datasets.Dataset.save_to_disk()` to save your processed dataset. For example:
    ```python
    # Assuming 'processed_dataset' is your Hugging Face Dataset object
    processed_dataset.save_to_disk("./prepared_data")
    ```
    The `train.py` script will then load data from the specified path (e.g., `./prepared_data`).

## Training

The `train.py` script handles both training from scratch and resuming from checkpoints.

**Key Training Arguments:**

*   `--data_path`: Path to the directory containing the preprocessed dataset (output of `save_to_disk`). (Required)
*   `--output_dir`: Directory where checkpoints and the final model will be saved. (Required)
*   `--tokenizer_path`: Path to the directory containing tokenizer files (default: current directory `.`).
*   `--config_path`: Path to the directory containing `config.json` (default: current directory `.`).
*   `--checkpoint_path`: Path to a specific checkpoint directory (e.g., `./output/epoch_5`) to resume training. (Optional)
*   `--epochs`: Number of training epochs.
*   `--learning_rate`: Initial learning rate.
*   `--batch_size`: Batch size per device (usually 1 due to sequence length).
*   `--accumulation_steps`: Gradient accumulation steps (effective batch size = `batch_size * accumulation_steps`).
*   `--wandb_project`: Your WandB project name to enable logging. (Optional)
*   `--wandb_run_name`: Name for the WandB run. (Optional)

**Example: Training from Scratch**

```bash
python train.py \
    --data_path ./prepared_data \
    --output_dir ./speech-voice-tts-output \
    --epochs 10 \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --accumulation_steps 32 \
    --save_every 1 \
    --wandb_project "my-tts-project" \
    --wandb_run_name "speech-voice-tts-run-1" \
    --device cuda
```

**Example: Continuing Training from Epoch 5**

```bash
python train.py \
    --data_path ./prepared_data \
    --output_dir ./speech-voice-tts-output \
    --checkpoint_path ./speech-voice-tts-output/epoch_5 \
    --epochs 10 \
    --learning_rate 5e-5 \
    --batch_size 1 \
    --accumulation_steps 32 \
    --save_every 1 \
    --wandb_project "my-tts-project" \
    --wandb_run_name "speech-voice-tts-continuation" \
    --wandb_id <previous_wandb_run_id> \
    --device cuda
```
*Replace `<previous_wandb_run_id>` if you want to resume the same WandB run.*

## Inference

Use the `infer.py` script to generate audio from text using a trained model checkpoint.

**Key Inference Arguments:**

*   `--model_path`: Path to the trained model directory (e.g., `./speech-voice-tts-output/epoch_10`). (Required)
*   `--input_text`: The text you want to convert to speech. (Required)
*   `--output_wav_path`: Path where the generated `.wav` file will be saved. (Required)
*   `--num_beams`, `--do_sample`, `--temperature`, `--top_p`, `--repetition_penalty`: Generation parameters to control the output quality.

**Example: Generating Speech**

```bash
python infer.py \
    --model_path ./speech-voice-tts-output/epoch_10 \
    --input_text "Hello, this is a test of the Speech Voice TTS model." \
    --output_wav_path ./output_audio.wav \
    --num_beams 4 \
    --repetition_penalty 2.0 \
    --device cuda
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
