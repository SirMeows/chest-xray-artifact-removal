# AAI Chest X-ray Pneumonia Classification

Applied Artificial Intelligence semester project investigating whether removing non-anatomical artifacts, such as R/L orientation markers, improves pneumonia classification from chest X-rays. The experiment trains the same custom PyTorch CNN on the original images and on a cleaned version of the dataset.

The [exam presentation](presentation.pdf) gives a brief explanation of the project, method, visual results, and limitations.

## Requirements

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage Python and the project environment. The supplied configuration selects Python 3.10 and CUDA 12.8 PyTorch wheels. The supplied setup targets Windows with an NVIDIA GPU. A compatible NVIDIA driver is required for CUDA.

Clone or download this repository and open a terminal in its root directory, containing `pyproject.toml`. Run all commands below from that directory. The `uv run` commands use the project environment without requiring manual activation.

## Repeat the experiment

### 1. Install dependencies

Install the versions recorded in `uv.lock`:

```bash
uv sync --locked
```

Check that the GPU is available to PyTorch:

```bash
uv run --locked python scripts/check_cuda.py
```

The check prints the Python, PyTorch, CUDA, and GPU details and performs a small GPU calculation. The notebook also has CPU/MPS device selection, but the supplied dependency configuration and saved models target NVIDIA CUDA.

### 2. Download the dataset

Download [Chest X-Ray Images (Pneumonia) from Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Kaggle may require a free account to download it.

Extract the images into the repository so that the layout is:

```text
chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

If the archive contains an extra `chest_xray/` directory, use the inner directory so that `train/`, `val/`, and `test/` sit directly inside the project's `chest_xray/`.

Use all three supplied splits. The notebook combines their 5,856 images and creates a shared 80/10/10 train/validation/test split for both experiments. The dataset and its archive are excluded from Git.

### 3. Create the cleaned images

```bash
uv run --locked python scripts/create_cleaned_dataset.py
```

This uses OpenCV marker detection and Telea inpainting to write `chest_xray_cleaned/`, preserving the original folder structure and filenames. Keep both dataset directories. The notebook needs the original and cleaned version of every image.

Existing cleaned images are skipped. To regenerate them after changing the cleaning settings, append `--overwrite`.

### 4. Start Jupyter and run the notebook

```bash
uv run --locked jupyter lab no_letters_nvidia.ipynb
```

Follow the browser link printed by Jupyter. Open `no_letters_nvidia.ipynb` and use the project's Python kernel. Restart the kernel and run all cells from top to bottom. In an IDE, select the interpreter from `.venv` instead.

The notebook performs data visualization, preprocessing, baseline training, a 20-configuration hyperparameter search, tuned-model training, and the final original-versus-cleaned experiment. The final two models use the same split, seed, architecture, preprocessing, and locked training settings, with 15 epochs per model. A full run includes several training runs and can take substantially longer without a GPU.

The presentation contains the visual explanation of the results. Fixed seeds make the experiment repeatable, but exact numerical results can vary with hardware and software versions.

The last cell saves `CNN_original.pkl` and `CNN_cleaned.pkl`, replacing the included files. Skip that final save cell, or change its output filenames, if you want to preserve the supplied models.

## Try an included model without retraining

Both trained models are included. They are complete `joblib`-serialized `PneumoniaCNN` objects containing CUDA tensors, so use the supplied CUDA environment to load them.

In the notebook, run only the import/configuration cell, the preprocessing definitions (`device` and `PadToSquare`), the transform definitions (`eval_transform`), and the **Model Definition** cell defining `PneumoniaCNN`. Then run this in a new cell:

```python
import joblib

saved_model = joblib.load("CNN_original.pkl").to(device).eval()
image = Image.open("chest_xray/test/NORMAL/IM-0001-0001.jpeg").convert("L")
batch = eval_transform(image).unsqueeze(0).to(device)

with torch.inference_mode():
    probabilities = torch.softmax(saved_model(batch), dim=1)[0]

dict(zip(classes, probabilities.cpu().tolist()))
```

This returns probabilities for `NORMAL` and `PNEUMONIA`. Replace the image path to try another image. To try `CNN_cleaned.pkl`, use the corresponding image from `chest_xray_cleaned/`.

## Main files

- `no_letters_nvidia.ipynb`: the experiment and model definition.
- `presentation.pdf`: brief project explanation and visual results.
- `scripts/create_cleaned_dataset.py`: dataset cleaning.
- `scripts/experiment_types.py`: shared experiment data structures.
- `scripts/check_cuda.py`: GPU setup check.
- `CNN_original.pkl` and `CNN_cleaned.pkl`: trained models.
- `pyproject.toml` and `uv.lock`: environment configuration and locked dependencies.
