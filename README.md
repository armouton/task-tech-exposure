# task-tech-exposure

This package implements the methodology of 'Measuring Task-Level Technological Exposure: A Language Model Approach' (2026). It handles file download and management, data classification, and exposure measurement.

Contained in this README:

* [Background](#background)
* [Quick Start](#quick-start)
* [Data Formatting](#data-formatting)
    * [generate_descriptions](#generate_descriptions)
    * [download_data](#download_data)
* [Measuring Exposure](#measuring-exposure)
    * [classify_patents](#classify_patents)
    * [classify_tasks](#classify_tasks)
    * [measure_exposure](#measure_exposure)
* [Model Training and Validation](#model-training-and-validation)
    * [create_sample](#create_sample)
    * [label_sample](#label_sample)
    * [train_model](#train_model)
    * [validate_model](#validate_model)
    * [embed_data](#embed_data)
* [Citation](#citation)
* [License and Data](#license-and-data)

## Background

This package provides tools for measuring the exposure of occupational tasks to technological change, using the accompanying data and resources at [10.5281/zenodo.17643646](https://doi.org/10.5281/zenodo.17643646).

The underlying methodology -- described at length in the paper ([link here](https://drive.google.com/file/d/1--GXGpKy3WdjjQbX1T398QZC2xdzYd3H/view?usp=drive_link)) -- matches USPTO patent abstracts with ONET occupational task statements and classifies them using sentence embedding models built on [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) via the [Sentence Transformers](https://sbert.net/) library. The matching process identifies patent-task pairs with high semantic similarity, while separate classification models categorize patents by technology type and tasks by functional category. Models are fine-tuned with Matryoshka Representation Learning (MRL) and validated against outputs from GPT-4.1, allowing them to achieve a comparatively high level of precision. Exposure indices are calculated for each task and technology category, conditional on a desired level of cross-sectional and temporal aggregation.

The package provides access to a living dataset that is updated quarterly with newly published patent applications and annual ONET revisions. This ensures minimal time lag and enables analysis at fine temporal resolution (monthly, quarterly, or annual). Users can define their own technology and task categories, customize weighting schemes, and generate exposure measures at various aggregation levels (aggregate economy-wide, by occupation, or by wage percentiles). All computations run locally, and the methodology is fully transparent and replicable.

## Quick Start

Package installation requires Git, and can be executed in the terminal:

```bash
python -m pip install --upgrade git+https://github.com/armouton/task-tech-exposure.git
```

*Sentencetransformers* and its dependencies will be installed if not already present. Note that some dependencies, such as *torch*, have different installation options, and if a particular option is desired then these should be installed separately.

Running the package requires, at a minimum, a pair of paths indicating the directories in which to store the data and the exposure measures.

**Step 1 — Data Formatting.** Download the dataset. The full dataset is approximately 10GB, but download can be limited to a particular date range. Users defining custom technology or task categories should also run `generate_descriptions()` to produce a GPT prompt for writing category descriptions, and add the returned descriptions to their categories CSV before proceeding to Step 3.

```python
import task_tech_exposure as tte

# Download/update the dataset (limit by year range to reduce download size)
tte.download_data(path_to_data="/path/to/data/",
                  from_year=2015,
                  to_year=2023)

# Optional: generate a prompt for writing custom category descriptions
tte.generate_descriptions(path_to_data="/path/to/data/",
                          path_to_results="/path/to/results/",
                          cat_type="tech",
                          path_to_categories="/path/to/my_categories.csv")
```

**Step 2 — Measuring Exposure.** Classify patents and task statements, then compute exposure indices. By default this uses the pre-trained models and category definitions shipped with the dataset. Custom models and category files can be specified via optional arguments (see [Measuring Exposure](#measuring-exposure)). When using custom technology categories, similarity thresholds for `classify_patents()` are set via the `cutoff` argument; if omitted, the package automatically uses per-category thresholds from the validation step in Step 3 if available, or the maximum of the manifest defaults otherwise (see [`classify_patents`](#classify_patents) for details).

```python
# Classify patents by technology category
tte.classify_patents(path_to_data="/path/to/data/",
                     path_to_results="/path/to/results/")

# Classify tasks by functional category
tte.classify_tasks(path_to_data="/path/to/data/",
                   path_to_results="/path/to/results/")

# Measure exposure
tte.measure_exposure(path_to_data="/path/to/data/",
                     path_to_results="/path/to/results/")
```

**Step 3 — Model Training and Validation (optional).** Two sub-workflows are available depending on whether you want to fine-tune a new embedding model or only find data-driven similarity thresholds for use with the existing model. Both require an OpenAI API key for GPT labeling.

*Option A — Find thresholds only.* Use this when you want to keep the pre-trained model but calibrate per-category similarity thresholds against labeled data (e.g. to validate the manifest defaults or adjust them for your application). The thresholds are written to `tte_samples/tech_thresholds.csv` and picked up automatically by `classify_patents()`.

```python
# Submit texts for GPT labeling (batch completes within 24 hours)
tte.create_sample(path_to_data="/path/to/data/",
                  path_to_results="/path/to/results/",
                  cat_type="tech",
                  api_key="sk-...",
                  path_to_categories="/path/to/my_categories.csv")

# Retrieve labeled results
tte.label_sample(path_to_results="/path/to/results/",
                 cat_type="tech",
                 api_key="sk-...")

# Compute thresholds using the existing pre-trained model
tte.validate_model(path_to_data="/path/to/data/",
                   path_to_results="/path/to/results/",
                   cat_type="tech")
```

*Option B — Fine-tune a new model.* Use this when using custom categories. Fine-tunes the embedding model on the labeled sample, validates thresholds, and regenerates the pre-stored embeddings. After re-embedding, pass the fine-tuned model path to `classify_patents()` via the `model` argument in Step 2.

```python
# Submit texts for GPT labeling (batch completes within 24 hours)
tte.create_sample(path_to_data="/path/to/data/",
                  path_to_results="/path/to/results/",
                  cat_type="tech",
                  api_key="sk-...",
                  path_to_categories="/path/to/my_categories.csv")

# Retrieve labeled results
tte.label_sample(path_to_results="/path/to/results/",
                 cat_type="tech",
                 api_key="sk-...")

# Fine-tune model and validate thresholds (validate_model() runs automatically)
tte.train_model(path_to_data="/path/to/data/",
                path_to_results="/path/to/results/",
                cat_type="tech")

# Regenerate pre-stored embeddings with the fine-tuned model
tte.embed_data(path_to_data="/path/to/data/",
               embed_type="patents")
```

## Data Formatting

Two preparatory steps are required before running either the standard or custom classification workflow. Users defining custom categories should call `generate_descriptions()` first to produce category descriptions; users relying on the default categories can skip directly to `download_data()`.

### `generate_descriptions`

Generates a GPT prompt for writing `gpt_description` entries for a custom categories CSV.

**Purpose:** Formats the category names into a prompt ready to paste into a web app (e.g., ChatGPT). The returned descriptions are added to the `gpt_description` column of the categories CSV before running `create_sample()`. A template file at `tte/tte_models/gpt_prompts/templates/{tech|task}_description_template.txt` is used if present, otherwise a built-in default is applied.

**Key Arguments:**
- `path_to_data` (str, required): Parent directory of the downloaded `tte/` dataset folder.
- `path_to_results` (str, required): Path where the output prompt file will be saved.
- `cat_type` (str, required): Classification type. `'tech'` for technology categories, `'task'` for occupational task categories.
- `path_to_categories` (str, optional): Path to a categories CSV with at least `name` and `category` columns. Defaults to the standard file in the downloaded dataset.

**Example:**
```python
import task_tech_exposure as tte

tte.generate_descriptions(path_to_data="/path/to/data/",
                          path_to_results="/path/to/results/",
                          cat_type="tech",
                          path_to_categories="/path/to/my_tech_categories.csv")
```

**Output:** A `.txt` file saved to `path_to_results` containing the formatted prompt. Paste the file contents into a web app to generate descriptions, then add the output as the `gpt_description` column in your categories CSV.

---

### `download_data`

Downloads the matched patent-task dataset from the DOI repository to a local directory.

**Purpose:** Retrieves matched USPTO patent applications and GPT-expanded ONET task statements, the Sentence-BERT embeddings for both sets of data, and supplementary employment and wage data. The dataset can be filtered by date range to focus on specific time periods and reduce download time.

**Key Arguments:**
- `from_year` (int, optional): Start date in YYYY format. If None, downloads full dataset from earliest available annual file (2001).
- `to_year` (int, optional): End date in YYYY format. If None, downloads full dataset through most recent available annual file.
- `path_to_data` (str, required): Parent directory where the `tte/` dataset folder will be created (e.g., `"/Users/username/tte_data/"` creates `"/Users/username/tte_data/tte/"`).
- `doi_url` (str, optional): Alternative URL if downloading from previous data version. Defaults to stable DOI for current version.
- `force_update` (bool, optional): If True, re-downloads files even if they already exist locally. Default is False.

**Example:**
```python
tte.download_data(from_year=2015, 
                  to_year=2020,
                  path_to_data="/Users/username/tte_data/")
```

**Output:** Downloads matched dataset files to the specified directory, including patent abstracts, task statements, sentence embeddings, and supplementary data files.

---

## Measuring Exposure

The three functions below cover the standard classification and measurement workflow. They operate on the default pre-trained models and category definitions shipped with the dataset, and require no OpenAI API key.

### `classify_patents`

Classifies patents into technology categories using fine-tuned sentence embedding models.

**Purpose:** Assigns patent applications to user-defined or default technology categories (*e.g.* AI, robotics, software) based on semantic similarity between patent abstracts and category descriptions. Uses a trained Sentence-BERT model with customizable similarity thresholds.

**Key Arguments:**
- `path_to_data` (str, required): Parent directory of the downloaded `tte/` dataset folder.
- `path_to_results` (str, required): Path where classification results will be saved.
- `path_to_output` (str, optional): Specific path and filename for output CSV. Defaults to 'tech_classification.csv' in results directory.
- `path_to_descriptions` (str, optional): Path to CSV file containing technology category descriptions. Uses default files if not specified (see [Quick Start](#quick-start) above).
- `model` (str, optional): Path to sentence transformer model for technology classification. Uses default fine-tuned model from manifest if None.
- `cutoff` (float or list of float, optional): Similarity threshold(s) for classifying patents into technology categories. A single float is applied uniformly to all categories; a list provides a per-category threshold and must have one entry per category in the descriptions file. When not specified, cutoffs are resolved in order: (1) per-category thresholds from `validate_model()` output (`tte_samples/tech_thresholds.csv`, `marg_prec_0.5` column) if present and category names match; (2) default values from `dataset_manifest.json` (based on 50% marginal precision); (3) the maximum of the manifest cutoffs applied uniformly, with a warning — this last fallback occurs when using custom categories whose count differs from the manifest defaults and no validation thresholds file is present.
- `groups` (list of lists, optional): Groups of technology categories for which classifications should be mutually exclusive (*e.g.* [[0,1], [3]] indicates that a patent should be matched to *at most* one of the first two categories in the descriptions file).
- `priority` (str, optional): Method for resolving grouped categories. `'order'` assigns each patent to the first matching category in list order; `'score'` assigns it to the highest-similarity category. Default is `'order'`.

**Example:**
```python
tte.classify_patents(path_to_data="/Users/username/tte_data/",
                     path_to_results="/Users/username/tte_results/",
                     cutoff=0.75,
                     groups=[[0,1], [3,2], [4]],
                     priority="score")
```

**Output:** CSV file containing patent IDs with assigned technology categories.

---

### `classify_tasks`

Classifies occupational task statements into functional categories using fine-tuned sentence embedding models.

**Purpose:** Assigns ONET task statements to user-defined or default task categories (e.g., cognitive, manual, routine) based on semantic similarity between task descriptions and category definitions.

**Key Arguments:**
- `path_to_data` (str, required): Parent directory of the downloaded `tte/` dataset folder.
- `path_to_results` (str, required): Path where classification results will be saved.
- `path_to_output` (str, optional): Specific path and filename for output CSV. Defaults to 'task_classification.csv' in results directory.
- `path_to_descriptions` (str, optional): Path to CSV file containing task category descriptions. Uses default files if not specified (see [Quick Start](#quick-start) above).
- `model` (str, optional): Path to sentence transformer model for task classification. Uses default fine-tuned model from manifest if None.

**Example:**
```python
tte.classify_tasks(path_to_data="/Users/username/tte_data/",
                   path_to_results="/Users/username/tte_results/")
```

**Output:** CSV file containing task statement IDs with assigned functional categories.

---

### `measure_exposure`

Calculates technological exposure measures at specified aggregation levels with customizable weighting schemes.

**Purpose:** Generates exposure indices that quantify the degree to which occupations or task categories face technological substitution. Combines patent classifications, task classifications, and patent-task matches to produce time-series measures at user-specified frequencies and aggregation levels.

**Key Arguments:**

*Input Files:*
- `path_to_data` (str, required): Parent directory of the downloaded `tte/` dataset folder.
- `path_to_results` (str, required): Path containing classification files and where exposure results will be saved.
- `path_to_tech_classifications` (str, optional): Path to technology classification CSV. Defaults to 'tech_classification.csv' in results directory.
- `path_to_task_classifications` (str, optional): Path to task classification CSV. Defaults to 'task_classification.csv' in results directory.

*Aggregation Options:*
- `level` (str, optional): Level of aggregation for exposure measures. Options:
  - `'aggregate'`: Economy-wide measure (default)
  - `'occupation'`: Occupation-level measures
  - `'percentiles'`: Measures by wage percentile groups
- `frequency` (str, optional): Time frequency for exposure calculation. Options: `'annual'` (default), `'quarterly'`, `'monthly'`, `'all'` (cumulative).
- `digits` (int, optional): Number of SOC occupation code digits if `level='occupation'`. Default is 6 (most detailed level).
- `num_percentiles` (int, optional): Number of percentile groups if `level='percentiles'`. Default is 20.
- `crosswalk` (str, optional): SOC code version for crosswalking. Options: `'2000'`, `'2019'`, or None. Results in occupation aggregation to ensure consistency across SOC versions.

*Weighting Schemes:*
- `weights` (str, optional): Weighting method for aggregation. Options:
  - `'both'`: Weight by both occupational employment and task importance (default)
  - `'occupation'`: Weight by occupational employment only
  - `'task'`: Weight by task importance only
  - `'none'`: Unweighted

*Exposure Calculation:*
- `measure` (str, optional): Type of exposure measure. Options:
  - `'exposed'`: Binary task-level indicator (task is exposed if number of matches exceeds sample average) (default)
  - `'counts'`: Continuous task-level measure based on number of matches
- `match_cutoff` (float, optional): Custom similarity threshold for determining patent-task matches. Overrides default from `dataset_manifest.json`.
- `drop_thresh` (int, optional): Drop patents with more than this number of matches (eliminates outliers). Default is None.

*Date Filters:*
- `start_date` (str, optional): Begin calculating exposure from this date ('YYYY-MM-DD'). If None, uses earliest date in dataset.
- `end_date` (str, optional): Stop calculating exposure at this date ('YYYY-MM-DD'). If None, uses latest date in dataset.

**Example:**
```python
# Economy-wide annual exposure with full weighting
tte.measure_exposure(path_to_data="/Users/username/tte_data/",
                     path_to_results="/Users/username/tte_results/",
                     level="aggregate",
                     frequency="annual",
                     weights="both",
                     measure="exposed")

# Occupation-level quarterly exposure
tte.measure_exposure(path_to_data="/Users/username/tte_data/",
                     path_to_results="/Users/username/tte_results/",
                     level="occupation",
                     frequency="quarterly",
                     digits=6,
                     weights="occupation")

# Wage percentile monthly exposure
tte.measure_exposure(path_to_data="/Users/username/tte_data/",
                     path_to_results="/Users/username/tte_results/",
                     level="percentiles",
                     frequency="monthly",
                     num_percentiles=20,
                     start_date="2015-01-01",
                     end_date="2020-12-31")
```

**Output:** CSV file containing exposure measures for each task and technology category, calculated at specified aggregation level and frequency. Also included are match, patent, and task counts.

---

## Model Training and Validation

The five functions below support two workflows: finding data-driven similarity thresholds for the existing pre-trained model, or fine-tuning a new embedding model for custom categories. Both start with GPT labeling via the OpenAI Batch API. An OpenAI API key is required for labeling. Category descriptions must be prepared first using `generate_descriptions()` (see [Data Formatting](#data-formatting)).

**Threshold-only workflow** (keep the pre-trained model, find data-driven cutoffs):

```python
import task_tech_exposure as tte

# 1. Submit texts for GPT labeling (returns within 24 hours)
tte.create_sample(path_to_data="/path/to/data/",
                  path_to_results="/path/to/results/",
                  cat_type="tech",
                  api_key="sk-...",
                  path_to_categories="/path/to/my_tech_categories.csv")

# 2. Retrieve labeled results once the batch is complete
tte.label_sample(path_to_results="/path/to/results/",
                 cat_type="tech",
                 api_key="sk-...")

# 3. Compute per-category thresholds using the existing model
tte.validate_model(path_to_data="/path/to/data/",
                   path_to_results="/path/to/results/",
                   cat_type="tech")
```

Thresholds are saved to `path_to_results/tte_samples/tech_thresholds.csv` and picked up automatically by `classify_patents()` — no `cutoff` argument needed.

**Full fine-tuning workflow** (custom categories, new embedding model):

```python
# 1. Submit texts for GPT labeling (returns within 24 hours)
tte.create_sample(path_to_data="/path/to/data/",
                  path_to_results="/path/to/results/",
                  cat_type="tech",
                  api_key="sk-...",
                  path_to_categories="/path/to/my_tech_categories.csv")

# 2. Retrieve labeled results once the batch is complete
tte.label_sample(path_to_results="/path/to/results/",
                 cat_type="tech",
                 api_key="sk-...")

# 3. Fine-tune the embedding model (validate_model() runs automatically)
tte.train_model(path_to_data="/path/to/data/",
                path_to_results="/path/to/results/",
                cat_type="tech")

# 4. Re-embed patents using the fine-tuned model
tte.embed_data(path_to_data="/path/to/data/",
               embed_type="patents")
```

Steps 1–3 are repeated for `cat_type="task"` if task categories are also being customized, followed by `embed_type="tasks"` in step 4. After re-embedding, pass the fine-tuned model path to `classify_patents()` or `classify_tasks()` via the `model` argument.

The categories CSV must contain at minimum the columns `name` (short identifier), `category` (display label), and `gpt_description` (longer description used by the embedding model). The `gpt_description` column is filled in by the user using the prompt generated by `generate_descriptions()`, and must be present before calling `create_sample()`.

---

### `create_sample`

Draws a sample of patent abstracts or task statements and submits them for GPT labeling via the OpenAI Batch API.

**Purpose:** Samples texts from the downloaded dataset, formats them with the classification prompt, and submits an OpenAI batch job. The batch ID is saved locally so `label_sample()` can retrieve results automatically. A snapshot of the categories used is also saved, ensuring that `train_model()` and `validate_model()` use exactly the same category definitions.

**Key Arguments:**
- `path_to_data` (str, required): Parent directory of the downloaded `tte/` dataset folder.
- `path_to_results` (str, required): Path where sample files will be saved.
- `cat_type` (str, required): Classification type (`'tech'` or `'task'`).
- `api_key` (str, required): OpenAI API key.
- `from_date` (str, optional): Start date for patent sampling (`'YYYY-MM-DD'`). Tech only. Defaults to January 1 of the most recent year in the dataset.
- `to_date` (str, optional): End date for patent sampling (`'YYYY-MM-DD'`). Tech only. Defaults to December 31 of the most recent year.
- `sample_size` (int, optional): Number of texts to submit. If None, submits all texts in the date range. Texts are drawn randomly.
- `gpt_model` (str, optional): OpenAI model for labeling. Default is `'gpt-4.1'`.
- `path_to_categories` (str, optional): Path to a categories CSV with columns `name`, `category`, and `gpt_description`. Defaults to the standard file in the downloaded dataset.

**Example:**
```python
tte.create_sample(path_to_data="/path/to/data/",
                  path_to_results="/path/to/results/",
                  cat_type="tech",
                  api_key="sk-...",
                  from_date="2018-01-01",
                  to_date="2022-12-31",
                  sample_size=1000,
                  path_to_categories="/path/to/my_tech_categories.csv")
```

**Output:** Saves sample texts and a categories snapshot to `path_to_results/tte_samples/`, and submits a batch job to the OpenAI API. The batch ID is saved to a local file for retrieval by `label_sample()`.

---

### `label_sample`

Retrieves completed GPT batch results and saves a labeled sample file.

**Purpose:** Checks the status of the OpenAI batch submitted by `create_sample()` and, once complete, downloads and parses the results into a labeled CSV for use by `train_model()`. If the batch is still in progress, prints the current status and returns without saving so the call can be retried later.

**Key Arguments:**
- `path_to_results` (str, required): Path where sample files are saved.
- `cat_type` (str, required): Classification type (`'tech'` or `'task'`).
- `api_key` (str, required): OpenAI API key.
- `batch_id` (str, optional): Batch ID to retrieve. If None, reads the ID saved by `create_sample()`. Pass explicitly to retrieve a specific earlier batch.

**Example:**
```python
tte.label_sample(path_to_results="/path/to/results/",
                 cat_type="tech",
                 api_key="sk-...")
```

**Output:** A labeled CSV saved to `path_to_results/tte_samples/`. Each row contains the original text plus GPT classification columns (`class_gpt_1`, `class_gpt_2`, …). For technology categories these are binary indicators; for task categories they are a ranked ordering of category indices.

---

### `train_model`

Fine-tunes a sentence embedding model for technology or task classification.

**Purpose:** Loads the labeled sample from `label_sample()`, formats it for contrastive training, and fine-tunes the base model using Matryoshka Representation Learning (MRL) and asymmetric instruct prompts. Optionally calls `validate_model()` after training to compute per-category similarity thresholds. The fine-tuned model is saved to `tte/tte_models/{cat_type}_custom/` and can be passed directly to `classify_patents()` or `classify_tasks()` via the `model` argument.

**Key Arguments:**
- `path_to_data` (str, required): Parent directory of the downloaded `tte/` dataset folder.
- `path_to_results` (str, required): Path where sample files are saved.
- `cat_type` (str, required): Classification type (`'tech'` or `'task'`).
- `val_frac` (float, optional): Fraction of formatted examples to hold out for validation. Default 0.1.
- `run_validation` (bool, optional): If True, calls `validate_model()` after training. Default True.
- `steps_total` (int, optional): Total training steps. Defaults to 2250 (tech) or 1000 (task).
- `steps_per_eval` (int, optional): Steps between evaluations and checkpoints. Default 125.
- `batch_size` (int, optional): Per-device batch size. Defaults to 4 (tech) or 8 (task).
- `learn_rate` (float, optional): Learning rate. Defaults to 8e-6 (tech) or 1e-5 (task).
- `gradient_accumulation_steps` (int, optional): Gradient accumulation steps, giving an effective batch of 32 at defaults. Defaults to 8 (tech) or 4 (task).
- `matryoshka_dims` (list, optional): MRL embedding dimensions. Default `[1024, 512, 384, 256]`. Pass None to disable MRL. Dimensions exceeding the model's native embedding size are removed automatically with a warning.
- `max_seq_length` (int, optional): Tokenizer sequence length cap. Default 512.
- `base_model` (str, optional): HuggingFace model ID or local path for the base embedding model. Defaults to `'Qwen/Qwen3-Embedding-0.6B'`.
- `use_bf16` (bool, optional): Whether to enable bf16 mixed precision training. Default True.
- `use_instruct` (bool, optional): Whether to apply asymmetric instruct prompts to the query side during training and evaluation. Default True.

**Example:**
```python
tte.train_model(path_to_data="/path/to/data/",
                path_to_results="/path/to/results/",
                cat_type="tech",
                steps_total=2250,
                run_validation=True)
```

**Output:** Fine-tuned model saved to `tte/tte_models/{cat_type}_custom/`. If `run_validation=True`, also produces threshold and performance CSV files in `path_to_results/tte_samples/`.

---

### `validate_model`

Evaluates model classification performance and computes per-category similarity thresholds.

**Purpose:** Loads the fine-tuned custom model (or the manifest default if no custom model exists), encodes validation texts and category descriptions, and produces per-category thresholds at multiple precision targets. For technology categories, the output thresholds file (`tte_samples/tech_thresholds.csv`) is automatically detected by `classify_patents()` when no `cutoff` is specified — no manual passing required. Thresholds can also be passed explicitly via the `cutoff` argument if a non-default column or value is preferred. For task categories, a classification accuracy report is produced instead (task classification uses argmax and has no threshold).

**Key Arguments:**
- `path_to_data` (str, required): Parent directory of the downloaded `tte/` dataset folder.
- `path_to_results` (str, required): Path where sample files are saved.
- `cat_type` (str, required): Classification type (`'tech'` or `'task'`).
- `beta` (list, optional): Beta values for F-score threshold computation. Tech only. Default `[0.5, 1]`.
- `prec_avg` (list, optional): Average precision targets for threshold computation. Tech only. Default `[0.25, 0.5, 0.75]`.
- `prec_mrg` (list, optional): Marginal precision targets (logit-based) for threshold computation. Tech only. Default `[0.25, 0.5, 0.75]`.

**Example:**
```python
tte.validate_model(path_to_data="/path/to/data/",
                   path_to_results="/path/to/results/",
                   cat_type="tech")
```

**Output:** For technology categories, a thresholds CSV and a performance summary CSV (accuracy, precision, recall, and F1 per category at the 50% marginal precision threshold) saved to `path_to_results/tte_samples/`. For task categories, a classification report CSV (accuracy, precision, recall, and F1 per category).

---

### `embed_data`

Re-generates the classification embedding files used by `classify_patents()` and `classify_tasks()`.

**Purpose:** Encodes patent abstracts and O\*NET task statements using the technology and task classification models respectively, and saves the resulting `.npy` files to the dataset directory. This must be run after `train_model()` when using a custom model, because `classify_patents()` and `classify_tasks()` load pre-computed embeddings from disk. Patents are encoded with the tech model using the technology instruct prompt; tasks are encoded with the task model using the task instruct prompt.

**Key Arguments:**
- `path_to_data` (str, required): Parent directory of the downloaded `tte/` dataset folder.
- `embed_type` (str, optional): What to re-embed. `'patents'` processes all year files using the tech model; `'tasks'` processes the O\*NET task file using the task model; `'both'` does both. Default `'both'`.
- `tech_model` (str, optional): Path to the technology classification model. Defaults to the `tech_model` entry in `dataset_manifest.json` (i.e. the fine-tuned custom model after `train_model()` is run). Ignored when `embed_type='tasks'`.
- `task_model` (str, optional): Path to the task classification model. Defaults to the `task_model` entry in `dataset_manifest.json`. Ignored when `embed_type='patents'`.
- `embed_dim` (int, optional): Embedding dimension to store. If less than the model's native dimension, MRL truncation is applied. Defaults to `class_embedding_dim_mrl` in `dataset_manifest.json` (384).
- `use_instruct` (bool, optional): Whether to apply asymmetric instruct prompts. Default True.
- `force` (bool, optional): If True, re-embeds files that already exist. Default False.

**Example:**
```python
tte.embed_data(path_to_data="/path/to/data/",
               embed_type="patents")
```

**Output:** Updated `.npy` embedding files written in-place to the dataset directory (`tte/tte_YYYY/patents/patent_embed_YYYY.npy` for patents; `tte/tte_onet_oews/tasks/task_embed.npy` for tasks).

---

## Citation

If you use this package in your research, please cite:

```
Mouton, Andre (2026). "Measuring Task-Level Technological Exposure: A Language Model Approach," Working Papers 132, Wake Forest University, Economics Department.
```

## License and Data

The matched dataset and trained models are available at DOI: [10.5281/zenodo.17643646](https://doi.org/10.5281/zenodo.17643646). The dataset is updated quarterly to incorporate newly published patent applications and annual ONET revisions.

The distributed sentence embedding models are fine-tuned derivatives of [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (Qwen Team, Alibaba Group), which is licensed under the Apache License 2.0. Attribution and citation details are provided in [NOTICES.md](NOTICES.md).

For questions or issues, please contact: moutona@wfu.edu
