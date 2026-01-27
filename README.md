# task-tech-exposure

This package implements the methodology of 'Measuring Task-Level Technological Exposure: A Language Model Approach' (2026). It handles file download and management, data classification, and exposure measurement.

Contained in this README:

* [Background](#background)
* [Quick Start](#quick-start)
* [Package Contents](#package-contents)
    * [download_data](#download_data)
    * [classify_patents](#classify_patents)
    * [classify_tasks](#classify_tasks)
    * [measure_exposure](#measure_exposure)
* [Citation](#citation)
* [License and Data](#license-and-data)

## Background

This package provides tools for measuring the exposure of occupational tasks to technological change, using the accompanying data and resources at [10.5281/zenodo.17643646](https://doi.org/10.5281/zenodo.17643646).

The underlying methodology -- described at length in the paper ([link here](https://drive.google.com/file/d/1--GXGpKy3WdjjQbX1T398QZC2xdzYd3H/view?usp=drive_link)) -- matches USPTO patent abstracts with ONET occupational task statements and classifies them using sentence embedding models from [Sentence Transformers](https://sbert.net/). The matching process identifies patent-task pairs with high semantic similarity, while separate classification models categorize patents by technology type and tasks by functional category. Sentence-BERT models are fine-tuned and validated against outputs from GPT-4.1, allowing them to achieve a comparatively high level precision. Exposure indices are calculated for each task and technology category, conditional on a desired level of cross-sectional and temporal aggregation.

The package provides access to a living dataset that is updated quarterly with newly published patent applications and annual ONET revisions. This ensures minimal time lag and enables analysis at fine temporal resolution (monthly, quarterly, or annual). Users can define their own technology and task categories, customize weighting schemes, and generate exposure measures at various aggregation levels (aggregate economy-wide, by occupation, or by wage percentiles). All computations run locally, and the methodology is fully transparent and replicable.

## Quick Start

Package installation requires Git, and can be executed in the terminal:

```bash
pip install --upgrade git+https://github.com/armouton/task-tech-exposure.git
```

Execution requires, at a minimum, a pair of paths indicating the directories in which to store the data and the exposure measures. Note that the full dataset is approximately 10GB, but that file download can be limited to a particular date range.

```python
import tte

# Download/update the dataset
tte.download_data(path_to_data="/path/to/data/", ...)

# Classify patents by technology category
tte.classify_patents(path_to_data="/path/to/data/",
                     path_to_results="/path/to/results/", ...)

# Classify tasks by functional category
tte.classify_tasks(path_to_data="/path/to/data/",
                   path_to_results="/path/to/results/", ...)

# Measure exposure
tte.measure_exposure(path_to_data="/path/to/data/",
                     path_to_results="/path/to/results/", ...)
```

By default, technology and task classification categories are specified in CSV files located at `tte/tte_models/category_descriptions/`. These can be modified directly. Also included are GenAI prompts for expanding short category definitions into longer descriptions more suitable for Sentence-BERT.

## Package Contents

### `download_data`

Downloads the matched patent-task dataset from the DOI repository to a local directory.

**Purpose:** Retrieves matched USPTO patent applications and GPT-expanded ONET task statements, the Sentence-BERT embeddings for both sets of data, and supplementary employment and wage data. The dataset can be filtered by date range to focus on specific time periods and reduce download time.

**Key Arguments:**
- `from_year` (str, optional): Start date in 'YYYY' format. If None, downloads full dataset from earliest available annual file (2001).
- `to_year` (str, optional): End date in 'YYYY' format. If None, downloads full dataset through most recent available annual file.
- `path_to_data` (str, required): Path where dataset files will be saved.
- `doi_url` (str, optional): Alternative URL if downloading from previous data version. Defaults to stable DOI for current version.
- `force_update` (bool, optional): If True, re-downloads files even if they already exist locally. Default is False.

**Example:**
```python
import tte

tte.download_data(from_year="2015", 
                  to_year="2020",
                  path_to_data="/Users/username/tte_data/")
```

**Output:** Downloads matched dataset files to the specified directory, including patent abstracts, task statements, sentence embeddings, and supplementary data files.

---

### `classify_patents`

Classifies patents into technology categories using fine-tuned sentence embedding models.

**Purpose:** Assigns patent applications to user-defined or default technology categories (*e.g.* AI, robotics, software) based on semantic similarity between patent abstracts and category descriptions. Uses a trained Sentence-BERT model with customizable similarity thresholds.

**Key Arguments:**
- `path_to_data` (str, required): Path to directory containing the downloaded dataset.
- `path_to_results` (str, required): Path where classification results will be saved.
- `path_to_output` (str, optional): Specific path and filename for output CSV. Defaults to 'tech_classification.csv' in results directory.
- `path_to_descriptions` (str, optional): Path to CSV file containing technology category descriptions. Uses default files if not specified (see [Quick Start](#quick-start) above).
- `sbert_tech` (str, optional): Path to sentence transformer model for technology classification. Uses default fine-tuned model from manifest if None.
- `tech_cutoff` (float, optional): Similarity threshold for classifying patents into technology categories. Default value is taken from the `dataset_manifest.json` file, and is based on a 50% marginal accuracy threshold.
- `tech_groups` (list of lists, optional): Groups of technology categories for which classifications should be mutually exclusive (*e.g.* [[0,1], [3]] indicates that a patent should be matched to *at most* one of the first two categories in the descriptions file).
- `tech_priority` (str, optional): Method for resolving grouped categories. 'score' selects highest similarity within group; 'order' uses list ordering. Default is 'score'. 

**Example:**
```python
tte.classify_patents(path_to_data="/Users/username/tte_data/",
                     path_to_results="/Users/username/tte_results/",
                     tech_cutoff=0.75,
                     tech_groups=[[0,1], [3,2]],
                     tech_priority="score")
```

**Output:** CSV file containing patent IDs with assigned technology categories.

---

### `classify_tasks`

Classifies occupational task statements into functional categories using fine-tuned sentence embedding models.

**Purpose:** Assigns ONET task statements to user-defined or default task categories (e.g., cognitive, manual, routine) based on semantic similarity between task descriptions and category definitions.

**Key Arguments:**
- `path_to_data` (str, required): Path to directory containing the downloaded dataset.
- `path_to_results` (str, required): Path where classification results will be saved.
- `path_to_output` (str, optional): Specific path and filename for output CSV. Defaults to 'task_classification.csv' in results directory.
- `path_to_descriptions` (str, optional): Path to CSV file containing task category descriptions. Uses default files if not specified (see [Quick Start](#quick-start) above).
- `sbert_task` (str, optional): Path to sentence transformer model for task classification. Uses default fine-tuned model from manifest if None.

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
- `path_to_data` (str, required): Path to directory containing the downloaded dataset.
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

## Citation

If you use this package in your research, please cite:

```
Mouton, Andre (2026). "Measuring Task-Level Technological Exposure: A Language Model Approach," Working Papers 132, Wake Forest University, Economics Department.
```

## License and Data

The matched dataset and trained models are available at DOI: [10.5281/zenodo.17643646](https://doi.org/10.5281/zenodo.17643646). The dataset is updated quarterly to incorporate newly published patent applications and annual ONET revisions.

For questions or issues, please contact: moutona@wfu.edu
