# Demographic Bias Transfer Metrics

Metric system for bias analysis in Facial Expression Recognition datasets

The details of this work are detailed in a paper, soon to be published in arXiv.


## Requirements

To replicate the experiments, the requirements are detailed in the file `conda_env.yml`. The environment can be created with conda:

```
conda env create -f conda_env.yml
```

The data for the experiments is not included, but can be downloaded from:

- [Affectnet](http://mohammadmahoor.com/affectnet/)
- [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- [FER+](https://github.com/Microsoft/FERPlus)

The experiments are dependent on a folder structure, customizable from the file `fer/paths.py`. The minimum folders to be generated are:

```
data
├─ affectnet
|  ├─ train_set
|  ├─ val_set
├─ fer2013
|  ├─ images_from_ferplus
processed
├─ data
|  ├─ aggregated
|  ├─ source
|  ├─ emolabel
|  ├─ demographic
|  ├─ figures
|  ├─ composed_datasets
|  ├─ final
├─ dataloaders
├─ learners
├─ results
├─ samples
├─ logs
```

For Affectnet, the data can be directly extracted to the folder. For FER+, the [github repo](https://github.com/Microsoft/FERPlus) includes a tool to extract the images from the original FER2013 data. In our experiments we have only used the images extracted that way.

The rest of the folders hold diverse artifacts of the experiments. The more relevant folders are:
- `processed/data/final` and `processed/data/composed_datasets`, which hold the `.csv` files genrated from the source datasets.
- `processed/data/results`, which hold the final results of the experiments
- `processed/data/logs`, with the execution logs

## Usage

With the source data in the correct folders, the pipeline is as follows:

- Prepare the datasets (proprocessing):
  ```
  conda activate <name_of_the_environment>
  python prepare_and_crop_datasets.py 
  ```

- Demographic analysis with [Fairface](https://github.com/joojs/fairface):
  ```
  python demographic_analysis_datasets.py 
  ```

- Run the experiments:
  ```
  python main_experiment.py
  ```
  Before running the experiments, edit the file commenting the appropriate lines for running either the Affectnet or the FER+ version.
  
Once the experiments have finished (it can take from hours to days, depending on the setup), the analysis is performed with the `analysis_affectnet.ipynb` and `analysis_ferplus.ipynb` notebooks, which generate the same tables and results from the paper.

## Results

The results are detailed in the paper. We have not included the artifacts and products mainly because of the file size limits of GitHub, and partly as they can be considered direct derivatives from the original datasets. By request and only for reproduction purposes, we can provide some of these intermediate files to other researchers.

## Code

The core functions to replicate our metrics are commented and provided on the `fer/metrics.py` file. The mathematical definitions can be found in our paper.

Unfortunately, the rest of the code is not fully commented. If needed, contact directly for any explanation or clarification.

## Acknowledments

We want to thank Karkkainen, K., & Joo, J. for their great work in [Fairface](https://github.com/joojs/fairface), which has made our research possible.

Our work was funded by a predoctoral fellowship of the Research Service of Universidad Publica de Navarra, the Spanish MICIN (PID2019-108392GB-I00 and PID2020-118014RB-I00 / AEI / 10.13039/501100011033), and the Government of Navarre (0011-1411-2020-000079 - Emotional Films).
