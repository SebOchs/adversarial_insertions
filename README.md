# Adversarial Insertions

## Short guide
### Install environment:
1. Create a conda environment with suitable name and python version 3.8
```
conda create --name adversarial_insertions python=3.8
conda activate adversarial_insertions
```
2. Install needed libraries such as pytorch, transformers etc.
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install transformers==4.2.1 bert-score==0.3.7 spacy==2.3.5 pytorch-lightning==1.1.2 matplotlib==3.3.3 jsonlines==1.2.0
```

### Preprocessing

Use preprocessing.py to preprocess the datasets used in this project.
The datasets were downloaded from these sources:
- MNLI: https://cims.nyu.edu/~sbowman/multinli/
- MRPC: https://www.microsoft.com/en-us/download/details.aspx?id=52398
- QQP: https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip
- RTE: https://dl.fbaipublicfiles.com/glue/data/RTE.zip
- WiC: https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WiC.zip
- SciEntsBank 3way seems to be only available here: https://www.kaggle.com/datasets/smiles28/semeval-2013-2-and-3-way

### Training
Models were trained with the reported hyperparameters using training.py

### Attack
1. Use find_top_words.py to find most common adjectives and adverbs
2. Get correct predictions for a given label and the respective model with get_correct_predictions.py
3. Prepare and execute the attack with prepare_attack.py and attack.py
4. Run plot_results.py to get confidence value histograms for BERT models and samples for successful adversarial examples for both T5 and BERT
5. Use analyze_results.py to get success rates of the attack
6. Print out samples with pretty_displaying.py
7. To analyze adjective and adverb occurrences in different classes, use frequency_analysis.py (currently WIP)

Make sure that each function gets proper input, such as folder or file paths as strings.

In cooperation with Anna Filighera and the Multimedia Communications Lab at TU Darmstadt.
