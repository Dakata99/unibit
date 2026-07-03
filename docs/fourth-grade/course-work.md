# Hepatitis C virus (HCV) monitoring assistant

## Prerequisites

1. Get WSL2 (Ubuntu24)

    Follow this guide: https://learn.microsoft.com/en-us/windows/wsl/install

    - Install Ubuntu 24 in WSL2

        1.1. Open Microsoft Store.

        1.2. Search for Ubuntu 24.04 LTS.

        1.3. Click Install.

        1.4. Once installed, launch it from Start Menu or run:
        ```bash
        wsl -d Ubuntu-24.04
        ```

    After installation, you need to set up a username and password. Then run:
    ```bash
    sudo apt update && sudo apt upgrade -y
    ```

2. Install needed dependencies
```bash
sudo apt install python3-venv python-is-python3 build-essential cmake ninja-build
```

> NOTE: currently, TFLite library is built for Ubuntu24 only, so
if trying to build the application with different distro, the `libstdc++6` will be problematic,
because it is with different verion in different distros.

## Formatting/linting

Use `black` to format the python scripts.
Use `ruff` to lint the python scripts.

## Topics

### Dataset Name

**URL:** <URL_HERE>

**# of Instances:** <NUMBER_OF_ROWS>

**# of Features (Columns):** <NUMBER_OF_COLUMNS>

**Feature List:**
- Feature 1
- Feature 2
- Feature 3
- ...

**Target Variable:** `<TARGET_COLUMN_NAME>`

**Target Classes / Values:**
- Class 1
- Class 2
- Class 3
- ...

**Dataset Type:**
- Classification / Multiclass / Regression
- Medical / Tabular / Imaging / etc.

**Notes:**
- Any quirks, missing values, imbalance, etc.


### Indian Liver Disease Dataset

**URL:** https://www.kaggle.com/datasets/paramjeetsinghds/indian-liver-disease-dataset

**# of Instances:** 68,000 (training CSV) +

**# of Features (Columns):** 29

**Feature List:**
- Patient_ID
- Age
- Gender
- Occupation
- BMI
- Obesity_Class
- Diet_Quality
- Physical_Activity
- Sleep_Hours
- Smoking_Status
- Alcohol_Consumption
- Sym_Fatigue
- Sym_Jaundice
- Sym_Abdominal_Pain
- Sym_Itching
- Sym_Ascites
- Sym_Dark_Urine
- Sym_Weight_Loss
- Comorb_Diabetes
- Comorb_Hypertension
- Comorb_Genetic_History
- ALT
- AST
- Bilirubin
- Albumin
- Platelets
- Alk_Phosphatase
- Liver_Disease_Type (target)

**Target Variable:**
`Liver_Disease_Type`

**Target Classes / Values:**
- Normal
- Fatty_Liver
- Alcoholic_Liver_Disease
- Cirrhosis
- Hepatitis_B
- Hepatitis_C

**Dataset Type:**
Multiclass classification — medical tabular dataset

**Notes:**
- Symptoms may be unreliable or optional
- Strong lab markers included (ALT, AST, Bilirubin, Albumin, ALP, Platelets)
- Contains lifestyle + comorbidity features

### Hepatitis C Virus (HCV) for Egyptian Patients

**Link/URL:**
https://www.kaggle.com/datasets/mdrakiburrahman10/hepatitis-c-virus-hcv-for-egyptian-patients

**# of Instances:**
Not specified on the page (dataset file is very small: ~1.88 kB)

**# of Features (Columns):**
29 features


**Feature List:**
The dataset page lists **29 feature names**, but the exact column names are not shown in the preview.
It does show that features are **discretized** into categories like:
- `Absent` / `Present`
- Numeric bins such as `[0; 20[`, `[20; 40]`, `]40; 128]`


**Target Variable:**
Not explicitly stated on the page.
Given the dataset title, the target is likely a **Hepatitis C diagnosis or stage**, but the page does not confirm this.

**Target Classes / Values:**
Not shown on the dataset page.

**Dataset Type:**
Medical tabular dataset
Likely classification (based on context), but not explicitly stated.

**Notes:**
- Dataset contains **29 discretized features**.
- Values include categorical indicators like `Absent` / `Present` and numeric ranges.
- No description, documentation, or column details are provided on the Kaggle page.
- Only one file is available: `Discretization-Criteria.csv` (1.88 kB).


### Others

- https://archive.ics.uci.edu/dataset/102/thyroid+disease
- Photos:
    - https://medmnist.com/
    - https://universe.roboflow.com/roboflow100vl-fsod/liver-diseases-fsod-gjyx
- https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008
- https://www.kaggle.com/datasets/nikee7/parkinsons-tremor-classification-dataset/data

- Liver disease related:
    - https://www.kaggle.com/datasets/davidechicco/hepatitis-c-ehrs-from-japan
        - has 123 instances
    - https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset
        - Same as mine


https://github.com/EvangeliaPetraki/Hepatitis_Dataset_Analysis_and_Classification
https://github.com/mauro-nievoff/MultiCaRe_Dataset
