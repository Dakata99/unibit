# 1. Which markers are important?

ChatGPT:
```
From the literature on this specific dataset and similar HCV cohorts, the lab markers that keep popping up as most important are:

AST
ALT
ALP
GGT
BIL
ALB
and sometimes CHOL.
```

# 2. On what basis do you drop / keep biomarkers?

Think of three layers:

### Domain & literature

Use medicine/biochem intuition (from papers) as a sanity check:

- Multiple ML studies on the UCI HCV data (and related HCV datasets) find that AST, ALP, ALT, GGT, BIL, ALB are consistently top contributors to models that distinguish healthy vs HCV / stages.

- CHOL and PROT can also carry signal, but often a bit less.
- Age and Sex often help but are usually weaker than the core liver biomarkers.

You don’t blindly trust this, but if Orange tells you AST, ALT, GGT, BIL are important… that matches clinical reality, which is nice.

### Statistical feature importance

Use feature scoring / selection to quantify:

- How strongly each feature is related to the class, e.g. Information Gain, ANOVA, ReliefF, etc.

- How redundant features are with each other, e.g. two biomarkers that are almost perfectly correlated → you might keep only one.

### Model performance vs #features

For your TFLite model, the real decision is:

```
“How many features can I drop before my validation performance falls off?”
```

So you want to compare:

- Model with all biomarkers (baseline).

- Model with top 3, top 5, top 7 features (according to ranking).

- Choose the smallest subset where metrics (AUC, F1, accuracy) are “good enough” compared to the full model (say within 1–2%).

# 3. How to do this in Orange step-by-step

## Step 0 – Define the problem

Decide what your target is:

- Binary: Blood donor vs any HCV (merge 1/2/3 into “HCV”).

- Or multi-class: 0, 0s, 1, 2, 3 as separate classes.

Orange can do both, but be clear because feature scores depend on the class definition.


> Decision: multi-class: 0, 0s, 1, 2, 3.

## Step 1 – Load and set roles

1. File widget → load hcvdat0.csv.
2. Connect File → Select Columns:
    - Set:
        - Target (class): Category
        - Attributes (features): Age, Sex, ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT
        - Meta: ID (you don’t want the model to use it)

This is important: your TFLite model must eventually expect exactly these chosen attributes in this order.

Can I just skip the ID field instead of making it meta?
```
Short answer:

- For the actual model (Python/TFLite) → yes, you should skip/drop ID completely.

- In Orange, it’s usually better to set ID as Meta instead of deleting it, because:
    - Meta attributes are never used for learning (they’re ignored by Rank, learners, Test & Score, etc.).
    - But they are still available:
        - in Data Table / Scatter Plot tooltips,
        - in saved data, so you can trace back misclassified samples later.
```

> Done!

## Step 2 – Handle missing values & scaling

Use the `Preprocess` widget.

Connect: `Select Columns → Preprocess`.

Inside `Preprocess`:
1. Impute missing values
    - “Average / Most Frequent” is a reasonable default:
        - Continuous (lab values) → mean.
        - Discrete (Sex, Category) → most frequent.
    - Alternatively, “Remove rows with missing values” if you want a clean subset, but you’ll then need a rule for missing values at inference time anyway. For the C++ app, it’s usually better to have a fixed imputation rule (like mean) you can hardcode.

2. Normalize (optional but helpful for many TF models):
    - Normalization to [0, 1] or standard score (zero mean, unit variance).
    - If you do this in Orange only for exploration, no problem. For the production pipeline, do final normalization in Python (or using a preprocessing layer in TensorFlow) so that the TFLite model includes it.

> Done!

## Step 3 – Rank the biomarkers in Orange

Now the fun part.

1. Add `Rank` widget.
2. Connect: `Preprocess → Rank`.

In Rank:
- Select scorers (for classification + numeric features):
    - Information Gain
    - Gain Ratio
    - Gini
    - ANOVA
    - ReliefF
    - FCBF

You’ll see a table: rows = features, columns = scoring methods.

Optionally, add model-based scorers:
- Add Random Forest and Logistic Regression Learner widgets.
- Connect them to Rank’s Scorer input.
- New columns appear with importance scores from those models.

Then in Rank:
- On the right, choose:
    - Output: Best ranked → N (e.g. 3, 5, 7).
- The Reduced Data output from Rank will contain only the selected top-N features.

This gives you a data-driven ranking of biomarkers.

## Step 4 – See how many features you actually need

Now compare model performance with and without feature selection.

Build two branches:
1. Branch A – all features

    `File → Select Columns → Preprocess → Test & Score`


2. Branch B – selected features

    `File → Select Columns → Preprocess → Rank → Test & Score`

In both Test & Score widgets:
- Connect the same learners (e.g. Logistic Regression, Random Forest, maybe SVM).
- Use default 10-fold cross-validation or similar.

Then:
- In Branch B’s Rank widget, start with:
    - Best ranked: 3 → check Test & Score metrics.
    - Best ranked: 5
    - Best ranked: 7, etc.

Compare:
- CA (classification accuracy)
- AUC
- F1 (especially if you care about minority HCV stages)

You’re looking for the smallest N where performance is close to Branch A.

Typical outcome (roughly, based on literature):
- Models using only ~4–6 biomarkers like AST, ALT, ALP, GGT, BIL, ALB can reach almost the same performance as using all 10+ features in HCV datasets.

## Step 5 – Fix the final biomarker subset for your project

Once you’re happy (“Top 5 features → performance is good enough”), do:
- In Rank, leave Best ranked: 5.
- Note down the names and order of those 5 biomarkers.
- Connect Rank to Save Data if you want an exported CSV with just those columns plus the label.

This list is now your Y biomarkers that the C++ / TFLite app will use.

# 4. How this integrates with Python + TensorFlow + TFLite + C++

Once you’ve picked the biomarkers with Orange, you replicate that pipeline in Python:

## In Python (training side)

1. Load the CSV.
2. Select the same columns as Orange’s top-N, in the same order.
3. Apply the same preprocessing you used conceptually in Orange:
    - Impute missing values (e.g. with SimpleImputer or manually using means).
    - Normalize (e.g. StandardScaler or build a tf.keras.layers.Normalization layer and adapt it).
4. Train your TensorFlow model on that feature vector.
5. Convert to TFLite.

If you use a `tf.keras.layers.Normalization` (or similar) as the first layer and adapt it on your training data, all scaling is inside the model, which makes C++ much simpler.

## In C++ (inference side)

Your TFLite C++ app must:
- Accept those Y biomarkers as input (e.g. 5 floats).
- Ensure they are:
    - In exactly the same order as during training.
    - Preprocessed in the same way:
        - If you put normalization in the model → just pass raw lab values.
        - If not, you must hardcode means/stds or other scaling constants.

Feed the feature vector into the TFLite model input tensor, invoke, and read the output (class probabilities, logits, etc.).

# 5. TL;DR – Direct answers

On what basis do I reduce the number of biomarkers?

Combine:
- medical sense (AST/ALT/ALP/GGT/BIL/ALB are key for HCV),
- feature scores (Info Gain, ANOVA, ReliefF, RF/LogReg importance),
- and final model performance vs number of features.



From the top table:

- AUC – how well the model separates the classes overall.
- CA (Classification Accuracy) – % of correctly classified examples.
- F1 – balance between precision/recall (averaged over classes).
- MCC – a single “correlation-like” score that handles imbalance well.

For making decisions:

- Pick one main metric (I’d choose F1 or MCC for multi-class + mild imbalance).
- Use AUC + CA as secondary checks.
- Compare the same model between:
    - Test & Score (FULL) – all biomarkers
    - Test & Score (REDUCED) – top-N biomarkers from Rank
