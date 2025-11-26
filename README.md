[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=dnafinder/roc&file=roc.m)

# roc

## 📌 Overview

roc performs Receiver Operating Characteristic (ROC) curve analysis for a continuous (or ordinal) diagnostic test or classifier with a binary outcome (healthy vs diseased).

Given a set of test results and class labels, the function:

- computes Sensitivity and Specificity at a grid of thresholds
- constructs the ROC curve (True Positive Rate vs False Positive Rate)
- estimates the Area Under the ROC Curve (AUC) via the Mann–Whitney–Wilcoxon statistic
- provides the standard error, confidence interval, and a z-test for AUC > 0.5
- optionally computes positive and negative predictive values (PPV, NPV) if the true prevalence is specified
- identifies several “optimal” cut-off points according to different criteria (max Se, max Sp, max efficiency, max PLR, min NLR, cost-effective Se = Sp)

The function reproduces the classical ROC analysis used in medical decision making and diagnostic test evaluation, while giving a rich numeric and graphical output.

---

## 📐 Syntax

The main calling forms are:

- ROCout = roc(X)
- ROCout = roc(X, THRESHOLD)
- ROCout = roc(X, THRESHOLD, ALPHA)
- ROCout = roc(X, THRESHOLD, ALPHA, VERBOSE)
- ROCout = roc(X, THRESHOLD, ALPHA, VERBOSE, PLOTTING)
- ROCout = roc(X, THRESHOLD, ALPHA, VERBOSE, PLOTTING, PREVALENCE)

If no output argument is requested, for example

- roc(X, THRESHOLD, ALPHA, VERBOSE, PLOTTING, PREVALENCE)

the function prints a detailed report in the Command Window and produces plots when PLOTTING is set to 1.

---

## 📥 Inputs

### X

- Type: numeric matrix, size N-by-2
- Columns:
  - X(:,1): test values (real, finite, non-NaN, non-empty)
  - X(:,2): binary class labels
    - 1 = unhealthy or diseased
    - 0 = healthy or non-diseased

All labels must be 0 or 1. If all labels are 0 (only healthy) or all are 1 (only unhealthy), the function throws an error.

---

### THRESHOLD (optional)

Controls how many distinct cut-off values are used for the ROC analysis.

- THRESHOLD = 0 (default)  
  All unique positive values in X(:,1) are used as candidate cut-offs.
- THRESHOLD > 2  
  That number of quantile-based unique values is used as candidate cut-offs (minimum allowed is 3).

This argument can be used to reduce computation time for very large datasets by using a coarser grid of thresholds.

---

### ALPHA (optional)

- Significance level for the AUC confidence interval and z-test.
- Default value: 0.05
- Must satisfy 0 < ALPHA < 1.

---

### VERBOSE (optional)

Controls textual output to the Command Window.

- 0  no textual report
- 1  print full report (default)

When VERBOSE = 1, the function prints:

- AUC with standard error and confidence interval
- p-value for AUC greater than 0.5 and an interpretative comment (Perfect, Excellent, Good, Fair, Poor, Fail, or less than chance)
- a detailed table of ROC curve data (cut-off, Sensitivity, Specificity, Efficiency, PLR, NLR, and optionally PPV, NPV).

---

### PLOTTING (optional)

Controls graphical output.

- 0  no plots
- 1  display ROC and cut-off plots (default)

When PLOTTING = 1, the function produces:

1. ROC curve plot (True Positive Rate vs False Positive Rate) with the area under the curve highlighted.
2. Cut-off analysis plot (test cut-off on the x-axis) showing:
   - fitted curves for Sensitivity, Specificity, and Efficiency
   - vertical lines at selected cut-off points (max Se, max Sp, max efficiency, cost-effective Se = Sp, max PLR, min NLR).

---

### PREVALENCE (optional)

- True prevalence of the condition, expressed as a probability in the open interval (0, 1).

If PREVALENCE is provided:

- prior odds POD = PREVALENCE / (1 − PREVALENCE) are computed
- for each threshold, the function computes:
  - PPV (Positive Predictive Value)
  - NPV (Negative Predictive Value)
- PPV and NPV are included in the output table as Pos_pred and Neg_Pred (in percent).

If PREVALENCE is omitted or empty:

- PPV and NPV are not computed
- ROC curve and AUC are still fully computed from Sensitivity and Specificity.

---

## 📤 Outputs

### Structure ROCout

If the function is called with an output argument, it returns a struct ROCout with the following fields:

- ROCout.AUC  
  Area Under the ROC Curve.

- ROCout.SE  
  Standard error of the AUC.

- ROCout.ci  
  Confidence interval for the AUC at level 1 minus ALPHA, as a row vector [lower upper].

- ROCout.p  
  p-value for the test:
  - H0: AUC = 0.5
  - H1: AUC > 0.5

- ROCout.xr  
  x-coordinates of ROC points (False Positive Rate).

- ROCout.yr  
  y-coordinates of ROC points (True Positive Rate).

When cut-off analysis is performed and AUC is significantly greater than 0.5:

- ROCout.matrix  
  Numeric matrix summarizing all evaluated cut-offs. Its columns include:
  - Cut_off        threshold value
  - Sensitivity
  - Specificity
  - Efficiency     Matthews-like index
  - PLR            Positive Likelihood Ratio
  - NLR            Negative Likelihood Ratio
  - Pos_pred       PPV in percent (only when PREVALENCE is set)
  - Neg_Pred       NPV in percent (only when PREVALENCE is set)

- ROCout.co  
  Selected cut-off points and their corresponding Sensitivity and Specificity, typically:
  - maximum Sensitivity cut-off
  - maximum Specificity cut-off
  - cost-effective cut-off with Sensitivity equal to Specificity
  - maximum Efficiency cut-off
  - maximum PLR cut-off
  - minimum NLR cut-off

- ROCout.table  
  Same as ROCout.matrix, provided for backward compatibility.

---

## 📊 Example

Example data matrix with two columns: test value and class label (1 for diseased, 0 for healthy):

x = [165 1;140 1;154 1;139 1;134 1;154 1;120 1;133 1;150 1; ...
     146 1;140 1;114 1;128 1;131 1;116 1;128 1;122 1;129 1;145 1;117 1; ...
     140 1;149 1;116 1;147 1;125 1;149 1;129 1;157 1;144 1;123 1;107 1; ...
     129 1;152 1;164 1;134 1;120 1;148 1;151 1;149 1;138 1;159 1;169 1; ...
     137 1;151 1;141 1;145 1;135 1;135 1;153 1;125 1;159 1;148 1;142 1; ...
     130 1;111 1;140 1;136 1;142 1;139 1;137 1;187 1;154 1;151 1;149 1; ...
     148 1;157 1;159 1;143 1;124 1;141 1;114 1;136 1;110 1;129 1;145 1; ...
     132 1;125 1;149 1;146 1;138 1;151 1;147 1;154 1;147 1;158 1;156 1; ...
     156 1;128 1;151 1;138 1;193 1;131 1;127 1;129 1;120 1;159 1;147 1; ...
     159 1;156 1;143 1;149 1;160 1;126 1;136 1;150 1;136 1;151 1;140 1; ...
     145 1;140 1;134 1;140 1;138 1;144 1;140 1;140 1;159 0;136 0;149 0; ...
     156 0;191 0;169 0;194 0;182 0;163 0;152 0;145 0;176 0;122 0;141 0; ...
     172 0;162 0;165 0;184 0;239 0;178 0;178 0;164 0;185 0;154 0;164 0; ...
     140 0;207 0;214 0;165 0;183 0;218 0;142 0;161 0;168 0;181 0;162 0; ...
     166 0;150 0;205 0;163 0;166 0;176 0];

Basic ROC analysis with full threshold grid:

ROCout = roc(x);

ROC analysis with prevalence equal to 30 percent:

ROCout_prev = roc(x, 0, 0.05, 1, 1, 0.30);

The function prints a summary of the AUC, its confidence interval, and a classification of the test quality (Perfect, Excellent, Good, Fair, Poor, Fail, or less than chance). When plotting is enabled, the ROC curve and cut-off analysis plots are displayed.

---

## 🧠 Method

1. Threshold selection  
   Test values are sorted and either all unique positive values (when THRESHOLD = 0) or a set of quantile-based values (when THRESHOLD > 2) are used as candidate cut-offs.

2. Confusion matrix at each threshold  
   For each candidate cut-off, the function computes:
   - TP, FP, FN, TN using cumulative sums over sorted labels
   - Sensitivity = TP / (TP + FN)
   - Specificity = TN / (TN + FP)
   - PLR  Sensitivity divided by (1 minus Specificity)
   - NLR  (1 minus Sensitivity) divided by Specificity

   When PREVALENCE is given, prior odds are used to derive PPV and NPV.

3. Efficiency or Matthews-like index  
   An efficiency measure is computed either:
   - from the confusion matrix alone (when prevalence is unknown), or
   - from Youden’s J and predictive values (when prevalence is known).

4. ROC curve construction  
   ROC points are defined as
   - False Positive Rate = 1 minus Specificity
   - True Positive Rate  = Sensitivity

   The ROC curve is optionally completed to pass through (0,0) and (1,1) depending on the observed trend.

5. AUC estimation via Mann–Whitney–Wilcoxon  
   AUC is computed from the Mann–Whitney–Wilcoxon U statistic (mwwtest), which is equivalent to the probability that a randomly selected diseased subject has a higher test value than a randomly selected healthy subject.

6. Standard error, confidence interval, and z-test  
   Using standard large-sample theory, the function computes:
   - standard error of AUC
   - confidence interval for AUC at level 1 minus ALPHA
   - standardized AUC and p-value for H0: AUC = 0.5 versus H1: AUC > 0.5.

7. Cut-off optimization and curve fitting  
   Several cut-off criteria are evaluated:
   - maximum Sensitivity
   - maximum Specificity
   - maximum Efficiency
   - maximum PLR
   - minimum NLR
   - cost-effective cut-off with Sensitivity equal to Specificity

   To obtain smooth Sensitivity and Specificity curves over the cut-off axis, a nonlinear model of the form

   1 − 1 / (1 + (x / C)^B)^E

   is fitted to Se and Sp after shifting the x-axis so that all cut-offs are positive. A smoothing spline is fitted to the efficiency values. The final plot is presented back in the original cut-off scale.

---

## 📦 Requirements

- MATLAB (tested on recent releases)
- Core functions:
  - inputParser, validateattributes, sortrows, quantile, cumsum
  - polyfit, erfc, erfcinv, realsqrt
  - table, array2table
- Dependency:
  - mwwtest.m (Mann–Whitney–Wilcoxon test) from the repository:
    https://github.com/dnafinder/mwwtest  
    This file must be on the MATLAB path before calling roc.

- Toolboxes:
  - Curve Fitting Toolbox (for fit, fittype, prepareCurveData, smoothing splines).

---

## 📚 References

- Cardillo G. (2008). ROC curve: compute a Receiver Operating Characteristics curve. MATLAB Central File Exchange.
- Hanley JA, McNeil BJ. (1982). The meaning and use of the area under a ROC curve. Radiology, 143(1), 29–36.
- Metz CE. (1978). Basic principles of ROC analysis. Seminars in Nuclear Medicine, 8(4), 283–298.

---

## 🧾 Citation

If you use this function in scientific work, you may cite:

Cardillo G. (2008). ROC curve: compute a Receiver Operating Characteristics curve. MATLAB Central File Exchange.

You may also acknowledge the GitHub version:

Cardillo G. (2025). ROC: Receiver Operating Characteristic curve analysis. GitHub repository dnafinder/roc.

---

## 👤 Author and Versioning

- Author: Giuseppe Cardillo
- Email: giuseppe.cardillo.75@gmail.com
- GitHub: https://github.com/dnafinder/roc

Version history:

- 1.0.0 (2008)  Initial release on MATLAB Central File Exchange.
- 2.0.0 (2025-11-26)  Refactored input handling; added prevalence argument; introduced cumulative computation for TP, FP, FN, TN; added dependency check on mwwtest; improved documentation, cut-off fitting and GitHub-ready formatting.
