# Predicting Customer Satisfaction in Automotive After-Sales Service

This repository contains the dataset and code used in the article
*Predicting Customer Satisfaction in Automotive After-Sales Service: A Large-Scale Comparison of Regression and Machine Learning Models*.

## Dataset

| Item | Description |
|---|---|
| File | `dataset.xlsx` (one sheet named `dataset`) |
| Size | 39,454 rows (one row = one customer survey) and 26 columns |
| Source | After-sales service network of an Iranian car manufacturer |
| Period | Farvardin to Esfand 1404 (Iranian calendar), i.e. March 2025 to March 2026 |
| Privacy | Fully anonymised. No names, contact data or other personal identifiers are included |
| Target variable | `q21` (overall satisfaction, 1 to 5) |

## Variables

Column names are kept as in the original file. The columns appear in the file in this order:
`q1, q2, q3, q4, q5, q6, q7, q10, q12, q13, q14, q15, q16, q18, q19, q21, q17, q8, q9, nps, q20, q22, q23, q24, q25, q26`.

### Service ratings (1 = very low, 5 = very high)

| Code | Meaning | Missing |
|---|---|---|
| q1 | Reception and check-in process | 0% |
| q2 | Physical environment of the service centre | 0% |
| q3 | Waiting time | 0% |
| q4 | Staff conduct | 0% |
| q5 | Repair quality | 0% |
| q6 | Cost transparency | 0% |
| q7 | Digital channels | 0% |
| q10 | Accessibility (asked only if `q9` = 1) | 29.6% |
| q12 | Interaction quality with the service adviser | 0% |
| q13 | Outcome of the work | 0% |
| q14 | Vehicle cleanliness | 0% |
| q15 | On-time delivery | 0% |
| q16 | Technical explanation given to the customer | 0% |
| q18 | Follow-up after the service (asked only if `q17` = 2) | 36.2% |
| q19 | Price value (asked only if `q17` = 2) | 36.0% |

### Target

| Code | Meaning | Values |
|---|---|---|
| q21 | Overall satisfaction (dependent variable) | 1 = very low, 2 = low, 3 = medium, 4 = high, 5 = very high |

### Binary process indicators

| Code | Meaning | Values |
|---|---|---|
| q8 | Service-process indicator | 1 or 2 |
| q9 | Routing question for `q10` | 1 or 2 (if 2, `q10` is not asked) |
| q17 | Routing question for `q18` and `q19` | 1 or 2 (if 1, `q18` and `q19` are not asked) |

### Recommendation measures

| Code | Meaning | Values |
|---|---|---|
| q20 | Likelihood to recommend | 0 to 10 |
| nps | Net promoter class | 1, 2 or 3 (3 is linked to the lowest satisfaction) |

### Respondent characteristics

| Code | Meaning | Values |
|---|---|---|
| q22 | Respondent characteristic (binary) | 0 or 1 |
| q23 | Age band | 4 bands (band 2 = 30 to 40 years) |
| q24 | Respondent characteristic | 5 categories |
| q25 | Respondent characteristic | 8 categories |
| q26 | Gender | 1 or 2 (about 68% / 32%) |

## Important notes

- **Empty cells are not errors.** `q10`, `q18` and `q19` are empty by questionnaire design (see the routing rules above), so an empty cell means "not asked". Please do not delete these rows.
- **The target is ordered.** `q21` is an ordinal scale, so ordinal metrics (for example quadratic weighted kappa) are recommended in addition to accuracy.
- **Identical rows.** 533 rows (1.4%) are identical to another row. They were kept in the analysis because anonymous answers on short scales can be the same by chance.
- **Same-survey data.** All variables come from the same questionnaire, so models built on this data classify survey answers. They are not early-warning tools.

## Quick start

```python
import pandas as pd

df = pd.read_excel("dataset.xlsx")
print(df.shape)                        # (39454, 26)
print(df["q21"].value_counts().sort_index())
print(df.isna().mean().round(3))       # missing only in q10, q18, q19
```

## Repository

https://github.com/khoji-code/Predicting-Customer-Satisfaction
