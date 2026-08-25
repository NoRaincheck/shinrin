# Categorical-awareness ablation

Accuracy on held-out data. `auto` detects integer-coded
categorical columns and applies CatBoost-style smoothed target-
statistic encoding (Mondrian) or target-statistic threshold axes
(SPOT/SPOTSET); `onehot` is the XGBoost-style indicator baseline;
`none` treats every column numerically (historical behaviour).

## pure-categorical

| model | mode | accuracy | derived columns | seconds |
|---|---|---|---|---|
| mondrian | auto | 1.0000 | 4 | 0.0 |
| mondrian | none | 0.9200 | 4 | 0.1 |
| spot | auto | 0.9560 | 4 | 0.7 |
| spot | none | 0.6320 | 4 | 1.1 |
| spot | onehot | 0.6320 | 4 | 1.1 |
| spotset | auto | 0.9560 | 2 | 0.2 |
| spotset | none | 0.6320 | 9 | 0.3 |
| spotset | onehot | 0.6320 | 9 | 0.3 |

## mixed

| model | mode | accuracy | derived columns | seconds |
|---|---|---|---|---|
| mondrian | auto | 0.9240 | 6 | 0.1 |
| mondrian | none | 0.8680 | 6 | 0.1 |
| spot | auto | 0.9170 | 6 | 35.4 |
| spot | none | 0.7820 | 6 | 38.1 |
| spot | onehot | 0.7820 | 6 | 38.1 |
| spotset | auto | 0.9100 | 28 | 0.5 |
| spotset | none | 0.7660 | 34 | 1.1 |
| spotset | onehot | 0.7660 | 34 | 1.1 |

## compas

| model | mode | accuracy | derived columns | seconds |
|---|---|---|---|---|
| mondrian | auto | 0.6486 | 27 | 0.2 |
| mondrian | none | 0.6447 | 27 | 0.2 |
| spot | auto | 0.6325 | 27 | 1.1 |
| spot | none | 0.6325 | 27 | 1.1 |
| spot | onehot | 0.6325 | 27 | 1.1 |
| spotset | auto | 0.6480 | 15 | 0.3 |
| spotset | none | 0.6480 | 15 | 0.3 |
| spotset | onehot | 0.6480 | 15 | 0.3 |
