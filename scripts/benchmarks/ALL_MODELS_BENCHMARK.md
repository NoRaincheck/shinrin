## Regression

*Fit time (seconds).*

| Dataset | MondrianTree | MondrianForest | RandomForest | ExtraTrees | RF-Quantile | ET-Quantile |
|---|---|---|---|---|---|---|
| diabetes | 0.000446 | 0.00686 | 0.112 | 0.0743 | 0.12 | 0.0999 |
| friedman1-2k | 0.00117 | 0.0224 | 0.479 | 0.29 | 0.687 | 0.554 |
| friedman1-10k | 0.00462 | 0.0892 | 3.14 | 1.5 | 8.35 | 7.23 |
| california-10k | 0.00361 | 0.0667 | 2.58 | 1.17 | 7.73 | 6.91 |
| make-regression-5k | 0.00322 | 0.0624 | 3.05 | 1.4 | 3.47 | 2.39 |

*Predict: ms per full test-set call / ms per 1k samples.*

| Dataset | MondrianTree | MondrianForest | RandomForest | ExtraTrees | RF-Quantile | ET-Quantile |
|---|---|---|---|---|---|---|
| diabetes | 0.0782 / 0.879 | 1.36 / 15.3 | 4.03 / 45.3 | 4.31 / 48.5 | 2.1 / 23.6 | 2.18 / 24.4 |
| friedman1-2k | 0.214 / 0.536 | 4.63 / 11.6 | 8.02 / 20 | 8.87 / 22.2 | 4.21 / 10.5 | 4.25 / 10.6 |
| friedman1-10k | 1.01 / 0.507 | 20.5 / 10.3 | 36.4 / 18.2 | 44.6 / 22.3 | 19.1 / 9.54 | 18.9 / 9.45 |
| california-10k | 0.803 / 0.401 | 15.4 / 7.69 | 35.7 / 17.9 | 43.6 / 21.8 | 18.2 / 9.1 | 18.2 / 9.1 |
| make-regression-5k | 0.712 / 0.712 | 14.4 / 14.4 | 17.1 / 17.1 | 21 / 21 | 8.73 / 8.73 | 8.94 / 8.94 |

*R^2 on the held-out test set.*

| Dataset | MondrianTree | MondrianForest | RandomForest | ExtraTrees | RF-Quantile | ET-Quantile |
|---|---|---|---|---|---|---|
| diabetes | -0.2540 | 0.2996 | 0.2687 | 0.2372 | 0.2648 | 0.3072 |
| friedman1-2k | -0.0170 | 0.6604 | 0.8502 | 0.8655 | 0.8510 | 0.8491 |
| friedman1-10k | 0.3615 | 0.6951 | 0.9089 | 0.9159 | 0.9074 | 0.9036 |
| california-10k | -0.0526 | 0.1940 | 0.7871 | 0.7878 | 0.7844 | 0.7747 |
| make-regression-5k | -0.0098 | 0.2703 | 0.7489 | 0.7747 | 0.7481 | 0.7346 |

## Classification

*Fit time (seconds).*

| Dataset | MondrianTree | MondrianForest | GOSDT | CORELS |
|---|---|---|---|---|
| breast-cancer | 0.000466 | 0.00941 | 0.00158 | 0.0122 |
| wine | 0.00028 | 0.0059 | - | - |
| digits | 0.00225 | 0.0442 | - | - |
| synthetic-binary-5k | 0.00256 | 0.0518 | 13 | 0.0438 |
| synthetic-binary-20k | 0.0197 | 0.394 | - | - |
| synthetic-multiclass-5k | 0.00281 | 0.0549 | - | - |

*Predict: ms per full test-set call / ms per 1k samples.*

| Dataset | MondrianTree | MondrianForest | GOSDT | CORELS |
|---|---|---|---|---|
| breast-cancer | 0.109 / 0.954 | 1.27 / 11.2 | 0.0772 / 0.677 | 0.0173 / 0.152 |
| wine | 0.0612 / 1.36 | 0.454 / 10.1 | - | - |
| digits | 0.587 / 1.63 | 10.3 / 28.5 | - | - |
| synthetic-binary-5k | 0.639 / 0.639 | 11.4 / 11.4 | 0.381 / 0.381 | 0.0506 / 0.0506 |
| synthetic-binary-20k | 4.57 / 1.14 | 88.6 / 22.1 | - | - |
| synthetic-multiclass-5k | 0.75 / 0.75 | 13.6 / 13.6 | - | - |

*Accuracy on the held-out test set.*

| Dataset | MondrianTree | MondrianForest | GOSDT | CORELS |
|---|---|---|---|---|
| breast-cancer | 0.9123 | 0.9123 | 0.9035 | 0.9386 |
| wine | 0.6222 | 0.7778 | - | - |
| digits | 0.7028 | 0.9500 | - | - |
| synthetic-binary-5k | 0.7300 | 0.8640 | 0.7330 | 0.7590 |
| synthetic-binary-20k | 0.6322 | 0.8117 | - | - |
| synthetic-multiclass-5k | 0.4000 | 0.6500 | - | - |

