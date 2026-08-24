"""Tree classifier representing one member of a SPOTSET Rashomon set.

Adapted from treeFARMS' ``treefarms/model/tree_classifier.py`` (BSD-3-Clause,
https://github.com/ubc-systopia/treeFARMS), renamed SPOTSET in this project.
Deviations: replaced invalid ``raise "<string>"`` statements with real
exceptions and trimmed unused imports.
"""

from json import JSONEncoder, dumps

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


# Supporting Override for Converting Numpy Types into Python Values
class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)


class TreeClassifier:
    """
    Unified representation of a tree classifier in Python.

    This class accepts a dictionary representation of a tree classifier and
    decodes it into an interactive object. Features are referenced by their
    integer column index in the (binarized) feature matrix used for fitting;
    a node tests ``sample[feature] == 1``.
    """

    def __init__(self, source, encoder=None, X=None, y=None):
        self.source = (
            source  # The classifier stored in a recursive dictionary structure
        )
        self.encoder = (
            encoder  # Optional encoder / decoder unit to run before / after prediction
        )
        if X is not None and y is not None:
            self.__initialize_training_loss__(X, y)

    def __initialize_training_loss__(self, X, y):
        """
        Compares every prediction y_hat against the labels y, then incorporates
        the misprediction into the stored loss values. This is used when parsing
        models from an algorithm that doesn't provide the training loss in the
        output.
        """
        for node in self.__all_leaves__():
            node["loss"] = 0.0
        n = X.shape[0]
        for i in range(n):
            node = self.__find_leaf__(X[i, :])
            label = y[i, -1]
            weight = 1 / n
            if node["prediction"] != label:
                node["loss"] += weight

    def __find_leaf__(self, sample):
        """Returns the leaf by which this sample would be classified."""
        nodes = [self.source]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                return node
            else:
                value = sample[node["feature"]]
                if value == 1:
                    nodes.append(node["true"])
                else:
                    nodes.append(node["false"])

    def __all_leaves__(self):
        """Returns a list of all leaves in this model."""
        nodes = [self.source]
        leaf_list = []
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                leaf_list.append(node)
            else:
                nodes.append(node["true"])
                nodes.append(node["false"])
        return leaf_list

    def classify(self, sample):
        """Returns ``(prediction, -1)`` for a single 1-by-m sample row."""
        node = self.__find_leaf__(sample)
        return node["prediction"], -1

    def predict(self, X):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            a matrix where each row is a sample to be predicted and each
            column is a feature to be used for prediction

        Returns
        ---
        array-like, shape = [n_samples] : the prediction associated with each row
        """
        if self.encoder is not None:
            import pandas as pd

            X = pd.DataFrame(self.encoder.encode(X[:, :]), columns=self.encoder.headers)

        predictions = []
        X = np.asanyarray(X)
        for i in range(X.shape[0]):
            prediction, _ = self.classify(X[i, :])
            predictions.append(prediction)
        return np.array(predictions)

    def error(self, X, y, weight=None):
        """Returns the inaccuracy produced by applying this model over the given dataset."""
        return 1 - self.score(X, y, weight=weight)

    def score(self, X, y, weight=None):
        """
        Returns
        ---
        real number : the accuracy produced by applying this model over the
        given dataset, with optionals for weighted accuracy
        """
        y_hat = self.predict(X)
        if weight == "balanced":
            return balanced_accuracy_score(y, y_hat)
        else:
            return accuracy_score(y, y_hat, normalize=True, sample_weight=weight)

    def confusion(self, X, y, weight=None):
        """Returns the confusion matrix of all classes present in the dataset."""
        return confusion_matrix(y, self.predict(X), sample_weight=weight)

    def __len__(self):
        """The number of terminal nodes present in this tree."""
        return self.leaves()

    def leaves(self):
        """The number of terminal nodes present in this tree."""
        leaves_counter = 0
        nodes = [self.source]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                leaves_counter += 1
            else:
                nodes.append(node["true"])
                nodes.append(node["false"])
        return leaves_counter

    def nodes(self):
        """The number of nodes present in this tree."""
        nodes_counter = 0
        nodes = [self.source]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                nodes_counter += 1
            else:
                nodes_counter += 1
                nodes.append(node["true"])
                nodes.append(node["false"])
        return nodes_counter

    def features(self):
        """A set of integers each describing a feature used by this model."""
        feature_set = set()
        nodes = [self.source]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                continue
            else:
                feature_set.add(node["feature"])
                nodes.append(node["true"])
                nodes.append(node["false"])
        return feature_set

    def maximum_depth(self, node=None):
        """The length of the longest decision path in this tree. A single-node tree will return 1."""
        if node is None:
            node = self.source
        if "prediction" in node:
            return 1
        else:
            return 1 + max(
                self.maximum_depth(node["true"]), self.maximum_depth(node["false"])
            )

    def __str__(self):
        """Pseudocode representing the logic of this classifier."""
        cases = []
        for group in self.__groups__():
            predicates = []
            for name in sorted(group["rules"].keys()):
                domain = group["rules"][name]
                if domain["type"] == "Categorical":
                    if len(domain["positive"]) > 0:
                        predicates.append(
                            "{} = {}".format(name, next(iter(domain["positive"])))
                        )
                    elif len(domain["negative"]) > 0:
                        if len(domain["negative"]) > 1:
                            predicates.append(
                                "{} not in {{ {} }}".format(
                                    name, ", ".join(str(v) for v in domain["negative"])
                                )
                            )
                        else:
                            predicates.append(
                                "{} != {}".format(
                                    name, str(next(iter(domain["negative"])))
                                )
                            )
                    else:
                        raise ValueError("Invalid rule: no positive or negative domain")
                elif domain["type"] == "Numerical":
                    predicate = name
                    if domain["min"] != -float("INF"):
                        predicate = "{} <= ".format(domain["min"]) + predicate
                    if domain["max"] != float("INF"):
                        predicate = predicate + " < {}".format(domain["max"])
                    predicates.append(predicate)

            if len(predicates) == 0:
                condition = "if true then:"
            else:
                condition = "if {} then:".format(" and ".join(predicates))
            outcomes = [
                "    predicted {}: {}".format(group["name"], group["prediction"])
            ]
            result = "\n".join(outcomes)
            cases.append(f"{condition}\n{result}")
        return "\n\nelse ".join(cases)

    def __repr__(self):
        """The recursive dictionary used to represent the model."""
        return dumps(self.source, indent=2, cls=NumpyEncoder)

    def latex(self, node=None):
        """
        Note: doesn't work well for label headers containing underscores
        (reserved character in LaTeX).

        Returns a LaTeX string representing the model.
        """
        if node is None:
            node = self.source
        if "prediction" in node:
            name = node.get("name", "feature_{}".format(node.get("feature")))
            return "[ ${}$ [ ${}$ ] ]".format(name, node["prediction"])
        else:
            if "name" in node:
                if "=" in node["name"]:
                    name = "{}".format(node["name"])
                else:
                    name = "{} {} {}".format(
                        node["name"], node["relation"], node["reference"]
                    )
            else:
                name = "feature_{} {} {}".format(
                    node["feature"], node["relation"], node["reference"]
                )
            return (
                "[ ${}$ {} {} ]".format(
                    name, self.latex(node["true"]), self.latex(node["false"])
                )
                .replace("==", " \\eq ")
                .replace(">=", " \\ge ")
                .replace("<=", " \\le ")
            )

    def json(self):
        """A JSON string representing the model."""
        return dumps(self.source, cls=NumpyEncoder)

    def __groups__(self, node=None):
        """Object representation of each leaf for conversion to a case in an if-then-else statement."""
        if node is None:
            node = self.source
        if "prediction" in node:
            node["rules"] = {}
            groups = [node]
            return groups
        else:
            if "name" in node:
                name = node["name"]
            else:
                name = "feature_{}".format(node["feature"])
            reference = node["reference"]
            groups = []
            for condition_result in ["true", "false"]:
                subtree = node[condition_result]
                for group in self.__groups__(subtree):
                    # For each group, add the corresponding rule
                    rules = group["rules"]
                    if name not in rules:
                        rules[name] = {}
                    rule = rules[name]
                    if node["relation"] == "==":
                        rule["type"] = "Categorical"
                        if "positive" not in rule:
                            rule["positive"] = set()
                        if "negative" not in rule:
                            rule["negative"] = set()
                        if condition_result == "true":
                            rule["positive"].add(reference)
                        elif condition_result == "false":
                            rule["negative"].add(reference)
                        else:
                            raise ValueError(f"Malformatted source: {node}")
                    elif node["relation"] == ">=":
                        rule["type"] = "Numerical"
                        if "max" not in rule:
                            rule["max"] = float("INF")
                        if "min" not in rule:
                            rule["min"] = -float("INF")
                        if condition_result == "true":
                            rule["min"] = max(reference, rule["min"])
                        elif condition_result == "false":
                            rule["max"] = min(reference, rule["max"])
                        else:
                            raise ValueError(f"Malformatted source: {node}")
                    else:
                        raise ValueError(
                            "Unsupported relational operator {}".format(
                                node["relation"]
                            )
                        )

                    # Add the modified group to the group list
                    groups.append(group)
            return groups
