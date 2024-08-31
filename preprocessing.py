import pandas as pd
from sklearn import preprocessing, model_selection, tree, metrics
import lab1

dataset = load_CSV_dataset()

print(
    f"{dataset.shape[0]} records read from {DATASET_FILE}\n{dataset.shape[1]} attributes found"
)
print(dataset.head(5))
print(dataset.cid.value_counts())

train, test = model_selection.train_test_split(dataset, test_size=0.2, random_state=42)

print(f"{train.shape[0]} samples for training, {test.shape[0]} samples for testing")
# train.head(10)

input_cols = dataset.columns[2:24] # No nos interesa el pidnum dado que unicamente identifica al paciente
my_tree = tree.DecisionTreeClassifier()
my_tree = my_tree.fit(train[input_cols], train.cid)

# predecimos los ejemplos del conjunto de test
test_pred = my_tree.predict(test[input_cols])

# y los comparamos contra los "reales"
print(f"\nAcierto: {metrics.accuracy_score(test.cid, test_pred)}")

# veamos precisión, recuperación...
print(metrics.classification_report(test.cid, test_pred))

