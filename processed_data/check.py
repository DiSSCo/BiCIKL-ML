import pandas as pd



print()


classes = pd.read_csv("SMOTE_Model_classes.csv")
class_strict = pd.read_csv("SMOTE_Model_Strict_classes.csv")
class_test = pd.read_csv("Investigating_classes.csv")

print(classes.value_counts())
#print(class_strict.shape)
print(class_test.value_counts())

breakpoint()

print(classes.value_counts())
print(class_strict.value_counts())

X = pd.read_csv("SMOTE_Model_int_mapping.csv")
X_strict = pd.read_csv("SMOTE_Model_Strict_int_mapping.csv")

print(X.shape)

print(classes.shape)

print(X_strict.shape)

print(class_strict.shape)

true_x = X.loc[classes["class"] == 1]
#true_x.to_csv("trueX.csv", index=False)

true_x_strict = X_strict.loc[class_strict["class"] == 1]
true_x_strict.to_csv("trueX_strict.csv", index=False)

print(X_strict.shape)
X_strict.drop_duplicates(inplace=True)
print(X_strict.shape)

print(X.shape)
X.drop_duplicates(inplace=True)
print(X.shape)
