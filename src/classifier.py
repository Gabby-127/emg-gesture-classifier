
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score)


class GestureClassifier:

    def __init__(self):
        self.models = {
            'SVM (RBF)': Pipeline([
                ('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', C=10, gamma='scale',
                                                          decision_function_shape='ovr', random_state=42))]),
            'Random Forest': Pipeline([
                ('scaler', StandardScaler()),

                ('clf', RandomForestClassifier(
                    n_estimators=100, min_samples_leaf=2,
                    random_state=42, n_jobs=-1))
            ]),
        }

        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.results = {}

    def cross_validate(self, X: np.ndarray, y: np.ndarray):
        print('=' * 55)
        print(f'5-Fold Cross-Validation   n_samples={len(y):,}')
        print('=' * 55)
        for name, model in self.models.items():
            acc = cross_val_score(model, X, y, cv=self.cv,
                                scoring='accuracy', n_jobs=-1)
            f1 = cross_val_score(model, X, y, cv=self.cv,
                                scoring='f1_weighted', n_jobs=-1)
            self.results[name] = {
                'cv_acc_mean': acc.mean(), 'cv_acc_std': acc.std(),
                'cv_f1_mean':  f1.mean(),  'cv_f1_std':  f1.std(),
            }
            print(f'\n{name}')
            print(f'  Accuracy: {acc.mean():.3f} +/- {acc.std():.3f}')
            print(f'  F1 (wtd): {f1.mean():.3f} +/- {f1.std():.3f}')
        return self.results

    def fit_and_evaluate(self, X_train, X_test, y_train, y_test):

        print('\n' + '=' * 55)
        print('Final Evaluation on Held-Out Test Set')
        print('=' * 55)

        gesture_names = [f'Gesture {i}' for i in range(1, 7)]

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            print(f'\n--- {name} ---')
            print(classification_report(y_test, y_pred,
                  target_names=gesture_names, digits=3))

            self.results[name]['confusion_matrix'] = confusion_matrix(
                y_test, y_pred)
            self.results[name]['y_test'] = y_test
            self.results[name]['y_pred'] = y_pred
            self.results[name]['model'] = model

        return self.results

    def get_feature_importance(self):
        rf_model = self.models['Random Forest'].named_steps['clf']
        importances = rf_model.feature_importances_
        return importances
