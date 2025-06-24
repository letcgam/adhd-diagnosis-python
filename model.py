import os
import json
import joblib
import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE


def get_train_test_split(dfs_to_merge=[]):
    path = 'tg-adhd-diagnosis/adhd-diagnosis-data/'

    # TABELA FEATURES
    def get_features_df():
        ft_df = pd.read_csv(path + 'features.csv', sep=';')

        # Identificando as colunas com apenas valores nulos para tratar
        null_sum = ft_df.isnull().sum()
        cols_with_null = null_sum[null_sum > 0]

        cols_to_drop = ft_df.columns[ft_df.isnull().sum() == len(ft_df)]
        ft_df = ft_df.drop(cols_to_drop, axis=1)

        null_count = ft_df.notnull().sum()
        colunas_nulas = null_count[null_count == 0].index

        ft_df = ft_df.drop(colunas_nulas, axis=1)

        return ft_df

    # TABELA CONNERSCONTINUOUSPERFORMANCETEST
    def get_cpt_2_df():
        # Conners Continuous Performance Test
        cpt_2 = pd.read_csv(path + 'CPT_II_ConnersContinuousPerformanceTest.csv', sep=';')
        raw_test_columns = [col for col in cpt_2.columns if (col.startswith('Trial') or col.startswith('Response'))]
        cpt_2_filtered = cpt_2.drop(raw_test_columns, axis=1)

        return cpt_2_filtered

    # TABELA HRV_FEATURES
    def get_hrv_df():
        hrv_df = pd.read_csv(path + 'hrv_features.csv')

        # Identificando as colunas com apenas valores nulos para tratar
        null_sum = hrv_df.isnull().sum()
        cols_with_null = null_sum[null_sum > 0]

        cols_to_drop = hrv_df.columns[hrv_df.isnull().sum() == len(hrv_df)]
        hrv_df = hrv_df.drop(cols_to_drop, axis=1)

        return hrv_df

    # TABELA PATIENT_INFO
    def get_patient_info_df():
        patient_info = pd.read_csv(path + 'patient_info.csv', sep=';')
        patient_info = patient_info[patient_info['filter_$'] == 1]

        # Função para converter HH:MM:SS para segundos
        def time_to_seconds(time_str):
            if pd.isna(time_str):
                return np.nan
            h, m, s = map(int, time_str.split(':'))
            return h * 3600 + m * 60 + s

        patient_info['ACC_TIME'] = patient_info['ACC_TIME'].apply(time_to_seconds)

        moda = int(patient_info['ACC_TIME'].mode()[0])
        patient_info.fillna({'ACC_TIME': moda}, inplace=True)

        patient_info['HRV_TIME'] = patient_info['HRV_TIME'].apply(time_to_seconds)

        moda = int(patient_info['HRV_TIME'].mode()[0])
        patient_info.fillna({'HRV_TIME': moda}, inplace=True)

        # Tratamento para numéricos com mediana
        for col in ['ACC_DAYS', 'HRV_HOURS', 'WURS', 'ASRS', 'MADRS', 'HADS_A', 'HADS_D']:
            patient_info.fillna({col: patient_info[col].median()}, inplace=True)

        # Tratamento para classes 1 e nan (para ficar 1 e 0)
        medication_cols = ['MED_Antidepr', 'MED_Moodstab', 'MED_Antipsych', 'MED_Anxiety_Benzo', 'MED_Sleep', 'MED_Analgesics_Opioids', 'MED_Stimulants']
        for col in medication_cols:
            patient_info.fillna({col: 0}, inplace=True)
            patient_info[col] = patient_info[col].astype(int)
        
        return patient_info

    # Preparação dos dados de treino e teste

    def get_merged_df():
        get_df = {
            'ft_df': get_features_df,
            'patient_info': get_patient_info_df,
            'cpt_2_filtered': get_cpt_2_df,
            'hrv_df': get_hrv_df,
        }

        df = get_df[dfs_to_merge[0]]().copy()

        for df_name in dfs_to_merge[1:]:
            df_to_merge = get_df[df_name]()

            df = pd.merge(df, df_to_merge, on='ID')
        
        if ('patient_info' not in dfs_to_merge):
            patient_info_df = get_patient_info_df()
            patient_info_df = patient_info_df[['filter_$', 'ADHD', 'ADD', 'ID']]
            
            df = pd.merge(df, patient_info_df, on='ID')
            df = df[df['filter_$'] == 1]

        return df
    
    df = get_merged_df()

    X = df.drop(['ID', 'filter_$', 'ADHD', 'ADD'], axis=1)
    y = df['ADHD']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=4
    )

    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'df_shape': df.shape
    }


def get_best_model_features(tts, merged_dfs):
    X_train, X_test, y_train, y_test, df_shape = tts.values()

    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=6),
        param_grid={
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        cv=10,
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X_train, y_train)

    best_model_full_features = grid.best_estimator_

    predictions_selected_features_rf = best_model_full_features.predict(X_test)
    
    auc_roc_selected_features_rf = best_model_full_features.predict_proba(X_test)[:, 1]
    report_dict_rf = classification_report(y_test, predictions_selected_features_rf, output_dict=True)
    accuracy_rf = report_dict_rf['accuracy']
    auc_roc_score_rf = roc_auc_score(y_test, auc_roc_selected_features_rf) # Calcula o AUC-ROC score

    # with open("tg-adhd-diagnosis/saida.txt", "a") as f:
    #     f.write("DataFrames utilizados: " + ", ".join(merged_dfs))
    #     f.write("Avaliação do Modelo com todas as features\n")
    #     f.write(f"\nAcurácia do RF com todas as features: {accuracy_rf:.4f}")
    #     f.write(f"\nAUC-ROC do RF com todas as features: {auc_roc_score_rf:.4f}")
    #     f.write("\nRelatório de Classificação do RF:\n")
    #     f.write(str(classification_report(y_test, predictions_selected_features_rf)))

    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model_full_features.feature_importances_
    })

    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    selected_features = feature_importance_df['feature'].tolist()
    feature_importances = feature_importance_df['importance'].tolist()

    best_model_features = {
        'selected_features': selected_features,
        'feature_importances': feature_importances,
        'n_estimators': best_model_full_features.n_estimators,
        'max_depth': best_model_full_features.max_depth,
        'min_samples_split': best_model_full_features.min_samples_split,
        'min_samples_leaf': best_model_full_features.min_samples_leaf
    }
    
    return best_model_features


def get_RF_model(tts, best_model_features, number_of_features=50):
    X_train, X_test, y_train, y_test, df_shape = tts.values()

    selected_features, feature_importances, n_estimators, max_depth, min_samples_split, min_samples_leaf = best_model_features.values()
    
    selected_features = selected_features[:number_of_features]

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    final_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    final_model.fit(X_train, y_train)

    predictions_selected_features_rf = final_model.predict(X_test)

    auc_roc_selected_features_rf = final_model.predict_proba(X_test)[:, 1]

    report_dict_rf = classification_report(y_test, predictions_selected_features_rf, output_dict=True)
    accuracy_rf = report_dict_rf['accuracy']
    auc_roc_score_rf = roc_auc_score(y_test, auc_roc_selected_features_rf)
        

    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_
    })

    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    selected_features = feature_importance_df['feature'].tolist()
    feature_importances = feature_importance_df['importance'].tolist()
    cm = confusion_matrix(y_test, predictions_selected_features_rf)
    
    best_model_features = {
        'selected_features': selected_features,
        'feature_importances': feature_importances,
        'n_estimators': final_model.n_estimators,
        'max_depth': final_model.max_depth,
        'min_samples_split': final_model.min_samples_split,
        'min_samples_leaf': final_model.min_samples_leaf
    }

    with open("tg-adhd-diagnosis/saida.txt", "a") as f:
        f.write(f"\n\n\nAvaliação do Modelo com {number_of_features} primeiras features\n")
        f.write(f"\nAcurácia do RF com features selecionadas: {accuracy_rf:.4f}")
        f.write(f"\nAUC-ROC do RF com features selecionadas: {auc_roc_score_rf:.4f}")
        f.write("\nMatriz de Confusão do RF:\n" + str(cm))
        f.write("\nRelatório de Classificação do RF:\n" + str(classification_report(y_test, predictions_selected_features_rf)))

    # Salvar o modelo
    joblib.dump(final_model, 'tg-adhd-diagnosis/models/modelo_RF.joblib')
    print("Modelo RF salvo")
    
    return best_model_features


dfs_to_merge = [
    'ft_df',
    'patient_info',
    'cpt_2_filtered',
    'hrv_df'
]

tts = get_train_test_split(dfs_to_merge)

with open("tg-adhd-diagnosis/saida.txt", "w") as f:
    f.write(f"\nModelo com as tabelas: " + ", ".join(dfs_to_merge) + "\n")
    f.write(f"\nShape do DataFrame: {tts['df_shape']}\n\n\n")

best_model_features = get_best_model_features(tts, dfs_to_merge)
final_model_features = get_RF_model(tts, best_model_features, 30)