import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


st.set_page_config(page_title='Подборка вин', layout='wide')
st.title('Подборка хороших вин')

model = load_model()
test = load_test()
features = list(model.feature_names_in_)
X_test = test[features]
y_test = test['target']
proba = test_proba(model, X_test)
config = json.loads((APP_DIR / 'config.json').read_text())

threshold = st.sidebar.slider(
    'Порог хорошего вина', 0.0, 1.0,
    value=float(config['threshold']), step=0.001, format='%.3f')

single_tab, metrics_tab, batch_tab, weights_tab = st.tabs([
    'Оценка одного вина', 'Модельные метрики',
    'Пакетное предсказание', 'Что выучила модель'])

with single_tab:
    values = {}
    columns = st.columns(2)
    for i, feature in enumerate(features):
        with columns[i % 2]:
            if feature == 'is_red':
                wine_type = st.radio(
                    'Цвет вина', ['Белое', 'Красное'],
                    index=int(X_test[feature].median()), horizontal=True)
                values[feature] = int(wine_type == 'Красное')
            else:
                values[feature] = st.number_input(
                    feature,
                    min_value=float(X_test[feature].min()),
                    max_value=float(X_test[feature].max()),
                    value=float(X_test[feature].median()), format='%.5f')

    wine = pd.DataFrame([values], columns=features)
    wine_proba = model.predict_proba(wine)[0, 1]
    st.metric('Вероятность хорошего вина', f'{wine_proba:.1%}')
    if wine_proba >= threshold:
        st.success('Добавить в подборку')
    else:
        st.info('Не добавлять в подборку')

with metrics_tab:
    prediction = (proba >= threshold).astype(int)
    precision = metrics.precision_score(y_test, prediction, zero_division=0)
    recall = metrics.recall_score(y_test, prediction, zero_division=0)
    f1 = metrics.f1_score(y_test, prediction, zero_division=0)

    columns = st.columns(3)
    columns[0].metric('Precision', f'{precision:.3f}')
    columns[1].metric('Recall', f'{recall:.3f}')
    columns[2].metric('F1', f'{f1:.3f}')

    cm = metrics.confusion_matrix(y_test, prediction, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(
        ax=ax, colorbar=False)
    ax.set_xlabel('Предсказание')
    ax.set_ylabel('Истинный класс')
    st.pyplot(fig)
    plt.close(fig)

    tn, fp, fn, tp = cm.ravel()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    metrics.RocCurveDisplay.from_predictions(y_test, proba, ax=axes[0])
    axes[0].scatter(fp / (fp + tn), recall, color='red', label='Текущий порог')
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].set_title('ROC-кривая')
    axes[0].legend()

    metrics.PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axes[1])
    axes[1].scatter(recall, precision, color='red', label='Текущий порог')
    axes[1].axhline(y_test.mean(), linestyle='--', color='gray', label='Случайная модель')
    axes[1].set_title('PR-кривая')
    axes[1].legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write(
        'Выше порог — меньше вин в подборке, обычно меньше посредственных вин, '
        'но больше пропущенных хороших. Ниже порог — подборка шире, '
        'но риск рекомендовать посредственное вино выше.')

with batch_tab:
    uploaded = st.file_uploader('CSV с признаками вин без target', type='csv')
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        missing = [feature for feature in features if feature not in batch.columns]
        if missing:
            st.error(f"В файле нет столбцов: {', '.join(missing)}")
        else:
            batch = batch.copy()
            batch['probability'] = model.predict_proba(batch[features])[:, 1]
            batch['prediction'] = (batch['probability'] >= threshold).astype(int)
            st.dataframe(batch, hide_index=True)
            st.download_button(
                'Скачать прогнозы', data=batch.to_csv(index=False).encode('utf-8'),
                file_name='predictions.csv', mime='text/csv')

with weights_tab:
    weights = pd.Series(
        model.named_steps['logisticregression'].coef_[0],
        index=features).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    weights.plot.barh(ax=ax)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Коэффициент после масштабирования')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    positive = weights[weights > 0].tail(3).sort_values(ascending=False).index
    negative = weights[weights < 0].head(3).index
    st.write(f"Сильнее всего за хорошее вино: {', '.join(positive)}.")
    st.write(f"Сильнее всего против: {', '.join(negative)}.")
