# Подборка вин

Проверено с Python 3.11.18. Из папки `wine_app`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

`model.joblib`, `config.json` и `test.csv` создаются ноутбуком `HW3_MMDA.ipynb`. Модель в приложении не обучается. Общий порог в боковой панели применяется во всех вкладках.

`st.cache_resource` сохраняет один объект модели для всех сессий. `st.cache_data` сохраняет таблицу и вероятности, возвращая их копии. Без декоратора `joblib.load` заново читал бы модель с диска при каждом изменении виджетов.
