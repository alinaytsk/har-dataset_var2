# Інтеграція har-dataset зі stm32ai-modelzoo-services

Пакет, щоб навчати модель на нашому датасеті (RealWorld, позиція `upperarm`,
104 Гц, вікно 208 × 6, 5 класів) у репозиторії `stm32ai-modelzoo-services`.

Нічого з наявного коду не переписується: додаються два файли і правиться конфіг.

---

## Що тут лежить

```
integrations/modelzoo/
├── README.md                     ← цей файл
├── user_config.yaml              ← готовий конфіг замість стокового
└── human_activity_recognition/tf/
    ├── src/datasets/realworld_npz.py    ← читає наші .npz
    ├── src/models/har_cnn.py            ← модель під вікно 208 × 6
    └── wrappers/datasets/realworld_npz.py  ← реєстрація завантажувача
```

Шляхи всередині `human_activity_recognition/` збігаються зі структурою
репозиторію, тому теку можна скопіювати цілком у його корінь.

---

## Шлях A: власний скрипт навчання (найшвидший)

`train_realworld_imu.py` уже вміє все потрібне: у ньому є `--scratch`
(навчання з нуля) і `--sequence-length`. Бракує лише читання наших `.npz`
замість CSV — це шість рядків.

**1. Скопіювати один файл:**
`human_activity_recognition/tf/src/datasets/realworld_npz.py`

**2. У `train_realworld_imu.py` додати імпорт і прапорець:**

```python
from human_activity_recognition.tf.src.datasets.realworld_npz import (
    prepare_realworld_npz_arrays)
...
parser.add_argument("--npz", help="тека data/processed з har-dataset")
...
# замість виклику prepare_realworld_imu_arrays:
if args.npz:
    data = prepare_realworld_npz_arrays(
        args.npz, target_classes, args.sequence_length, normalization=True)
else:
    data = prepare_realworld_imu_arrays(
        args.csv, target_classes, args.sequence_length,
        args.validation_split, args.test_split, args.seed, normalization=True)
```

`prepare_realworld_npz_arrays` повертає рівно ті самі ключі
(`x_train`, `y_train`, `x_valid`, `y_valid`, `x_test`, `y_test`, `mean`, `std`,
`class_names`), тому решта скрипта не змінюється. Ще варто зробити `--csv`
необов'язковим, якщо передано `--npz`.

**3. Запуск:**

```bash
python human_activity_recognition/tf/src/training/train_realworld_imu.py \
  --npz ../har-dataset/data/processed \
  --classes Lying,Stationary,Walking,Running,Stairs \
  --sequence-length 208 \
  --scratch \
  --output-dir runs/har_upperarm_104hz
```

`--scratch` тут обов'язковий: перенесення ваг (`imu_transfer.py`) працює лише
коли часовий вимір не змінюється, а вихідна ST-модель має 24 відліки проти
наших 208. Деталі — у розділі «Чому без transfer learning».

---

## Шлях B: через фреймворк (`stm32ai_main.py`)

Потрібен, якщо хочеться `chain_tb`, MLflow і бенчмаркінг.

**1. Скопіювати теку `human_activity_recognition/` з цього пакета в корінь
репозиторію** (три файли, нічого не перезаписується).

**2. Зареєструвати модель.** У `human_activity_recognition/tf/src/models/__init__.py`:

```python
from .har_cnn import get_har_cnn
```

У `human_activity_recognition/tf/wrappers/models/custom_models/models.py`
додати рядок у таблицю:

```python
TF_MODEL_FNS = {
    'st_gmp': (get_gmp, {}),
    'st_ign': (get_ign, {}),
    'custom_model': (get_custom_model, {}),
    'har_cnn': (get_har_cnn, {}),        # ← додати
}
```

**3. Імпортувати обгортку завантажувача** там, де імпортується `realworld_imu`
(поруч, у тому самому місці — щоб реєстрація відбулася до виклику
`get_dataloaders`).

**4. Покласти `user_config.yaml` з цього пакета** замість
`human_activity_recognition/user_config.yaml` і виправити в ньому
`dataset.training_path` під свої шляхи.

```bash
python human_activity_recognition/stm32ai_main.py
```

---

## Які конфлікти це закриває

| Конфлікт | Було | Стало |
|---|---|---|
| Формат даних | завантажувач читає склеєний CSV і сам ріже вікна | `realworld_npz.py` бере готові вікна з `.npz` |
| Перекриття вікон | CSV-шлях ріже без перекриття | наші вікна з перекриттям 50 % |
| Спліт | випадковий `GroupShuffleSplit` | фіксований за людьми: train 1–13, val 6/10, test 12/14/15 |
| Нормалізація | рахується заново на кожному запуску | `norm_stats.json` — ті самі числа, що в прошивці |
| Класи | 4 класи WISDM | 5 наших, порядок звіряється з датасетом |
| Форма входу | `(24, 3, 1)` | `(208, 6, 1)` |
| `num_classes` | не було в конфігу → модель отримувала `None` | `dataset.num_classes: 5` |
| `gravity_rot_sup` | `True` | `False`, інакше завантажувач падає з поясненням |

---

## Чому без transfer learning

`imu_transfer.py` має жорстку умову:

```python
if source_shape[0] != target_shape[0] ...:
    raise ValueError("Only axis-count expansion is supported")
```

Він розширює лише кількість осей (3 → 6), а часовий вимір мусить лишитися
незмінним. Вихідна ST-модель навчена на 24 відліках, наші вікна — 208, тому
перенесення просто впаде.

Зберегти його можна було б лише повернувши вікно до 24 відліків, але 24 при
104 Гц — це 0.23 с, тобто менше одного кроку. В оригіналі ST це 24 при 26 Гц
(0.92 с), тож довелося б відкотити весь проєкт на 26 Гц разом із прошивкою.
Ментор затвердив 104 Гц, тому цей варіант відпадає.

Втрата невелика: переносяться ваги моделі, навченої на іншому датасеті, іншій
позиції на тілі й іншій частоті, причому гіроскопна гілка при перенесенні
однаково занулюється.

---

## Про модель

`--scratch` зі `get_ign((208, 6, 1), 5)` працюватиме, але архітектура
розрахована на короткі вікна: після `Flatten` виходить 9 216 ознак, і
`Dense(12)` розростається до **≈111 тисяч ваг** — 99 % моделі в одному шарі.
Ядро `(16, 1)` при 104 Гц охоплює лише 0.15 с замість 0.6 с при 26 Гц.

`har_cnn.py` із цього пакета — та сама ідея (згортки по часу, канали не
змішуються), але під довге вікно: три блоки зі зростаючим полем огляду і
усереднення по часу замість `Flatten`. Близько **14 тисяч ваг** замість 111
тисяч. Викликається так само: `get_har_cnn((208, 6, 1), num_classes=5)`.

Що вибрати — на розсуд: `get_ign` дає точку відліку для порівняння, `har_cnn`
дешевший для плати. Обидва влізають у STM32U585, різниця у швидкості інференсу.

---

## Чого НЕ робити

- **Не вмикати `gravity_rot_sup`.** Обертання по гравітації прибирає постійну
  складову акселерометра, якою `Lying` відрізняється від `Stationary`. Прошивка
  його не робить, тож вхід моделі розійдеться з тим, що буде на платі.
- **Не перераховувати mean/std.** Вони з train і зашиті в прошивку через
  згенерований `har_model_config.h`. Інші числа = зсунутий вхід на платі.
- **Не користуватися `operation_mode: deployment`.** Він збирає застосунок ST,
  який робить власне препроцесування на мікроконтролері. У нас своя прошивка,
  яка вже робить ФНЧ 25 Гц, вікно 208 × 6 і нормалізацію. Модель ставиться в
  неї через X-CUBE-AI, у позначене місце `TODO` в `main.c`.
- **Не ділити дані самостійно.** Спліт за людьми вже зроблений; випадковий
  поділ вікон дасть завищену точність через 50 % перекриття.

---

## Про дисбаланс класів

`Stationary` і `Stairs` приблизно вдвічі численніші за інші (кожен зібраний з
двох активностей). Ні `har_trainer.py`, ні `train_realworld_imu.py` не
передають `class_weight` у `model.fit`. Варіанти: додати
`class_weight=...` у виклик `fit` (одна строчка, ваги рахуються як
`N / (K * n_клас)`) або просто дивитися **macro-F1**, який скрипт і так рахує,
замість accuracy. Вирівнювати сам датасет не треба — це спотворить розподіл.

---

## Що перевірено, а що ні

Перевірено на справжніх файлах датасету (`python tests/test_modelzoo_loader.py`
у har-dataset): форми й типи масивів, порядок класів, збереження спліту за
людьми, що нормалізація — це рівно `(x - mean) / std` з `norm_stats.json`, що
без нормалізації `|acc|` для `Lying` дорівнює 0.976 g, і що неправильний
порядок класів або довжина вікна дають зрозумілу помилку, а не тиху.

**Не перевірено:** код, який потребує TensorFlow — сама `tf.data`-обгортка і
`har_cnn.py`. TensorFlow не встановлений на машині, де готувався пакет.
Перший запуск варто зробити на двох-трьох епохах і переконатися, що модель
будується і навчання стартує.
