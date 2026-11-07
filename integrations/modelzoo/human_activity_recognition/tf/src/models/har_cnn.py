"""Згорткова мережа під вікно 208 x 6 (2 с при 104 Гц) з har-dataset.

Чому не можна просто взяти st_ign з іншим input_shape. Її ядро (16, 1)
розраховане на 26 Гц: 16 відліків це 0.6 с. При 104 Гц ті самі 16 відліків —
0.15 с, тобто перший шар перестає бачити навіть один крок. А `Flatten` після
одного згорткового блоку на вікні 208 дає ~9 тисяч ознак, і `Dense(12)` після
нього розростається до ~111 тисяч ваг — 99 % моделі осідає в одному шарі.

Тут три згорткові блоки зі зростаючим полем огляду і усереднення по часу
замість Flatten. Важливо: усі ядра мають ширину 1 по осі каналів, тому
канали не змішуються між собою до самого кінця — ax..gz лишаються окремими
ознаками, як у st_ign.

Оцінка за замовчуванням для (208, 6, 1) і 5 класів:
    conv1  (16,1)/2 -> 97 x 6 x 24       408 ваг
    pool   (2,1)    -> 48 x 6 x 24
    conv2  (8,1)    -> 41 x 6 x 32     6 176 ваг
    pool   (2,1)    -> 20 x 6 x 32
    conv3  (4,1)    -> 17 x 6 x 48     6 192 ваг
    avgpool по часу -> 1 x 6 x 48
    dense                              1 445 ваг
    разом                           ~14 200 ваг (~57 КБ у float32)
"""
import tensorflow as tf
import tensorflow.keras.layers as layers

SEED = 0
tf.random.set_seed(SEED)


def _as_shape(value):
    """input_shape з YAML приходить рядком '(208, 6, 1)' — приймаємо і це."""
    if isinstance(value, str):
        value = value.strip().strip("()[]")
        return tuple(int(v) for v in value.split(",") if v.strip())
    return tuple(int(v) for v in value)


def get_har_cnn(input_shape=(208, 6, 1),
                num_classes: int = 5,
                dropout: float = 0.3,
                filters=(24, 32, 48),
                kernels=(16, 8, 4),
                head: str = "avgpool",
                **kwargs):
    """Будує модель для вікна (час, осі, 1).

    Args:
        input_shape: (timesteps, axes, 1); осі не змішуються згортками.
        num_classes: кількість цільових класів.
        dropout: перед класифікатором; 0 або None — без шару.
        filters, kernels: три згорткові блоки; кернел задається по часу.
        head: 'avgpool' — усереднення по часу (мало ваг, стійкіше),
              'flatten' — як у st_ign (більше ваг, чутливіше до положення
              події у вікні).
    """
    shape = _as_shape(input_shape)
    if len(shape) != 3 or shape[2] != 1:
        raise ValueError(f"очікується (timesteps, axes, 1), отримано {shape}")
    if num_classes is None:
        raise ValueError("num_classes не заданий: додай dataset.num_classes у конфіг")
    if len(filters) != 3 or len(kernels) != 3:
        raise ValueError("filters і kernels мають бути по три значення")

    inputs = layers.Input(shape=shape)

    # Перший блок із кроком 2 по часу: вікно довге, і прорідити його одразу
    # дешевше, ніж тягнути повну довжину крізь усю мережу.
    x = layers.Conv2D(filters[0], kernel_size=(kernels[0], 1), strides=(2, 1),
                      padding="valid", name="conv_time_1")(inputs)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)

    x = layers.Conv2D(filters[1], kernel_size=(kernels[1], 1), padding="valid",
                      name="conv_time_2")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)

    x = layers.Conv2D(filters[2], kernel_size=(kernels[2], 1), padding="valid",
                      name="conv_time_3")(x)
    x = layers.Activation("relu")(x)

    if head == "avgpool":
        # Усереднення ТІЛЬКИ по часу: вісь каналів лишається окремою, інакше
        # ax..gz злилися б в одну ознаку.
        steps = int(x.shape[1])
        x = layers.AveragePooling2D(pool_size=(steps, 1), name="time_pool")(x)
    elif head != "flatten":
        raise ValueError("head має бути 'avgpool' або 'flatten'")
    x = layers.Flatten()(x)

    if dropout:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=[outputs], name="har_cnn")
