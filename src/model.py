import os, sys
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config

def build_model(input_shape, num_classes):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output, name='ASL_MobileNetV2')

    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"[MODELE] MobileNetV2 + tête ASL")
    print(f"[MODELE] Paramètres entraînables : {sum([tf.size(v).numpy() for v in model.trainable_variables]):,}")
    print(f"[MODELE] Paramètres gelés (base) : {sum([tf.size(v).numpy() for v in base_model.non_trainable_variables]):,}")

    return model

if __name__ == "__main__":
    model = build_model((128, 128, 3), 28)
    print(f"Input: {model.input_shape}  Output: {model.output_shape}")
