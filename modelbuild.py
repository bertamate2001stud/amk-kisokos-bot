import os
import json
import tensorflow as tf
import tensorflow_hub as tf_hub

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
with open('./templates/intents.json','r',encoding="utf-8") as f:
    intents = json.load(f)

patterns: list[str] = []
labels : list[int] = []
[patterns.append(pattern) or labels.append(i) for i, intent in enumerate(intents) for pattern in intent['patterns']]
outputs = tf.one_hot(labels, len(intents))
dataset = tf.data.Dataset.from_tensor_slices((patterns, outputs)).shuffle(len(patterns))
train_dataset, valid_dataset = dataset.take(int(0.8 * len(dataset))), dataset.skip(int(0.8 * len(dataset)))
train_dataset = train_dataset.batch(batch_size=32).prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(batch_size=32).prefetch(tf.data.AUTOTUNE)

print(train_dataset)
print(valid_dataset)


tf_hub_embedding_layer = tf_hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4', trainable=False)

inputs = tf.keras.layers.Input(shape=[], dtype=tf.string, name='InputLayer')
pretrained_embedding = tf_hub_embedding_layer(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_embedding)
outputs = tf.keras.layers.Dense(len(intents), activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

model.summary()

model.fit(train_dataset, epochs=35, validation_data=valid_dataset)
model.save('amk_model')
