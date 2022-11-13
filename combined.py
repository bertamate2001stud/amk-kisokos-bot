# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import sys
import traceback
from datetime import datetime
from http import HTTPStatus

from aiohttp import web
from aiohttp.web import (
    Request, 
    Response, 
    json_response)
from botbuilder.core import (
    BotFrameworkAdapterSettings,
    BotFrameworkAdapter,
    TurnContext
)
from botbuilder.schema import (
    Activity, 
    ActivityTypes
)

from bots import AMKBot
from config import DefaultConfig

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

# Catch-all for errors.
async def on_error(context: TurnContext, error: Exception):
    # This check writes out errors to console log .vs. app insights.
    # NOTE: In production environment, you should consider logging this to Azure
    #       application insights.
    print(f"\n [on_turn_error] unhandled error: {error}", file=sys.stderr)
    traceback.print_exc()

    # Send a message to the user
    await context.send_activity("The bot encountered an error or bug.")
    await context.send_activity(
        "To continue to run this bot, please fix the bot source code."
    )
    # Send a trace activity if we're talking to the Bot Framework Emulator
    if context.activity.channel_id == "emulator":
        # Create a trace activity that contains the error object
        trace_activity = Activity(
            label="TurnError",
            name="on_turn_error Trace",
            timestamp=datetime.utcnow(),
            type=ActivityTypes.trace,
            value=f"{error}",
            value_type="https://www.botframework.com/schemas/error",
        )
        # Send a trace activity, which will be displayed in Bot Framework Emulator
        await context.send_activity(trace_activity)


# Listen for incoming requests on /api/messages
async def messages(req: Request) -> Response:
    # Main bot message handler.
    if "application/json" in req.headers["Content-Type"]:
        body = await req.json()
    else:
        return Response(status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE)

    activity = Activity().deserialize(body)
    auth_header = req.headers["Authorization"] if "Authorization" in req.headers else ""

    response = await ADAPTER.process_activity(activity, auth_header, BOT.on_turn)
    if response:
        return json_response(data=response.body, status=response.status)
    return Response(status=HTTPStatus.OK)

def init_app(argv = None):
    """
    Initializie an aiohttp web application with basic settings.
    """
    APP = web.Application()
    APP.router.add_post(CONFIG.MESSAGE_ROUTE, messages)

    return APP

# Create configuration
CONFIG = DefaultConfig()


# Create the Bot
BOT = AMKBot(CONFIG)

# Create adapter.
# See https://aka.ms/about-bot-adapter to learn more about how bots work.
SETTINGS = BotFrameworkAdapterSettings(CONFIG.APP_ID, CONFIG.APP_PASSWORD)
ADAPTER = BotFrameworkAdapter(SETTINGS)
ADAPTER.on_turn_error = on_error

# Startup commands:
# 1.) gunicorn --bind 0.0.0.0 --worker-class aiohttp.worker.GunicornWebWorker --timeout 600 app:APP
# Anything after the APP obbject wll be ignored, APP object must be outside of 'if __name__ == "__main__":'
# 2.) python -m aiohttp.web -H 0.0.0.0 -P 8000 app:init_app
# Works with the init_app function and does not block exection of code after the function

# Create application
APP = init_app()

if __name__ == "__main__":
    try:
        web.run_app(APP, host="localhost", port=CONFIG.PORT)
    except Exception as error:
        raise error
