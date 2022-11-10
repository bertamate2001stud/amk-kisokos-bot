#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

class DefaultConfig:
    """ Bot Configuration """

    PORT = 3978
    MESSAGE_ROUTE = '/api/messages'
    APP_ID = os.environ.get("MicrosoftAppId", "")
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "")
    NAME = 'AMKBot'

    # Office timezone and workhours

    OFFICE_TIMEZONE_TZDBNAME = "Europe/Budapest"
    OFFICE_TIMEZONE_ABBREV = "CET"
    OFFICE_START_HOUR = 0
    OFFICE_END_HOUR = 1

    # Bot text variables

    BOTTEXT_GREETING="Kedves OE AMK Hallgatók!\n\
    Ez a Teams csatorna azért lett létrehozva, hogy Projektmunka keretein belül a létrehozóim bemutassanak egy Deep Learninget használó Chatbotot, aminek keretein belül képes vagyok válaszolni valamennyi kérdéseitekre."
    BOTTEXT_SPACER="\n--------------------\n"
    BOTTEXT_NOQUESTIONFOUND="Úgy látom még nem kérdeztél, kérdezz bátran, meglátjuk, hogy tudok-e válaszolni."
    BOTTEXT_QUESTION_FOUND="Sajnos nem tudok erre a kérdésedre válaszolni, de továbbítom a fejlesztőimnek és majd ők segítenek neked."
    BOTTEXT_FALLBACK='Sajnálom, jelenleg nem tudok a kérdésedre válaszolni\n'


    # Bot command related variables
    
    ADMINS = []
    COMMAND_PREFIX = "command:"
    COMMANDS = {
        'shutdown': 'Goodbye.',
        'restart': 'See you soon!',
        'tetrain': 'Retraining started. This might take a while...'
    }

    # Bot logging variables
    LOGFILE = 'logs/events.log'
    LOG_TIME_FORMAT = '%Y %b %e %H:%M:%S'
    CLI_PREFIX = f'[{NAME}]'

    # Bot statistic variables
    QUESTIONS_FILE = 'logs/questions{stamp}.json'
    UNKNOWN_QUESTION_FILE = 'logs/unkown{stamp}.json'
    FILE_TIMESTAMP_FORMAT = '%Y%m%d'

    # Bot variables for scheduling reports
    REPORT_START_DATE = {       # The date and time, from which the count the frist interval, comment out unsused parameters.
        #'year': ...,
        #'month': ...,
        #'day:': ...,
        'hour': OFFICE_START_HOUR,
        'minute': 0,
        'second': 0,
        'microsecond': 0
    }
    REPORT_INTERVALS = {        # Time to wait between reports
        'weeks': 0,
        'days': 0,
        'hours': 0,
        'minutes': 5,
        'seconds': 0
    }

    # Converation reference variables for reporting
    REPORT_CONVERSATION_ACQUSITION = 'DYNAMIC'
    DEFAULT_REPORT_CONVERSTAION_TEMPLATE = 'templates/default_report_conversation.json'

    # Bot ML variables
    RETARIN = True
    INTENTS_FILEPATH = 'templates/intents.json'
    DATA_FOLDER='data/'
    MODELS_FOLDER=DATA_FOLDER + 'models/'
    DEFAULT_MODEL_PATH = MODELS_FOLDER + 'default_model.tf'
    DEFAULT_MODEL_TEMPLATE = 'templates/default_model.json'
    VALIDATION_RATE = 0.2
    TEST_RATE = 0.0
    COMPILE_PARAMS = {  # Paramteters for the tensorflow.keras.Model().compile() function
        'optimizer': 'Adam',
        'loss': 'categorical_crossentropy',
        'metrics': [
            'accuracy'
        ]
    }
    # Do not specify 'x', 'y' and 'validation_data'
    FIT_PARAMS = {  # Additional key word arguments to pass onto 'tensorflow.keras.Model().fit()'
        'epochs': 50,
        'shuffle': True,
        'verbose': 0
    }
    # Do not specify 'model' and 'filepath' 
    SAVE_MODEL_PARAMETERS = {
        'save_format': 'h5'
    }
    PRED_THRESHOLD = 0.70
    NON_LEARNING_CATEGORIES = [
        'greeting',
        'goodbye',
        'thanks'
    ]

    # Regex
    MENTION_REGEX = "^<@(|[WU].+?)>(.*)"

    CONVERSATION_EXPIRE = 600