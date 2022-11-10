# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import json
from random import choice
import tensorflow as tf


from botbuilder.core import (
    ActivityHandler, 
    TurnContext
)
from botbuilder.schema import ChannelAccount

from config import DefaultConfig

class AMKBot(ActivityHandler):
    """
    AMK Bot
    """
    # See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.
    def __init__(
        self, 
        config: DefaultConfig
    ):
        """
        Initialize an instance of AMKBot

        :param DefaultConfig config: Configuration of the bot
        """
        super(AMKBot, self).__init__()
        self.model = tf.keras.models.load_model('amk_model')
        with open('templates\intents.json','r',encoding="utf-8") as ffile:
            self.intents = json.load(ffile)
        if config is None:
            raise TypeError(
                f"[{type(self).__name__}]: Missing parameter. config is required but none was given"
            )

        self.__config = config
    async def on_turn(self, turn_context: TurnContext):
        """
        Manage recieved activities and save changes made to BotStates.
        """
        
        if turn_context.activity.type == "message":
            #if is_active() or is_holiday('HU'):
            await self.on_message_activity(turn_context)

            #else:
            #    print(f"{self.__config.CLI_PREFIX}: Recieved a message while dormant.")

        elif turn_context.activity.type == 'conversationUpdate':
            await self.on_members_added_activity(turn_context.activity.members_added, turn_context)

    async def on_message_activity(self, turn_context: TurnContext):
        """
        Find the appropriate reply to the recieved message.
        """


        if turn_context.activity.text == None:
            return
        valami = self.model.predict([turn_context.activity.text])[0].argmax(0)
        response = choice(self.intents[valami]['responses'])
        await turn_context.send_activity(response)

    async def on_members_added_activity(
        self,
        members_added: list[ChannelAccount],
        turn_context: TurnContext
    ):
        """
        Greet when users are added to the conversation.
        Note that all channels do not send the conversation update activity.
        If you find that this bot works in the emulator, but does not in
        another channel the reason is most likely that the channel does not
        send this activity.
        """
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(f"Hello {turn_context.activity.from_property.name}!\n{self.__config.BOTTEXT_GREETING}")