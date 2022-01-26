import time

from aiogram import Bot, Dispatcher, executor, types # импортируем 4 классa
from timedetect_api_token import API_TOKEN  #tg-bot token
from main import predictor

# Включаем логирование
import logging
logging.basicConfig(level=logging.INFO)


# Инициализируем бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def start(message: types.Message):  
    user_first_name = message.from_user.first_name
    user_id = message.from_user.id
    logging.info(f'{user_id} {user_first_name} started the Time_Detect_Bot at {time.asctime()}')
    await message.reply(f"Hi, {user_first_name} !\nI'm Time_Detect_Bot!\nSend me a picture of the watch")


@dp.message_handler(content_types=['photo'])
async def handleR_photo(message):
    chat_id = message.chat.id
    # media_group_id is None means single photo at message
    if message.media_group_id is None:
        user_id = message.from_user.id
        message_id = message.message_id
        user_first_name = message.from_user.first_name
        img_path = 'uploaded_images/%s_%s_%s.jpg' % (user_id, user_first_name, message_id)
        await message.photo[-1].download(img_path)
        await message.reply("It's " + predictor(img_path) + " o'clock")
    else:
        await message.reply("Send me just one photo, please")


@dp.message_handler()
async def echo(message: types.Message):
    # await message.answer(message.text)
    await message.reply("Send me image, please")


if __name__ == '__main__':
    executor.start_polling(dp)