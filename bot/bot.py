import telegram
import telebot
from settings.constants import TOKEN, WELCOME_TEXT, COVID_TEXT, SYMPTOM_PRED, SPEC_LIST, DOC_FIND
from utils import datacleaner, predictor, searcher
import random
from telebot import types
import sys
import os

sys.path.append(os.getcwd())
READ_SYMPTOM = 0
READ_DOCTOR = 0

# updater and dispatcher
bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=['start'])
def welcome(message):
    # keyboard
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.KeyboardButton("Доктор по симптомах")
    item2 = types.KeyboardButton("Доктор по специализации")
    item4 = types.KeyboardButton("У меня симптомы COVID-19")
    markup.add(item1, item2, item4)

    bot.send_message(message.chat.id, WELCOME_TEXT, reply_markup=markup)


@bot.message_handler(content_types=['text'])
def lalala(message):
    global READ_SYMPTOM, READ_DOCTOR
    if message.chat.type == 'private':
        if READ_SYMPTOM:
            text = message.text
            # нейронка работает тут
            model = predictor.Predictor()
            pred = model.predict(text)
            doc_pred = searcher.search(pred[0], pred[1], pred[2])
            if len(doc_pred) < 5:
                doc_pred += searcher.search(pred[0], pred[1])
                if len(doc_pred) < 5:
                    doc_pred += searcher.search(pred[0], pred[2])
                    if len(doc_pred) < 5:
                        doc_pred += searcher.search(pred[1], pred[2])
                        if len(doc_pred) < 5:
                            doc_pred += searcher.search(pred[0])
                            if len(doc_pred) < 5:
                                doc_pred += searcher.search(pred[1])
                                if len(doc_pred) < 5:
                                    doc_pred += searcher.search(pred[2])

            while len(doc_pred) < 5:
                if len(doc_pred) >= 1:
                    doc_pred.append(random.choice(doc_pred))
                else:
                    doc_pred.append(["Не ", "знайдено"])
            # поиск лучших врачей по специализации
            bot.send_message(message.chat.id,
                             SYMPTOM_PRED.format(pred[0], pred[1], pred[2], doc_pred[0][0], doc_pred[0][1],
                                                 doc_pred[1][0], doc_pred[1][1], doc_pred[2][0], doc_pred[2][1],
                                                 doc_pred[3][0], doc_pred[3][1], doc_pred[4][0], doc_pred[4][1]))
            READ_SYMPTOM = 0

        elif READ_DOCTOR:
            text = message.text
            text = text.capitalize()
            if text not in SPEC_LIST:
                if text == "Выход":
                    READ_DOCTOR = 0
                else:
                    bot.send_message(message.chat.id,
                                     'Специализация не найдена. Попробуйте еще раз или введите "выход"')
                    READ_DOCTOR = 0
            else:
                doc_pred = searcher.search(text)
                while len(doc_pred) < 5:
                    if len(doc_pred) >= 1:
                        doc_pred.append(random.choice(doc_pred))
                    else:
                        doc_pred.append(["Не ", "знайдено"])

                bot.send_message(message.chat.id, DOC_FIND.format(text, doc_pred[0][0], doc_pred[0][1], doc_pred[1][0],
                                                                  doc_pred[1][1], doc_pred[2][0], doc_pred[2][1],
                                                                  doc_pred[3][0], doc_pred[3][1], doc_pred[4][0],
                                                                  doc_pred[4][1]))
                READ_DOCTOR = 0

        elif message.text == 'Доктор по симптомах':
            bot.send_message(message.chat.id, 'Подробно опишите симптомы:')
            READ_SYMPTOM = 1
        elif message.text == 'Доктор по специализации':
            bot.send_message(message.chat.id, 'Введите название специализации:')
            READ_DOCTOR = 1
        elif message.text == 'У меня симптомы COVID-19':
            bot.send_message(message.chat.id, COVID_TEXT)
        else:
            bot.send_message(message.chat.id, 'Не знаю что ответить, попробуйте еще раз.')


bot.polling(none_stop=True)
