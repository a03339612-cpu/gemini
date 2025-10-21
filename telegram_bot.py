# -*- coding: utf-8 -*-

import logging
import os
import requests
import json
import asyncio
import time
import threading
import http.server
import socketserver
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ConversationHandler,
    MessageHandler,
    filters,
    ContextTypes,
    PicklePersistence,
)
from io import BytesIO
import mimetypes
import google.generativeai as genai

# --- Настройки ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
load_dotenv()

# --- Загрузка ключей ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
KIE_API_KEY = os.getenv("KIE_API_KEY")

if not all([TELEGRAM_BOT_TOKEN, GOOGLE_API_KEY, KIE_API_KEY]):
    raise ValueError("Please fill all required environment variables")

genai.configure(api_key=GOOGLE_API_KEY)

# --- Состояния для диалога ---
(
    SELECTING_ACTION, SELECTING_IMAGE_TYPE, SELECTING_VIDEO_TYPE, CHAT_WITH_AI,
    AWAITING_IMAGE_TEXT_PROMPT, AWAITING_SINGLE_IMAGE_WITH_PROMPT,
    COLLECTING_PHOTOS, AWAITING_COMBINE_PROMPT,
    AWAITING_VIDEO_TEXT_PROMPT, EDITING_IMAGE,
) = range(10)

# --- ФУНКЦИИ ГЕНЕРАЦИИ (БЕЗ ИЗМЕНЕНИЙ) ---
async def generate_video_from_text_kie(prompt: str) -> BytesIO | None:
    headers = {"Authorization": f"Bearer {KIE_API_KEY}", "Content-Type": "application/json"}
    create_task_url = "https://api.kie.ai/api/v1/jobs/createTask"
    payload = {"model": "sora-2-text-to-video", "input": {"prompt": prompt, "aspect_ratio": "landscape"}}
    try:
        response_create = requests.post(create_task_url, headers=headers, json=payload)
        if response_create.status_code != 200: return None
        task_id = response_create.json().get("data", {}).get("taskId")
        if not task_id: return None
        record_info_url = f"https://api.kie.ai/api/v1/jobs/recordInfo?taskId={task_id}"
        for _ in range(24):
            await asyncio.sleep(20)
            response_record = requests.get(record_info_url, headers=headers)
            if response_record.status_code == 200:
                data = response_record.json().get("data", {})
                state = data.get("state")
                if state == "success":
                    video_url = json.loads(data.get("resultJson")).get("resultUrls", [None])[0]
                    if video_url:
                        response_video = requests.get(video_url, stream=True)
                        if response_video.status_code == 200: return BytesIO(response_video.content)
                    return None
                elif state == "fail": return None
        return None
    except Exception as e:
        logger.error(f"KIE.AI video generation error: {e}")
        return None

async def generate_chat_response(prompt: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")
        if "chat_session" not in context.user_data:
            context.user_data["chat_session"] = model.start_chat(history=[])
        response = await context.user_data["chat_session"].send_message_async(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Chat generation error: {e}")
        return "Произошла ошибка при общении с ИИ."

async def generate_image_from_text(prompt: str) -> BytesIO | None:
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")
        response = await model.generate_content_async(prompt)
        part = next((p for p in response.candidates[0].content.parts if "inline_data" in p), None)
        if part: return BytesIO(part.inline_data.data)
        return None
    except Exception as e:
        logger.error(f"Text-to-image generation error: {e}")
        return None

async def generate_image_from_image(prompt: str, image_bytes: bytes, mime_type: str) -> BytesIO | None:
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")
        image_part = {"mime_type": mime_type, "data": image_bytes}
        response = await model.generate_content_async([prompt, image_part])
        part = next((p for p in response.candidates[0].content.parts if "inline_data" in p), None)
        if part: return BytesIO(part.inline_data.data)
        return None
    except Exception as e:
        logger.error(f"Image-to-image generation error: {e}")
        return None

async def generate_combined_image(prompt: str, image_parts: list) -> BytesIO | None:
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")
        content = [prompt, *image_parts]
        response = await model.generate_content_async(content)
        part = next((p for p in response.candidates[0].content.parts if "inline_data" in p), None)
        if part: return BytesIO(part.inline_data.data)
        return None
    except Exception as e:
        logger.error(f"Image combination error: {e}")
        return None


# --- ОБРАБОТЧИКИ ТЕЛЕГРАМ ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    keyboard = [[InlineKeyboardButton("🖼️ Изображение", callback_data="main_image")],
                [InlineKeyboardButton("🎬 Видео", callback_data="main_video")],
                [InlineKeyboardButton("💬 Чат", callback_data="main_chat")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    if "chat_session" in context.user_data: del context.user_data["chat_session"]
    message_text = "👋 Добро пожаловать!\n\nВыберите, что вы хотите сделать:"
    if update.callback_query:
        try:
            await update.callback_query.edit_message_text(message_text, reply_markup=reply_markup)
        except BadRequest:
            await update.callback_query.message.reply_text(message_text, reply_markup=reply_markup)
    else:
        await update.message.reply_text(message_text, reply_markup=reply_markup)
    return SELECTING_ACTION

async def handle_main_menu_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == "main_image": return await show_image_menu(update, context)
    if query.data == "main_video": return await show_video_menu(update, context)
    if query.data == "main_chat":
        await query.edit_message_text("Вы вошли в режим чата с ИИ...")
        return CHAT_WITH_AI
    return SELECTING_ACTION

async def show_image_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    keyboard = [
        [InlineKeyboardButton("📝 По тексту", callback_data="img_from_text")],
        [InlineKeyboardButton("🖼️ По изображению", callback_data="img_from_image")],
        [InlineKeyboardButton("🎨 Комбинировать фото", callback_data="img_combine")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")]
    ]
    await update.callback_query.edit_message_text("Выберите способ генерации изображения:", reply_markup=InlineKeyboardMarkup(keyboard))
    return SELECTING_IMAGE_TYPE

async def handle_image_menu_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == "img_from_text":
        await query.edit_message_text("Пришлите текстовое описание...")
        return AWAITING_IMAGE_TEXT_PROMPT
    if query.data == "img_from_image":
        await query.edit_message_text("Отправьте одно изображение с подписью, описывающей, что нужно сделать.")
        return AWAITING_SINGLE_IMAGE_WITH_PROMPT
    if query.data == "img_combine":
        context.user_data['combine_photos'] = [] 
        context.user_data['control_message_id'] = None 
        keyboard = [[InlineKeyboardButton("✅ Готово (0 фото)", callback_data="done_collecting")]]
        sent_message = await query.edit_message_text(
            "Отправьте мне от 2 до 5 фотографий (по одной). Когда закончите, нажмите 'Готово'.",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        context.user_data['control_message_id'] = sent_message.message_id
        return COLLECTING_PHOTOS
    if query.data == "back_to_main": return await start(update, context)
    return SELECTING_IMAGE_TYPE

async def collect_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if 'combine_photos' not in context.user_data: context.user_data['combine_photos'] = []
    
    if len(context.user_data['combine_photos']) >= 5:
        await update.message.reply_text("Достигнут лимит в 5 фотографий. Нажмите 'Готово', чтобы продолжить.")
        return COLLECTING_PHOTOS

    photo_file_id = update.message.photo[-1].file_id
    context.user_data['combine_photos'].append(photo_file_id)
    
    count = len(context.user_data['combine_photos'])
    control_message_id = context.user_data.get('control_message_id')
    
    keyboard = [[InlineKeyboardButton(f"✅ Готово ({count} фото)", callback_data="done_collecting")]]
    
    if control_message_id:
        try:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=control_message_id,
                text=f"Фото {count} добавлено. Отправьте следующее или нажмите 'Готово'.",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        except BadRequest: 
            sent_message = await update.message.reply_text("Фото добавлено.", reply_markup=InlineKeyboardMarkup(keyboard))
            context.user_data['control_message_id'] = sent_message.message_id
    else:
        sent_message = await update.message.reply_text(
            f"Фото {count} добавлено. Отправьте следующее или нажмите 'Готово'.",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        context.user_data['control_message_id'] = sent_message.message_id

    return COLLECTING_PHOTOS

async def done_collecting(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    
    if len(context.user_data.get('combine_photos', [])) < 2:
        await query.message.reply_text("Нужно добавить хотя бы 2 фотографии.")
        return COLLECTING_PHOTOS
        
    await query.edit_message_text("Отлично! Теперь напишите, что нужно сделать с этими фото (например, 'помести человека со второго фото на фон с первого').")
    return AWAITING_COMBINE_PROMPT

async def handle_combine_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    prompt = update.message.text
    await update.message.reply_text("🎨 Комбинирую изображения... Это может занять некоторое время.")
    
    image_parts = []
    for file_id in context.user_data.get('combine_photos', []):
        photo_file = await context.bot.get_file(file_id)
        image_bytes = bytes(await photo_file.download_as_bytearray())
        image_parts.append({"mime_type": "image/jpeg", "data": image_bytes})
        
    result_image = await generate_combined_image(prompt, image_parts)
    
    context.user_data.pop('combine_photos', None)
    context.user_data.pop('control_message_id', None)

    if result_image:
        context.user_data['last_generated_image_bytes'] = result_image.getvalue()
        context.user_data['last_generated_image_prompt'] = prompt
        keyboard = [
            [InlineKeyboardButton("✏️ Редактировать", callback_data="edit_image")],
            [InlineKeyboardButton("🏠 На главное меню", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_photo(
            photo=BytesIO(context.user_data['last_generated_image_bytes']), 
            caption=f"Результат вашего творчества!\n\nЗапрос: `{prompt}`",
            reply_markup=reply_markup
        )
        return EDITING_IMAGE
    else:
        await update.message.reply_text("Не удалось скомбинировать изображения.")
        return await start(update, context)

async def handle_text_prompt_for_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    prompt = update.message.text
    await update.message.reply_text("🎨 Генерирую изображение...")
    image_file = await generate_image_from_text(prompt)
    if image_file:
        context.user_data['last_generated_image_bytes'] = image_file.getvalue()
        context.user_data['last_generated_image_prompt'] = prompt
        keyboard = [[InlineKeyboardButton("✏️ Редактировать", callback_data="edit_image")], [InlineKeyboardButton("🏠 На главное меню", callback_data="back_to_main")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_photo(photo=BytesIO(context.user_data['last_generated_image_bytes']), caption=f"Запрос: `{prompt}`", reply_markup=reply_markup)
        return EDITING_IMAGE
    else:
        await update.message.reply_text("Не удалось создать изображение.")
        return await start(update, context)

async def edit_image_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if 'last_generated_image_prompt' not in context.user_data:
        await query.message.reply_text("Не могу найти предыдущее изображение. Начните заново.")
        return await start(update, context)
    current_prompt = context.user_data['last_generated_image_prompt']
    await query.edit_message_caption(caption=f"Текущий запрос: `{current_prompt}`\n\nПришлите новый текст для редактирования.", reply_markup=None)
    return EDITING_IMAGE

async def handle_edited_image_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    new_prompt = update.message.text
    if 'last_generated_image_bytes' not in context.user_data:
        await update.message.reply_text("Не могу найти предыдущее изображение. Начните заново.")
        return await start(update, context)
    last_image_bytes = context.user_data['last_generated_image_bytes']
    await update.message.reply_text(f"🎨 Редактирую изображение с новым запросом: `{new_prompt}`")
    edited_image = await generate_image_from_image(new_prompt, last_image_bytes, 'image/jpeg')
    if edited_image:
        context.user_data['last_generated_image_bytes'] = edited_image.getvalue()
        context.user_data['last_generated_image_prompt'] = new_prompt
        keyboard = [[InlineKeyboardButton("✏️ Редактировать еще", callback_data="edit_image")], [InlineKeyboardButton("🏠 На главное меню", callback_data="back_to_main")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_photo(photo=BytesIO(context.user_data['last_generated_image_bytes']), caption=f"Отредактировано: `{new_prompt}`", reply_markup=reply_markup)
        return EDITING_IMAGE
    else:
        await update.message.reply_text("Не удалось отредактировать изображение.")
        return await start(update, context)

async def handle_single_image_with_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not update.message.photo or not update.message.caption:
        await update.message.reply_text("Нужно отправить изображение с подписью.")
        return AWAITING_SINGLE_IMAGE_WITH_PROMPT
    prompt = update.message.caption
    await update.message.reply_text("🎨 Обрабатываю...")
    photo_file = await update.message.photo[-1].get_file()
    image_bytes = bytes(await photo_file.download_as_bytearray())
    generated_image = await generate_image_from_image(prompt, image_bytes, 'image/jpeg')
    if generated_image:
        context.user_data['last_generated_image_bytes'] = generated_image.getvalue()
        context.user_data['last_generated_image_prompt'] = prompt
        keyboard = [[InlineKeyboardButton("✏️ Редактировать", callback_data="edit_image")], [InlineKeyboardButton("🏠 На главное меню", callback_data="back_to_main")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_photo(photo=BytesIO(context.user_data['last_generated_image_bytes']), caption=f"Запрос: `{prompt}`", reply_markup=reply_markup)
        return EDITING_IMAGE
    else:
        await update.message.reply_text("Не удалось создать изображение.")
        return await start(update, context)

async def show_video_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    keyboard = [[InlineKeyboardButton("📝 По тексту", callback_data="video_from_text")],
                [InlineKeyboardButton("🖼️ По изображению (скоро)", callback_data="video_from_image_soon")],
                [InlineKeyboardButton("⬅️ Назад", callback_data="back_to_main")]]
    await update.callback_query.edit_message_text("Выберите способ генерации видео:", reply_markup=InlineKeyboardMarkup(keyboard))
    return SELECTING_VIDEO_TYPE

async def handle_video_menu_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == "video_from_text":
        await query.edit_message_text("Пришлите текстовое описание для видео...")
        return AWAITING_VIDEO_TEXT_PROMPT
    if query.data == "video_from_image_soon":
        await query.edit_message_text("Эта функция скоро появится!")
        return SELECTING_VIDEO_TYPE
    if query.data == "back_to_main": return await start(update, context)
    return SELECTING_VIDEO_TYPE

async def handle_text_prompt_for_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    prompt = update.message.text
    await update.message.reply_text("🎬 Задача на генерацию видео Sora 2 создана...")
    video_file = await generate_video_from_text_kie(prompt)
    if video_file:
        await update.message.reply_video(video=video_file, caption=f"Ваше видео по запросу:\n\n`{prompt}`")
    else:
        await update.message.reply_text("Не удалось создать видео.")
    return await start(update, context)

async def handle_chat_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_message = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    response_text = await generate_chat_response(user_message, context)
    await update.message.reply_text(response_text)
    return CHAT_WITH_AI

# НОВАЯ ФУНКЦИЯ ДЛЯ ВЕБ-СЕРВЕРА
def run_health_check_server():
    """Запускает простой HTTP-сервер для проверки состояния Render."""
    # Render предоставляет порт в переменной окружения PORT
    PORT = int(os.environ.get("PORT", 8080))
    
    class HealthCheckHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Bot is alive!")

    with socketserver.TCPServer(("", PORT), HealthCheckHandler) as httpd:
        logger.info(f"Health check server running on port {PORT}")
        httpd.serve_forever()

def main() -> None:
    persistence = PicklePersistence(filepath="bot_persistence")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).persistence(persistence).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            SELECTING_ACTION: [CallbackQueryHandler(handle_main_menu_selection, pattern="^main_")],
            SELECTING_IMAGE_TYPE: [CallbackQueryHandler(handle_image_menu_selection), CallbackQueryHandler(start, pattern="^back_to_main$")],
            SELECTING_VIDEO_TYPE: [CallbackQueryHandler(handle_video_menu_selection), CallbackQueryHandler(start, pattern="^back_to_main$")],
            CHAT_WITH_AI: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_chat_message)],
            AWAITING_IMAGE_TEXT_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_prompt_for_image)],
            AWAITING_SINGLE_IMAGE_WITH_PROMPT: [MessageHandler(filters.PHOTO, handle_single_image_with_prompt)],
            AWAITING_VIDEO_TEXT_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_prompt_for_video)],
            EDITING_IMAGE: [
                CallbackQueryHandler(edit_image_prompt, pattern="^edit_image$"),
                CallbackQueryHandler(start, pattern="^back_to_main$"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_edited_image_prompt)
            ],
            COLLECTING_PHOTOS: [
                MessageHandler(filters.PHOTO, collect_photo),
                CallbackQueryHandler(done_collecting, pattern="^done_collecting$")
            ],
            AWAITING_COMBINE_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_combine_prompt)]
        },
        fallbacks=[CommandHandler("start", start)],
        name="main_conversation",
        persistent=True,
    )
    application.add_handler(conv_handler)
    
    # НОВЫЙ БЛОК: Запуск веб-сервера в отдельном потоке
    health_thread = threading.Thread(target=run_health_check_server)
    health_thread.daemon = True
    health_thread.start()

    print("Бот запущен...")
    application.run_polling()

if __name__ == "__main__":
    main()

