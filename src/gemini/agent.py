import google.generativeai as genai
from google.generativeai.types.generation_types import StopCandidateException

import os, time, threading, pickle
from datetime import datetime

from util.typing import Session
from tools import loader
import logging
logger = logging.getLogger(__name__)

genai.configure(api_key=os.environ["API_KEY"])

model = None          # Hold A global GenAI model
sessions_cache = {}   # Hold a session cache
HISTORY_MAX_SIZE = 100 * 1024 # 100 KB
last_dump_time = None

GRACEFULLY_STOP = False 

with open('./resources/system_instruction.txt') as f:
    instruction = f.read ()
    logger.debug(instruction)



def loadChatSession ():
    directory_path = 'history'
    files_and_dirs = os.listdir(directory_path)

    # 過濾出檔案
    files = [f for f in files_and_dirs]    
    logger.debug(files) 

    for f in files_and_dirs:
         filename = os.path.join(directory_path, f)
         if os.path.isfile(filename):
             with open(filename, 'rb') as file:
                 history = pickle.load(file)
                 logger.debug(history)
                 if history != '' :
                    model = getModel() 
                    session = Session()
                    session.chat = model.start_chat(history=history, enable_automatic_function_calling=True)
                    session.timestamp = time.time() 
                    sessions_cache[str(f)] = session 


def ask (id, content):
    answer = "哇，不知道怎麼回答這個問題"
    if id not in sessions_cache:
        logger.debug ("No Chat session is found for : " + str(id) + ". Starts a new chat session")

        model = getModel()
        
        chat_session = Session()
        chat_session.chat = model.start_chat(history=[], enable_automatic_function_calling=True)  # Load history from here
        chat_session.timestamp = time.time()

        sessions_cache[id] = chat_session

        
    session  = sessions_cache[id]
    
    # Check if history size exceeds limit
    if get_history_size(session.chat.history) > HISTORY_MAX_SIZE:
        logger.info(f"History for {id} exceeds {HISTORY_MAX_SIZE} bytes. Compressing...")
        archive_to_markdown(id, session.chat.history)
        summarize_and_compress(id, session)

    logger.debug (session)
    #logger.debug (chat.history)
    try:
        response = session.chat.send_message(content)
        session.timestamp = time.time()
        answer = response.text
    except StopCandidateException as safety_exception :
        logger.error ("Error occurred when user ask : " + content + "  with exception : " + str(safety_exception)) 
        answer = "為了保護你，這個問題就不回答了"
    except Exception as e:
        logger.error ("Error occurred when user ask : " + content + "  with exception : " + str(e)) 

    return answer

def archive_to_markdown(user_id, history):
    """Archive chat history to a Markdown file organized by date."""
    today = datetime.now().strftime('%Y-%m-%d')
    folder_path = os.path.join('history', today)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    
    file_path = os.path.join(folder_path, f"{user_id}.md")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"# Chat History for {user_id}\n")
        f.write(f"Archived on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for content in history:
            role = "User" if content.role == "user" else "Bread (AI)"
            f.write(f"### {role}:\n")
            for part in content.parts:
                if hasattr(part, 'text'):
                    f.write(f"{part.text}\n")
            f.write("\n---\n\n")
    logger.info(f"Archived history for {user_id} to {file_path}")

def summarize_and_compress(user_id, session):
    """Summarize the chat history and replace it with the summary."""
    model = getModel()
    summary_prompt = (
        "請簡短地總結以上的對話內容。這個總結將作為之後對話的上下文背景。"
        "請務必保持簡潔，總結內容不要超過 2000 個字元。"
    )
    
    try:
        # We use the existing chat session to ask for a summary of itself
        response = session.chat.send_message(summary_prompt)
        summary_content = response.text
        
        # Safeguard: If the summary is still over 100KB, forcefully truncate it
        # Actually, if it's over 100KB, it's very wrong, so we'll truncate it to 50KB to be safe.
        summary_bytes = summary_content.encode('utf-8')
        if len(summary_bytes) > HISTORY_MAX_SIZE:
             logger.warning(f"Summary for {user_id} was too large ({len(summary_bytes)}), truncating.")
             # Keep the end of the summary if it was long, or a fixed portion
             summary_content = summary_bytes[-50000:].decode('utf-8', 'ignore') + "\n\n(Note: Summary truncated due to size)"

        # Start a fresh chat with the summary as the initial context
        new_history = [
            {"role": "user", "parts": [f"這是我們之前的對話總結：\n{summary_content}"]},
            {"role": "model", "parts": ["好的，我已經記住了目前的對話背景。請繼續。"]}
        ]
        session.chat = model.start_chat(history=new_history, enable_automatic_function_calling=True)
        session.timestamp = time.time()
        logger.info(f"Compressed history for {user_id} using AI summary.")
    except Exception as e:
        logger.error(f"Failed to summarize history for {user_id}: {e}")
        # Fallback: just clear history or keep only the last few messages if summarization fails
        session.chat = model.start_chat(history=[], enable_automatic_function_calling=True)


def getModel ():
    global model
    if model is None:
        registry = loader.load_plugins()
        plugin_tools = registry.get("tools", [])
        
        # When using tools with the Python SDK, if the list is empty,
        # it's best to omit the tools parameter entirely rather than passing [].
        kwargs = {
            'model_name': os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite'),
            'system_instruction': instruction,
            'safety_settings': {
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE, 
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH, 
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE
            }
        }
        if plugin_tools:
            kwargs['tools'] = plugin_tools
            logger.info(f"Registered {len(plugin_tools)} tools with Gemini model.")
            
        model = genai.GenerativeModel(**kwargs)
    return model

def get_history_size(history):
    """Estimate the size of the chat history in bytes."""
    size = 0
    for content in history:
        for part in content.parts:
            if hasattr(part, 'text'):
                size += len(part.text.encode('utf-8'))
    return size

def history (id):
    data = '' 
    session = sessions_cache[id]
    if session:
        data = str(session.chat.history) 
    return data


def historyPersistentJob () :

    while not GRACEFULLY_STOP:
        historyPersistent()
        time.sleep(60)

def historyPersistent():
    global last_dump_time
    if not sessions_cache is None:
            if not last_dump_time is None:
                for key in sessions_cache:
                    if last_dump_time < sessions_cache[key].timestamp:
                        history = sessions_cache[key].chat.history
                        if history :
                            with open('history/'+key, 'wb') as file:
                                pickle.dump(history, file)
    last_dump_time = time.time()


job = threading.Thread(target=historyPersistentJob)    
job.start()
