import os
from bare_functions.supabase_class import SupabaseAgent
import json
import traceback
from loguru import logger
from bare_functions.update_task_status import TaskStatus
from loggers.timer_decorator import timer_decorator
from modules.agreement_writer.background_tasks import generate_whole_agreement
from modules.chat_processor.new_chat import start_new_chat_agreement_writer, start_new_chat_case_research, start_new_chat_cross_examination, start_new_chat_quick_writer
from modules.cross_examination.background_tasks import generate_cross_examination
from modules.doc_writer.background_tasks.generate_draft import generate_draft_background
from modules.doc_writer.processing_functions.process_chat import process_chat_doc_writer
from modules.doc_writer.processing_functions.start_rechat import start_rechat_doc_writer
from modules.legal_research.background_tasks import generate_case_research
from modules.quick_witer.draft_generator import generate_draft
from modules.chat_processor.process_chat import process_chat_agreement_writer, process_chat_case_research, process_chat_cross_examination, process_chat_quick_writer

supabase_agent = SupabaseAgent()
task_status = TaskStatus()


async def fetch_session_data(session_id: str, table_name: str):
    fetch_data = {
        "input_table_name": table_name,
        "where_conditions": {
            "id": session_id
        },
    }
    return await supabase_agent.call_supabase_function(
        schemaname="public",
        functionname="get_with_where_conditions",
        params=fetch_data,
    )


async def fetch_module_details(module_id: int):

    file_path = os.path.join("modules", "chat_processor", "module_info.json")
    with open(file_path, "r") as file:
        module_dict = json.load(file)

    module_name = None
    table_name = None

    for module in module_dict:
        if module["module_id"] == module_id:
            module_name = module["module_name"]
            table_name = module["table_name"]
            break
    return module_name, table_name


MODULE_PROCESSOR = {
    "quick_writer": generate_draft,
    "cross_examination": generate_cross_examination,
    "agreement_writer": generate_whole_agreement,
    "case_research": generate_case_research,
    "document_writer": generate_draft_background,
}

NEW_CHAT_PROCESSOR = {
    "quick_writer": start_new_chat_quick_writer,
    "cross_examination": start_new_chat_cross_examination,
    "agreement_writer": start_new_chat_agreement_writer,
    "case_research": start_new_chat_case_research,
    "document_writer": start_rechat_doc_writer
}

CHAT_PROCESSOR = {
    "quick_writer": process_chat_quick_writer,
    "cross_examination": process_chat_cross_examination,
    "agreement_writer": process_chat_agreement_writer,
    "case_research": process_chat_case_research,
    "document_writer": process_chat_doc_writer
}


@timer_decorator
async def chat_process_background(
    module_id: int,
    session_id: int,
    message: str = None,
    overall_status: bool = False
):

    try:

        module_name, table_name = await fetch_module_details(
            module_id=module_id
        )

        fetched_data_from_supabase = await fetch_session_data(
            session_id=session_id, table_name=table_name
        )

        session_data = fetched_data_from_supabase[0]

        chat_id = session_data.get("chat_id", None)

        if chat_id:
            # chat id present check for messages data
            logger.debug("chat id found so fetching previous chat data")
            previous_chat_data = await task_status.get_task_status(chat_id)

            if previous_chat_data:

                previous_chat_messages = previous_chat_data['chat_messages']

                if message:
                    logger.debug(
                        "Appending the user message to previous chat messages."
                    )
                    previous_chat_messages.extend(
                        [
                            {"role": "user", "content": message},
                        ]
                    )

                    chat_data = {
                        "chat_id": chat_id,
                        "chat_messages": previous_chat_messages,
                        "overall_status": overall_status
                    }

                    await task_status.update_task_status(chat_id, chat_data)

                else:
                    logger.debug(
                        "No message found from user side."
                    )

                if overall_status == True:
                    logger.debug(
                        "overall status is true, so activating {module_name} module")

                    # TODO: change function

                    res = await MODULE_PROCESSOR[module_name](previous_chat_messages=previous_chat_messages, session_data=session_data)

                else:
                    logger.debug(
                        "overall status is false, so continuing with user query"
                    )
                    await CHAT_PROCESSOR[module_name](previous_chat_messages=previous_chat_messages, session_data=session_data)
            else:
                logger.debug(
                    "previous chat data not found, starting new chat with same chat_id")

                await NEW_CHAT_PROCESSOR[module_name](session_data=session_data)

        else:
            logger.warning("chat id not found in session data.")
    except Exception as e:
        logger.critical(
            f"An error occurred processing query_chat_process_background, TRACEBACK:--------- {traceback.format_exc()}")
