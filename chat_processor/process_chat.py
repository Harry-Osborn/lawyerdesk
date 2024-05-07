import traceback
from loguru import logger
from bare_functions.summarize import summarize
from bare_functions.get_title import get_title
from bare_functions.llm_call import LlmCall
from bare_functions.supabase_class import SupabaseAgent
from bare_functions.update_task_status import TaskStatus
from loggers.timer_decorator import timer_decorator
from modules.agreement_writer.get_chat_section import chat_section_finder
from modules.chat_processor.fetch_agreement import fetch_agreement_from_supabase

supabase_agent = SupabaseAgent()
task_status = TaskStatus()
llm_call = LlmCall()

generate_question = {
    "name": "generate_question",
    "description": "Generates a question based on the provided user query and previous chat messages",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to be generated, as per the previous chat messages and user query."
            },
            "is_question": {
                "type": "boolean",
                "description": "True if the question is generate else False",
                "enum": [True, False]
            }
        },
        "required": ["question", "is_question"]
    }
}


async def format_chat_messages(previous_chat_messages):
    formatted_messages = []
    for message in previous_chat_messages:
        formatted_messages.append(
            f"{message['role'].capitalize()} : {message['content']}")
    return '\n'.join(formatted_messages)

# async def format_chat_messages(previous_chat_messages):
#     formatted_messages = []
#     for message in previous_chat_messages:
#         if message['role'] != 'user':
#             formatted_messages.append(message['content'])
#     return '\n'.join(formatted_messages)


@timer_decorator
async def process_chat_quick_writer(previous_chat_messages: list, session_data: list):

    try:
        chat_id = session_data['chat_id']

        system_content = """

        You are tasked with drafting legal documents based on user queries, specializing in Indian legal law. Utilize the 'generate_question' function to interact with users, ensuring all necessary details are collected to draft a document accurately. Your primary objectives are:

        1. Identify and gather all essential information from the user required for the legal document. This includes, but is not limited to, names, purposes, addresses, and contact details of the involved parties.
        2. Ensure you have a clear understanding of the user's needs. If any details are missing or clarification is needed, generate a question. Keep in mind to generate only one question at a time, focusing on the most critical information missing.
        
        ### Instructions ###
        -Explicit Tracking: After each user interaction, record the essence of the question asked. This serves as a log to avoid repetition.
        -Question Diversification: Before generating a new question, review the log of previously asked questions. If a user's response to a previous question was negative, ensure the next question explores a different aspect of the information needed. This aims to cover various facets of the incident or information required for the legal document without circling back to already answered inquiries.
        -Contextual Shifts: In cases where multiple negative responses have led to a dead end in a particular line of questioning, intentionally shift the context. 
        -Regular Review: Periodically review the interaction flow to identify patterns of negative responses. Use these insights to adjust the direction of questioning, focusing on areas yet to be explored or that might yield more productive responses.
        """
        overall_status = False
        print(f"the previous chat messages: {previous_chat_messages}")

        messages = [
            {"role": "system", "content": system_content}
        ]

        messages.extend(previous_chat_messages)

        logger.debug(f"Messages: {messages}")

        assistant_response = await llm_call.openai_llm_call(
            messages=messages,
            structured_function=generate_question,
            module_name="Quick Writer",
            model="gpt-4-1106-preview"
        )
        logger.debug(assistant_response)

        is_question = assistant_response.get('is_question', None)
        if is_question:
            if is_question == False:
                assistant_response = "Based on the information provided, it seems that I have sufficient details to proceed."
                overall_status = True
            else:
                assistant_response = assistant_response['question']

        else:
            assistant_response = assistant_response['question']

        previous_chat_messages.extend(
            [
                {"role": "assistant", "content": assistant_response}
            ]
        )

        chat_data = {
            "chat_id": chat_id,
            "chat_messages": previous_chat_messages,
            "overall_status": overall_status
        }

        await task_status.update_task_status(chat_id, chat_data)

    except Exception as e:
        logger.error(f"An error occurred: {str(traceback.format_exc())}")


@timer_decorator
async def process_chat_agreement_writer(previous_chat_messages: list, session_data: list):

    try:
        chat_id = session_data['chat_id']
        agreement_id = session_data['legal_agreement_id']

        fetch_agreement = await fetch_agreement_from_supabase(agreement_id=agreement_id)

        agreement_name = fetch_agreement[0]['agreement_name']

        # Dynamically get the first key regardless of its name
        main_key = next(iter(fetch_agreement[0]['agreement_structure']))

        # Extract section names
        section_names = [item['section_name']
                         for item in fetch_agreement[0]['agreement_structure'][main_key]]
        logger.debug(f"The section names are: {section_names}")
        section_names = ", ".join(section_names).replace("'", "")
        temp_agreement = session_data['temp_agreement']
        if not temp_agreement:
            system_content = f"""
            As an AI assistant, you are tasked with drafting a legal {agreement_name} under Indian law by collecting necessary details from the user. 
            Utilize the 'generate_question' function to ask for critical information needed to draft the agreement comprehensively. 
            This agreement includes sections such as {section_names}.

            ### Instructions ###
            -Explicit Tracking: After each user interaction, record the essence of the question asked. This serves as a log to avoid repetition.
            -Question Diversification: Before generating a new question, review the log of previously asked questions. If a user's response to a previous question was negative, ensure the next question explores a different aspect of the information needed. This aims to cover various facets of the incident or information required for the legal document without circling back to already answered inquiries.
            -Contextual Shifts: In cases where multiple negative responses have led to a dead end in a particular line of questioning, intentionally shift the context. 
            -Regular Review: Periodically review the interaction flow to identify patterns of negative responses. Use these insights to adjust the direction of questioning, focusing on areas yet to be explored or that might yield more productive responses.
            """
        else:
            section_name = await chat_section_finder(session_data=session_data, previous_chat_messages=previous_chat_messages)
            section_content = [section["content"]
                               for section in temp_agreement if section['section_name'] == section_name]
            system_content = f"""
                        As an agreement writer assistant, you're currently assisting in the revision of a legal agreement under Indian law based on user feedback.
                        Your task involves interacting with the user to understand their needs for modifying specific sections or content within the agreement.
                        The agreement includes sections such as {section_names}, and you've been discussing modifications related to {section_name}.
                            Below is past "section_content" of "section_name" being discussed by the user:
                                {section_name} Section Content: {section_content}
                        Consider the {previous_chat_messages} to understand the user's feedback or request to change what modifications he want to change in the "section_content".
                        Tell the user already you have this in the "section_content" and ask what and where you would like to change in the "section_content".
                        ### Instructions ###
                        - Feedback Integration: After considering the past interactions "previous_chat_messages", summarize the user's feedback or requested changes to ensure clarity on what modifications are sought in the "section_content".
                        - Clarification and Confirmation: Communicate to the user what is currently included in the "section_content" and ask for detailed instructions on what and where they would like changes to be made. This promotes a precise understanding of the desired revisions.
                        - Iterative Interaction: Continue asking for the user's preferences on modifications in an iterative manner. Encourage the user to specify their changes until they feel all desired revisions have been adequately addressed.
                        - Explicit Tracking: After each user interaction, document the essence of the feedback or request. This serves as a log to ensure that revisions are accurately captured and to avoid overlooking any modifications.
                        - Question Diversification: Before posing a new question or suggestion, review the log of previously discussed revisions. If the user has declined a suggestion or indicated satisfaction with a section, move on to explore different areas of the agreement that may need adjustment.
                        - Contextual Shifts: If it becomes evident that no further modifications are desired in the current section, propose moving on to another section that has not yet been reviewed or where the user previously expressed a desire for changes.
                        - Regular Review: Periodically summarize the modifications agreed upon or discussed so far. This helps both you and the user keep track of the changes and ensures that the final document reflects all desired amendments.
                        Remember, your goal is to facilitate the user in refining the agreement to their satisfaction, ensuring all legal and personal preferences are accurately incorporated.
                        """

        overall_status = False

        messages = [
            {"role": "system", "content": system_content},
        ]

        messages.extend(previous_chat_messages)

        logger.debug(f"Messages: {messages}")

        assistant_response = await llm_call.openai_llm_call(
            messages=messages,
            structured_function=generate_question,
            module_name="Agreement Writer",
            model="gpt-4-1106-preview"
        )
        logger.debug(assistant_response)

        is_question = assistant_response.get('is_question', None)
        if is_question:
            if is_question == False:
                assistant_response = "Based on the information provided, it seems that I have sufficient details to proceed."
                overall_status = True
            else:
                assistant_response = assistant_response['question']

        else:
            assistant_response = assistant_response['question']

        previous_chat_messages.extend(
            [
                {"role": "assistant", "content": assistant_response}
            ]
        )
        chat_data = {
            "chat_id": chat_id,
            "chat_messages": previous_chat_messages,
            "overall_status": overall_status
        }

        await task_status.update_task_status(chat_id, chat_data)

    except Exception as e:
        logger.error(f"An error occurred: {str(traceback.format_exc())}")


@timer_decorator
async def process_chat_cross_examination(previous_chat_messages: list, session_data: list):

    try:
        chat_id = session_data['chat_id']
        summary = session_data['summary']

        system_content = f"""
        You are tasked with preparing for cross-examinations in the realm of Indian legal law. Utilize the 'generate_question' function to interact with users, with the aim of collecting all necessary details to formulate effective cross-examination strategies. Your primary objectives are:

        1. Identify and gather all essential information from the user required for the cross-examination process. This includes, but is not limited to, the nature of the dispute, relevant facts, dates, and any specific allegations or defenses.
        2. Ensure you have a comprehensive understanding of the user's case and objectives. If any details are missing or clarification is needed, generate a question. Remember to generate only one question at a time, prioritizing the acquisition of the most critical information missing.

        ### Instructions ###
        -Explicit Tracking: After each user interaction, record the essence of the question asked. This serves as a log to avoid repetition.
        -Question Diversification: Before generating a new question, review the log of previously asked questions. If a user's response to a previous question was negative, ensure the next question explores a different aspect of the information needed. This aims to cover various facets of the incident or information required for the legal document without circling back to already answered inquiries.
        -Contextual Shifts: In cases where multiple negative responses have led to a dead end in a particular line of questioning, intentionally shift the context. 
        -Regular Review: Periodically review the interaction flow to identify patterns of negative responses. Use these insights to adjust the direction of questioning, focusing on areas yet to be explored or that might yield more productive responses.

        Note: Your goal is to efficiently collect the necessary information for an effective cross-examination strategy, ensuring the generated question is pertinent, succinct, and crucial for understanding the case at hand.
        """
        overall_status = False
        print(f"the previous chat messages: {previous_chat_messages}")

        if summary:
            system_content = system_content + \
                f"Below is document provided by user : {summary}"
        messages = [
            {"role": "system", "content": system_content}
        ]

        messages.extend(previous_chat_messages)

        logger.debug(f"Messages: {messages}")
        assistant_response = await llm_call.openai_llm_call(
            messages=messages,
            structured_function=generate_question,
            module_name="Cross Examination",
            model="gpt-4-1106-preview"
        )
        logger.debug(assistant_response)
        is_question = assistant_response.get('is_question', None)
        if is_question:
            if is_question == False:
                assistant_response = "Based on the information provided, it seems that I have sufficient details to proceed."
                overall_status = True
            else:
                assistant_response = assistant_response['question']

        else:
            assistant_response = assistant_response['question']

        previous_chat_messages.extend(
            [
                {"role": "assistant", "content": assistant_response}
            ]
        )
        chat_data = {
            "chat_id": chat_id,
            "chat_messages": previous_chat_messages,
            "overall_status": overall_status
        }

        await task_status.update_task_status(chat_id, chat_data)

    except Exception as e:
        logger.error(f"An error occurred: {str(traceback.format_exc())}")


@timer_decorator
async def process_chat_case_research(previous_chat_messages: list, session_data: list):

    try:
        chat_id = session_data['chat_id']
        summary = session_data['summary']

        system_content = f"""
        You are tasked with performing a research on a case in the realm of Indian legal law. 
        Utilize the 'generate_question' function to interact with users, with the aim of collecting all necessary details 
        to formulate effective case research strategies. 
        
        Your primary objectives are:

        1. Identify and gather all essential information from the user required for the Case Research process. 
        This includes, but is not limited to, the nature of the dispute, relevant facts, dates, and any specific allegations or defenses.
        2. Ensure you have a comprehensive understanding of the user's case and objectives. If any details are missing or clarification is needed, 
        generate a question. Remember to generate only one question at a time, prioritizing the acquisition of the most critical information missing.

        ### Instructions ###
        -Explicit Tracking: After each user interaction, record the essence of the question asked. This serves as a log to avoid repetition.
        -Question Diversification: Before generating a new question, review the log of previously asked questions. If a user's response to a previous question was negative, ensure the next question explores a different aspect of the information needed. This aims to cover various facets of the incident or information required for the legal document without circling back to already answered inquiries.
        -Contextual Shifts: In cases where multiple negative responses have led to a dead end in a particular line of questioning, intentionally shift the context. 
        -Regular Review: Periodically review the interaction flow to identify patterns of negative responses. Use these insights to adjust the direction of questioning, focusing on areas yet to be explored or that might yield more productive responses.

        """
        overall_status = False
        print(f"the previous chat messages: {previous_chat_messages}")

        if summary:
            system_content = system_content + \
                f"Below is document provided by user : {summary}"
        messages = [
            {"role": "system", "content": system_content}
        ]

        messages.extend(previous_chat_messages)

        logger.debug(f"Messages: {messages}")
        assistant_response = await llm_call.openai_llm_call(
            messages=messages,
            structured_function=generate_question,
            module_name="Case Research",
            model="gpt-4-1106-preview"
        )
        logger.debug(assistant_response)
        is_question = assistant_response.get('is_question', None)
        if is_question:
            if is_question == False:
                assistant_response = "Based on the information provided, it seems that I have sufficient details to proceed."
                overall_status = True
            else:
                assistant_response = assistant_response['question']

        else:
            assistant_response = assistant_response['question']

        previous_chat_messages.extend(
            [
                {"role": "assistant", "content": assistant_response}
            ]
        )
        chat_data = {
            "chat_id": chat_id,
            "chat_messages": previous_chat_messages,
            "overall_status": overall_status
        }

        await task_status.update_task_status(chat_id, chat_data)

    except Exception as e:
        logger.error(f"An error occurred: {traceback.format_exc()}")
