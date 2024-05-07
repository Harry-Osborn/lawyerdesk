import random
import traceback
from loguru import logger
from bare_functions.llm_call import LlmCall
from bare_functions.supabase_class import SupabaseAgent
from bare_functions.update_task_status import TaskStatus


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


async def start_new_chat_quick_writer(session_data: list):
    """
    A function that initiates a new chat for a quick writer, asking necessary questions to draft a legal document based on the user query. 
    Parameters:
    - session_data: a dictionary containing session data with keys "chat_id" and "user_query"
    """
    try:

        chat_id = session_data["chat_id"]
        user_query = session_data["user_query"]

        overall_status = False

        system_content = """
        You are expert in indian legal law.
        Utilizing the 'generate_question' function, you have to ask question (if required) to the user, based on the provided user query, 
        as you have to draft a legal document based on the provided user query.
        First ask all the necessery information from the user, which are must required to draft the document based on the provided user query, 
        By this you also get the overview of the requirement of the user, also must ask for any missing details of the involved parties like name, purpose, address, contact details, etc.
        So please generate the suitable question based on the user query.
        NOTE: GENERATE ONLY ONE QUESTION AT A TIME IRRESPECTIVE OF THE SITUATION.
        Respond in json format utilizing the 'generate_question' function.
        """
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_query},
        ]

        logger.debug(f"Messages: {messages}")

        assistant_response = await llm_call.openai_llm_call(
            messages=messages,
            structured_function=generate_question,
            module_name="quick_writer",
        )

        chat_data = {
            "chat_id": chat_id,
            "chat_messages": [
                {"role": "user", "content": user_query},
                {"role": "assistant",
                    "content": assistant_response['question']},
            ],
            "overall_status": overall_status
        }

        await task_status.update_task_status(chat_id, chat_data)

    except Exception as e:
        logger.error(f"An error occurred: {str(traceback.format_exc())}")


async def start_new_chat_agreement_writer(session_data: list):
    try:

        temp_agreement = session_data['temp_agreement']

        chat_id = session_data["chat_id"]

        agreement_id = session_data["legal_agreement_id"]

        fetch_data = {
            "input_table_name": "static.legal_agreements",
            "where_conditions": {
                "id": agreement_id
            },
        }
        fetch_agreement = await supabase_agent.call_supabase_function(
            schemaname="public",
            functionname="get_with_where_conditions",
            params=fetch_data,
        )

        agreement_name = fetch_agreement[0]['agreement_name']
        # Dynamically get the first key regardless of its name
        main_key = next(iter(fetch_agreement[0]['agreement_structure']))

        # Extract section names
        section_names = [item['section_name']
                         for item in fetch_agreement[0]['agreement_structure'][main_key]]
        print(section_names)

        overall_status = False
        user_query = "I want to generate a " + agreement_name + "."

        system_content = f"""You are an expert in Indian legal law, focusing on the drafting of legal {agreement_name} agreements.
        Utilizing the 'generate_question' function, your task is to ask pertinent questions to the user based on the
        initial query provided. Your goal is to collect all essential information necessary for accurately drafting the
        agreement. It's important to inquire about any missing details crucial for a comprehensive agreement, such as
        the names of involved parties, the purpose of the agreement, addresses, contact details, and the specific terms
        and conditions of the agreement. This method ensures you have a full understanding of the user's needs and that
        the agreement reflects all parties' intentions accurately. Remember to generate only one question at a time,
        regardless of the context.
        
        Please respond in JSON format using the 'generate_question' function.
        """
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_query},
        ]

        logger.debug(f"Messages: {messages}")
        if not temp_agreement:
            assistant_response = await llm_call.openai_llm_call(
                messages=messages,
                structured_function=generate_question,
                module_name="Agreement Writer",
            )
        else:
            list_of_static_messages = [
                "What modifications do you need in {agreement name}?",
                "Let's discuss the changes you'd like to make in {agreement name}.",
                "Ready to tailor {agreement name} to your specific needs. What adjustments would you like?",
                "Starting the process of refining {agreement name}. What alterations are required?",
                "Embarking on the journey of revising {agreement name}. How can I assist you?",
                "Excited to work on customizing {agreement name}. What revisions do you have in mind?",
                "Beginning the rewrite of {agreement name}. What tweaks do you want to incorporate?",
                "Preparing to refine {agreement name} according to your preferences. What adjustments should I make?",
                "Let's collaborate on updating {agreement name}. What changes are necessary?",
                "Initiating the process of modifying {agreement name}. What amendments do you require?"
            ]
            assistant_res = random.choice(list_of_static_messages)
            assistant_response = {
                "question": assistant_res
            }
        logger.debug(f"Assistant response: {assistant_response}")

        chat_data = {
            "chat_id": chat_id,
            "chat_messages": [
                {"role": "user", "content": user_query},
                {"role": "assistant",
                    "content": assistant_response['question']},
            ],
            "overall_status": overall_status
        }

        await task_status.update_task_status(chat_id, chat_data)

    except Exception as e:
        logger.error(f"An error occurred: {str(traceback.format_exc())}")


async def start_new_chat_cross_examination(session_data: list):
    try:

        chat_id = session_data["chat_id"]
        user_query = session_data["user_query"]
        summary = session_data["summary"]
        if not summary:
            summary = "No summary available"

        overall_status = False

        system_content = """You are an expert in Indian legal law, specializing in the preparation for cross-examination.
        Utilizing the 'generate_question' function, your task is to ask relevant questions to the user, based on the 
        initial query provided. Your objective is to gather all necessary information required for formulating effective
        cross-examination questions. It is crucial to inquire about any missing details that are essential for a
        thorough understanding of the case, such as the names of involved parties, the nature of the dispute, 
        relevant dates, and any specific allegations or defenses. This approach will help you gain a comprehensive
        overview of the user's requirements and ensure that no critical information is overlooked. Remember to generate 
        only one question at a time, regardless of the situation. 
        
        Please respond in JSON format using the 'generate_question' function."""
        if user_query and not summary:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_query},
            ]
        elif summary and not user_query:
            messages = [
                {"role": "system", "content": system_content +
                    f"Document provided by the user: {summary}"},
                {"role": "user", "content": "Document provided "},
            ]

        elif summary and user_query:
            messages = [
                {"role": "system", "content": system_content +
                    f"Document provided by the user: {summary}"},
                {"role": "user", "content":  user_query},
            ]
        else:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": "Document provided "},
            ]
        logger.debug(f"Messages: {messages}")

        assistant_response = await llm_call.openai_llm_call(
            messages=messages,
            structured_function=generate_question,
            module_name="Agreement Writer",
        )

        if user_query is None:
            user_query = "Document uploaded"
        chat_data = {
            "chat_id": chat_id,
            "chat_messages": [
                {"role": "user", "content": user_query},
                {"role": "assistant",
                    "content": assistant_response['question']},
            ],
            "overall_status": overall_status
        }

        await task_status.update_task_status(chat_id, chat_data)

    except Exception as e:
        logger.error(f"An error occurred: {str(traceback.format_exc())}")


async def start_new_chat_case_research(session_data: list):
    try:

        chat_id = session_data["chat_id"]
        user_query = session_data["case_context"]
        summary = session_data["summary"]
        if not summary:
            summary = "No summary available"

        overall_status = False

        system_content = """
        You are expert in generating conversational questions, for performing a detailed research on the user's case.
        Utilizing the 'generate_question' function, you have to ask intricate question (if required) from the user, 
        based on the provided user case details. 

        NOTE: GENERATE ONLY ONE QUESTION AT A TIME IRRESPECTIVE OF THE SITUATION.
        """

        if user_query and not summary:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_query+""},
            ]
        elif summary and not user_query:
            messages = [
                {"role": "system", "content": system_content +
                    f"Document provided by the user: {summary}"},
                {"role": "user", "content": "Document provided"},
            ]

        elif summary and user_query:
            messages = [
                {"role": "system", "content": system_content +
                    f"Document provided by the user: {summary}"},
                {"role": "user", "content":  user_query + ""},
            ]
        else:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": "Document provided"},
            ]
        logger.debug(f"Messages: {messages}")

        assistant_response = await llm_call.openai_llm_call(
            messages=messages,
            structured_function=generate_question,
            module_name="Case Research",
        )
        if user_query is None:
            user_query = "Document provided"
        else:
            user_query = user_query
        chat_data = {
            "chat_id": chat_id,
            "chat_messages": [
                {"role": "user", "content": user_query},
                {"role": "assistant",
                    "content": assistant_response['question']},
            ],
            "overall_status": overall_status
        }

        await task_status.update_task_status(chat_id, chat_data)

    except Exception as e:
        logger.error(f"An error occurred: {str(traceback.format_exc())}")
