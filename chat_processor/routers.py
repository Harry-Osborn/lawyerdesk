from http import HTTPStatus
import traceback
from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger
from modules.chat_processor.background_tasks import chat_process_background


router = APIRouter()


@router.post("/process_chat")
async def chat_processor_endpoint(
    module_id: int,
    session_id: int,
    message: str = None,
    overall_status: bool = False,
    background_tasks: BackgroundTasks = None
):
    """
    # CHAT PROCESSING ROUTE

    - This will take input the module_id and session_id of the current session and process the chat message.
    - Only process if there is a chat_id in the chat_id column of the provided module id sessions column.
    - message can be none, for the case if user needs to skips the chat, and then you have set the overall_status to true.
    - Once You will hit the endpoint with overall status as true it will stop the chat processing, and give the final response for that module's session.

    # Sample Request

    ```json
    {
        "module_id": 4,
        "session_id": 1,
        "message": "Hello",
        "overall_status": false
    }
    ```

    # Sample Response

    ```json
    {
        "message": "Chat processing initiated"
    }
    ```

    # ACCEPTED MODULES BELOW

    ```json
    [
        {
            "module_name": "quick_writer",
            "module_id": 32,
            "table_name": "doc_writer.quick_writer"
        },
        {
            "module_name": "cross_examination",
            "module_id": 33,
            "table_name": "cross_examination.cross_examin"
        },
        {
            "module_name": "agreement_writer",
            "module_id": 16,
            "table_name": "documentation.agreement_writer"
        },
        {
            "module_name": "case_research",
            "module_id": 3,
            "table_name": "research.legal_research"
        },
        {
            "module_name": "document_writer",
            "module_id": 4,
            "table_name": "doc_writer.sessions"
        }
    ]
    ```
    """

    try:

        background_tasks.add_task(
            chat_process_background,
            module_id,
            session_id,
            message,
            overall_status
        )

        logger.debug(
            f"Processing new message in background."
        )
        return {"message": "Chat processing initiated"}

    except Exception as e:
        logger.critical(
            f"Error processing new message in background: {traceback.format_exc(e)}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e
