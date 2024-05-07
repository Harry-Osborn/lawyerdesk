import asyncio
import functools

from bare_functions.supabase_class import SupabaseAgent

supabase_agent = SupabaseAgent()


def async_lru_cache(maxsize=128, typed=False):
    def decorator(func):
        cache = functools.lru_cache(maxsize=maxsize, typed=typed)(func)

        @functools.wraps(func)
        async def wrapped(*args, __cache=cache, **kwargs):
            return await asyncio.get_event_loop().run_in_executor(None, functools.partial(__cache, *args, **kwargs))
        return wrapped
    return decorator


# @async_lru_cache(maxsize=32)  # Adjust the cache size as needed
async def fetch_agreement_from_supabase(agreement_id):
    # Your code to fetch data from Supabase, similar to what you currently have
    fetch_data = {
        "input_table_name": "static.legal_agreements",
        "where_conditions": {
            "id": agreement_id
        },
    }
    agreement_details = await supabase_agent.call_supabase_function(
        schemaname="public",
        functionname="get_with_where_conditions",
        params=fetch_data,
    )
    return agreement_details
