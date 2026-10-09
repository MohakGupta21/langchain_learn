import re
import inspect
import ollama
from langsmith import traceable

#React Prompt

MODEL = 'gemma3:4b'
MAX_ITERATIONS=10

def get_product_price(product:str) -> float:
    "Look up the price of a product in the catalog"
    prices = {"laptop":1000, "tv":800, "refrigerator":20}
    print(f"The price of the product:{product} is {prices.get(product)}")
    return prices.get(product)

def apply_discount(price:float, discount_tier:str)-> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f".    Executing apply_discount(price:{price}, discount_tier:{discount_tier})")
    discount_percentages = {"bronze":5, "silver":12, "gold":23}
    discount = discount_percentages.get(discount_tier,0)
    return round(price * (1- discount/100),2)


def get_tool_description(tools_dict):
    descriptions=[]
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tools = {"get_product_price":get_product_price, "apply_discount":apply_discount}
tool_descriptions = get_tool_description(tools)
tool_names = ", ".join(tools.keys())

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model,messages, options):
    return ollama.chat(model=model, messages=messages, options=options)

react_prompt=f"""
 STRICT RULES - you must follow these instructions:
                1. Never guess or assume any product price
                You must call get_product_price first to get the real price.\n
                2. Only call apply_discount AFTER you have received 
                a price from get_product_price. Pass the exact price 
                returned by get_product_price - do NOT pass a made-up number. \n
                3. Never calculate discounts yourself using math. 
                Always use the apply_discount_tool.\n
                4. If the user does not specify a discount tier, 
                ask them which tier to use - do not assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{input}}
Thought:
"""

def run_agent(question:str):
    print(f"Question: {question}")
    print("="*60)

    # CHANGE 5: One prompt string replaces the system/user message split
    prompt = react_prompt.format(input=question)
    scratchpad = ""
    for iteration in range(1,MAX_ITERATIONS+1):
        print(f"\n--- Iteration {iteration} ----")
        full_prompt = prompt + scratchpad
        # Difference 5: ollama.chat() directly instead of llm_with_tools.invoke()
        response = ollama_chat_traced(
            model = MODEL,
            messages = [{"role":"user", "content":full_prompt}],
            options={"stop":["\nObservation"], "temperature":0}
        )
        output = response.message.content
        print(f"LLM Output:\n{output}")

        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)

        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("\n"+"="*60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        action_match = re.search(r"Action:\s*(\w+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print("No action found in output, stopping.")
            return output

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f". [Tool Selected] {tool_name} with args: {tool_input_raw}")

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.strip("w",1)[-1].strip().strip("'\'") for x in raw_args]

    
        print(f"  [Tool Executing] {tool_name}{{args}}...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list[str][tools.keys()]}"
        else:
            observation = str(tools[tool_name][*args])
        print(f"  [Tool Result] {observation}")

        scratchpad =f"{output}\nObservation: {observation}\nThought:"
if __name__=="__main__":
    run_agent(question="What is the price of laptop with silver discount tier?")
