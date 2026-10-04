from __future__ import annotations


REACT_PAPER_WEBSHOP_PROMPT1 = """Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
Observation: OK.

Action: click[B078GWRC1J]
Observation: 
[Back to Search] 
[< Prev] 
scent [assorted scents][bright citrus][calming lavender][ginger fresh][simply non-scents]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
Price: $10.99 
Rating: N.A. 
[Description] 
[Features] 
[Reviews] 
[Buy Now]  

Action: think[For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus. 

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1). 

Action: click[Buy Now]
"""


WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


WEBSHOP_FEWSHOT_HEADER = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.

Use these rules:
- For search, use a short keyword query with the product type and one or two core descriptors.
- Do not put price filters, size filters, color codes, or every requirement into the search query.
- If search results are zero, go back and broaden the query.
- On a product page, select the required color, size, style, or other options from the available click actions before buying.
- Click Buy Now only after the visible product and selected options match the task as well as possible.
- Your final line must contain exactly one action in this format: <action>search[keywords]</action> or <action>click[option]</action>.

Example 1:
Task: Find me machine wash men's dress shirts with cotton spandex, classic fit, short sleeve with color: patina green, and size: x-large, and price lower than 60.00 dollars.
Observation: 'Search'
Available actions:
[
'search[<your query>]',
'click[search]',
]
Response:
<think>I should start broad. The searchable product type is men's dress shirts; color and size can be selected later on a product page.</think>
<action>search[men's dress shirts]</action>

Example 2:
Task: Find me hand crafted candy & chocolate for gift set, valentine day with style: holiday truffle chocolate, and price lower than 40.00 dollars.
Observation: 'Back to Search' [SEP] 'style' [SEP] 'happy birthday truffle chocolate' [SEP] 'holiday truffle chocolate' [SEP] 'truffle collection chocolate' [SEP] 'CRAVINGS BY ZOE Valentine's Day Chocolate Truffles' [SEP] 'Price: $24.95' [SEP] 'Buy Now'
Available actions:
[
'click[back to search]',
'click[happy birthday truffle chocolate]',
'click[holiday truffle chocolate]',
'click[truffle collection chocolate]',
'click[buy now]',
]
Response:
<think>The product is a Valentine's chocolate gift under budget. I must select the requested style before buying.</think>
<action>click[holiday truffle chocolate]</action>

Example 3:
Task: Find me machine wash men's t-shirts with long sleeve with color: country, and size: 4x-large big, and price lower than 70.00 dollars.
Observation: 'Back to Search' [SEP] 'color' [SEP] 'country' [SEP] 'mossy oak country' [SEP] 'size' [SEP] '4x-large big' [SEP] 'Legendary Whitetails Men's Non-Typical Long Sleeve T-Shirt' [SEP] 'Price: $10.52 to $40.5' [SEP] 'Buy Now'
Available actions:
[
'click[back to search]',
'click[country]',
'click[mossy oak country]',
'click[4x-large big]',
'click[buy now]',
]
Response:
<think>The product is a long sleeve men's t-shirt under budget. The requested color and size are visible, so I should select any missing required option before buying.</think>
<action>click[country]</action>
"""


WEBSHOP_FEWSHOT_PAGEAWARE_HEADER = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.

Use these rules:
- You must choose an action that is valid in the current Available actions.
- Only use search[...] when Available actions contains search[<your query>].
- If Available actions does not contain search[<your query>], do not output search[...].
- If the page shows Total results: 0 and Available actions contains click[back to search], output click[back to search].
- After returning to the Search page, use a broader short keyword query.
- On result pages, click a product or navigation action from Available actions instead of issuing search directly.
- For search, use a short keyword query with the product type and one or two core descriptors.
- Do not put price filters, size filters, color codes, or every requirement into the search query.
- On a product page, select the required color, size, style, or other options from the available click actions before buying.
- Click Buy Now only after the visible product and selected options match the task as well as possible.
- Your final line must contain exactly one action in this format: <action>search[keywords]</action> or <action>click[option]</action>.

Example 1:
Task: Find me machine wash men's dress shirts with cotton spandex, classic fit, short sleeve with color: patina green, and size: x-large, and price lower than 60.00 dollars.
Observation: 'Search'
Available actions:
[
'search[<your query>]',
'click[search]',
]
Response:
<think>I should start broad. The searchable product type is men's dress shirts; color and size can be selected later on a product page.</think>
<action>search[men's dress shirts]</action>

Example 2:
Task: Find me hand crafted candy & chocolate for gift set, valentine day with style: holiday truffle chocolate, and price lower than 40.00 dollars.
Observation: 'Back to Search' [SEP] 'style' [SEP] 'happy birthday truffle chocolate' [SEP] 'holiday truffle chocolate' [SEP] 'truffle collection chocolate' [SEP] 'CRAVINGS BY ZOE Valentine's Day Chocolate Truffles' [SEP] 'Price: $24.95' [SEP] 'Buy Now'
Available actions:
[
'click[back to search]',
'click[happy birthday truffle chocolate]',
'click[holiday truffle chocolate]',
'click[truffle collection chocolate]',
'click[buy now]',
]
Response:
<think>The product is a Valentine's chocolate gift under budget. I must select the requested style before buying.</think>
<action>click[holiday truffle chocolate]</action>

Example 3:
Task: Find me machine wash men's t-shirts with long sleeve with color: country, and size: 4x-large big, and price lower than 70.00 dollars.
Observation: 'Back to Search' [SEP] 'color' [SEP] 'country' [SEP] 'mossy oak country' [SEP] 'size' [SEP] '4x-large big' [SEP] 'Legendary Whitetails Men's Non-Typical Long Sleeve T-Shirt' [SEP] 'Price: $10.52 to $40.5' [SEP] 'Buy Now'
Available actions:
[
'click[back to search]',
'click[country]',
'click[mossy oak country]',
'click[4x-large big]',
'click[buy now]',
]
Response:
<think>The product is a long sleeve men's t-shirt under budget. The requested color and size are visible, so I should select any missing required option before buying.</think>
<action>click[country]</action>

Example 4:
Task: Find me washable women's walking shoes with color: black, and size: 8, and price lower than 60.00 dollars.
Observation: 'Back to Search' [SEP] 'Page 1 (Total results: 0)' [SEP] 'Next >'
Available actions:
[
'click[back to search]',
'click[next >]',
]
Response:
<think>The current page has zero results and search is not an available action. I must go back to the Search page before trying a broader query.</think>
<action>click[back to search]</action>
"""


WEBSHOP_FEWSHOT_TEMPLATE_NO_HIS = WEBSHOP_FEWSHOT_HEADER + """
Now solve the current task.
Task: {task_description}
Observation: {current_observation}
Available actions:
[
{available_actions}
]
Response:
"""


WEBSHOP_FEWSHOT_PAGEAWARE_TEMPLATE_NO_HIS = WEBSHOP_FEWSHOT_PAGEAWARE_HEADER + """
Now solve the current task.
Task: {task_description}
Observation: {current_observation}
Available actions:
[
{available_actions}
]
Response:
"""


WEBSHOP_FEWSHOT_PAGEAWARE_TEMPLATE = WEBSHOP_FEWSHOT_PAGEAWARE_HEADER + """
Now solve the current task.
Task: {task_description}
Recent history:
{action_history}
Current step: {current_step}
Observation: {current_observation}
Available actions:
[
{available_actions}
]
Response:
"""


WEBSHOP_REFLECTION_TEMPLATE = """
You are a conservative WebShop action reviewer.

Your job is not to solve the task from scratch. Your job is to decide whether
the proposed action is clearly unsafe or clearly unhelpful, using the current
observation, recent history, and the predicted next observation.

Task:
{task_description}

Current observation:
{current_observation}

Recent history:
{recent_history}

Current available actions:
[
{available_actions}
]

Proposed action:
{proposed_action}

Predicted next observation if the proposed action is executed:
{predicted_next_observation}

Predicted reward: {predicted_reward}
Predicted done: {predicted_done}

Decision rules:
- Default decision is KEEP.
- Change the proposed action only if there is concrete evidence that it is bad.
- Treat the predicted next observation as imperfect. Do not reject an action based on speculation.
- The replacement action must be executable from the CURRENT state, not from the predicted next state.
- Only output search[...] if search[<your query>] is currently available.
- Only output click[...] if the clicked label appears in Current available actions.
- If keeping the proposed action, the final action must be exactly the proposed action.

Reject the proposed action only in these cases:
1. It will likely produce zero search results because the query is too long or over-specific.
2. It repeats a recent failed loop, such as search -> zero results -> back to search -> similar search.
3. It clicks a product that clearly mismatches the task type or required constraints.
4. It clicks Buy Now before required visible options such as size, color, style, flavor, or pack count are selected.
5. It navigates away from a useful product page without a clear reason.
6. It is invalid or unavailable in the current state.

Search guidance:
- Good search queries are short: product type plus one or two core descriptors.
- Do not include price limits, exact size, color codes, long material lists, or every task constraint in search.
- If a previous query got zero results, broaden it by removing constraints.

Product-page guidance:
- If the product type and price look reasonable, prefer selecting missing required options over going back.
- If all required visible options are selected and the product matches the task well enough, Buy Now is appropriate.
- Do not demand perfect evidence for hidden attributes if the visible product title/details already match well.

Output format:
First write one short reason inside <think>...</think>.
Then output exactly one final action inside <action>...</action>.
"""


WEBSHOP_FEWSHOT_TEMPLATE = WEBSHOP_FEWSHOT_HEADER + """
Now solve the current task.
Task: {task_description}
Recent history:
{action_history}
Current step: {current_step}
Observation: {current_observation}
Available actions:
[
{available_actions}
]
Response:
"""
