"""LLM prompt template for web crawler navigation.

A single NAVIGATION_PROMPT drives every navigation step.

The caller injects two things via format kwargs:
  {objective}  — what to find and what URL to use when FOUND
  {hints}      — optional site-specific guidance (empty string if none)

All other placeholders ({visited}, {current_url}, {content}) are standard.
"""

NAVIGATION_PROMPT = """\
Below is the full rendered content of a web page in markdown format.
URLs appear as markdown links, e.g. [anchor text](https://...) or bare https://... strings.

{objective}
{hints}
Rules:
1. A direct match for the objective always takes priority over navigating to a section page.
2. If the page appears empty or JS-rendered (no content visible), look for navigation links
   in menus or tabs and use NEXT — do not give up with NONE while unvisited links remain.

!! CRITICAL — you MUST NOT suggest any URL that appears in the already-visited list below.
   If all promising links have already been visited, return NONE immediately. !!

Already visited:
{visited}

Set action to:
  "FOUND" with the URL — objective achieved (see objective above for what URL to set)
  "NEXT"  with the URL — most promising UNVISITED link toward the objective
  "NONE"  with url null — no unvisited links remain toward the objective

Always set confidence_message to a brief explanation:
  - For "FOUND": what you found and why it satisfies the objective
  - For "NEXT": which link you chose and why it is the most promising unvisited path
  - For "NONE": what you looked for, every link you considered, and why none could lead there

Current page: {current_url}

Page content:
{content}
"""
