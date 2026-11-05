import os
import json
import ast
import requests

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Recommended to avoid strict API rate limits
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

# Query searching for functions decorated with @tool across GitHub Python files
SEARCH_QUERY = "language:python @tool def"
MAX_TOOLS = 50  # Target number of tools to extract[cite: 3]


def parse_tool_code(code_str: str) -> list[dict]:
    """
    Parses Python file content to extract functions with a docstring and `@tool` decorator.
    Returns a list of dicts with 'interface' (description) and 'code' (implementation).
    """
    tools = []
    try:
        tree = ast.parse(code_str)
    except Exception:
        return []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if function has a docstring and any decorator matching 'tool'
            has_tool_decorator = any(
                (isinstance(dec, ast.Name) and dec.id == "tool") or
                (isinstance(dec, ast.Attribute) and dec.attr == "tool") or
                (isinstance(dec, ast.Call) and (
                    (isinstance(dec.func, ast.Name) and dec.func.id == "tool") or
                    (isinstance(dec.func, ast.Attribute) and dec.func.attr == "tool")
                ))
                for dec in node.decorator_list
            )
            docstring = ast.get_docstring(node)

            if has_tool_decorator and docstring:
                func_code = ast.unparse(node)
                tools.append({
                    "interface": docstring.strip(),
                    "code": func_code.strip()
                })
    return tools


def fetch_github_tools(target_count: int = 50) -> list[dict]:
    """Queries GitHub REST API to find Python tool implementations."""
    collected_tools = []
    url = f"https://api.github.com/search/code?q={SEARCH_QUERY}&per_page=30"
    page = 1

    while len(collected_tools) < target_count:
        response = requests.get(f"{url}&page={page}", headers=HEADERS)
        if response.status_code != 200:
            print(f"API Request Failed: {response.status_code} - {response.json().get('message')}")
            break

        items = response.json().get("items", [])
        if not items:
            break

        for item in items:
            raw_url = item["html_url"].replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            code_res = requests.get(raw_url, headers=HEADERS)
            
            if code_res.status_code == 200:
                extracted = parse_tool_code(code_res.text)
                for tool in extracted:
                    collected_tools.append(tool)
                    print(f"Extracted Tool {len(collected_tools)}/{target_count}")
                    if len(collected_tools) >= target_count:
                        break

        page += 1

    return collected_tools[:target_count]


if __name__ == "__main__":
    print(f"Pulling {MAX_TOOLS} tool definitions from GitHub...")
    tools_dataset = fetch_github_tools(target_count=MAX_TOOLS)

    output_filename = "github_tools_dataset.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(tools_dataset, f, indent=2)

    print(f"Successfully saved {len(tools_dataset)} tools to {output_filename}")