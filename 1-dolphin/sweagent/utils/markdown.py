"""
    Utility functions and classes to process markdown input and output.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

import mistune
from bs4 import BeautifulSoup


@dataclass
class CodeBlock:
    language: str
    content: str


@dataclass
class MarkdownElement:
    type: str
    content: str
    level: Optional[int] = None


class MarkdownParser:
    def __init__(self):
        self._parser = mistune.create_markdown()

    def parse(self, markdown_text: str) -> List[MarkdownElement]:
        """Parse markdown text into a list of markdown elements using mistune"""
        # Convert markdown to HTML
        html = self._parser(markdown_text)
        soup = BeautifulSoup(html, "html.parser")

        elements = []

        # Parse headings
        for i in range(1, 7):
            for heading in soup.find_all(f"h{i}"):
                elements.append(
                    MarkdownElement(type="heading", content=heading.get_text(), level=i)
                )

        # Parse code blocks
        for code in soup.find_all("pre"):
            code_content = code.find("code")
            if code_content:
                # Try to get language from class
                classes = code_content.get("class", [])
                language = "text"
                print(classes)
                if classes:
                    # mistune adds 'language-python' like classes
                    lang_class = [c for c in classes if c.startswith("language-")]
                    if lang_class:
                        language = lang_class[0].replace("language-", "")

                elements.append(
                    MarkdownElement(
                        type="code", content=code_content.get_text(), level=None
                    )
                )

        # Parse lists
        for ul in soup.find_all(["ul", "ol"]):
            for li in ul.find_all("li", recursive=False):
                elements.append(MarkdownElement(type="list", content=li.get_text()))

        # Parse paragraphs
        for p in soup.find_all("p"):
            elements.append(MarkdownElement(type="paragraph", content=p.get_text()))

        return elements

    def get_code_blocks(self, markdown_text: str) -> List[CodeBlock]:
        """Extract only code blocks from markdown text"""
        html = self._parser(markdown_text)
        soup = BeautifulSoup(html, "html.parser")

        code_blocks = []
        for code in soup.find_all("pre"):

            code_content = code.find("code")
            if code_content:
                classes = code_content.get("class", [])
                language = "text"
                if classes:
                    lang_class = [c for c in classes if c.startswith("language-")]
                    if lang_class:
                        language = lang_class[0].replace("language-", "")

                code_blocks.append(
                    CodeBlock(language=language, content=code_content.get_text())
                )

        return code_blocks

    def get_json_code(self, markdown_text: str) -> list[str]:
        blocks= self.get_code_blocks(markdown_text)
        res = []
        for b in blocks:
            if b.language == "json":
                res.append(b.content)

        return res


# Precompute the escaped characters pattern
MARKDOWN_CHARS = r'\`*_{}[]()#+-.!|<>'
ESCAPED_MARKDOWN_CHARS = re.escape(MARKDOWN_CHARS)
ESCAPE_MD_REGEX = re.compile(f'([{ESCAPED_MARKDOWN_CHARS}])')

def markdown_escape(text: str) -> str:
    return ESCAPE_MD_REGEX.sub(r'\\\1', text)

def markdown_escape_list(lines: list[str]) -> list[str]:
    new_lines = [markdown_escape(line) for line in lines]
    return new_lines



# ------------------------ Below are test cases -------------------------------------
def test_markdown_parser():
    parser = MarkdownParser()
    markdown_text = """
# Heading 1
This is some text.

```command
print("Hello, world!")
```
## Heading 2
This is some more text.
- List item 1
- List item 2
    """
    elements = parser.parse(markdown_text)
    for element in elements:
        if isinstance(element, CodeBlock):
            print(f"===Code Block (Language: {element.language})===\n{element.content}")
        else:
            print(f"==={element.type}===\n{element.content}")

    elements = parser.get_code_blocks(markdown_text)
    for element in elements:
        if isinstance(element, CodeBlock):
            print(f"===Code Block (Language: {element.language})===\n{element.content}")
        else:
            print(f"==={element.type}===\n{element.content}")


if __name__ == "__main__":
    test_markdown_parser()
