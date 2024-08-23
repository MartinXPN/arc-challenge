"""
The extraction module is responsible for extracting Python code from the raw text output of a model.
"""
from textwrap import dedent


def extract_code(text: str) -> str:
    """
    Extract the Python code from the text output.
    :param text: The text output
    :return: The extracted Python code
    """
    # Find the first ```python
    start = text.find('```python')
    if start == -1:
        # Find the first ```
        start = text.find('```')
    if start == -1:
        return ''

    # Go to the next line
    start = text.find('\n', start)
    if start == -1:
        return ''

    # Find the next ```
    end = text.find('```', start)
    if end == -1:
        end = len(text)

    return text[start:end].strip()


if __name__ == '__main__':
    print(extract_code(dedent('''
    This is some random text.
    ```
    def foo():
        print('Hello, World!')
    foo()
    ''').strip()))
    print('-' * 50)

    print(extract_code(dedent('''
    This is some random text.
    ```
    
    def foo():
        print('Hello, World!')
    foo()
    ```
    ''').strip()))
    print('-' * 50)

    print(extract_code(dedent('''
    This is some random text.
    ```python
    def foo():
        print('Hello, World!')
    foo()
    ```
    ''').strip()))
    print('-' * 50)

    print(extract_code(dedent('''
    This is some distraction text.
    ```
    ...
    ```
    This is some real code:
    ```python
    def foo():
        print('Hello, World!')
    foo()
    ```
    ''').strip()))
    print('-' * 50)

    # Empty code
    print(extract_code(dedent('''
    This is some distraction text.
    ''').strip()))
    print('-' * 50)

    # No code block
    print(extract_code(dedent('''
    This is some distraction text.
    def foo():
        print('Hello, World!')
    foo()
    ''').strip()))
    print('-' * 50)
