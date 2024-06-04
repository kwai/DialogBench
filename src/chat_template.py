DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your\
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not\
correct. If you don't know the answer to a question, please don't share false information."""


def default_chat_template():
    """
    LLaMA uses [INST] and [/INST] to indicate user messages, and <<SYS>> and <</SYS>> to indicate system messages.
    Assistant messages do not have special tokens, because LLaMA chat models are generally trained with strict
    user/assistant/user/assistant message ordering, and so assistant messages can be identified from the ordering
    rather than needing special tokens. The system message is partly 'embedded' in the first user message, which
    results in an unusual token ordering when it is present. This template should definitely be changed if you wish
    to fine-tune a model with more flexible role ordering!

    The output should look something like:

    <bos>[INST] B_SYS SystemPrompt E_SYS Prompt [/INST] Answer <eos> <bos>[INST] Prompt [/INST] Answer <eos>
    <bos>[INST] Prompt [/INST]
    """

    template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
        "{% set system_message = messages[0]['content'] %}"
        "{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}"
        "{% set loop_messages = messages %}"  # Or use the default system message if the flag is set
        "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}"
        "{% else %}"
        "{% set loop_messages = messages %}"
        "{% set system_message = false %}"
        "{% endif %}"
        "{% for message in loop_messages %}"  # Loop over all non-system messages
        "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
        "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
        "{% endif %}"
        "{% if loop.index0 == 0 and system_message != false %}"  # Embed system message in first message
        "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
        "{% else %}"
        "{% set content = message['content'] %}"
        "{% endif %}"
        "{% if message['role'] == 'user' %}"  # After all of that, handle messages/roles in a fairly normal way
        "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
        "{% elif message['role'] == 'system' %}"
        "{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ ' '  + content.strip() + ' ' + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
    )
    template = template.replace("USE_DEFAULT_PROMPT", "true" if True else "false")
    default_message = DEFAULT_SYSTEM_PROMPT.replace("\n", "\\n").replace("'", "\\'")
    template = template.replace("DEFAULT_SYSTEM_MESSAGE", default_message)

    return template
