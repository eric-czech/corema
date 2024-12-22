from openai import OpenAI
from corema.config import get_config
from corema.utils.openai_models import DEFAULT_MODEL


def test_openai_api() -> None:
    # Get API key from config
    config = get_config()
    api_key = config.get("OPENAI_API_KEY")

    if not api_key:
        print("No API key found in config")
        return

    # Initialize client
    client = OpenAI(api_key=api_key)

    try:
        # Make a simple request
        response = client.chat.completions.create(
            # model="gpt-4-turbo-preview",
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            temperature=0,
        )

        # Print response
        print("Response:", response.choices[0].message.content)
        print("Model used:", response.model)
        print("Usage:", response.usage)

    except Exception as e:
        print("Error:", str(e))


if __name__ == "__main__":
    test_openai_api()
