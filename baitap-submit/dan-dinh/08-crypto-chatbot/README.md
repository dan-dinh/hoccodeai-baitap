## Name
Dan Dinh

## Exercise
08-crypto-chatbot.md

## Prerequisites:
- Install dependencies in requirements.txt
- Add key/value pair in the .env file
    - `API_KEY`: `<<Your Groq API Key>>`
    - `MODEL_NAME`: `<<Your Groq Model Name>>`
    - `WEATHER_API_KEY`: `<<Your Weather API Key>>`

## Installation & How-to-test:
1. Run the practice_function_calling_31_62c.py in Terminal to test
2. Test with current weather in `<<somewhere>>` (e.g., Melbourne)
3. Test with stock price of `<<any name>>` (e.g., Vinfast, BTC, ETH)
4. Test with website summary

## Challenges:
1. Sometimes I encountered the error message -
`groq.BadRequestError: Error code: 400 - {'error': {'message': "Failed to call a function. Please adjust your prompt. See 'failed_generation' for more details.", 'type': 'invalid_request_error', 'code': 'tool_use_failed', 'failed_generation': '<function=get_symbol>{"company": "Vinfast", "country": "VN"}</function>\n<function=get_stock_price>{"symbol": "VFS"}></function>'}}`. I tried to address the issue by improving my prompt, but I'm not sure if this is the most effective solution. I would greatly appreciate any additional feedback or insights you may have to help resolve it properly.

2. Do you have any experience regarding Web OpenUI and function callings via API?

## Screenshot or Video:
Demo with getting weather, and stock prices:

![Result](image.png)

## Checklist:
- [x] I tested my code with happy case only.
- [x] I handled only main functions, skipped some of error exception handlings.