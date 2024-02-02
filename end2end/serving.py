from text_generation import Client


def main():

    client = Client("http://54.167.93.63:8011")

    # Prompt the base LLM
    prompt = "[INST] Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? [/INST]"
    print(client.generate(prompt, max_new_tokens=64).generated_text)

    # Prompt a LoRA adapter
    adapter_id = "kyryl-georgian/flan-base-sql"
    print(client.generate(prompt, max_new_tokens=64, adapter_id=adapter_id).generated_text)


if __name__ == "__main__":
    main()
