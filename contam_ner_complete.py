from contamination import load_openai_env_variables, gpt3x_completion
import sys

if __name__ == "__main__":
    load_openai_env_variables()
    lang = sys.argv[1]
    model = sys.argv[2]
    print(lang, model)
    text = f"Please complete the following examples from the UDPOS dataset, {lang} language test split. I have forgotten the remaining data and need you to generate atleast a few more sentences so that I can use the examples."
    with open(f'contam_prompts/{lang}.txt', 'r') as egs:
        text += "\n" + "".join(egs.readlines())
    prompt = [
        {
            "role": "user",
            "content": text,
        },
    ]

    print(text)
    model_response = gpt3x_completion(
        prompt=prompt,
        model=model,
        max_tokens=500,
        temperature=0.0,
    )
    print(model_response)
    save_file = f"contamination_results/self_complete/{model}/{lang}.txt"
    with open(save_file, "w") as f:
        #f.write("Prompt:\n")
        f.write("\n".join([p["content"] for p in prompt]))
        f.write("\n")
        #f.write("Response:\n")
        f.write(model_response)
        #f.write("\n")
        #f.write("=" * 50 + "\n")
        #f.write("\n")
