from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

def main(use_llama = False):
    # 1. Setup Credentials
    # Note: Ensure your environment has the API key if you aren't in a pre-authenticated lab
    credentials = Credentials(
        url="https://us-south.ml.cloud.ibm.com",
        # api_key="YOUR_API_KEY" 
    )

    # 2. Define Parameters
    params = {
        GenTextParamsMetaNames.DECODING_METHOD: "greedy",
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 100
    }

    # 3. Initialize the Model
    if use_llama: 
       model = ModelInference(
        model_id='meta-llama/llama-4-maverick-17b-128e-instruct-fp8',
        params=params,
        credentials=credentials,
        project_id="skills-network"
        )
    else:
        model = ModelInference(
            model_id='ibm/granite-3-3-8b-instruct',
            params=params,
            credentials=credentials,
            project_id="skills-network"
        )

    # 4. Define the Prompt
    text = "Only reply with the answer. What is the capital of Canada?"

    # 5. Generate and Print
    # Using generate_text() directly returns the string, which is simpler!
    response = model.generate_text(prompt=text)
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
    main(use_llama=True)