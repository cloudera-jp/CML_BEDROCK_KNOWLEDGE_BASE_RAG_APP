import gradio as gr
import os
import json
from utils import bedrock
from utils import bedrock_agent

  
# Initializing the bedrock client using AWS credentials
boto3_bedrock = bedrock.get_bedrock_client(
    region=os.environ.get("AWS_DEFAULT_REGION", None))
boto3_bedrock_agent = bedrock_agent.get_bedrock_client(
    region=os.environ.get("AWS_DEFAULT_REGION", None))

accept = 'application/json'
contentType = 'application/json'

with open('amp_2_app/example.txt', 'r') as file:
    example_text = file.read()
examples = {'CML Documentation': example_text}
def example_lookup(text):
  if text:
    return examples[text]
  return ''

# example_instruction = "Please provide a summary of the following text. Do not add any information that is not mentioned in the text below."
example_instruction = "以下のテキストのJSONから発電所の設備名称として適当なものだけを選んで、同じようなJSON形式で回答してください。"

def clear_out():
  cleared_tuple = (gr.Textbox.update(value=""), gr.Textbox.update(value=""), gr.Textbox.update(value=""), gr.Textbox.update(value=""))
  return cleared_tuple

# List of LLM models to use for text summarization
models = ['amazon.titan-tg1-large', 'anthropic.claude-v2:1', 'anthropic.claude-3-5-sonnet-20240620-v1:0']


#
# retrieveの実行
#
def retrieve_from_knowledge_base(query, knowledge_base_id, max_results=3):
    try:
        response = boto3_bedrock_agent.retrieve(
            knowledgeBaseId=knowledge_base_id,
            retrievalQuery={
                'text': query
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': max_results
                }
            }
        )
        return response['retrievalResults']
    except Exception as e:
        print(f"Error in retrieve: {str(e)}")
        return None

#
# Generate部の実行
#
def invoke_model(model_id, prompt, max_tokens, temperature, top_p):
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": prompt,
            "temperature": temperature,
            "top_p": top_p,
        },ensure_ascii=False, indent=2)
        print(f"Debug Prompt: {prompt}")
        print(f"Debug Body: {body}")
        response = boto3_bedrock.invoke_model(
            modelId=model_id,
            body=body
        )
        response_body = json.loads(response['body'].read())
        
        # Extract the output from the API response for the corresponding model
        if model_id == 'amazon.titan-tg1-large':
          result = response_body.get('results')[0].get('outputText')
        elif model_id == 'anthropic.claude-v2:1':
          result = response_body.get('completion')
        elif model_id == 'anthropic.claude-3-5-sonnet-20240620-v1:0':
          result = response_body['content'][0]['text']
        
        return result
    except Exception as e:
        print(f"Error in invoke_model: {str(e)}")
        return None

#
# retrieve and generate の実行
#
# Base:                    summarize(modelId, input_text,        instruction_text, max_tokens, temperature, top_p):
# Origin: get_knowledgeable_response(query,   knowledge_base_id, model_id):
#
def get_knowledgeable_response(knowledge_base_id, model_id, query, input_text, max_tokens, temperature, top_p):
    # ナレッジベースから情報を取得
    retrieved_info = retrieve_from_knowledge_base(query, knowledge_base_id)
    
    if not retrieved_info:
        return "申し訳ありませんが、関連情報を取得できませんでした。"

    # 取得した情報をコンテキストとして整形
    knowledge_context = [
        {
            "content": result['content'],
            "source": result['location'],
            "score": result['score']
       } for result in retrieved_info
    ]
    
    # PASS2 # knowledge_context = "\n".join([result['content']['text'] for result in retrieved_info])
    
    print(f"Debug knowledge_context: {knowledge_context}")

    # モデルに問い合わせ
    # full_prompt = prompt_construction(modelId, instruction_text, input_text)
    # body = json_format(modelId, max_tokens, temperature, top_p, full_prompt)
    prompt = [
        {
          "role": "user", 
          "content": f"質問: {query}\n\n提供情報:\n{json.dumps(knowledge_context, ensure_ascii=False, indent=2)}\n\n入力データ:\n{input_text}"
        }
    ]
    # PASS1 # "content": "質問:" + query + "\n\n提供情報:\n\n入力データ:\n<text>" + input_text + "</text>"
    # PASS2 # content": f"質問: {query}\n\n提供情報:\n{knowledge_context}\n\n入力データ:\n{input_text}"
    # "content": f"質問: {query}\n\n提供情報:\n{json.dumps(knowledge_context, ensure_ascii=False, indent=2)}\n\n入力データ:\n{input_text}"
    
    model_response = invoke_model(model_id, prompt, max_tokens, temperature, top_p)

    if model_response:
        return model_response
    else:
        return "申し訳ありませんが、回答を生成できませんでした。"


# Setting up the prompt syntax for the corresponding model
def prompt_construction(modelId, instruction="[instruction]", prompt="[input_text]"):
  if modelId == 'amazon.titan-tg1-large':
    full_prompt = instruction + """\n<text>""" + prompt + """</text>"""
  elif modelId == 'anthropic.claude-v2:1':
    full_prompt = """Human: """ + instruction + """\n<text>""" + prompt + """</text>
Assistant:"""
  elif modelId == 'anthropic.claude-3-5-sonnet-20240620-v1:0':
        full_prompt = [
            {"role": "user", "content": instruction + "\n<text>" + prompt + "</text>"}
        ]
    
  return full_prompt

# Setting up the API call in the correct format for the corresponding model
def json_format(modelId, tokens, temperature, top_p, full_prompt="[input text]"):
  if modelId == 'amazon.titan-tg1-large':
    body = json.dumps({"inputText": full_prompt, 
                   "textGenerationConfig":{
                       "maxTokenCount":tokens,
                       "stopSequences":[],
                       "temperature":temperature,
                       "topP":top_p}})
  elif modelId == 'anthropic.claude-v2:1':
    body = json.dumps({"prompt": full_prompt,
                 "max_tokens_to_sample":tokens,
                 "temperature":temperature,
                 "top_k":250,
                 "top_p":top_p,
                 "stop_sequences":[]
                  })
  elif modelId == 'anthropic.claude-3-5-sonnet-20240620-v1:0':
    body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": tokens,
            "messages": full_prompt,
            "temperature": temperature,
            "top_p": top_p
            })
    
  return body

def display_format(modelId):
  if modelId == 'amazon.titan-tg1-large':
    body = json.dumps({"inputText": "[input_text]", 
                   "textGenerationConfig":{
                       "maxTokenCount":"[max_tokens]",
                       "stopSequences":[],
                       "temperature":"[temperature]",
                       "topP":"[top_p]"}})
  elif modelId == 'anthropic.claude-v2:1':
    body = json.dumps({"prompt": "[input_text]",
                 "max_tokens_to_sample":"[max_tokens]",
                 "temperature":"[temperature]",
                 "top_k":250,
                 "top_p":"[top_p]",
                 "stop_sequences":[]
                  })
  elif modelId == 'anthropic.claude-3-5-sonnet-20240620-v1:0':
    body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": "[max_tokens]",
            "messages": [{"role": "user", "content": "[input_text]"}],
            "temperature": "[temperature]",
            "top_p": "[top_p]"
            })
  return body

#
# summarize:
#  - input_text: 要約対象のテキスト・今回はOCR情報
#  - instruction_text: プロンプト
#
def summarize(modelId, input_text, instruction_text, max_tokens, temperature, top_p):
  
  # Params
  knowledge_base_id="WTRH0XX1PA"
  
  result = get_knowledgeable_response(knowledge_base_id, modelId, instruction_text, input_text, max_tokens, temperature, top_p)
  
  #以下はget_knowledgeable_response()に置き換え
  # full_prompt = prompt_construction(modelId, instruction_text, input_text)
  # body = json_format(modelId, max_tokens, temperature, top_p, full_prompt)

  # # Foundation model is invoked here to generate a response
  # response = boto3_bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
  # response_body = json.loads(response.get('body').read())

  # # Extract the output from the API response for the corresponding model
  # if modelId == 'amazon.titan-tg1-large':
  #   result = response_body.get('results')[0].get('outputText')
  # elif modelId == 'anthropic.claude-v2:1':
  #   result = response_body.get('completion')
  # elif modelId == 'anthropic.claude-3-5-sonnet-20240620-v1:0':
  #   result = response_body['content'][0]['text']

  return result.strip('\n')



#
# メイン画面
#
with gr.Blocks() as demo:
  with gr.Row():
    gr.Markdown("# 設備階層ナレッジベース テストアプリ v2")
    example_holder = gr.Textbox(visible=False, label="サンプルテキスト", value="example")
  with gr.Row():
    modelId = gr.Dropdown(label="Bedrock Modelの選択", choices=models, value='anthropic.claude-3-5-sonnet-20240620-v1:0')
  with gr.Row():
    with gr.Column(scale=4):
      custom_instruction = gr.Textbox(label="プロンプト:", value=example_instruction)
      input_text = gr.Textbox(label="OCR抽出情報", placeholder="クレンジング対象のテキストを入力")
      example = gr.Examples(examples=[[example_instruction, "CML Documentation"]], inputs=[custom_instruction, example_holder])
    with gr.Column(scale=4):
      with gr.Accordion("Advanced Generation Options", open=False):
        max_new_tokens = gr.Slider(minimum=0, maximum=4096, step=1, value=512, label="Max Tokens")
        temperature = gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.5, label="Temperature")
        top_p = gr.Slider(minimum=0, maximum=1.0, step=0.01, value=1.0, label="Top P")
      with gr.Accordion("Bedrock API Request Details", open=False):
        instruction_prompt = gr.Code(label="Instruction Prompt", value=prompt_construction('amazon.titan-tg1-large'))
        input_format = gr.JSON(label="Input Format", value=display_format('amazon.titan-tg1-large'))
        with gr.Accordion("AWS Credentials", open=False):
          label = gr.Markdown("These can be set from the project env vars")
          region = gr.Markdown("**Region**: "+os.getenv('AWS_DEFAULT_REGION'))
          access_key = gr.Markdown("**Access Key**: "+os.getenv('AWS_ACCESS_KEY_ID'))
          secret_key = gr.Markdown("**Secret Key**: *****")
      summarize_btn = gr.Button("実行", variant='primary')
      reset_btn = gr.Button("リセット")
    with gr.Column(scale=4):
      output = gr.Textbox(label="Bedrockからの応答")
  summarize_btn.click(fn=summarize, inputs=[modelId, input_text, custom_instruction, max_new_tokens, temperature, top_p], outputs=output, 
                            api_name="summarize")
  reset_btn.click(fn=clear_out, inputs=[], outputs=[input_text, output, example_holder, custom_instruction], show_progress=False)
  modelId.change(fn=prompt_construction, inputs=[modelId], outputs=instruction_prompt)
  modelId.change(fn=display_format, inputs=modelId, outputs=input_format)
  example_holder.change(fn=example_lookup, inputs=example_holder, outputs=input_text, show_progress=False)

demo.launch(server_port=int(os.getenv('CDSW_APP_PORT')),
           enable_queue=True,
           show_error=True,
           server_name='127.0.0.1',
)
