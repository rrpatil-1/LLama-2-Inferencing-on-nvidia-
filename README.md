
# Meta LLama 2 inferencing with Nvidia TensorRT-LLM and Triton Inference server

### Prerequisites:
1. Install NVIDIA Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit
2. Install Docker: https://docs.docker.com/engine/install/
3. Create an account with NVIDIA: https://catalog.ngc.nvidia.com/
4. Generate an API Key: https://org.ngc.nvidia.com/setup/api-key
5. Docker login with your NGC API key:
```bash
$ docker login nvcr.io --username='$oauthtoken' --password=${NGC_CLI_API_KEY}
```
6. Download the NGC CLI tool for your OS: https://org.ngc.nvidia.com/setup/installers/cli
```bash
$ wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.41.3/files/ngccli_linux.zip -O ~/ngccli_linux.zip &&
unzip ~/ngccli_linux.zip -d ~/ngc &&
chmod u+x /ngc/ngc-cli/ngc &&
echo "export PATH="$PATH:/ngc/ngc-cli"" >> ~/.bash_profile && source ~/.bash_profile
```
7. Set up your NGC CLI Tool locally: 

```bash
$ ngc config set
```
**Note:** After you enter your API key, you may see multiple options for the org and team. Select as desired or hit enter to accept the default.

### Create Hugging Face account
1. Create account in: https://huggingface.co/login
2. Create access token: https://huggingface.co/settings/tokens
**Fill the form for accessing:** https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

**After 1 hr you will get the access for llama as per meta policy**


 
## step1: Installation

1. Create directory llama
2. Clone the model Llama-2-7b-chat-hf model
3. Clone the TensorRT-LLM repository
4. Clone the tensorrtllm_backend repository

#### command to run
```bash
$ mkdir llama

$ git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

$ git clone https://github.com/NVIDIA/TensorRT-LLM.git

$ git clone https://github.com/triton-inference-server/tensorrtllm_backend
```
## Step 2: Triton Setup
**1. Install Triton Inference Server**
**2. Docker exec into the container**

```bash
$ docker run --rm -itd --name trtllm -v `pwd`:/mnt -w /mnt --gpus all -p 8000:8000 - p 8002:8002 -p 8001:8001 nvcr.io/nvidia/tritonserver:24.07-trtllm

$ docker exec -it trtllm bash
```
**Once inside the container, run the following commands:**
```bash
# Replace 'HF_LLAMA_MODE' with your model path
$ export HF_LLAMA_MODEL=/mnt/Llama-2-7b-chat-hf/

$ export UNIFIED_CKPT_PATH=/mnt/llama/ckpt/llama/7b/

$ export ENGINE_PATH=/llama/engines/llama/7b/
```
```bash
$ python3 TensorRT-LLM/examples/llama/convert_checkpoint.py --model_dir ${HF_LLAMA_MODEL} --output_dir ${UNIFIED_CKPT_PATH} --dtype float16
```
```bash
$ trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
             --remove_input_padding enable \
             --gpt_attention_plugin float16 \
             --context_fmha enable \
             --gemm_plugin float16 \
             --output_dir ${ENGINE_PATH} \
             --paged_kv_cache enable \
             --max_batch_size 64
```
```bash
$ cd tensorrtllm_backend

$ cp all_models/inflight_batcher_llm/ llama_ifb -r
```
```bash
$ python3 tools/fill_template.py -i llama_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_LLAMA_MODEL},triton_max_batch_size:64,preprocessing_instance_count:1
```
```bash
$ python3 tools/fill_template.py -i llama_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False
```
```bash
$ python3 tools/fill_template.py -i llama_ifb/ensemble/config.pbtxt triton_max_batch_size:64
```
```bash
$ python3 tools/fill_template.py -i llama_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0
```
**Launch Triton Server**
```bash
$ pip install SentencePiece

$ python3 scripts/launch_triton_server.py --world_size 1 --model_repo=llama_ifb/ --http_port 8000
```
**Note:** If you don't **pass http_port 8000** arguments then it will deploy to default port 8000

**Once server up and running press CTRL+D to exit the docker container**

### Send Request
```bash
$ curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": "", "pad_id": 2, "end_id": 2}' | json_pp
```
### Output
```bash
{
  "cum_log_probs": 0.0,
  "model_name": "ensemble",
  "model_version": "1",
  "output_log_probs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "sequence_end": false,
  "sequence_id": 0,
  "sequence_start": false,
  "text_output": "\nMachine learning is a subset of artificial intelligence (AI) that uses algorithms to learn from data and"
}
```
### Python Code
```python
import requests

def ask_meta_llama_2(question):
    url = "http://localhost:8000/v2/models/ensemble/generate"
    payload = {"inputs": [{"name": "input_text", "shape": [1], "datatype": "BYTES", "data": [question.encode()]}]}
    response = requests.post(url, json=payload)
    answer = response.json()["outputs"][0]["data"][0].decode()
    return answer

# Example usage:
user_question = "What is the meaning of life?"
meta_llama_answer = ask_meta_llama_2(user_question)
print(f"Meta Llama 2 says: {meta_llama_answer}")
```

