import requests
import json
import torch
from typing import Generator, Dict, List, Optional, Union

class VLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...

    def generate(
        self,
        prompt_token_ids: List[int],
        prompt_embeds: Union[List[float], torch.Tensor],
        stream: bool = False,
        temperature: float = 0,
        top_k: int = 50,
        max_tokens: int = 1024,
        min_tokens: int = 10,
        stop: Optional[List[str]] = None,
        trace_id: str = None,
    ) -> Generator[Dict, None, None]:
        """
        Generate text using the VLLM server.

        Args:
            prompt_token_ids: List of token IDs for the prompt
            prompt_embeds: Audio embeddings for the prompt (can be list or tensor)
            stream: Whether to stream the results
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            max_tokens: Maximum number of tokens to generate
            stop: List of stop sequences

        Yields:
            Dictionary containing the generated text and metadata
        """
        # Convert tensor to list if necessary
        if isinstance(prompt_embeds, torch.Tensor):
            prompt_embeds = prompt_embeds.cpu().to(torch.float32).tolist()

        # Prepare request data
        data = {
            "prompt_token_ids": prompt_token_ids,
            "prompt_embeds": prompt_embeds,
            "stream": stream,
            # "temperature": temperature,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "stop": stop or ["<audio_eos>"],
            "trace_id": trace_id,
        }

        with requests.post('http://localhost:8816/generate', json=data, stream=stream) as res:
            if stream:
                for chunk in res.iter_lines(chunk_size=8192,
                                            decode_unicode=False,
                                            delimiter=b"\0"):
                    if chunk:
                        data = json.loads(chunk.decode("utf-8"))

                        # print(data)
                        yield data
            else:
                data = res.json()
                # print(data)
                yield data

    def abort_request():
        ...

def main():
    prompt_token_ids = torch.load('/mnt1/zhoubofan/prompt_token_ids.pt')
    prompt_embeds = torch.load('/mnt1/zhoubofan/prompt_embeds.pt')
    with VLLMClient("http://localhost:8816") as client:
        # Stream mode
        print("Streaming mode:")
        result_list = []
        for result in client.generate(
            prompt_token_ids=prompt_token_ids,
            prompt_embeds=prompt_embeds,
            temperature=0.0,
            top_k = 1,
            stream=True
        ):
            result_list.append(result)
            print(f"Generated token: {result['token_ids']}")
            print(f"Finish reason: {result['finish_reason']}")


if __name__ == "__main__":
    main() 
