## Omni SDK

We have unified the addition of VLLM inference acceleration capabilities for multimodal question answering, image generation, speech generation, and other modules in the Omni model. By installing the ming_sdk wheel package and specifying the ming-omni model path, users can easily load the model for VLLM inference.

### Build ming_sdk for python whl
```
cd Ming
python ming_sdk/setup.py sdist bdist_wheel
```

### Install ming_sdk
```
pip install inclusionAI/Ming-Lite-Omni-FP8/ming_sdk-1.0.0-py3-none-any.whl
```

### Install dependencies

```
pip install inclusionAI/Ming-Lite-Omni-FP8/flash_attn-2.7.0.post1%2Bcu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install inclusionAI/Ming-Lite-Omni-FP8/vllm-0.8.6.dev1+ga37daf9e9.d20250730-cp310-cp310-linux_x86_64.whl

```

### Inference with the SDK

For more examples, please refer to the official the examples `ming_sdk/ming_test.py`



## Supported functions
- [x] Single-machine multi-card
- [x] text Q&A returns all at once
- [x] text question-and-answer stream return
- [x] Voice Q&A returns all at once
- [x] Voice question-and-answer stream return
- [x] Streaming support interruption
- [x] Image generation
- [x] Image editing
- [x] ASR task

## Example
- download model checkppoint
  ```
  from modelscope import snapshot_download
  model_path = snapshot_download('inclusionAI/Ming-lite-omni-1.5-FP8')
  ```

- textQA
  ```
  
  ming = Ming(model_path=model_path, driver="0", talker_with_vllm = True, gpu_memory_utilization={"moe": 0.55,"talker": 0.17}
  res = ming.generate(text="介绍一下杭州")
  print(res)
  ```
- textQA stream
  ```
  
  ming = Ming(model_path=model_path, driver="0", talker_with_vllm = True, gpu_memory_utilization={"moe": 0.55,"talker": 0.17}
  all_text = ""
  request_id = ""
  for text, request_id in ming.generate_stream(
      text="介绍一下杭州", max_new_tokens=128
  ):
      all_text += text
  print(f"request_id:{request_id},text={all_text}")
  ```
- imageQA
  ```
  
  ming = Ming(model_path=model_path, driver="0", talker_with_vllm = True, gpu_memory_utilization={"moe": 0.55,"talker": 0.17}
  asr_res = ming.generate(
      text="描述下这个图",
      image=["auto_t2i.jpg"],
  )
  print(asr_res)
  ```
- textQA & speech output
  ```
  
  ming = Ming(model_path=model_path, driver="0", talker_with_vllm = True, gpu_memory_utilization={"moe": 0.55,"talker": 0.17}
  output_audio_path = "test.wav"
  waveform = ming.generate(
      text="介绍一下杭州", output_type="speech", max_new_tokens=128
  )
  torchaudio.save(output_audio_path, waveform, 24000)
  ```
- textQA & speech output stream
  ```
  
  ming = Ming(model_path=model_path, driver="0", talker_with_vllm = True, gpu_memory_utilization={"moe": 0.55,"talker": 0.17}
  all_wavs = []
  all_text = ""
  request_id = ""
  output_audio_path = "test_stream.wav"
  for tts_speech, text, request_id in ming.generate_stream(
      text="介绍一下杭州", output_type="speech", max_new_tokens=128
  ):
      all_text += text
      all_wavs.append(tts_speech)
  waveform = torch.cat(all_wavs, dim=-1)
  torchaudio.save(output_audio_path, waveform, 24000)
  print(f"request_id:{request_id},audio:{output_audio_path},text={all_text}")
  ```
- textQA & speech output stream & interrupte
  ```
  
  ming = Ming(model_path=model_path, driver="0", talker_with_vllm = True, gpu_memory_utilization={"moe": 0.55,"talker": 0.17}
  all_wavs = []
  all_text = ""
  request_id = ""
  output_audio_path = "test_stream.wav"
  for tts_speech, text, request_id in ming.generate_stream(
      text="介绍一下杭州", output_type="speech", max_new_tokens=128
  ):
      all_text += text
      all_wavs.append(tts_speech)
      if len(all_text) > 20:
          ming.generate_interrupte(request_id)
  waveform = torch.cat(all_wavs, dim=-1)
  torchaudio.save(output_audio_path, waveform, 24000)
  print(f"request_id:{request_id},audio:{output_audio_path},text={all_text}")
  ```
- image generate
  ```
  
  ming = Ming(model_path=model_path, driver="0", talker_with_vllm = True, gpu_memory_utilization={"moe": 0.55,"talker": 0.17}
  image_t2i = ming.generate(text="生成一张图 美女在沙滩上奔跑", output_type="image")
  image_t2i.save("auto_t2i.jpg")
  ```
- image edit
  ```
  
  ming = Ming(model_path=model_path, driver="0", talker_with_vllm = True, gpu_memory_utilization={"moe": 0.55,"talker": 0.17}
  image_t2i = "auto_t2i.jpg"
  image_edit = ming.generate(
      text="给人物戴上眼镜", image=[image_t2i], output_type="image"
  )
  image_edit.save("auto_edit.jpg")
  ```
- asr recognize
  ```
  
  ming = Ming(model_path=model_path, driver="0", talker_with_vllm = True, gpu_memory_utilization={"moe": 0.55,"talker": 0.17}
  audio_path = "xwlb-20s.wav"
  audio = np.array(
      AudioSegment.from_file(audio_path)
      .set_channels(1)
      .set_frame_rate(16000)
      .get_array_of_samples()
  )
  asr_res = ming.generate(
      text="Please recognize the language of this speech and transcribe it. Format: oral.",
      audio=audio,
  )
  print(asr_res)
  ```


- tts
  ```
  ming = Ming(model_path=model_path, driver="0", talker_with_vllm = True, gpu_memory_utilization={"moe": 0.55,"talker": 0.17}
  output_audio_path = "test_tts.wav"
  waveform = ming.tts(text="我爱北京天安门")
  torchaudio.save(output_audio_path, waveform, 24000)
  ```

##  model path
- Ming-Lite-Omni-1.5 https://modelscope.cn/models/inclusionAI/Ming-Lite-Omni-1.5         A100
- Ming-Lite-Omni-1.5-FP8 https://modelscope.cn/models/inclusionAI/Ming-lite-omni-1.5-FP8 L20