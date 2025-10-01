import os
import re
import sys
import uuid
import time
import torch
import shutil
import logging
import warnings
import subprocess
from PIL import Image
from hyperpyyaml import load_hyperpyyaml
from typing import Any, Dict, List, Optional, Tuple, Union, Generator

from vllm import LLM, SamplingParams
from vllm.inputs import TextPrompt as LLMInputs
from transformers import AutoTokenizer, AutoProcessor, AutoConfig


logger = logging.getLogger()
warnings.filterwarnings("ignore")
current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)
sys.path.insert(0, current_dir_path)


class MingUtils(object):
    def __init__(self, model_path: str, limit_mm_per_prompt={"image": 10, "video":2},sample_rate=16000, sys_prompt=None):
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.limit_mm_per_prompt = limit_mm_per_prompt
        if 'image' in limit_mm_per_prompt:
            self.limit_images = limit_mm_per_prompt['image']
        else:
            self.limit_images = None

        if 'video' in limit_mm_per_prompt:
            self.limit_videos = limit_mm_per_prompt['video']
        else:
            self.limit_videos = None
        self.sys_prompt = sys_prompt
        self.sample_rate = sample_rate
        self.max_frames = 40

    def filter_message(self, data, limit_images=10, limit_videos=2, limit_audios=1):
        # history not support audio
        history_audio_max_len = 0
        total_image_count = 0
        total_video_count = 0
        total_audio_count = 0
        total_audio_count_except_last_input = 0

        filtered_data = []
        last_item = data[-1] if data else None

        if last_item and last_item['role'] == 'HUMAN':
            last_item_images = sum(1 for content in last_item['content'] if content['type'] == 'image')
            last_item_videos = sum(1 for content in last_item['content'] if content['type'] == 'video')
            last_item_audios = sum(1 for content in last_item['content'] if content['type'] == 'audio')

            if total_image_count + last_item_images <= limit_images and total_video_count + last_item_videos <= limit_videos and last_item_audios + total_audio_count <= limit_audios:
                filtered_data.append(last_item)
                total_image_count += last_item_images
                total_video_count += last_item_videos
                total_audio_count += last_item_audios

        temp_human = None
        temp_assistant = None
        for entry in reversed(data[:-1]):
            if entry['role'] == 'HUMAN':
                temp_human = entry
                
                if temp_human and temp_assistant:
                    human_images = sum(1 for content in temp_human['content'] if content['type'] == 'image')
                    human_videos = sum(1 for content in temp_human['content'] if content['type'] == 'video')
                    human_audios = sum(1 for content in temp_human['content'] if content['type'] == 'audio')
                    assistant_images = sum(1 for content in temp_assistant['content'] if content['type'] == 'image')
                    assistant_videos = sum(1 for content in temp_assistant['content'] if content['type'] == 'video')
                    assistant_audios = sum(1 for content in temp_assistant['content'] if content['type'] == 'audio')

                    new_image_count = total_image_count + human_images + assistant_images
                    new_video_count = total_video_count + human_videos + assistant_videos

                    if new_image_count > limit_images or new_video_count > limit_videos or assistant_audios + human_audios > 0:  
                        temp_human = None
                        temp_assistant = None
                        continue
                    else:
                        filtered_data.append(temp_assistant)
                        filtered_data.append(temp_human)
                        total_image_count = new_image_count
                        total_video_count = new_video_count
                        # total_audio_count_except_last_input = new_audio_count

                        temp_human = None
                        temp_assistant = None

            elif entry['role'] == 'ASSISTANT':
                temp_assistant = entry

        return filtered_data[::-1]

    def build_prompt(
        self,
        prompt: str,
        audio: Optional[Union[str, bytes]] = None,
        video: Optional[Union[str, bytes]] = None,
        image: Optional[Union[str, bytes, Image.Image]] = None,
        history: list = [],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Build a prompt input for the model (common logic for text/audio/image generation).

        Args:
            prompt (str): User input text.
            audio (Optional[Union[str, bytes]]): Audio data (e.g., file path or binary).
            video (Optional[Union[str, bytes]]): Video data (e.g., file path or binary).
            image (Optional[Union[str, bytes, Image.Image]]): Image data (file path, binary, or PIL Image).
            history (list, optional): Conversation history. Defaults to empty list.
            **kwargs: Additional parameters for prompt building.

        Returns:
            Dict[str, Any]: A dictionary containing the built prompt.
        """

        os.environ["IMAGE_GEN_MODE"] = ""
        for key, value in kwargs.items():
            if key == "audio":
                audio = value
            if key == "video":
                video = value
            if key == "image":
                image = value

        messages_video_and_audio = [{"role": "HUMAN", "content": []}]
        if self.sys_prompt is not None:
            messages_video_and_audio[0]["content"].append(
                {"type": "text", "text": "SYSTEM: %s" % self.sys_prompt}
            )

        if video is not None:
            logger.info("llm activate video input")
            if True:
                with_history_video = video
            else:
                with_history_video = self.add_history_video(video)
            messages_video_and_audio[0]["content"].append(
                {
                    "type": "video",
                    "video": with_history_video,
                    "sample": "uniform",
                    "max_frames": len(with_history_video) if type(with_history_video) is list else self.max_frames,
                }
            )
        # image is path list
        if image is not None:
            logger.info("llm activate image input")
            for single_image in image:
                messages_video_and_audio[0]["content"].append(
                    {"type": "image", "image": single_image}
                )
        if audio is not None:
            logger.info("llm activate audio input")
            # audio = torch.from_numpy(audio).unsqueeze(0)
            messages_video_and_audio[0]["content"].append(
                {"type": "audio", "audio": audio, "sample_rate": self.sample_rate}
            )
        if prompt is not None:
            logger.info("llm activate text input")
            messages_video_and_audio[0]["content"].append(
                {"type": "text", "text": prompt}
            )
        # self.manage_history_query_message()
        if len(history) > 0 and video is None:
            messages_video_and_audio = history + messages_video_and_audio

        logger.info("prompt: " + str(messages_video_and_audio))

        if self.limit_images and self.limit_videos:
            messages_video_and_audio = self.filter_message(messages_video_and_audio, self.limit_images, self.limit_videos)
        prompt = self.processor.apply_chat_template(
            messages_video_and_audio,
            tokenize=False,
            add_generation_prompt=True,
            use_system=True,
        )
        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(
            messages_video_and_audio
        )

        requests = []
        inputs = LLMInputs(
            {
                "prompt": prompt,
            }
        )

        """"
        "image": image_inputs,
        "video": video_inputs,
        "audio": audio_inputs,
        """
        if image is not None or image_inputs is not None:
            if "multi_modal_data" in inputs.keys():
                inputs["multi_modal_data"]["image"] = image_inputs
            else:
                inputs["multi_modal_data"] = {"image": image_inputs}
        if video is not None or video_inputs is not None:
            if "multi_modal_data" in inputs.keys():
                inputs["multi_modal_data"]["video"] = video_inputs
            else:
                inputs["multi_modal_data"] = {"video": video_inputs}
        if audio is not None or audio_inputs is not None:
            if "multi_modal_data" in inputs.keys():
                inputs["multi_modal_data"]["audio"] = audio_inputs
            else:
                inputs["multi_modal_data"] = {"audio": audio_inputs}
        requests.append(inputs)
        return requests

    def build_img_prompt(
        self,
        prompt: str,
        image: Optional[Union[str, bytes, Image.Image]] = None,
        **kwargs,
    ):
        """
        Build a prompt for image generation based on input text and optional image.

        Args:
            text (str): The text prompt for image generation.
            image (Optional[Union[str, bytes, Image.Image]]): Optional input image (for editing mode).
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            List[LLMInputs]: A list of LLM input objects containing the generated prompt and image data.

        Description:
            - Constructs a message structure for the model. If no image is provided, a dummy image is used.
            - The message order depends on whether an image is provided:
            - If `image is None`: [Text, Dummy Image]
            - Else: [Image, Text]
            - Applies the chat template to generate a text prompt.
            - Processes vision-related information (e.g., image inputs).
            - Returns LLM input objects with the prompt and multi-modal data.
        """
        messages = (
            [
                {
                    "role": "HUMAN",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": Image.new("RGB", (1, 1), (0, 0, 0))},
                    ],
                },
            ]
            if image is None
            else [
                {
                    "role": "HUMAN",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
        )

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, use_system=True
        )
        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(
            messages
        )
        requests = [
            LLMInputs({"prompt": text, "multi_modal_data": {"image": image_inputs}}),
        ]
        return requests

    def build_img_gen_prompt(
        self,
        prompt: str,
        image: Optional[Union[str, bytes, Image.Image]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare input data for image generation or editing.

        Args:
            text (str): The text prompt for image generation.
            image (Optional[Union[str, bytes, Image.Image]]): Optional input image (for editing mode).
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            Dict[str, torch.Tensor]: A dictionary of processed inputs in tensor format, including text and multi-modal data.

        Description:
            - Constructs a message structure for the model. If no image is provided, only the text is included.
            - Applies the chat template to generate a text prompt.
            - Processes vision-related information (e.g., image inputs).
            - Converts the inputs into PyTorch tensors and moves them to the GPU.
            - Converts specific tensor types (e.g., pixel values) to `torch.bfloat16` for efficient inference.
        """
        messages = (
            [
                {
                    "role": "HUMAN",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            if image is None
            else [
                {
                    "role": "HUMAN",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
        )
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(
            messages
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
        ).to("cuda")

        for k in inputs.keys():
            if k in [
                "pixel_values",
                "pixel_values_videos",
                "audio_feats",
                "pixel_values_reference",
            ]:
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)
        return inputs


class MingTalker(object):
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        spk_info: Dict[str, torch.Tensor] = None,
        device: str = "cuda",
    ) -> None:
        """
        Initialize the BailingTalker instance.

        Args:
            model_path (str): Path to the model directory containing TTS components.
            tensor_parallel_size (int, optional): Number of GPU devices for tensor parallelism. Defaults to 1.
            spk_info (Dict[str, torch.Tensor], optional): Speaker-specific embeddings. Defaults to None.
            device (str, optional): Device to run the model on (e.g., "cuda" or "cpu"). Defaults to "cuda".
        """
        super().__init__()
        from modeling_bailing_talker import (
            AudioDetokenizer,
            BailingTalkerForConditionalGeneration,
        )
        from audio_detokenizer.cli.frontend import TTSFrontEnd

        luna_path = os.path.join(current_dir_path, "data/spks/luna_v2.pt")
        eng_path = os.path.join(current_dir_path, "data/spks/eng_v2.pt")

        logger.info(f"luna_path={luna_path}, eng_path={eng_path}")
        if spk_info is None:
            spk_info = {
                "luna": torch.load(luna_path),
                "eng": torch.load(eng_path),
            }
        self.spk_info = spk_info
        self.model_name_or_path = model_path
        self.device = device

        talker_path = os.path.join(model_path, "talker")
        self.talker = (
            BailingTalkerForConditionalGeneration.from_pretrained(talker_path)
            .cuda()
            .to(torch.bfloat16)
        )

        audio_detokenizer_path = os.path.join(
            model_path, "talker/audio_detokenizer_stream.yaml"
        )
        with open(audio_detokenizer_path, "r") as f:
            configs = load_hyperpyyaml(f)

        self.audio_detokenizer = AudioDetokenizer(
            audio_detokenizer_path,
            flow_model_path=os.path.join(model_path, "talker/flow_stream.pt"),
            hifigan_model_path=os.path.join(model_path, "talker/hift_v2.pt"),
            spk_info=spk_info,
        )

        # new mel
        self.audio_frontend = TTSFrontEnd(
            configs["feat_extractor"],
            f"{model_path}/talker/campplus.onnx",
            f"{model_path}/talker/speech_tokenizer_v1.onnx",
        )

        # try:
        #     use_fp16 = False
        #     trt_file_name = 'flow.decoder.estimator.fp16.plan' if use_fp16 else "flow.decoder.estimator.fp32.plan"
        #     flow_decoder_onnx_model = os.path.join(model_path, 'talker', 'flow.decoder.estimator.fp32.onnx')
        #     flow_decoder_trt_model = os.path.join(model_path, 'talker', trt_file_name)
        #     self.audio_detokenizer.model.load_trt(flow_decoder_trt_model, flow_decoder_onnx_model, fp16=use_fp16)
        # except Exception as e:
        #     print(f"load tensorrt file failed: {e}")

        logger.info("init talker success")

    def contains_chinese(self, text: str) -> bool:
        """
        Check if the input text contains Chinese characters.

        Args:
            text (str): Input text to check.

        Returns:
            bool: True if Chinese characters are present, False otherwise.
        """
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    def generate(self, text: str, speaker: str = "luna") -> torch.Tensor:
        """
        Generate audio from text using the TTS model.

        Args:
            text (str): Input text to convert to speech.
            speaker (str, optional): Speaker identifier (e.g., "luna", "eng"). Defaults to "luna".

        Returns:
            torch.Tensor: Generated audio waveform as a tensor.
        """
        spk_input = self.spk_info.get(speaker, "luna")
        is_chinese = self.contains_chinese(text)
        if not is_chinese:
            text = text.split()

        all_wavs = []
        for tts_speech, text_list in self.talker.omni_audio_generation(
            text,
            audio_detokenizer=self.audio_detokenizer,
            thinker_reply_part=None,
            speaker=speaker,
            stream=False,
            **spk_input,
        ):
            all_wavs.append(tts_speech)
        waveform = torch.cat(all_wavs, dim=-1)
        return waveform

    def generate_stream(
        self, text: Any, speaker: str = "luna"
    ) -> Generator[Tuple[torch.Tensor, List[str]], None, None]:
        """
        Stream audio generation from text incrementally.

        Args:
            text (Any): Input text to convert to speech. Can be a string or other compatible type.
            speaker (str, optional): Speaker identifier (e.g., "luna", "eng"). Defaults to "luna".

        Yields:
            Tuple[torch.Tensor, List[str]]: Generated audio segment and corresponding text segments.
        """
        spk_input = self.spk_info.get(speaker, "luna")
        # is_chinese = self.contains_chinese(text)
        # if not is_chinese:
        #     text = text.split()

        for tts_speech, text_list in self.talker.omni_audio_generation(
            text, audio_detokenizer=self.audio_detokenizer, stream=True, **spk_input
        ):
            yield tts_speech, text_list


class MingMOE(object):
    max_new_tokens = 1024

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.5,
        sys_prompt: str = "",
    ) -> None:
        """
        Initialize the BailigMOE instance.

        Args:
            model_path (str): Path to the LLM model directory or Hugging Face repository.
            tensor_parallel_size (int, optional): Number of GPU devices for tensor parallelism. Defaults to 1.
            gpu_memory_utilization (float, optional): Fraction of GPU memory to use (0.0-1.0). Defaults to 0.6.
            sys_prompt (str, optional): System-level prompt to prepend to user inputs. Defaults to empty.
        """
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"image": 10, "video":2}
        )
        self.sys_prompt = sys_prompt

    def set_sys_prompt(self, sys_prompt: str) -> None:
        """
        Set or update the system-level prompt.

        Args:
            sys_prompt (str): New system prompt to use for subsequent generations.
        """
        self.sys_prompt = sys_prompt

    def create_request_id(self) -> str:
        """
        Generate a unique request ID for tracking.

        Returns:
            str: A UUID4 string representing a unique request identifier.
        """
        return str(uuid.uuid4())

    def generate(
        self, requests: List[LLMInputs], with_hidden_status: bool = False, **kwargs
    ) -> Any:
        """
        Generate text responses from the LLM.

        Args:
            requests (List[LLMInputs]): List of input prompts for generation.
            with_hidden_status (bool, optional): Whether to return hidden states from the LLM. Defaults to False.
            **kwargs: Additional parameters for sampling (e.g., max_new_tokens, temperature).

        Returns:
            Any: Generated text or hidden states, depending on `with_hidden_status`.
        """
        (
            temperature,
            presence_penalty,
            repetition_penalty,
            return_hidden_states,
        ) = (0, 0, 1, False)
        max_new_tokens = self.max_new_tokens
        for key, value in kwargs.items():
            if key == "max_new_tokens" and value is not None:
                max_new_tokens = value
            if key == "temperature" and value is not None:
                temperature = value
            if key == "presence_penalty" and value is not None:
                presence_penalty = value
            if key == "repetition_penalty" and value is not None:
                repetition_penalty = value
            if key == "return_hidden_states" and value is not None:
                return_hidden_states = value
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            return_hidden_states=return_hidden_states,
        )
        res = self.llm.generate(requests, sampling_params)
        if with_hidden_status:
            return res[0].prefill_hidden_states
        return res[0].outputs[0].text

    def generate_stream(
        self, requests: List[LLMInputs], request_id: int = 0, **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream text responses from the LLM incrementally.

        Args:
            requests (List[LLMInputs]): List of input prompts for generation.
            request_id (int, optional): Unique identifier for the request. Defaults to 0.
            **kwargs: Additional parameters for sampling (e.g., max_new_tokens).

        Yields:
            str: Incremental text output as it is generated.
        """
        max_new_tokens = self.max_new_tokens
        for key, value in kwargs.items():
            if key == "max_new_tokens" and value > 0:
                max_new_tokens = value
        inputs = [
            (
                request_id,
                requests[0],
                SamplingParams(temperature=0, max_tokens=max_new_tokens),
            )
        ]
        req_id, prompt_text, sampling_params = inputs.pop(0)
        llm_engine = self.llm.llm_engine
        llm_engine.add_request(str(req_id), prompt_text, sampling_params)
        logger.info("start to inference llm")

        history_sentence_index = 0
        while llm_engine.has_unfinished_requests():
            request_outputs = llm_engine.step()
            new_sentence = request_outputs[0].outputs[0].text
            sentence = new_sentence[history_sentence_index:]
            history_sentence_index = len(new_sentence)
            yield sentence

    def generate_interrupte(self, request_id: str) -> None:
        """
        Interrupt an ongoing request.

        Args:
            request_id (str): Unique identifier of the request to abort.
        """
        llm_engine = self.llm.llm_engine
        llm_engine.abort_request(str(request_id))


class MingImg(object):

    def __init__(
        self,
        model_path: str,
        **kwargs,
    ) -> None:
        """
        Initialize the BailingImg module.

        Args:
            model_path (str): Path to the model directory.
            **kwargs: Additional arguments (e.g., `dit_type`, `torch_dtype`).
        """
        os.environ["IMAGE_GEN_MODE"] = "None"
        from modeling_bailingmm import BailingMMNativeForConditionalGeneration
        model_diffusion = BailingMMNativeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
            attn_implementation="flash_attention_2",
            load_image_gen=True,
            low_cpu_mem_usage=True,       # Minimize CPU memory during loading
            load_vlm=False,               # No vlm, only diffusion
        ).to("cuda").to(torch.bfloat16)                     # Run on GPU
        self.model_diffusion = model_diffusion


class Ming(object):

    """
    Initialize the class with model components and configuration.

    Args:
        model_path (str): Path to the model directory.
        sys_prompt (str, optional): System prompt for the model. Defaults to empty.
        driver (str, optional): GPU device identifiers (e.g., "0,1"). Defaults to "0".
        talker_with_vllm (bool, optional): Whether to enable VLLM for the talker. Defaults to False.
    """
    proc = None
    def __init__(
        self,
        model_path: str,
        sys_prompt: str = "",
        driver: str = "0",
        gpu_memory_utilization: dict = {
            "moe": 0.6,
            "talker": 0.1
        },
        talker_with_vllm: bool = True,
    ):  
        logger.info(f"gpu_memory_utilization={gpu_memory_utilization},model_path={model_path},driver={driver}")
        tensor_parallel_size = (
            len(driver.split(",")) if len(driver.split(",")) > 0 else 1
        )
        shutil.copy(model_path + "/config.json", current_dir_path)
        am_path = os.path.join(model_path,"am.mvn")
        shutil.copy(am_path, ".")
        self.utils = MingUtils(model_path=current_dir_path)
        self.talker = MingTalker(model_path=model_path)
        if talker_with_vllm:
            self.kill_with_spawn()
            os.environ["VLLM_USE_V1"] = "1"
            current_dir = os.path.dirname(os.path.abspath(__file__))
            proc = subprocess.Popen(["python3", 
                f"{current_dir}/talker/talker_vllm_server.py",
                "--model", 
                f"{model_path}/talker",
                "--gpu-memory-utilization",
                f"{gpu_memory_utilization['talker']}",
                "--port",
                "8816"],
            )
            self.proc = proc
            self.wait_for_talker_ready(proc, timeout=90, check_interval=5)
            self.talker.talker.set_use_vllm(use_vllm=True, vllm_in_process=True)
        else:
            self.talker.talker.set_use_vllm(use_vllm=False, vllm_in_process=False)

        os.environ["VLLM_USE_V1"] = "0"
        self.moe = MingMOE(
            model_path, tensor_parallel_size=tensor_parallel_size,
            sys_prompt=sys_prompt,
            gpu_memory_utilization=gpu_memory_utilization["moe"]
        )
        self.img = MingImg(model_path)

    def kill_with_spawn(self):
        result = os.popen("ps aux | grep 'multiprocessing.spawn' | grep -v grep").read()
        lines = result.strip().split('\n')
        for line in lines:
            parts = line.split()
            if len(parts) > 1:
                pid = parts[1]
                try:
                    os.kill(int(pid), 9)  # SIGKILL
                    logger.info(f"kill success PID={pid}")
                except Exception as e:
                    logger.error(f"kill fail PID={pid}: {e}")

    def wait_for_talker_ready(self, proc, timeout=90, check_interval=5):
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                raise RuntimeError(f"talker service init fail: {proc.returncode}\nStderr: {stderr}")
            time.sleep(check_interval)

    def __del__(self):
        if self.proc is not None:
            logger.info("close talker spawn")
            self.proc.kill() 

    def _generate_text(
        self,
        prompt: str,
        audio: Optional[Union[str, bytes]] = None,
        video: Optional[Union[str, bytes]] = None,
        image: Optional[Union[str, bytes, Image.Image]] = None,
        history: list = [],
        **kwargs
    ) -> str:
        """
        Generate text output based on the input prompt.

        Args:
            prompt (str): User input text.
            audio (Optional[Union[str, bytes]]): Audio data (e.g., file path or binary).
            video (Optional[Union[str, bytes]]): Video data (e.g., file path or binary).
            image (Optional[Union[str, bytes, Image.Image]]): Image data (file path, binary, or PIL Image).
            history (list, optional): Conversation history. Defaults to empty list.
            **kwargs: Additional parameters for the model.

        Returns:
            str: Generated text output.
        """
        inputs = self.utils.build_prompt(
            prompt=prompt,
            audio=audio,
            video=video,
            image=image,
            history=history,
            **kwargs,
        )
        return self.moe.generate(inputs, **kwargs)

    def _generate_image(
        self,
        prompt: str,
        image: Optional[Union[str, bytes, Image.Image]] = None,
        **kwargs
    ) -> Image.Image:
        """
        Generate an image or edit an existing one based on the input prompt.

        Args:
            prompt (str): User input text.
            image (Optional[Union[str, bytes, Image.Image]]): Input image (for editing). Defaults to None.
            **kwargs: Additional parameters for image generation.

        Returns:
            Image.Image: Generated or edited image.
        """
        inputs = self.utils.build_img_prompt(prompt=prompt, image=image, **kwargs)

        os.environ["IMAGE_GEN_MODE"] = "T2I" if image is None else "EDIT"
        # generate tokenhidden_states
        image_gen_llm_hidden_states = self.moe.generate(
            requests=inputs,
            with_hidden_status=True,
            max_new_tokens=1,
            return_hidden_states=True,
        )

        inputs = self.utils.build_img_gen_prompt(prompt=prompt, image=image)
        image = self.img.model_diffusion.generate(
            **inputs,
            image_gen_llm_hidden_states=image_gen_llm_hidden_states.unsqueeze(0),
            image_gen=True,
        )
        return image

    def _generate_audio(
        self,
        prompt: str,
        audio: Optional[Union[str, bytes]] = None,
        video: Optional[Union[str, bytes]] = None,
        image: Optional[Union[str, bytes, Image.Image]] = None,
        history: list = [],
        **kwargs
    ) -> Union[bytes, Generator[bytes, None, None]]:
        """
        Generate audio (text-to-speech or speech-to-speech).

        Args:
            prompt (str): User input text.
            audio (Optional[Union[str, bytes]]): Audio data (e.g., file path or binary).
            video (Optional[Union[str, bytes]]): Video data (e.g., file path or binary).
            image (Optional[Union[str, bytes, Image.Image]]): Image data (file path, binary, or PIL Image).
            history (list, optional): Conversation history. Defaults to empty list.
            **kwargs: Additional parameters for the model.

        Returns:
            Union[bytes, Generator[bytes, None, None]]: Generated audio data or a stream generator.
        """

        # generate text
        inputs = self.utils.build_prompt(
            prompt=prompt,
            audio=audio,
            video=video,
            image=image,
            history=history,
            **kwargs,
        )
        gen_text = self.moe.generate(inputs, **kwargs)

        # generate audio
        audio = self.talker.generate(text=gen_text)
        return audio

    def generate(
        self,
        text: Optional[str] = None,
        audio: Optional[Union[str, bytes]] = None,
        video: Optional[Union[str, bytes]] = None,
        image: Optional[Union[str, bytes, Image.Image]] = None,
        history: list = [],
        output_type: str = "text",
        **kwargs
    ) -> Union[str, Image.Image, bytes, Generator]:
        """
        Generate content based on the specified output type.

        Args:
            text (Optional[str]): User input text.
            audio (Optional[Union[str, bytes]]): Audio data (e.g., file path or binary).
            video (Optional[Union[str, bytes]]): Video data (e.g., file path or binary).
            image (Optional[Union[str, bytes, Image.Image]]): Image data (file path, binary, or PIL Image).
            history (list, optional): Conversation history. Defaults to empty list.
            output_type (str, optional): Output type ("text", "speech", "image"). Defaults to "text".
            **kwargs: Additional parameters for the model.

        Returns:
            Union[str, Image.Image, bytes, Generator]: Generated content (text, image, or audio).

        Raises:
            ValueError: If `output_type` is not supported.
        """
        if output_type == "text":
            return self._generate_text(
                prompt=text,
                audio=audio,
                video=video,
                image=image,
                history=history,
                **kwargs,
            )

        elif output_type == "speech":
            return self._generate_audio(
                prompt=text,
                audio=audio,
                video=video,
                image=image,
                history=history,
                **kwargs,
            )

        elif output_type == "image":
            return self._generate_image(
                prompt=text,
                audio=audio,
                video=video,
                image=image,
                history=history,
                **kwargs,
            )

        else:
            raise Exception("not support output_type")

    def generate_stream(
        self,
        text: Optional[str] = None,
        audio: Optional[Union[str, bytes]] = None,
        video: Optional[Union[str, bytes]] = None,
        image: Optional[Union[str, bytes, Image.Image]] = None,
        history: list = [],
        output_type: str = "text",
        **kwargs
    ) -> Generator[Tuple[Union[bytes, str], str], None, None]:
        """
        Stream generated content (text or speech).

        Args:
            text (Optional[str]): User input text.
            audio (Optional[Union[str, bytes]]): Audio data (e.g., file path or binary).
            video (Optional[Union[str, bytes]]): Video data (e.g., file path or binary).
            image (Optional[Union[str, bytes, Image.Image]]): Image data (file path, binary, or PIL Image).
            history (list, optional): Conversation history. Defaults to empty list.
            output_type (str, optional): Output type ("text", "speech"). Defaults to "text".
            **kwargs: Additional parameters for the model.

        Yields:
            Tuple[Union[bytes, str], str]: Generated content (text or audio) and request ID.

        Raises:
            ValueError: If `output_type` is not supported for streaming.
        """
        if output_type == "text":
            inputs = self.utils.build_prompt(
                prompt=text,
                audio=audio,
                video=video,
                image=image,
                history=history,
                **kwargs,
            )
            request_id = self.moe.create_request_id()
            for text in self.moe.generate_stream(
                requests=inputs, request_id=request_id, **kwargs
            ):
                yield text, request_id

        elif output_type == "speech":
            inputs = self.utils.build_prompt(
                prompt=text,
                audio=audio,
                video=video,
                image=image,
                history=history,
                **kwargs,
            )
            request_id = self.moe.create_request_id()
            text_generator = self.moe.generate_stream(
                requests=inputs, request_id=request_id, **kwargs
            )
            for tts_speech, sentence in self.talker.generate_stream(
                text=text_generator
            ):
                yield tts_speech, sentence, request_id
        else:
            raise Exception("not support output_type")

    def generate_interrupte(self, request_id: str) -> None:
        """
        Interrupt a specific request.

        Args:
            request_id (str): ID of the request to interrupt.

        Raises:
            ValueError: If `request_id` is empty.
        """

        self.moe.generate_interrupte(request_id)
    
    def tts(self, text: str)-> Union[bytes, Generator[bytes, None, None]]:
        # generate audio
        audio = self.talker.generate(text=text)
        return audio