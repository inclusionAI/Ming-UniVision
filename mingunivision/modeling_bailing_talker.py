from dataclasses import dataclass
from typing import Optional, Tuple, List
import os
import yaml
import re
import torch
import torch.nn as nn
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from contextlib import nullcontext
import threading
import numpy as np
import time
import torch
import uuid

from transformers import Qwen2Config, PreTrainedModel
from transformers import Qwen2ForCausalLM, AutoTokenizer
from audio_detokenizer.cli.flow_stream_model import AudioDetokenizerModel
from audio_detokenizer.utils.common import fade_in_out
from s3bpe_tokenizer import S3BpeTokenizer
from configuration_bailing_talker import BailingTalkerConfig
from transformers.utils import ModelOutput
from sentence_manager.sentence_manager import SentenceNormalizer
import logging
from talker.talker_vllm_client import VLLMClient
from talker.sync_vllm_infer import (
    construct_vllm,
    vllm_infer_generator,
    SamplingParams,
    get_vllm_request_id,
)


@dataclass
class BailingTalkerOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[torch.FloatTensor] = None


class BailingTalkerForConditionalGeneration(PreTrainedModel):
    config_class = BailingTalkerConfig
    base_model_prefix = "model"

    def __init__(self, config: BailingTalkerConfig):
        super().__init__(config)

        self.config = config
        self.vocab_size = self.config.vocab_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.config._name_or_path)
        self.tokenizer_len = len(self.tokenizer)
        if self.tokenizer_len == 184445:
            # for vllm, tokenizer contains audio codec token
            self.tokenizer_len = self.tokenizer_len - 32768
        self.model_config = Qwen2Config.from_pretrained(self.config._name_or_path)
        if self.model_config.vocab_size == 151936:
            # give up resize embedding operation
            self.model_config.vocab_size = 184445
        self.model = Qwen2ForCausalLM(self.model_config)
        self.thinker_to_talker_proj = nn.Linear(
            self.config.qa_model_hidden_size, self.model_config.hidden_size
        )
        self.vp_head = nn.Conv1d(
            self.config.vp_feature_size,
            self.model_config.hidden_size,
            kernel_size=self.config.vp_kernel_size,
            stride=self.config.vp_stride,
            padding=self.config.vp_kernel_size // 2,
        )
        self.s3bpe_tokenizer = S3BpeTokenizer(
            bpe_model=f"{self.config._name_or_path}/s3_bpe/tokenizer.json",
            mapping_file=f"{self.config._name_or_path}/s3_bpe/char_mapping.txt",
        )

        self.loss_function = nn.CrossEntropyLoss()

        default_config_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "sentence_manager/default_config.yaml",
        )
        # print(f"default_config_path: {default_config_path}")
        self.sentence_manager_config = yaml.safe_load(open(default_config_path))
        if "split_token" not in self.sentence_manager_config:
            self.sentence_manager_config["split_token"] = []
        assert isinstance(self.sentence_manager_config["split_token"], list)
        self.sentence_manager_config["split_token"].append(
            re.escape(self.tokenizer.eos_token)
        )
        self.normalizer = SentenceNormalizer(
            self.sentence_manager_config.get("text_norm", {})
        )

        self.device_new = torch.device("cuda")
        self.overlap = 879
        self.window = np.hamming(2 * self.overlap)
        self.llm_context = (
            torch.cuda.stream(torch.cuda.Stream(self.device_new))
            if torch.cuda.is_available()
            else nullcontext()
        )
        self.flow_hift_context = (
            torch.cuda.stream(torch.cuda.Stream(self.device_new))
            if torch.cuda.is_available()
            else nullcontext()
        )
        self.lock = threading.Lock()
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}

        self.use_vllm = True  # 是否使用 vllm，或者 torch
        self.vllm_in_process = False  # 使用同一进程下的 vllm engine
        self.vllm_engine = None

        if self.use_vllm:
            if self.vllm_in_process:
                self.vllm_engine = construct_vllm(
                    model_path=self.config._name_or_path,
                    enforce_eager=False,
                    gpu_memory_utilization=0.2,
                )

        # if self.use_vllm:
        #     original_embeds = self.model.get_input_embeddings()
        #     vocab_size, embed_dim = original_embeds.weight.shape
        #     original_embeds.pad
        #     src_weight = self.model.get_input_embeddings().weight.data
        #
        #     # 初始化新嵌入层（权重随机）
        #     self.input_embeds = nn.Embedding(vocab_size, embed_dim)
        #     self.input_embeds.weight.data.copy_(src_weight)
        #     del self.model

    def set_use_vllm(self, use_vllm: bool, vllm_in_process: bool = False):
        self.use_vllm = use_vllm

        if not vllm_in_process:
            if self.vllm_engine is not None:
                del self.vllm_engine
                self.vllm_engine = None

        if self.use_vllm:
            if self.vllm_in_process:
                if self.vllm_engine is None:
                    self.vllm_engine = construct_vllm(
                        model_path=self.config._name_or_path,
                        enforce_eager=False,
                        gpu_memory_utilization=0.2,
                    )
            else:
                self.vllm_engine = VLLMClient("http://localhost:8816")
        else:
            if self.vllm_engine is not None:
                del self.vllm_engine
                self.vllm_engine = None

    def get_input_embeddings(self):
        # if self.use_vllm:
        #     return self.input_embeds
        # else:
        return self.model.get_input_embeddings()

    def encode_audio_segments(
        self,
        inputs_embeds: torch.FloatTensor,
        vp_emb: torch.FloatTensor,
        vp_insert_loc: torch.LongTensor,
        thinker_reply_part: Optional[torch.FloatTensor] = None,
        thinker_reply_length: Optional[List] = None,
        thinker_prefix_insert_loc: Optional[torch.LongTensor] = None,
    ):
        vp_emb_encoded = self.vp_head(vp_emb.transpose(-1, -2)).transpose(-1, -2)

        for idx in range(vp_insert_loc.shape[0]):
            inputs_embeds[
                idx, vp_insert_loc[idx].item() : vp_insert_loc[idx].item() + 1, :
            ] = vp_emb_encoded[idx, :, :]

        if thinker_prefix_insert_loc is not None:
            thinker_reply_part = self.thinker_to_talker_proj(thinker_reply_part)
            for idx in range(thinker_prefix_insert_loc.shape[0]):
                real_length = thinker_reply_length[idx]
                inputs_embeds[
                    idx,
                    thinker_prefix_insert_loc[idx]
                    .item() : thinker_prefix_insert_loc[idx]
                    .item()
                    + real_length,
                    :,
                ] = thinker_reply_part[idx, :real_length, :]

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[dict] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        text_input_ids: Optional[torch.LongTensor] = None,
        vp_emb: Optional[torch.FloatTensor] = None,
        vp_insert_loc: Optional[torch.LongTensor] = None,
        thinker_reply_part: Optional[torch.FloatTensor] = None,
        thinker_reply_length: Optional[torch.FloatTensor] = None,
        thinker_prefix_insert_loc: Optional[torch.LongTensor] = None,
    ):

        if inputs_embeds is None:
            audio_input_embeds = self.model.get_input_embeddings()(input_ids)
            text_input_embeds = self.model.get_input_embeddings()(text_input_ids)
            inputs_embeds = audio_input_embeds + text_input_embeds
            if past_key_values is None:
                inputs_embeds = self.encode_audio_segments(
                    inputs_embeds,
                    vp_emb,
                    vp_insert_loc,
                    thinker_reply_part=thinker_reply_part,
                    thinker_reply_length=thinker_reply_length,
                    thinker_prefix_insert_loc=thinker_prefix_insert_loc,
                )

            if position_ids is None:
                position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_(
                    (attention_mask == 0), 1
                )

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs.logits

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )

        return BailingTalkerOutputWithPast(
            loss=loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            logits=logits,
        )

    def sample(
        self, logits, topk=20, filter_value=-float("Inf"), refuse=False, eos_id=151666
    ):
        """
        从topk中采样，返回采样的id
        Args:
            logits: [1, V]
            topk: int
        """
        logits = logits.reshape(1, -1)  # [1, V]
        indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
        if refuse is True:
            indices_to_remove[0][eos_id] = True
        logits[indices_to_remove] = filter_value
        token_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).to(
            torch.long
        )
        return token_id

    def text_length(self, text):
        return len(re.findall("[\u4e00-\u4e27\u4e29-\u4e3e\u4e42-\u9fa4]", text))

    def cut_text(self, text, max_length, tail_min_length=5):
        def text_append(text_list, text, max_length):
            if len(text_list) == 0:
                text_list.append(text)
            else:
                if len(text_list[-1]) + self.text_length(text) <= max_length:
                    if (
                        text_list[-1].endswith("。")
                        and self.text_length(text) < tail_min_length
                    ):
                        text_list.append(text.lstrip("，"))
                    else:
                        text_list[-1] += text
                else:
                    text_list.append(text.lstrip("，"))
            return text_list

        text = text.replace("\n", " ")
        text = self.normalizer.normalize(text)
        text = text.replace("。，", "。")
        if len(text) <= max_length:
            return [text]
        text_list = []
        text = text.replace(".", "。").replace(",", "，")

        sps1 = []
        for t in text.split("。"):
            t = t.strip()
            if len(t) > 0:
                if t[-1] not in "！？，。!?,.":
                    t += "。"
                sps1.append(t)

        for text_piece1 in sps1:
            sps2 = []
            for t in text_piece1.split("，"):
                t = t.strip()
                if len(t) > 0:
                    if t[-1] not in "！？，。!?,.":
                        t += "，"
                    sps2.append(t)

            for text_piece2 in sps2:
                text_piece2 = text_piece2.replace("。，", "。")
                if self.text_length(text_piece2) > max_length:
                    for i in range(0, len(text_piece2), max_length):
                        text_list = text_append(
                            text_list, text_piece2[i : i + max_length], max_length
                        )
                else:
                    text_list = text_append(text_list, text_piece2, max_length)
        return text_list

    @torch.no_grad()
    def generate(
        self,
        talker_audio_prefix: torch.LongTensor,
        talker_text_prefix: torch.LongTensor,
        talker_text_input_part: List,
        position_ids: Optional[torch.LongTensor] = None,
        vp_emb: Optional[torch.FloatTensor] = None,
        vp_insert_loc: Optional[torch.LongTensor] = None,
        thinker_reply_part: Optional[torch.FloatTensor] = None,
        thinker_reply_length: Optional[torch.FloatTensor] = None,
        thinker_prefix_insert_loc: Optional[torch.LongTensor] = None,
        prompt_wav_token_len: int = 0,
        min_new_token=10,
    ):
        result = []
        step = 0
        eos_id = self.tokenizer.encode("<audio_eos>")[0]

        def get_prompt_embeds(
            input_ids, text_input_ids, vp_emb, vp_insert_loc, thinker_reply_length
        ):
            audio_input_embeds = self.get_input_embeddings()(input_ids)
            text_input_embeds = self.get_input_embeddings()(text_input_ids)
            inputs_embeds = audio_input_embeds + text_input_embeds

            inputs_embeds = self.encode_audio_segments(
                inputs_embeds,
                vp_emb,
                vp_insert_loc,
                thinker_reply_part=thinker_reply_part,
                thinker_reply_length=thinker_reply_length,
                thinker_prefix_insert_loc=thinker_prefix_insert_loc,
            )
            return inputs_embeds

        if self.use_vllm:
            prompt_tokens = talker_audio_prefix[0].cpu().tolist()
            inputs_embeds = get_prompt_embeds(
                talker_audio_prefix,
                talker_text_prefix,
                vp_emb=vp_emb,
                vp_insert_loc=vp_insert_loc,
                thinker_reply_length=thinker_reply_length,
            )[
                0
            ]  # (90, 896, )
            if self.vllm_in_process:
                sampling_params = SamplingParams(
                    top_k=20,
                    skip_special_tokens=True,
                    max_tokens=1024,
                    min_tokens=10,
                    stop=["<audio_eos>"],
                )
                output_generator = vllm_infer_generator(
                    self.vllm_engine,
                    prompt_tokens,
                    inputs_embeds,
                    get_vllm_request_id(),
                    sampling_params,
                )
                for result in output_generator:
                    if result["finish_reason"] == "stop":
                        break
                    yield [result["token_ids"]]
            else:
                with VLLMClient("http://localhost:8816") as client:
                    for result in client.generate(
                        prompt_token_ids=prompt_tokens,
                        prompt_embeds=inputs_embeds,
                        top_k=20,
                        min_tokens=10,
                        max_tokens=1024,
                        stream=True,
                    ):
                        if (
                            "finish_reason" in result
                            and result["finish_reason"][0] == "stop"
                        ):
                            break
                        yield result["token_ids"]
        else:

            while step < 1000:
                if step == 0:
                    talker_audio_input_ids = talker_audio_prefix
                    talker_text_input_ids = talker_text_prefix
                    attention_mask = torch.ones(talker_audio_input_ids.shape).to(
                        talker_audio_prefix.device
                    )

                else:
                    talker_audio_input_ids = next_token
                    talker_text_input_ids = (
                        torch.tensor(talker_text_input_part[0], dtype=torch.long)
                        .reshape(1, -1)
                        .to(talker_audio_prefix.device)
                    )
                    attention_mask = torch.ones(next_token.shape[0], 1).to(
                        talker_audio_prefix.device
                    )
                    position_ids = (position_ids[0][-1] + 1).view(1, -1)
                    thinker_prefix_insert_loc = None

                    if len(talker_text_input_part) > 1:
                        talker_text_input_part = talker_text_input_part[1:]

                outputs = self(
                    input_ids=talker_audio_input_ids,
                    text_input_ids=talker_text_input_ids,
                    thinker_reply_part=thinker_reply_part,
                    thinker_reply_length=thinker_reply_length,
                    thinker_prefix_insert_loc=thinker_prefix_insert_loc,
                    vp_emb=vp_emb,
                    vp_insert_loc=vp_insert_loc,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                    past_key_values=outputs.past_key_values if step > 0 else None,
                )
                # 采样
                logits = outputs.logits[:, -1, :]
                if step < min_new_token:
                    refuse = True
                else:
                    refuse = False
                next_token = self.sample(logits, refuse=refuse)
                if next_token.item() == eos_id:
                    break

                yield [next_token.item()]
                step += 1

    def omni_audio_generation_func(
        self,
        tts_text,
        prompt,
        prefix_from_thinker,
        vp,
        position_ids,
        talker_audio_prefix,
        vp_insert_loc,
        thinker_length,
        vp_emb=None,
        thinker_reply_part=None,
        prompt_text_input_part=None,
        prompt_wav_token_len=None,
    ):
        text_input_part = self.tokenizer.encode(tts_text)
        # audio_prefix and text_prefix for first step generation
        talker_text_prefix = (
            prompt
            + prefix_from_thinker
            + vp
            + prompt_text_input_part
            + text_input_part
            + self.tokenizer.encode("<text_eos>")
            + self.tokenizer.encode("<text_pad>")
            * (
                prompt_wav_token_len
                - len(prompt_text_input_part)
                - len(text_input_part)
            )
        )
        # the rest of input_text
        talker_text_input_part = self.tokenizer.encode("<text_pad>")
        talker_text_prefix = (
            torch.tensor(talker_text_prefix).reshape(1, -1).to(self.device)
        )
        record_list = []
        for audio_token in self.generate(
            talker_audio_prefix=talker_audio_prefix,
            talker_text_prefix=talker_text_prefix,
            talker_text_input_part=talker_text_input_part,
            position_ids=position_ids,
            vp_emb=vp_emb,
            vp_insert_loc=vp_insert_loc,
            thinker_reply_part=thinker_reply_part,
            thinker_reply_length=torch.tensor([thinker_length]).to(self.device),
            thinker_prefix_insert_loc=(
                torch.tensor([len(prompt) + 1]).to(self.device)
                if thinker_reply_part is not None
                else None
            ),
            prompt_wav_token_len=prompt_wav_token_len,
        ):
            audio_token = [ele - self.tokenizer_len for ele in audio_token]
            audio_token = self.s3bpe_tokenizer.decode(audio_token)

            yield audio_token
            record_list += audio_token

        if len(record_list) == 0:
            yield [0, 0, 0]

    def token2wav(
        self,
        audio_detokenizer,
        token,
        prompt_token,
        prompt_feat,
        embedding,
        token_offset,
        uuid,
        stream=False,
        finalize=False,
        speed=1.0,
    ):
        self.fp16 = audio_detokenizer.model.fp16
        self.speech_window = audio_detokenizer.model.speech_window
        self.mel_cache_len = audio_detokenizer.model.mel_cache_len
        self.source_cache_len = audio_detokenizer.model.source_cache_len

        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, _ = audio_detokenizer.model.flow.inference(
                token=token.to(self.device_new),
                token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(
                    self.device_new
                ),
                prompt_token=prompt_token.to(self.device_new),
                prompt_token_len=torch.tensor(
                    [prompt_token.shape[1]], dtype=torch.int32
                ).to(self.device_new),
                prompt_feat=prompt_feat.to(self.device_new),
                prompt_feat_len=torch.tensor(
                    [prompt_feat.shape[1]], dtype=torch.int32
                ).to(self.device_new),
                embedding=embedding.to(torch.float32).to(self.device_new),
                streaming=stream,
                finalize=finalize,
            )

        tts_mel = tts_mel[
            :, :, token_offset * audio_detokenizer.model.flow.token_mel_ratio :
        ]
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = (
                self.hift_cache_dict[uuid]["mel"],
                self.hift_cache_dict[uuid]["source"],
            )
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            tts_speech, tts_source = audio_detokenizer.model.hift.inference(
                speech_feat=tts_mel, cache_source=hift_cache_source
            )
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(
                    tts_speech, self.hift_cache_dict[uuid]["speech"], self.speech_window
                )
            self.hift_cache_dict[uuid] = {
                "mel": tts_mel[:, :, -self.mel_cache_len :],
                "source": tts_source[:, :, -self.source_cache_len :],
                "speech": tts_speech[:, -self.source_cache_len :],
            }
            tts_speech = tts_speech[:, : -self.source_cache_len]
        else:
            if speed != 1.0:
                assert (
                    self.hift_cache_dict[uuid] is None
                ), "speed change only support non-stream inference mode"
                tts_mel = F.interpolate(
                    tts_mel, size=int(tts_mel.shape[2] / speed), mode="linear"
                )
            tts_speech, tts_source = audio_detokenizer.model.hift.inference(
                speech_feat=tts_mel, cache_source=hift_cache_source
            )
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(
                    tts_speech, self.hift_cache_dict[uuid]["speech"], self.speech_window
                )

        return tts_speech

    def llm_job(
        self,
        text,
        prompt,
        prefix_from_thinker,
        vp,
        position_ids,
        talker_audio_prefix,
        vp_insert_loc,
        thinker_length,
        vp_emb,
        thinker_reply_part,
        prompt_text_input_part,
        prompt_wav_token_len,
        this_uuid,
    ):
        with self.llm_context:
            audio_token_list = []
            for audio_token in self.omni_audio_generation_func(
                tts_text=text,
                prompt=prompt,
                prefix_from_thinker=prefix_from_thinker,
                vp=vp,
                position_ids=position_ids,
                talker_audio_prefix=talker_audio_prefix,
                vp_insert_loc=vp_insert_loc,
                thinker_length=thinker_length,
                vp_emb=vp_emb,
                thinker_reply_part=thinker_reply_part,
                prompt_text_input_part=prompt_text_input_part,
                prompt_wav_token_len=prompt_wav_token_len,
            ):
                for item in audio_token:
                    self.tts_speech_token_dict[this_uuid].append(item)
                    audio_token_list.append(item)
        self.llm_end_dict[this_uuid] = True

    def tts_job(
        self,
        text,
        prompt,
        prefix_from_thinker,
        vp,
        position_ids,
        talker_audio_prefix,
        vp_insert_loc,
        thinker_length,
        vp_emb,
        thinker_reply_part,
        audio_detokenizer,
        speaker,
        prompt_text_input_part,
        prompt_wav_token_len,
        stream,
        flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
        prompt_speech_feat=torch.zeros(1, 0, 80),
        speed=1.0,
    ):

        self.token_hop_len = audio_detokenizer.model.token_hop_len
        self.pre_lookahead_len = audio_detokenizer.model.flow.pre_lookahead_len
        this_uuid = str(uuid.uuid1())
        flow_embedding = vp_emb.squeeze(0)

        if speaker in audio_detokenizer.spk_info:
            prompt_dict = audio_detokenizer.spk_info[speaker]
            flow_prompt_speech_token = prompt_dict["flow_prompt_speech_token"]
            prompt_speech_feat = prompt_dict["prompt_speech_feat"]
            flow_embedding = prompt_dict["flow_embedding"]

        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = (
                [],
                False,
            )
            self.hift_cache_dict[this_uuid] = None

        p = threading.Thread(
            target=self.llm_job,
            args=(
                text,
                prompt,
                prefix_from_thinker,
                vp,
                position_ids,
                talker_audio_prefix,
                vp_insert_loc,
                thinker_length,
                vp_emb,
                thinker_reply_part,
                prompt_text_input_part,
                prompt_wav_token_len,
                this_uuid,
            ),
        )
        p.start()

        if stream is True:
            token_offset = 0
            prompt_token_pad = int(
                np.ceil(flow_prompt_speech_token.shape[1] / self.token_hop_len)
                * self.token_hop_len
                - flow_prompt_speech_token.shape[1]
            )
            while True:
                time.sleep(0.1)
                this_token_hop_len = (
                    self.token_hop_len + prompt_token_pad
                    if token_offset == 0
                    else self.token_hop_len
                )
                if (
                    len(self.tts_speech_token_dict[this_uuid]) - token_offset
                    >= this_token_hop_len + self.pre_lookahead_len
                ):
                    this_tts_speech_token = torch.tensor(
                        self.tts_speech_token_dict[this_uuid][
                            : token_offset + this_token_hop_len + self.pre_lookahead_len
                        ]
                    ).unsqueeze(dim=0)
                    t0 = time.time()
                    this_tts_speech = self.token2wav(
                        audio_detokenizer=audio_detokenizer,
                        token=this_tts_speech_token,
                        prompt_token=flow_prompt_speech_token,
                        prompt_feat=prompt_speech_feat,
                        embedding=flow_embedding,
                        token_offset=token_offset,
                        uuid=this_uuid,
                        stream=stream,
                        finalize=False,
                    )
                    token_offset += this_token_hop_len

                    yield {"tts_speech": this_tts_speech.cpu()}

                if (
                    self.llm_end_dict[this_uuid] is True
                    and len(self.tts_speech_token_dict[this_uuid]) - token_offset
                    < this_token_hop_len + self.pre_lookahead_len
                ):
                    break
            p.join()

            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(
                self.tts_speech_token_dict[this_uuid]
            ).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(
                audio_detokenizer=audio_detokenizer,
                token=this_tts_speech_token,
                prompt_token=flow_prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=flow_embedding,
                token_offset=token_offset,
                uuid=this_uuid,
                finalize=True,
            )
            yield {"tts_speech": this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(
                self.tts_speech_token_dict[this_uuid]
            ).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(
                audio_detokenizer=audio_detokenizer,
                token=this_tts_speech_token,
                prompt_token=flow_prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=flow_embedding,
                token_offset=0,
                uuid=this_uuid,
                finalize=True,
                speed=speed,
            )
            # this_tts_speech = self.adjust_volume(this_tts_speech, target_db=-20)
            yield {"tts_speech": this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.current_stream().synchronize()

    def omni_audio_generation(
        self,
        tts_text,
        vp_emb=None,
        thinker_reply_part=None,
        max_length=50,
        audio_detokenizer=None,
        speaker=None,
        stream=False,
        **kwargs,
    ):
        prompt_text = kwargs["prompt_text"]
        prompt_text_input_part = self.tokenizer.encode(prompt_text)
        prompt_wav_token = kwargs["prompt_speech_token"][0].tolist()
        prompt_wav_token_bpe = self.s3bpe_tokenizer.encode(prompt_wav_token)[0]
        prompt_wav_token_bpe = (
            np.array(prompt_wav_token_bpe, dtype=np.int64) + self.tokenizer_len
        ).tolist()

        # thinker_reply_part: [B, T, d]
        # get text_emb and hidden_states from thinker
        thinker_length = (
            thinker_reply_part.size(1) if thinker_reply_part is not None else 0
        )
        prefix_from_thinker = (
            self.tokenizer.encode("<thinker_prefix>")
            + self.tokenizer.encode("<audio_pad>")
            * thinker_length  # placeholder for prefix emb from thinker
            + self.tokenizer.encode("</thinker_prefix>")
        )

        prompt = self.tokenizer.encode("<prompt>") + self.tokenizer.encode("</prompt>")
        vp = (
            self.tokenizer.encode("<vp>")
            + self.tokenizer.encode("<audio_pad>")
            + self.tokenizer.encode("</vp>")
        )
        talker_audio_prefix = (
            prompt
            + prefix_from_thinker
            + vp
            + self.tokenizer.encode("<audio_bos>")
            + prompt_wav_token_bpe
        )
        attention_mask = (
            torch.ones(len(talker_audio_prefix)).reshape(1, -1).to(self.device)
        )
        position_ids = (
            (attention_mask.cumsum(-1) - 1)
            .masked_fill_((attention_mask == 0), 1)[:, :]
            .view(1, -1)
        )
        talker_audio_prefix = (
            torch.tensor(talker_audio_prefix).reshape(1, -1).to(self.device)
        )
        vp_insert_loc = torch.tensor(
            len(prompt) + len(prefix_from_thinker) + 1, dtype=torch.long
        ).reshape(1, -1)
        vp_emb = vp_emb.unsqueeze(0).to(torch.bfloat16).to(self.device)

        assert (
            max_length > 0
        ), f"max_length must be greater than 0, but here is {max_length}"
        streaming_text = []
        count = 0

        for ele in tts_text:
            # 正常的分句
            if ele[-1] in "！？，。!?," and (
                len(streaming_text) >= 12 or count > 0 and len(streaming_text) >= 8
            ):
                streaming_text.append(ele)
                if bool(re.search(r"[\u4e00-\u9fff]", "".join(streaming_text))):
                    streaming_text = "".join(streaming_text)
                else:
                    streaming_text = " ".join(streaming_text)
                text_list = self.cut_text(streaming_text, max_length)

                for text in text_list:
                    if text[0] == "，":
                        text = text[1:]
                    # 首句流式，其余句子非流式
                    if count == 0:
                        for idx, this_tts_speech_dict in enumerate(
                            self.tts_job(
                                text,
                                prompt,
                                prefix_from_thinker,
                                vp,
                                position_ids,
                                talker_audio_prefix,
                                vp_insert_loc,
                                thinker_length,
                                vp_emb,
                                thinker_reply_part,
                                audio_detokenizer,
                                speaker,
                                prompt_text_input_part,
                                len(prompt_wav_token_bpe),
                                stream=stream & True,
                            )
                        ):
                            yield this_tts_speech_dict["tts_speech"], (
                                text if idx == 0 else ""
                            )
                    else:
                        for idx, this_tts_speech_dict in enumerate(
                            self.tts_job(
                                text,
                                prompt,
                                prefix_from_thinker,
                                vp,
                                position_ids,
                                talker_audio_prefix,
                                vp_insert_loc,
                                thinker_length,
                                vp_emb,
                                thinker_reply_part,
                                audio_detokenizer,
                                speaker,
                                prompt_text_input_part,
                                len(prompt_wav_token_bpe),
                                stream=False,
                            )
                        ):
                            yield this_tts_speech_dict["tts_speech"], (
                                text if idx == 0 else ""
                            )

                streaming_text = []
                count += 1

            # 判断 . 为小数点还是英文结束
            elif (
                ele[-1] in "."
                and (
                    len(streaming_text) >= 12 or count > 0 and len(streaming_text) >= 8
                )
                and bool(re.search(r"[0-9]", streaming_text[-1][-1])) is False
            ):
                streaming_text.append(ele)
                # 中文
                if bool(re.search(r"[\u4e00-\u9fff]", "".join(streaming_text))):
                    streaming_text = "".join(streaming_text)
                # 英文
                else:
                    streaming_text = " ".join(streaming_text)
                text_list = self.cut_text(streaming_text, max_length)
                logging.info(f"判断 . 为小数点还是英文结束")
                audio_tokens = []
                for text in text_list:
                    if text[0] == "，":
                        text = text[1:]
                    if count == 0:  # 首句流式
                        for idx, this_tts_speech_dict in enumerate(
                            self.tts_job(
                                text,
                                prompt,
                                prefix_from_thinker,
                                vp,
                                position_ids,
                                talker_audio_prefix,
                                vp_insert_loc,
                                thinker_length,
                                vp_emb,
                                thinker_reply_part,
                                audio_detokenizer,
                                speaker,
                                prompt_text_input_part,
                                len(prompt_wav_token_bpe),
                                stream=stream & True,
                            )
                        ):
                            yield this_tts_speech_dict["tts_speech"], (
                                text if idx == 0 else ""
                            )
                    else:  # 非流式
                        for idx, this_tts_speech_dict in enumerate(
                            self.tts_job(
                                text,
                                prompt,
                                prefix_from_thinker,
                                vp,
                                position_ids,
                                talker_audio_prefix,
                                vp_insert_loc,
                                thinker_length,
                                vp_emb,
                                thinker_reply_part,
                                audio_detokenizer,
                                speaker,
                                prompt_text_input_part,
                                len(prompt_wav_token_bpe),
                                stream=False,
                            )
                        ):
                            yield this_tts_speech_dict["tts_speech"], (
                                text if idx == 0 else ""
                            )

                streaming_text = []
                count += 1

            # 针对换行符的判断
            elif ele[-1] == "\n":
                if len(streaming_text) > 0:
                    # 中文
                    if bool(re.search(r"[\u4e00-\u9fff]", "".join(streaming_text))):
                        if len(streaming_text) > 0 and bool(
                            re.search(r"[\u4e00-\u9fff]", streaming_text[-1][-1])
                        ):
                            ele = "，"
                            streaming_text.append(ele)
                    # 英文
                    else:
                        # 当前单词尾部无符号
                        if len(ele) > 1 and bool(re.search(r"[a-zA-Z]", ele[-2])):
                            ele = ele[:-1] + "."
                        # 当前单词尾部有符号
                        else:
                            ele = ele[:-1]
                        streaming_text.append(ele)
                # 触发分句条件
                if len(streaming_text) >= 12 or count > 0 and len(streaming_text) >= 8:
                    if bool(re.search(r"[\u4e00-\u9fff]", "".join(streaming_text))):
                        streaming_text = "".join(streaming_text)
                    else:
                        streaming_text = " ".join(streaming_text)
                    text_list = self.cut_text(streaming_text, max_length)
                    logging.info(f"针对换行符的判断")
                    audio_tokens = []
                    for text in text_list:
                        if text[0] == "，":
                            text = text[1:]
                        if count == 0:  # 首句流式
                            for idx, this_tts_speech_dict in enumerate(
                                self.tts_job(
                                    text,
                                    prompt,
                                    prefix_from_thinker,
                                    vp,
                                    position_ids,
                                    talker_audio_prefix,
                                    vp_insert_loc,
                                    thinker_length,
                                    vp_emb,
                                    thinker_reply_part,
                                    audio_detokenizer,
                                    speaker,
                                    prompt_text_input_part,
                                    len(prompt_wav_token_bpe),
                                    stream=stream & True,
                                )
                            ):
                                yield this_tts_speech_dict["tts_speech"], (
                                    text if idx == 0 else ""
                                )
                        else:  # 非流式
                            for idx, this_tts_speech_dict in enumerate(
                                self.tts_job(
                                    text,
                                    prompt,
                                    prefix_from_thinker,
                                    vp,
                                    position_ids,
                                    talker_audio_prefix,
                                    vp_insert_loc,
                                    thinker_length,
                                    vp_emb,
                                    thinker_reply_part,
                                    audio_detokenizer,
                                    speaker,
                                    prompt_text_input_part,
                                    len(prompt_wav_token_bpe),
                                    stream=False,
                                )
                            ):
                                yield this_tts_speech_dict["tts_speech"], (
                                    text if idx == 0 else ""
                                )

                    streaming_text = []
                    count += 1
            elif ele != " ":
                streaming_text.append(ele)

        # for last sentence, if contain meaningful content
        if len(streaming_text) > 0 and re.search(
            r"[a-zA-Z\u4e00-\u9fff1-9]", "".join(streaming_text)
        ):
            # 中文
            if bool(re.search(r"[\u4e00-\u9fff]", "".join(streaming_text))):
                streaming_text = "".join(streaming_text)
            # 英文
            else:
                streaming_text = " ".join(streaming_text)
            text_list = self.cut_text(streaming_text, max_length)
            logging.info(f"for last sentence")
            audio_tokens = []
            for text in text_list:
                if text[0] == "，":
                    text = text[1:]
                if count == 0:  # 首句流式
                    for idx, this_tts_speech_dict in enumerate(
                        self.tts_job(
                            text,
                            prompt,
                            prefix_from_thinker,
                            vp,
                            position_ids,
                            talker_audio_prefix,
                            vp_insert_loc,
                            thinker_length,
                            vp_emb,
                            thinker_reply_part,
                            audio_detokenizer,
                            speaker,
                            prompt_text_input_part,
                            len(prompt_wav_token_bpe),
                            stream=stream & True,
                        )
                    ):
                        yield this_tts_speech_dict["tts_speech"], (
                            text if idx == 0 else ""
                        )
                else:  # 非流式
                    for idx, this_tts_speech_dict in enumerate(
                        self.tts_job(
                            text,
                            prompt,
                            prefix_from_thinker,
                            vp,
                            position_ids,
                            talker_audio_prefix,
                            vp_insert_loc,
                            thinker_length,
                            vp_emb,
                            thinker_reply_part,
                            audio_detokenizer,
                            speaker,
                            prompt_text_input_part,
                            len(prompt_wav_token_bpe),
                            stream=False,
                        )
                    ):
                        yield this_tts_speech_dict["tts_speech"], (
                            text if idx == 0 else ""
                        )
            streaming_text = []


class AudioDetokenizer:
    def __init__(self, config_path, flow_model_path, hifigan_model_path, spk_info=None):
        with open(config_path, "r") as f:
            configs = load_hyperpyyaml(f)
        self.model = AudioDetokenizerModel(configs["flow"], configs["hift"])
        self.model.load(flow_model_path, hifigan_model_path)
        self.sr = configs["sample_rate"]
        self.spk_info = spk_info
