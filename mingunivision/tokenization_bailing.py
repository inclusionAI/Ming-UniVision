#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast
from transformers.tokenization_utils_base import AddedToken, BatchEncoding
from transformers.utils import TensorType, logging

from chat_format import Chat

logger = logging.get_logger(__name__)


def is_system(msg):
    return msg['role'].lower() == 'system'


def is_user(msg):
    return msg['role'].lower() in ['human', 'user']


def is_assistant(msg):
    return msg['role'].lower() == 'assistant'


def _convert_to_conversation(query, system=None):
    conversation = []
    if system:
        conversation.append({"role": "SYSTEM", "content": system})
    if isinstance(query, str):
        conversation.append({"role": "HUMAN", "content": query})
    elif isinstance(query, List):
        conversation.extend(query)
    elif isinstance(query, Dict):
        if "messages" in query:
            conversation.extend(query["messages"])
            if "system_message" in query and len(conversation) > 0 and not is_system(conversation[0]):
                conversation.insert(0, {"role": "SYSTEM", "content": query["system_message"]})
        else:
            conversation.append(query)
    return conversation


class BailingTokenizer(PreTrainedTokenizerFast):
    is_bailing_tokenizer = True
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    # add gmask_token
    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "gmask_token",
        "additional_special_tokens",
    ]

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        clean_up_tokenization_spaces=False,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        cls_token="[CLS]",
        pad_token="<|endoftext|>",
        gmask_token="[gMASK]",
        add_bos_token=False,
        add_eos_token=False,
        **kwargs,
    ):
        self.add_bos_token = add_bos_token

        self._gmask_token = (
            AddedToken(gmask_token, lstrip=False, rstrip=False, normalized=False)
            if isinstance(gmask_token, str)
            else gmask_token
        )

        self._sop_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, normalized=False)
            if isinstance(bos_token, str)
            else bos_token
        )

        self._eop_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, normalized=False)
            if isinstance(eos_token, str)
            else eos_token
        )

        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            bos_token=bos_token,
            eos_token=eos_token,
            cls_token=cls_token,
            pad_token=eos_token,
            gmask_token=gmask_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            **kwargs,
        )

        self.check_special_tokens()

    def check_special_tokens(self):
        '''
        eos_token, cls_token, mask_token
        special tokens should init, check special token is not None
        '''
        for name, special_token in zip(
            ['eos', 'bos', 'cls', 'gmask'],
            [self.eos_token, self.bos_token, self.cls_token, self.gmask_token],
        ):
            assert special_token is not None, f'should init special token [{name}] in tokenizer_config.json'

    @property
    def gmask_token(self) -> Optional[str]:
        if self._gmask_token is None:
            if self.verbose:
                logger.error("Using gmask_token, but it is not set yet.")
            return None
        return str(self._gmask_token)

    @gmask_token.setter
    def gmask_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the gmask token")
        self._gmask_token = value

    @property
    def gmask_token_id(self) -> Optional[int]:
        if self._gmask_token is None:
            return None
        return self.convert_tokens_to_ids(self.gmask_token)

    @property
    def sop_token(self) -> Optional[str]:
        if self._sop_token is None:
            if self.verbose:
                logger.error("Using sop_token, but it is not set yet.")
            return None
        return str(self._sop_token)

    @sop_token.setter
    def sop_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the sop token")
        self._sop_token = value

    @property
    def sop_token_id(self) -> Optional[int]:
        if self._sop_token is None:
            return None
        return self.convert_tokens_to_ids(self.sop_token)

    @property
    def eop_token(self) -> Optional[str]:
        if self._eop_token is None:
            if self.verbose:
                logger.error("Using eop_token, but it is not set yet.")
            return None
        return str(self._eop_token)

    @eop_token.setter
    def eop_token(self, value):
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the eop token")
        self._eop_token = value

    @property
    def eop_token_id(self) -> Optional[int]:
        if self._eop_token is None:
            return None
        return self.convert_tokens_to_ids(self.eop_token)

    @property
    def vocab_size(self):
        return len(self.get_vocab())

    def _chat_from_json(self, chat, chat_format="antglm_chat", system=None):
        msgs = chat if "messages" not in chat else chat["messages"]
        _msgs = []
        sys_msg = None
        for msg in msgs:
            if is_system(msg):
                sys_msg = msg['content']
            else:
                _msgs.append(msg)
        chat = {"messages": _msgs}
        system = system or sys_msg
        if system:
            chat['system_message'] = system
        return Chat.from_json(chat, name=chat_format)

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        tools: Optional[List[Dict]] = None,
        documents: Optional[List[Dict[str, str]]] = None,
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        system: str = None,  # only used for legacy chatml
        tokenize=False,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if hasattr(PreTrainedTokenizerFast, "apply_chat_template"):
            # use transformers built-in method
            return super().apply_chat_template(
                conversation=conversation,
                tools=tools,
                documents=documents,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
                return_dict=return_dict,
                return_assistant_tokens_mask=return_assistant_tokens_mask,
                tokenizer_kwargs=tokenizer_kwargs,
            )
        chat_format = kwargs.get('chat_format', 'antglm_chat')

        is_batched = False

        if isinstance(conversation, List) and (
            isinstance(conversation[0], (list, tuple)) or "messages" in conversation[0]
        ):
            conversations = conversation
            is_batched = True

        if not is_batched:
            conversations = [conversation]

        rendered = []
        for chat in conversations:
            rendered_chat = self._chat_from_json(chat, chat_format=chat_format, system=system).prompt_str
            rendered.append(rendered_chat)

        if not is_batched:
            rendered = rendered[0]

        if tokenize:
            out = self(
                rendered,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                add_special_tokens=False,
                return_tensors=return_tensors,
            )
            if return_dict:
                return out
            else:
                return out["input_ids"]
        else:
            return rendered

    def _build_position_ids(
        self,
        mask_pos: int,
        bos_pos: int,
        max_output_length: int,
        rotary_type: Optional[str] = "none",
        **kwargs,
    ) -> List[List[int]]:
        window_size = kwargs.get("window_size", 1024) - 1
        block_position_ids = [0] * bos_pos

        # 获得mask所在的位置，用于后面output positionid的构造
        if "1d" in rotary_type:
            position_ids = list(range(bos_pos)) + list(range(mask_pos + 1, mask_pos + max_output_length + 2))
            block_position_ids = block_position_ids + list(range(1, max_output_length + 2))
        elif "2d" in rotary_type:
            # 后面input_ids要加一个bos_id
            position_ids = list(range(bos_pos))
            position_ids = position_ids + [mask_pos] * (1 + max_output_length)
            block_position_ids = block_position_ids + list(range(1, max_output_length + 2))
        else:
            # build position ids
            position_ids = []
            repeat_times = bos_pos // window_size
            for _ in range(repeat_times):
                position_ids += list(range(window_size))
            position_ids += list(range(bos_pos - window_size * repeat_times))
            # need consider additional bos_id after input_ids
            mask_pos = position_ids[-1]
            position_ids += [mask_pos] * (max_output_length + 1)

            block_repeat_times = max_output_length // (window_size - 1)
            additional_block_position_ids = []
            for _ in range(block_repeat_times):
                additional_block_position_ids += list(range(1, window_size))
            additional_block_position_ids += list(
                range(1, max_output_length + 2 - (window_size - 1) * block_repeat_times)
            )
            block_position_ids = block_position_ids + additional_block_position_ids

        position_ids = [position_ids, block_position_ids]
        return position_ids

    def _build_inputs_for_generation(
        self,
        input_ids: List[int],
        max_input_length=None,
        left_truncate=True,
        max_output_length=1024,
        rotary_type="none",
        unidirectional_attention: bool = True,
        **kwargs,
    ):
        if max_input_length and len(input_ids) > max_input_length:
            if left_truncate:
                input_ids = input_ids[-max_input_length:]
            else:
                input_ids = input_ids[:max_input_length]

        is_left_padding = input_ids[0] == self.eos_token_id
        if not unidirectional_attention:
            if input_ids[0] != self.cls_token_id:
                input_ids = [self.cls_token_id] + input_ids

            if self.gmask_token_id not in set(input_ids):
                input_ids = input_ids + [self.gmask_token_id]

            mask_pos = input_ids.index(self.gmask_token_id)
            sep = len(input_ids)
        else:
            if self.add_bos_token:
                input_ids = input_ids + [self.bos_token_id]
                if self.eos_token_id in input_ids:
                    mask_pos = input_ids.index(self.eos_token_id) - 1
                else:
                    mask_pos = len(input_ids) - 1
                sep = len(input_ids) - 1
            else:
                sep = len(input_ids)
                if self.eos_token_id in input_ids:
                    if is_left_padding:
                        ori_input_ids = input_ids
                        input_ids = input_ids[::-1]
                    mask_pos = input_ids.index(self.eos_token_id) - 1
                    mask_pos = max(0, mask_pos)  # for empty sequence
                    if is_left_padding:
                        input_ids = ori_input_ids
                        mask_pos = sep - 1 - mask_pos  # the first non-eos token

                else:
                    mask_pos = len(input_ids) - 1

        position_ids = self._build_position_ids(mask_pos, sep, max_output_length, rotary_type, **kwargs)

        if is_left_padding:
            position_ids[0] = [max(0, i - mask_pos) for i in range(len(position_ids[0]))]

        # 后面input_ids要加一个bos_id
        total_length = sep + max_output_length
        if self.add_bos_token:
            total_length += 1

        def build_mask_matrix(seq_length, sep, mask_pos, unidirectional_attention):

            if unidirectional_attention:
                attention_mask = np.ones([seq_length, seq_length])
                attention_mask = np.tril(attention_mask)
                if is_left_padding:
                    attention_mask[:, :mask_pos] = 0
                else:
                    attention_mask[:, mask_pos + 1 : sep] = 0
            else:
                attention_mask = np.zeros([seq_length, seq_length])
                attention_mask[:, : mask_pos + 1] = 1
                for i in range(sep, total_length):
                    attention_mask[i, sep : i + 1] = 1
            return attention_mask

        if self.add_bos_token:
            attention_mask = build_mask_matrix(total_length, sep + 1, mask_pos, unidirectional_attention)
        else:
            attention_mask = build_mask_matrix(total_length, sep, mask_pos, unidirectional_attention)

        inputs = {
            "input_ids": torch.Tensor([input_ids]).long(),
            "position_ids": torch.Tensor([position_ids]).long(),
            "attention_mask": torch.Tensor(np.expand_dims(attention_mask, axis=[0, 1])).long(),
        }

        return BatchEncoding(inputs)

    def build_inputs_for_generation(
        self,
        input_ids: Union[List[int], List[List[int]], torch.Tensor],
        max_input_length=None,
        left_truncate=True,
        max_output_length=1024,
        rotary_type="1d",
        unidirectional_attention=True,
        **kwargs,
    ):
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        if isinstance(input_ids[0], list):
            input_ids_list = []
            position_ids_list = []
            attention_mask_list = []
            for _input_ids in input_ids:
                inputs = self._build_inputs_for_generation(
                    _input_ids,
                    max_input_length=max_input_length,
                    left_truncate=left_truncate,
                    max_output_length=max_output_length,
                    rotary_type=rotary_type,
                    unidirectional_attention=unidirectional_attention,
                    **kwargs,
                )
                input_ids_list.append(inputs['input_ids'])
                position_ids_list.append(inputs['position_ids'])
                attention_mask_list.append(inputs["attention_mask"])

            max_ids_length = max([input.size(1) for input in input_ids_list])

            for i in range(len(input_ids)):
                cur_ids_length = input_ids_list[i].size(1)
                if cur_ids_length < max_ids_length:
                    # pad input ids
                    pad_input_ids = input_ids_list[i].new_zeros((1, max_ids_length - cur_ids_length))
                    input_ids_list[i] = torch.cat([pad_input_ids, input_ids_list[i]], dim=-1)

                    # pad postition ids with left pad
                    # 0, 1, 2, 3, 4 ... -> 0, ..., 0, 1, 2, 3, 4, ...
                    pad_position_ids = input_ids_list[i].new_zeros((1, 2, max_ids_length - cur_ids_length))
                    position_ids_list[i] = torch.cat([pad_position_ids, position_ids_list[i]], dim=-1)

                    # pad generation attention mask with left and bottom pad
                    new_attention_mask = input_ids_list[i].new_zeros(
                        1,
                        1,
                        max_ids_length + max_output_length,
                        max_ids_length + max_output_length,
                    )
                    new_attention_mask[
                        :,
                        :,
                        max_ids_length - cur_ids_length :,
                        max_ids_length - cur_ids_length :,
                    ] = attention_mask_list[i]
                    attention_mask_list[i] = new_attention_mask.contiguous()

            input_ids_list = torch.cat(input_ids_list, dim=0)
            position_ids_list = torch.cat(position_ids_list, dim=0)
            attention_mask_list = torch.cat(attention_mask_list, dim=0)

            inputs = {
                "input_ids": input_ids_list,
                "position_ids": position_ids_list,
                "attention_mask": attention_mask_list,
            }

            return BatchEncoding(inputs)
        else:
            return self._build_inputs_for_generation(
                input_ids,
                max_input_length=max_input_length,
                left_truncate=left_truncate,
                max_output_length=max_output_length,
                rotary_type=rotary_type,
                unidirectional_attention=unidirectional_attention,
                **kwargs,
            )

    def _build_inputs_for_train(
        self,
        inputs: Union[str, List[str]],
        outputs: Union[str, List[str]],
        new_conversation_offset: List[int] = None,
        num_query_token: int = None,
        max_num_query_token: int = None,
        max_length: int = 2048,
        rotary_type: str = "1d",
        left_truncate: bool = True,
        unidirectional_attention: bool = True,
        isolation_position_ids: bool = False,
        padding: bool = True,
        use_fa2: bool = True,
        use_packed: bool = True,
    ):
        r"""
        Build tensor input for model training. If inputs and outputs are list, will pack them.

        Args:
            inputs (str, List[str], List[Dict], List[List[Dict]]): the input prompts.
            outputs (str, List[str]): the output responses.
            max_length (int, Optional): the maximum length of the final input ids for training. Default: 2048
            rotary_type (str, Optional): the rotary type of position embedding. Default: 1d
            left_truncate (bool, Optional): whether truncate the inputs from left. Default: True
            use_fa2 (bool, Optional): whether to build attention mask under flash attention 2.
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]

        assert len(inputs) == len(outputs)

        inputs = [item.replace('\\n', '\n') for item in inputs]
        input_ids = [self(item)['input_ids'] for item in inputs]

        outputs = [item.replace('\\n', '\n') for item in outputs]
        output_ids = [self(item)['input_ids'] for item in outputs]

        packed_input_ids = []
        packed_output_ids = []
        current_len = 0

        for idx, (input, output) in enumerate(zip(input_ids, output_ids)):
            num_special_tokens = 0
            if not unidirectional_attention:
                if (
                    idx == 0
                    or not new_conversation_offset
                    or (new_conversation_offset and idx in new_conversation_offset)
                ):
                    # cls and gmask
                    num_special_tokens += 2
                else:
                    # only gmask
                    num_special_tokens += 1
            else:
                # sop and eos
                if self.add_bos_token:
                    num_special_tokens += 2
                else:
                    num_special_tokens += 1

            # truncate
            if len(input) + len(output) + current_len > max_length - num_special_tokens:
                left_len = max_length - num_special_tokens - current_len
                if len(input) > left_len // 2 and len(output) > left_len // 2:
                    # 如果都超过了最大长度的一半,那都截取到最大长度的一半
                    if left_truncate:
                        input = input[-left_len // 2 :]
                    else:
                        input = input[: left_len // 2]
                    output = output[: left_len // 2]
                else:
                    # 从input_ids和output_ids中比较长的那一个截断,input_ids可以选择从左边或右边阶段,output_ids默认从右边截断
                    if len(input) >= len(output):
                        if left_truncate:
                            input = input[-(left_len - len(output)) :]
                        else:
                            input = input[: left_len - len(output)]
                    else:
                        output = output[: left_len - len(input)]
                if unidirectional_attention:
                    packed_input_ids.append(list(input))
                else:
                    if num_special_tokens == 4:
                        packed_input_ids.append([self.cls_token_id] + list(input) + [self.gmask_token_id])
                    else:
                        packed_input_ids.append(list(input) + [self.gmask_token_id])

                packed_output_ids.append(list(output) + [self.eos_token_id])
                current_len += len(input) + len(output) + num_special_tokens
                break
            if unidirectional_attention:
                packed_input_ids.append(list(input))
            else:
                if num_special_tokens == 4:
                    packed_input_ids.append([self.cls_token_id] + list(input) + [self.gmask_token_id])
                else:
                    packed_input_ids.append(list(input) + [self.gmask_token_id])

            packed_output_ids.append(list(output) + [self.eos_token_id])
            current_len += len(input) + len(output) + num_special_tokens

        assert current_len <= max_length

        if use_packed:
            # pack模式
            def build_mask_matrix(seq_length, sep):
                # https://github.com/pytorch/pytorch/issues/101932, fix triu/tril bf16 support
                m = torch.ones((1, seq_length, seq_length))
                mask = torch.arange(1, m.shape[-1] + 1).reshape(1, -1, 1).to(m.device)
                ids = torch.arange(1, m.shape[-1] + 1).reshape(1, 1, -1).expand(1, m.shape[-1], -1).to(m.device)
                m = (ids <= mask).type_as(m)

                m[0, :, : int(sep)] = 1
                m = m.squeeze(0)
                return m

            tokens = []
            attention_mask_list = []
            input_length_list = []
            position_id_list = []
            block_position_id_list = []

            for input, output in zip(packed_input_ids, packed_output_ids):
                if self.add_bos_token:
                    data = input + [self.sop_token_id] + output
                    mask_pos = len(input) - 1
                else:
                    data = input + output
                    mask_pos = len(input) - 2
                if unidirectional_attention:
                    attention_mask = build_mask_matrix(len(data), 0)
                else:
                    attention_mask = build_mask_matrix(len(data), len(input))
                attention_mask = attention_mask.squeeze((0, 1))

                attention_mask_list.append(attention_mask)
                input_length_list.append(len(input))
                tokens += data

                sop_pos = mask_pos + 1
                position_ids, block_position_ids = self._build_position_ids(
                    mask_pos=mask_pos, bos_pos=sop_pos, max_output_length=len(output), rotary_type=rotary_type
                )

                position_id_list.append(position_ids)
                block_position_id_list.append(block_position_ids)

            labels = []
            for i in range(len(packed_input_ids)):
                if self.add_bos_token:
                    labels += [-100] * len(packed_input_ids[i]) + packed_output_ids[i] + [-100]
                else:
                    labels += [-100] * (len(packed_input_ids[i]) - 1) + packed_output_ids[i] + [-100]

            total_len = 0
            if use_fa2:
                pack_attention_mask = -1 * torch.ones([2, current_len])
            else:
                pack_attention_mask = torch.tril(torch.ones([current_len, current_len]))

            pack_position_ids = []
            pack_block_position_ids = []
            total_len = 0
            max_index = 0
            for i in range(len(attention_mask_list)):
                attention_mask = attention_mask_list[i]
                if use_fa2:
                    pack_attention_mask[0][i] = total_len
                    pack_attention_mask[1][i] = total_len + input_length_list[i]
                else:
                    pack_attention_mask[
                        total_len : total_len + attention_mask.shape[0],
                        total_len : total_len + attention_mask.shape[0],
                    ] = attention_mask
                position_ids = [pid + max_index for pid in position_id_list[i]]
                block_position_ids = block_position_id_list[i]
                pack_position_ids.extend(position_ids)
                pack_block_position_ids.extend(block_position_ids)
                if not isolation_position_ids:
                    max_index = pack_position_ids[-1] + 1
                total_len += len(attention_mask_list[i])
            position_ids = [pack_position_ids, pack_block_position_ids]
        else:
            # 单输入模式
            input, output = packed_input_ids[0], packed_output_ids[0]
            if self.add_bos_token:
                tokens = input + [self.sop_token_id] + output
            else:
                tokens = input + output

            attention_mask = len(input)
            if self.add_bos_token:
                labels = [-100] * len(input) + output + [-100]
                position_ids = self._build_position_ids(
                    mask_pos=len(input) - 1, bos_pos=len(input), max_output_length=len(output), rotary_type=rotary_type
                )
            else:
                labels = [-100] * (len(input) - 1) + output + [-100]
                position_ids = self._build_position_ids(
                    mask_pos=len(input) - 2,
                    bos_pos=len(input) - 1,
                    max_output_length=len(output),
                    rotary_type=rotary_type,
                )

        assert len(tokens) == current_len

        # 最大长度补全
        if max_length > 0 and len(tokens) < max_length and padding:
            pad_length = max_length - len(tokens)
            tokens += [self.pad_token_id] * pad_length
            labels.extend([-100] * pad_length)
            position_ids[0] += [0] * pad_length
            position_ids[1] += [0] * pad_length

            if use_packed:
                if use_fa2:
                    new_attention_mask = -1 * torch.ones([2, max_length])
                    new_attention_mask[:, :current_len] = pack_attention_mask
                else:
                    new_attention_mask = torch.tril(torch.ones([max_length, max_length]))
                    new_attention_mask[:current_len, :current_len] = pack_attention_mask
                pack_attention_mask = new_attention_mask.contiguous()

        assert len(tokens) == len(labels)

        if max_length > 0 and padding:
            assert len(tokens) == max_length

        if use_fa2 and unidirectional_attention:
            # pack_attention_mask = torch.zeros([1], dtype=torch.long)
            pack_attention_mask = 0

        if use_packed:
            if not use_fa2:
                attention_mask = pack_attention_mask.unsqueeze(0).long()
            else:
                attention_mask = pack_attention_mask
        else:
            attention_mask = torch.tensor(attention_mask).long()
        return {
            'input_ids': torch.tensor(tokens).long(),
            'position_ids': torch.tensor(position_ids).long(),
            'attention_mask': attention_mask,
            'labels': torch.tensor(labels).long(),
        }

    def build_inputs_for_train(
        self,
        data: Union[Dict, List[Dict]],
        new_conversation_offset: List[int] = None,
        chat_format="antglm_chat",
        is_chat_format=True,  # 如果传入的是字符串，用于说明是否已经是
        use_true_multiturn=False,
        max_length: int = 2048,
        rotary_type: str = "1d",
        left_truncate: bool = True,
        unidirectional_attention: bool = True,
        isolation_position_ids: bool = False,
        padding: bool = True,
        use_fa2: bool = True,
        use_packed: bool = True,
    ):
        r"""
        Build tensor input for model training. If inputs and outputs are list, will pack them.

        Args:
            inputs (str, List[str], List[Dict], List[List[Dict]]): the input prompts.
            outputs (str, List[str]): the output responses.
            new_conversation_offset (List[int]): the offset index of the new conversation turn.
            is_chat_format (bool): whether the input is already chatml format
            max_length (int, Optional): the maximum length of the final input ids for training. Default: 2048
            rotary_type (str, Optional): the rotary type of position embedding. Default: 1d
            left_truncate (bool, Optional): whether truncate the inputs from left. Default: True
            use_fa2 (bool, Optional): whether to build attention mask under flash attention 2.
        """
        if isinstance(data, List):
            # chatml list
            _inputs = []
            _outputs = []
            new_conversation_offset = []
            for _input in data:
                if use_true_multiturn:
                    chat = self._chat_from_json(_input, chat_format=chat_format)
                    chat_data = chat.prompt_pack
                    new_conversation_offset.append(len(_inputs))
                    _inputs.extend(chat_data['input'])
                    _outputs.extend(chat_data['output'])
                else:
                    _conversation = _convert_to_conversation(_input)
                    assert is_assistant(_conversation[-1])

                    _inputs.append(
                        self.apply_chat_template(_conversation[:-1], tokenize=False, add_generation_prompt=True)
                    )
                    _outputs.append(_conversation[-1]['content'])

            return self._build_inputs_for_train(
                inputs=_inputs,
                outputs=_outputs,
                new_conversation_offset=new_conversation_offset,
                max_length=max_length,
                rotary_type=rotary_type,
                left_truncate=left_truncate,
                unidirectional_attention=unidirectional_attention,
                isolation_position_ids=isolation_position_ids,
                padding=padding,
                use_fa2=use_fa2,
                use_packed=use_packed,
            )
        elif isinstance(data, Dict):
            if 'messages' in data:
                # chatml format
                if use_true_multiturn:
                    chat = self._chat_from_json(data, chat_format=chat_format)
                    chat_data = chat.prompt_pack
                else:
                    _conversation = _convert_to_conversation(data)
                    assert is_assistant(_conversation[-1])

                    chat_data = {
                        "input": self.apply_chat_template(
                            _conversation[:-1], tokenize=False, add_generation_prompt=True
                        ),
                        "output": _conversation[-1]['content'],
                    }

                return self._build_inputs_for_train(
                    inputs=chat_data['input'],
                    outputs=chat_data['output'],
                    max_length=max_length,
                    rotary_type=rotary_type,
                    left_truncate=left_truncate,
                    unidirectional_attention=unidirectional_attention,
                    isolation_position_ids=isolation_position_ids,
                    padding=padding,
                    use_fa2=use_fa2,
                    use_packed=use_packed,
                )
            else:
                inputs = data['input']
                outputs = data['output']

                if isinstance(inputs, str):
                    inputs = [inputs]
                if isinstance(outputs, str):
                    outputs = [outputs]

                if not is_chat_format and chat_format:
                    inputs = [
                        self.apply_chat_template(
                            [{"role": "HUMAN", "content": item}], tokenize=False, chat_format=chat_format
                        )
                        for item in inputs
                    ]

                return self._build_inputs_for_train(
                    inputs=inputs,
                    outputs=outputs,
                    new_conversation_offset=new_conversation_offset,
                    max_length=max_length,
                    rotary_type=rotary_type,
                    left_truncate=left_truncate,
                    unidirectional_attention=unidirectional_attention,
                    isolation_position_ids=isolation_position_ids,
                    padding=padding,
                    use_fa2=use_fa2,
                    use_packed=use_packed,
                )

