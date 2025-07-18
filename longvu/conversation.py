import base64
import dataclasses
from enum import auto, Enum
from io import BytesIO
from typing import Any, Dict, List, Tuple, Union

from longvu.file_io import PathManager

from PIL import Image
from transformers import AutoTokenizer


class LongVUSeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()
    LLAMA_3 = auto()
    LLAMA_3_1 = auto()
    LLAMA_3_2 = auto()
    QWEN = auto()
    CHATML = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: LongVUSeparatorStyle = LongVUSeparatorStyle.SINGLE
    sep: str = "###"
    # pyre-fixme[8]: Attribute has type `str`; used as `None`.
    sep2: str = None
    version: str = "Unknown"

    tokenizer: Any = None
    # Stop criteria (the default one is EOS token)
    # pyre-fixme[8]: Attribute has type `Union[List[str], str]`; used as `None`.
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    # pyre-fixme[8]: Attribute has type `List[int]`; used as `None`.
    stop_token_ids: List[int] = None

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if "mmtag" in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == LongVUSeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == LongVUSeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"

        elif self.sep_style == LongVUSeparatorStyle.CHATML:
            ret = "" if self.system == "" else self.system + self.sep + "\n"
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images, _ = message
                        message = "<image>" * len(images) + message
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret

        elif self.sep_style == LongVUSeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == LongVUSeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: (
                f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            )
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0:
                        message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == LongVUSeparatorStyle.LLAMA_3:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "Vision-CAIR/LongVU_Llama3_2_3B"
                )
            chat_template_messages = [{"role": "system", "content": self.system}]
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = "<image>" * len(images) + message
                    chat_template_messages.append({"role": role, "content": message})

            # print("chat", chat_template_messages, flush=True)
            return self.tokenizer.apply_chat_template(
                chat_template_messages, tokenize=False, add_generation_prompt=True
            )
        elif self.sep_style == LongVUSeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def process_image(
        self,
        image,
        image_process_mode,
        return_pil=False,
        image_format="PNG",
        max_len=1344,
        min_len=672,
    ):
        if image_process_mode == "Pad":

            def expand2square(pil_img, background_color=(122, 116, 104)):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
        if max(image.size) > max_len:
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
            longest_edge = int(shortest_edge * aspect_ratio)
            W, H = image.size
            if H > W:
                H, W = longest_edge, shortest_edge
            else:
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))
        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    image = self.process_image(
                        image, image_process_mode, return_pil=return_pil
                    )
                    images.append(image)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    img_b64_str = self.process_image(
                        image, "Default", return_pil=False, image_format="JPEG"
                    )
                    img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace("<image>", "").strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [
                    [x, y[0] if type(y) is tuple else y] for x, y in self.messages
                ],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    # pyre-fixme[6]: For 2nd argument expected `List[str]` but got `Tuple[str, str]`.
    roles=("Human", "Assistant"),
    # pyre-fixme[6]: For 3rd argument expected `List[List[str]]` but got
    #  `Tuple[Tuple[str, str], Tuple[str, str]]`.
    messages=(
        (
            "Human",
            "What are the key differences between renewable and non-renewable energy sources?",
        ),
        (
            "Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n",
        ),
    ),
    offset=2,
    sep_style=LongVUSeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    # pyre-fixme[6]: For 2nd argument expected `List[str]` but got `Tuple[str, str]`.
    roles=("USER", "ASSISTANT"),
    version="v1",
    # pyre-fixme[6]: For 4th argument expected `List[List[str]]` but got `Tuple[]`.
    messages=(),
    offset=0,
    sep_style=LongVUSeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    # pyre-fixme[6]: For 2nd argument expected `List[str]` but got `Tuple[str, str]`.
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    # pyre-fixme[6]: For 4th argument expected `List[List[str]]` but got `Tuple[]`.
    messages=(),
    offset=0,
    sep_style=LongVUSeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language.",
    # pyre-fixme[6]: For 2nd argument expected `List[str]` but got `Tuple[str, str]`.
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    # pyre-fixme[6]: For 4th argument expected `List[List[str]]` but got `Tuple[]`.
    messages=(),
    offset=0,
    sep_style=LongVUSeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    # pyre-fixme[6]: For 2nd argument expected `List[str]` but got `Tuple[str, str]`.
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    # pyre-fixme[6]: For 4th argument expected `List[List[str]]` but got `Tuple[]`.
    messages=(),
    offset=0,
    sep_style=LongVUSeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llava_plain = Conversation(
    system="",
    # pyre-fixme[6]: For 2nd argument expected `List[str]` but got `Tuple[str, str]`.
    roles=("", ""),
    # pyre-fixme[6]: For 3rd argument expected `List[List[str]]` but got `Tuple[]`.
    messages=(),
    offset=0,
    sep_style=LongVUSeparatorStyle.PLAIN,
    sep="\n",
    version="plain",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    # pyre-fixme[6]: For 2nd argument expected `List[str]` but got `Tuple[str, str]`.
    roles=("Human", "Assistant"),
    # pyre-fixme[6]: For 3rd argument expected `List[List[str]]` but got `Tuple[]`.
    messages=(),
    offset=0,
    sep_style=LongVUSeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    "The visual content will be provided with the following format: <Image>visual content</Image>.",
    # pyre-fixme[6]: For 2nd argument expected `List[str]` but got `Tuple[str, str]`.
    roles=("Human", "Assistant"),
    # pyre-fixme[6]: For 3rd argument expected `List[List[str]]` but got `Tuple[]`.
    messages=(),
    offset=0,
    sep_style=LongVUSeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    # pyre-fixme[6]: For 2nd argument expected `List[str]` but got `Tuple[str, str]`.
    roles=("USER", "ASSISTANT"),
    version="v1",
    # pyre-fixme[6]: For 4th argument expected `List[List[str]]` but got `Tuple[]`.
    messages=(),
    offset=0,
    sep_style=LongVUSeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    "The visual content will be provided with the following format: <Image>visual content</Image>.",
    # pyre-fixme[6]: For 2nd argument expected `List[str]` but got `Tuple[str, str]`.
    roles=("USER", "ASSISTANT"),
    # pyre-fixme[6]: For 3rd argument expected `List[List[str]]` but got `Tuple[]`.
    messages=(),
    offset=0,
    sep_style=LongVUSeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

conv_mistral_instruct = Conversation(
    system="",
    # pyre-fixme[6]: For 2nd argument expected `List[str]` but got `Tuple[str, str]`.
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    # pyre-fixme[6]: For 4th argument expected `List[List[str]]` but got `Tuple[]`.
    messages=(),
    offset=0,
    sep_style=LongVUSeparatorStyle.LLAMA_2,
    sep="",
    sep2="</s>",
)

conv_chatml_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    # pyre-fixme[6]: For 2nd argument expected `List[str]` but got `Tuple[str, str]`.
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    # pyre-fixme[6]: For 4th argument expected `List[List[str]]` but got `Tuple[]`.
    messages=(),
    offset=0,
    sep_style=LongVUSeparatorStyle.MPT,
    sep="<|im_end|>",
)

llama3_tokenizer = AutoTokenizer.from_pretrained(
    "Vision-CAIR/LongVU_Llama3_2_3B"
)

conv_llama3 = Conversation(
    system="""You are a helpful assistant.""",
    # pyre-fixme[6]: For 2nd argument expected `List[str]` but got `Tuple[str, str]`.
    roles=("user", "assistant"),
    version="llama3",
    # pyre-fixme[6]: For 4th argument expected `List[List[str]]` but got `Tuple[]`.
    messages=(),
    offset=0,
    sep_style=LongVUSeparatorStyle.LLAMA_3,
    tokenizer=llama3_tokenizer,
    sep="<|eot_id|>",
)

conv_llama3_2 = Conversation(
    system="""You are a helpful assistant.""",
    # pyre-fixme[6]: For 2nd argument expected `List[str]` but got `Tuple[str, str]`.
    roles=("user", "assistant"),
    version="llama3_2",
    # pyre-fixme[6]: For 4th argument expected `List[List[str]]` but got `Tuple[]`.
    messages=(),
    offset=0,
    sep_style=LongVUSeparatorStyle.LLAMA_3_2,
    sep="<|eot_id|>",
)

conv_phi3_instruct = Conversation(
    system="""<|system|>\nYou are a helpful AI assistant.""",
    # pyre-fixme[6]: For 2nd argument expected `List[str]` but got `Tuple[str, str]`.
    roles=("\n<|user|>\n", "\n<|assistant|>\n"),
    version="phi3",
    # pyre-fixme[6]: For 4th argument expected `List[List[str]]` but got `Tuple[]`.
    messages=(),
    offset=0,
    sep_style=LongVUSeparatorStyle.MPT,
    sep="<|end|>",
)

conv_qwen = Conversation(
    system="""<|im_start|>system
You are a helpful assistant.""",
    # pyre-fixme[6]: For 2nd argument expected `List[str]` but got `Tuple[str, str]`.
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    version="qwen",
    messages=[],
    offset=0,
    sep_style=LongVUSeparatorStyle.CHATML,
    sep="<|im_end|>",
)

default_conversation = conv_vicuna_v1
longvu_conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llama_2": conv_llama_2,
    "mistral_instruct": conv_mistral_instruct,
    "chatml_direct": conv_chatml_direct,
    "mistral_direct": conv_chatml_direct,
    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "llava_v0": conv_llava_v0,
    "v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,
    "mpt": conv_mpt,
    "llama3": conv_llama3,
    "llama3_2": conv_llama3_2,
    "phi3": conv_phi3_instruct,
    "qwen": conv_qwen,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())
