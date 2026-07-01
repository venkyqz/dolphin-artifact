import traceback
from typing import ClassVar, List, Optional

import httpx
import litellm
import loguru
import openai
import yaml
from litellm.types.utils import ModelResponse
from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field

from sweagent.agent.models import GenericAPIModelConfig, LiteLLMModel
from sweagent.config import env
from sweagent.exceptions import (
    ContextWindowExceededError,
    ContentPolicyViolationError,
)
from sweagent.tools.tools import ToolConfig

log = loguru.logger

client_pool = {}


class Message(BaseModel):
    role: str = ...
    content: str = ...
    enable: bool = True
    mid: str = ""  # message id
    message_type: Optional[str] = None

    @classmethod
    def sample(cls):
        return Message(role="user", content="sample")

    def next_role(self):
        if self.role == "user":
            return "assistant"
        else:
            return "user"


class ProviderModel(BaseModel):
    name: str = ""
    api_base: str = ""
    api_key: str = ""
    models: List[str] = []
    proxy: str = ""
    enabled: bool = True


class SystemConfigModel(BaseModel):
    __parsed: ClassVar["SystemConfigModel"] = None

    models: List[str] = []
    default_model: str = Field(alias="default_model", default="")
    api_key: str = ""
    api_base: str = ""
    providers: List[ProviderModel] = []
    file: str = ""

    def get_provider(self, model):
        for provider in self.providers:
            if provider.enabled:
                if model in provider.models:
                    return provider
        return None

    def list_models(self):
        models = []
        for provider in self.providers:
            if provider.enabled:
                models.extend(provider.models)
        return models

    @classmethod
    def singleton(cls) -> "SystemConfigModel":
        if not cls.__parsed:
            return cls.parse_from(env.settings.SWE_LLM_CONFIG)
        return cls.__parsed

    @classmethod
    def parse_from(cls, file):
        with open(file) as fd:
            data = yaml.load(fd, yaml.SafeLoader)
            cls.__parsed = SystemConfigModel(**data)
        cls.__parsed.file = file
        return cls.__parsed


class Conversation(BaseModel):
    messages: List[Message] = Field(default_factory=list)

    def set_system(self, m: Message):
        if self.messages:
            if self.messages[0].role == "system":
                self.messages[0] = m
                return
        self.messages.insert(0, m)
        return

    @staticmethod
    def to_message(messages):
        res = []
        for i in messages:
            if i.enable:
                res.append(dict(role=i.role, content=i.content))
        return res

    def to_payload(self):
        return self.to_message(self.messages)

    def pp_str(self):
        res = []
        for m in self.messages:
            res.append("{}:\n{}".format(m.role, m.content))
        return "\n".join(res)

    def append_content(self, content, role="user"):
        if not self.messages:
            self.messages.append(
                Message(
                    role=role,
                    content=content,
                )
            )
        else:
            if role:
                self.messages.append(Message(content=content, role=role))
            else:
                self.messages.append(
                    Message(content=content, role=self.messages[-1].next_role())
                )


def build_conversation(
    messages, role="user", system="", message_type=None
) -> Conversation:
    res = []
    if system:
        res.append(Message(role="system", content=system))
    if role == "user":
        order = ["user", "assistant"]
    else:
        order = ["assistant", "user"]

    for idx, m in enumerate(messages):
        role = order[idx % 2]
        res.append(Message(role=role, content=m, message_type=message_type))

    return Conversation(messages=res)


async def simple_chat(content):
    messages = [dict(role="user", content=content)]
    return completion(messages)


def list_llm_models():
    config = SystemConfigModel.singleton()
    for provider in config.providers:
        proxy = provider.proxy

        if proxy:
            client = OpenAI(
                api_key=provider.api_key,
                base_url=provider.api_base,
                http_client=httpx.Client(proxy=str(proxy)),
            )
        else:
            client = OpenAI(api_key=provider.api_key, base_url=provider.api_base)

        res = client.models.list()
        for m in res:
            print(m)


def completion(
    messages,
    model="",
    top_p=None,
    temperature=None,
    tools=None,
    tool_choice=None,
    **kwargs,
) -> ChatCompletion:
    config = SystemConfigModel.singleton()

    if not model:
        model = SystemConfigModel.singleton().default_model

    provider: ProviderModel = config.get_provider(model)

    proxy = provider.proxy

    key = "{}+{}".format(provider.api_base, proxy)
    client = client_pool.get(key, None)

    if not client:
        # TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url=provider.api_base)'
        if proxy:
            client = OpenAI(
                api_key=provider.api_key,
                base_url=provider.api_base,
                http_client=httpx.Client(proxy=str(proxy)),
            )
        else:
            client = OpenAI(api_key=provider.api_key, base_url=provider.api_base)

        client_pool[key] = client

    try:
        messages = [dict(role=i["role"], content=i["content"]) for i in messages]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=30,
            top_p=top_p,
            temperature=temperature,
            tools=tools or openai.NOT_GIVEN,
            tool_choice=tool_choice or openai.NOT_GIVEN,
        )

        log.info("response: {}", response)

        choice = response.choices[0]
        log.info("content: {}", choice.message.content)

        return response
    except Exception as e:
        raise e

class SimpleLLMModel(LiteLLMModel):
    def __init__(self, args: GenericAPIModelConfig, tools: ToolConfig):
        super().__init__(args, tools)


    def _update_stats(
        self, *, input_tokens: int, output_tokens: int, cost: float
    ) -> None:
        ...

    def _sleep(self) -> None:
        ...

    def _single_query(
        self,
        messages: list[dict[str, str]],
        n: int | None = None,
        temperature: float | None = None,
    ) -> list[dict]:
        self._sleep()
        input_tokens: int = litellm.utils.token_counter(
            messages=messages, model=self.config.name
        )
        if self.model_max_input_tokens is None:
            msg = (
                f"No max input tokens found for model {self.config.name!r}. "
                "If you are using a local model, you can set `max_input_token` in the model config to override this."
            )
            self.logger.warning(msg)
        elif input_tokens > self.model_max_input_tokens > 0:
            msg = f"Input tokens {input_tokens} exceed max tokens {self.model_max_input_tokens}"
            raise ContextWindowExceededError(msg)
        extra_args = {}
        if self.config.api_base:
            # Not assigned a default value in litellm, so only pass this if it's set
            extra_args["api_base"] = self.config.api_base
            extra_args["base_url"] = self.config.api_base
        if self.tools.use_function_calling:
            extra_args["tools"] = self.tools.tools
        # We need to always set max_tokens for anthropic models
        completion_kwargs = self.config.completion_kwargs
        if self.lm_provider == "anthropic":
            completion_kwargs["max_tokens"] = self.model_max_output_tokens
        try:
            response: openai.ChatCompletion = completion(  # type: ignore
                model=self.config.name,
                messages=messages,
                temperature=self.config.temperature
                if temperature is None
                else temperature,
                top_p=self.config.top_p,
                **extra_args,
            )
        except litellm.exceptions.ContextWindowExceededError as e:
            raise ContextWindowExceededError from e
        except litellm.exceptions.ContentPolicyViolationError as e:
            raise ContentPolicyViolationError from e
        except litellm.exceptions.BadRequestError as e:
            if "is longer than the model's context length" in str(e):
                raise ContextWindowExceededError from e
            raise
        except Exception as e:
            traceback.print_exception(e)
            raise e
        cost = 0
        # try:
        #     cost = litellm.cost_calculator.completion_cost(response)
        # except Exception as e:
        #     self.logger.debug(f"Error calculating cost: {e}, setting cost to 0.")
        #     if (
        #         self.config.per_instance_cost_limit > 0
        #         or self.config.total_cost_limit > 0
        #     ):
        #         msg = (
        #             f"Error calculating cost: {e} for your model {self.config.name}. If this is ok "
        #             "(local models, etc.), please make sure you set `per_instance_cost_limit` and "
        #             "`total_cost_limit` to 0 to disable this safety check."
        #         )
        #         self.logger.error(msg)
        #         raise ModelConfigurationError(msg)
        #     cost = 0
        choices: litellm.types.utils.Choices = response.choices  # type: ignore
        n_choices = n if n is not None else 1
        outputs = []
        output_tokens = 0
        for i in range(n_choices):
            output = choices[i].message.content or ""
            output_tokens += litellm.utils.token_counter(
                text=output, model=self.config.name
            )
            output_dict = {"message": output}
            if self.tools.use_function_calling:
                if response.choices[i].message.tool_calls:  # type: ignore
                    tool_calls = [call.to_dict() for call in response.choices[i].message.tool_calls]  # type: ignore
                else:
                    tool_calls = []
                output_dict["tool_calls"] = tool_calls
            outputs.append(output_dict)
        self._update_stats(
            input_tokens=input_tokens, output_tokens=output_tokens, cost=cost
        )
        return outputs
