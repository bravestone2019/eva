import torch

from app.config import (
    MAX_NEW_TOKENS,
    TEMPERATURE
)


class LLMEngine:

    def __init__(
        self,
        model_manager
    ):

        self.llm = (
            model_manager.llm
        )

        self.tokenizer = (
            model_manager.tokenizer
        )

    def generate(

        self,

        user_message,

        emotion_state=None,

        conversation_history=None,

        rag_context=None
    ):

        messages = [

            {

                "role":
                "system",

                "content":
                """
You are EVA, a voice-first
personal AI assistant.

Be helpful, natural and conversational.

Answer straightforward factual questions directly and lead with
the answer. Use your general knowledge for stable facts such as
geography, history, science, and common concepts.

Do not say that you are an AI, apologize for being unable to
access information, or mention limitations unless the user asks
for current, live, private, or otherwise unavailable information.

Do not invent facts. When a fact may be time-sensitive or you are
genuinely uncertain, say so briefly and explain what would verify it.

Emotional signals are uncertain
observations rather than facts about
the user's internal mental state.

Never claim certainty about what
the user is feeling.

Never diagnose medical or
psychological conditions.

Use emotional information only to
adapt communication style.
"""
            }
        ]

        if conversation_history:

            for conversation in (
                conversation_history
            ):

                messages.append({

                    "role":
                    "user",

                    "content":
                    conversation["user"]
                })

                messages.append({

                    "role":
                    "assistant",

                    "content":
                    conversation["assistant"]
                })

        if rag_context:

            messages.append({

                "role":
                "system",

                "content":
                f"""
Relevant retrieved context:

{rag_context}
"""
            })

        if emotion_state:

            messages.append({

                "role":
                "system",

                "content":
                f"""
Possible vocal interaction signals:

Arousal:
{emotion_state.get("arousal", 0):.2f}

Intensity:
{emotion_state.get("intensity", 0):.2f}

Stress indicator:
{emotion_state.get("stress", 0):.2f}

Treat these as uncertain signals.
"""
            })

        messages.append({

            "role":
            "user",

            "content":
            user_message
        })

        inputs = (
            self.tokenizer
            .apply_chat_template(

                messages,

                add_generation_prompt=True,

                tokenize=True,

                return_dict=True,

                return_tensors="pt"
            )
        )

        inputs = inputs.to(
            self.llm.device
        )

        with torch.no_grad():

            outputs = self.llm.generate(

                **inputs,

                max_new_tokens=
                MAX_NEW_TOKENS,

                do_sample=True,

                temperature=
                TEMPERATURE
            )

        input_length = (
            inputs["input_ids"]
            .shape[-1]
        )

        response = (
            self.tokenizer.decode(

                outputs[0][input_length:],

                skip_special_tokens=True
            )
        )

        return response.strip()
