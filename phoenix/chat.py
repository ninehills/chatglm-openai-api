#!/usr/bin/env python3
import torch
from .conversation import get_default_conv_template, SeparatorStyle

def init_model_args(model_args = None):
    if model_args is None:
        model_args = {}
    model_args['temperature'] = model_args['temperature'] if model_args.get('temperature') != None else 0.7
    model_args['max_tokens'] = model_args['max_tokens'] if model_args.get('max_tokens') != None else 512

    return model_args


def do_chat(model, tokenizer, question, history, model_args = None):
    ret = ""
    for char in do_chat_stream(model, tokenizer, question, history, model_args):
        ret += char
    return ret


def do_chat_stream(model, tokenizer, question, history, model_args = None):
    model_args = init_model_args(model_args)
    conv = get_default_conv_template().copy()

    for (human, ai) in history:
        conv.append_message(conv.roles[0], human)
        # NOTE: strip is important to align with the training data.
        conv.append_message(conv.roles[1], ai.strip())
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)

    generate_stream_func = generate_stream
    prompt = conv.get_prompt()

    params = {
        "model": model,
        "prompt": prompt,
        "temperature": model_args['temperature'],
        "max_new_tokens": model_args['max_tokens'],
        "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else None,
    }

    output_stream = generate_stream_func(model, tokenizer, params, model.running_device)

    pre = 0
    for outputs in output_stream:
        now = len(outputs) - 1
        if now > pre:
            yield(outputs[pre:now])
            pre = now
    yield(outputs[pre:])


@torch.inference_mode()
def generate_stream(model, tokenizer, params, device, context_len=2048, stream_interval=2):
    prompt = params["prompt"]
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_ids", [tokenizer.eos_token_id])

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    l_prompt = len(tokenizer.decode(input_ids, skip_special_tokens=False))

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            if model.config.is_encoder_decoder:
                encoder_outputs = model.encoder(
                    input_ids=torch.as_tensor([input_ids], device=device)
                )
                out = model(
                    torch.as_tensor([input_ids], device=device),
                    decoder_input_ids=torch.as_tensor(
                        [[model.generation_config.decoder_start_token_id]],
                        device=device,
                    ),
                    encoder_outputs=encoder_outputs,
                    use_cache=True,
                )
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
        else:
            if model.config.is_encoder_decoder:
                out = model(
                    input_ids=torch.as_tensor([input_ids], device=device),
                    use_cache=True,
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=torch.as_tensor([[token]], device=device),
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=False)
            if stop_str:
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[l_prompt:pos]
                    stopped = True
                else:
                    output = output[l_prompt:]
                yield output
            else:
                raise NotImplementedError

        if stopped:
            break

    del past_key_values