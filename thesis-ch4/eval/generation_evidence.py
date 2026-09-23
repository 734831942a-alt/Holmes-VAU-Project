"""Observe actual generated IDs without editing the frozen SPEC-01 module.

The wrapper forwards the original call unchanged, restores the instance method
even on exceptions, and permits only one capture at a time. No text re-encoding.
"""
from contextlib import contextmanager
from threading import Lock


_capture_lock = Lock()


def single_sequence(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        raise ValueError('Evidence capture requires one output sequence')
    return list(value[0])


def stopping_evidence(output_ids, input_prefix_ids, eos_token_id, max_new_tokens):
    """input_prefix_ids means the prefix actually included in output_ids."""
    if output_ids[:len(input_prefix_ids)] != input_prefix_ids:
        raise ValueError('Output does not contain the declared input prefix')
    generated = list(output_ids[len(input_prefix_ids):])
    count = len(output_ids) - len(input_prefix_ids)
    if count > max_new_tokens or max_new_tokens <= 0:
        raise ValueError('Generated token count exceeds the effective budget')
    eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)
    if not eos_ids or not all(isinstance(v, int) for v in eos_ids + generated):
        raise ValueError('Missing or invalid authoritative token IDs')
    reason = ('eos' if any(v in eos_ids for v in generated) else
              'max_new_tokens' if count == max_new_tokens else 'other')
    return {'stop_reason': reason, 'generated_token_count': count,
            'max_new_tokens': max_new_tokens, 'last_token_is_eos': bool(generated and generated[-1] in eos_ids),
            'generated_token_ids': generated, 'truncated': reason == 'max_new_tokens',
            'input_prefix_token_count': len(input_prefix_ids), 'output_token_count': len(output_ids),
            'eos_token_id': eos_token_id}


@contextmanager
def capture_generation(language_model, expected_budget):
    if not _capture_lock.acquire(blocking=False):
        raise RuntimeError('Concurrent/reentrant generation evidence capture is unsupported')
    original = language_model.generate
    had_instance_value = 'generate' in vars(language_model)
    previous_instance_value = vars(language_model).get('generate')
    captured = []

    def observed(*args, **kwargs):
        if args:
            raise RuntimeError('Unverified positional generation inputs; stop instead of guessing prefix')
        generation_config = kwargs.get('generation_config')
        def effective(name):
            if name in kwargs:
                return kwargs[name]
            if isinstance(generation_config, dict):
                return generation_config.get(name)
            return getattr(generation_config, name, None)
        budget, eos = effective('max_new_tokens'), effective('eos_token_id')
        if budget != expected_budget or eos is None:
            raise RuntimeError('Missing or conflicting generation budget/EOS evidence')
        if kwargs.get('input_ids') is not None:
            prefix = single_sequence(kwargs['input_ids'])
            source = 'input_ids_in_returned_sequence'
        elif kwargs.get('inputs_embeds') is not None:
            prefix = []
            source = 'inputs_embeds_only_empty_token_prefix'
        else:
            raise RuntimeError('Unknown generation input layout')
        if captured:
            raise RuntimeError('More than one generation call; no retry allowed')
        result = original(**kwargs)
        sequences = getattr(result, 'sequences', result)
        evidence = stopping_evidence(single_sequence(sequences), prefix, eos, budget)
        evidence['input_prefix_source'] = source
        captured.append(evidence)
        return result

    try:
        language_model.generate = observed
        yield captured
    finally:
        if had_instance_value:
            language_model.generate = previous_instance_value
        else:
            delattr(language_model, 'generate')
        _capture_lock.release()


def generate_with_evidence(video_path, prompt, model, tokenizer, *, generate_fn=None, **kwargs):
    if generate_fn is None:
        from src.model.holmesvau_infer import generate as generate_fn
    with capture_generation(model.language_model, kwargs['max_new_tokens']) as captured:
        raw, indices = generate_fn(video_path, prompt, model, tokenizer, **kwargs)
    if len(captured) != 1:
        raise RuntimeError('Frozen generate did not expose exactly one authoritative token sequence')
    return raw, indices, captured[0]
