"""Upload identity and comparison state across Streamlit reruns."""
import hashlib


SINGLE_RESULT_KEYS = (
    "last_page", "last_sensitivity", "processed", "img_a_highlight",
    "img_b_highlight", "regions", "has_diff", "result", "page_selector",
)
BATCH_RESULT_KEYS = ("batch_results", "batch_zip_bytes", "batch_output_dir")


def clear_uploads(state):
    """Reset both modes and recreate upload widgets with empty file selections."""
    generation = state.get('upload_generation', 0)
    for key in (*SINGLE_RESULT_KEYS, *BATCH_RESULT_KEYS,
                'single_upload_identity', 'batch_previous_files'):
        state.pop(key, None)
    for key in ('pdf_a', 'pdf_b', 'pdf_a_batch', 'pdf_b_batch'):
        state.pop(f'{key}_{generation}', None)
    state['upload_generation'] = generation + 1


def upload_identity(uploaded_file):
    """Include content so replacing a file with the same name invalidates results."""
    return uploaded_file.name, hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def reset_results_on_upload_change(state, identity_key, uploads, result_keys):
    identity = tuple(tuple(upload_identity(f) for f in group) for group in uploads)
    if state.get(identity_key) != identity:
        for key in result_keys:
            state.pop(key, None)
        state[identity_key] = identity
    return identity
