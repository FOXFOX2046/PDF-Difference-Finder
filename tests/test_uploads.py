import io
import unittest

from src.core.uploads import (
    BATCH_RESULT_KEYS, SINGLE_RESULT_KEYS, reset_results_on_upload_change,
)


def upload(name, content):
    file = io.BytesIO(content)
    file.name = name
    return file


class UploadStateTests(unittest.TestCase):
    def test_rerun_preserves_results_and_stream_position(self):
        file = upload('drawing.pdf', b'%PDF-original')
        state = {}
        reset_results_on_upload_change(state, 'identity', ([file], [file]), SINGLE_RESULT_KEYS)
        state['result'] = 'comparison'
        file.read()
        reset_results_on_upload_change(state, 'identity', ([file], [file]), SINGLE_RESULT_KEYS)
        self.assertEqual(state['result'], 'comparison')
        self.assertEqual(file.tell(), len(file.getvalue()))
        self.assertEqual(file.getvalue(), b'%PDF-original')

    def test_same_filename_new_content_resets_single_and_batch(self):
        for keys in (SINGLE_RESULT_KEYS, BATCH_RESULT_KEYS):
            with self.subTest(keys=keys):
                state = {'unrelated': 'keep'}
                reset_results_on_upload_change(state, 'identity', ([upload('a.pdf', b'old')], []), keys)
                state.update(dict.fromkeys(keys, 'old result'))
                reset_results_on_upload_change(state, 'identity', ([upload('a.pdf', b'new')], []), keys)
                self.assertTrue(all(key not in state for key in keys))
                self.assertEqual(state['unrelated'], 'keep')

    def test_removing_upload_clears_results(self):
        state = {}
        reset_results_on_upload_change(state, 'identity', ([upload('a.pdf', b'a')], []), SINGLE_RESULT_KEYS)
        state['processed'] = True
        reset_results_on_upload_change(state, 'identity', ([], []), SINGLE_RESULT_KEYS)
        self.assertNotIn('processed', state)


if __name__ == '__main__':
    unittest.main()
