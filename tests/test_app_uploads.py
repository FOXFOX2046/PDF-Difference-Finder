"""Exercise the real app with uploads injected into Streamlit's test runner."""
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
from streamlit.delta_generator import DeltaGenerator
from streamlit.testing.v1 import AppTest


def pdf_upload(text):
    with fitz.open() as doc:
        page = doc.new_page(width=120, height=120)
        page.insert_text((10, 30), text)
        file = io.BytesIO(doc.tobytes())
    file.name = 'drawing.pdf'
    return file


class AppUploadTests(unittest.TestCase):
    def setUp(self):
        self.files = {'pdf_a': pdf_upload('Original'), 'pdf_b': pdf_upload('Revised')}
        self.app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30)
        def injected_uploader(generator, label, **kwargs):
            return self.uploader(label, **kwargs)

        self.patcher = patch.object(DeltaGenerator, 'file_uploader', injected_uploader)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def uploader(self, label, **kwargs):
        key = kwargs['key']
        if key.endswith('_batch'):
            file = self.files.get(key.removesuffix('_batch'))
            return [file] if file is not None else []
        return self.files.get(key)

    def assert_success(self):
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 0)

    def test_single_upload_rerun_replace_remove_and_invalid(self):
        self.app.run()
        self.assert_success()
        original_identity = self.app.session_state['single_upload_identity']
        self.assertTrue(self.app.session_state['processed'])
        # Simulate an already-consumed stream and trigger a control rerun.
        for file in self.files.values():
            file.read()
        self.app.sidebar.slider[0].set_value(0.7).run()
        self.assert_success()
        self.assertTrue(self.app.session_state['processed'])
        self.files['pdf_b'] = pdf_upload('Replacement')
        self.app.run()
        self.assert_success()
        self.assertNotEqual(original_identity, self.app.session_state['single_upload_identity'])
        self.assertTrue(self.app.session_state['processed'])
        del self.files['pdf_b']
        self.app.run()
        self.assert_success()
        self.assertNotIn('processed', self.app.session_state)
        self.files['pdf_b'] = io.BytesIO(b'not a PDF')
        self.files['pdf_b'].name = 'broken.pdf'
        self.app.run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 1)

    def test_batch_can_process_same_uploads_twice(self):
        self.app.run()
        self.app.sidebar.radio[0].set_value('Batch Mode').run()
        for _ in range(2):
            self.app.sidebar.button[0].click().run()
            self.assert_success()
            self.assertTrue(self.app.session_state['batch_zip_bytes'].startswith(b'PK'))
            for file in self.files.values():
                file.read()
        self.files['pdf_b'] = pdf_upload('Replacement')
        self.app.run()
        self.assert_success()
        self.assertNotIn('batch_zip_bytes', self.app.session_state)


if __name__ == '__main__':
    unittest.main()
