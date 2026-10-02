import unittest
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from src.core.annotate import add_green_overlay
from test_app_uploads import pdf_upload


class ReuploadTests(unittest.TestCase):
    def test_blend_matches_original_without_float_page_arrays(self):
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)
        image = np.repeat(image[:, :, None], 3, axis=2)
        mask = np.full((16, 16), 255, dtype=np.uint8)
        mask[::2] = 0
        overlay = np.zeros_like(image)
        overlay[:, :, 1] = 255
        for alpha in (0, 0.3, 0.7, 1):
            blend = (mask[:, :, None] > 0).astype(float) * alpha
            expected = (image * (1 - blend) + overlay * blend).astype(np.uint8)
            np.testing.assert_array_equal(add_green_overlay(image, mask, alpha), expected)

    def test_real_upload_widgets_clear_and_reupload(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()
        for cycle in range(3):
            for index, text in enumerate(('Original', f'Revision {cycle}')):
                app.sidebar.file_uploader[index].set_value(
                    ('drawing.pdf', pdf_upload(text).getvalue(), 'application/pdf')
                ).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.error), 0)
            self.assertTrue(app.session_state['processed'])
            app.sidebar.button(key='clear_uploads').click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertNotIn('result', app.session_state)
            self.assertTrue(all(widget.value is None for widget in app.sidebar.file_uploader))
