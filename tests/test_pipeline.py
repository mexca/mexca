""" Test Pipeline class and methods """

import os
import pytest
from mexca.core.output import Multimodal
from mexca.core.pipeline import Pipeline

class TestPipeline:
    obj = Pipeline(
        enabled_video=True,
        enabled_audio=True,
        enabled_text=True
    )

    filepath = os.path.join('test', 'path')

    obj_no_filepath = Pipeline(
        enabled_video=True,
        enabled_audio=True,
        enabled_text=True
    )

    result = Multimodal(None, None, None)

    def test_get_enabled_video(self):
        assert self.obj.get_enabled_video()


    def test_get_enabled_audio(self):
        assert self.obj.get_enabled_audio()


    def test_get_enabled_text(self):
        assert self.obj.get_enabled_text()


    def test_set_filepath(self):
        self.obj.set_filepath(self.filepath)
        assert self.obj.filepath == self.filepath


    def test_get_filepath(self):
        self.obj.set_filepath(self.filepath)
        assert self.obj.get_filepath() == self.filepath

        with pytest.raises(AttributeError):
            self.obj_no_filepath.get_filepath()


    @pytest.mark.skip(
        reason='Subpipelines for modalities are not implemented yet; cannot apply entire pipeline'
    )
    def test_apply(self):
        result_pipeline = self.obj.apply(self.filepath)
        assert result_pipeline == self.result
