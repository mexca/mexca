""" Test output classes and methods """

from mexca.core.output import Multimodal

class TestMultimodal:

    @staticmethod
    def test_add():
        multi = Multimodal()
        multi.add({'time': [0.0, 0.1]})

        assert isinstance(multi, Multimodal)
        assert multi.features == {'time': [0.0, 0.1]}

    @staticmethod
    def test_add_replace():
        multi = Multimodal()
        multi.add({'time': [0.0, 0.1]})
        multi.add({'time': [0.0, 0.2]})

        assert multi.features == {'time': [0.0, 0.1]}

        multi.add({'time': [0.0, 0.2]}, replace=True)

        assert multi.features == {'time': [0.0, 0.2]}
