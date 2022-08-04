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

    @staticmethod
    def test_match_faces_speakers():
        multi = Multimodal()
        multi.add(
            feature_dict = {
                'time': [0.1, 0.3, 0.4, 0.5],
                'face_id': [2, 2, 3, 3],
                'speaker_id': ['A', 'A', 'B', 'B']
            }
        )
        multi.match_faces_speakers()

        # Test set due to switching id assignment
        assert set(multi.features['match_id'].tolist()) == set([1.0, 1.0, 2.0, 2.0])
