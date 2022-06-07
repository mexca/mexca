""" Test dummy function """

from mexca.audio.dummy import dummy_fun

def test_dummy():
    dummy = dummy_fun()

    assert dummy == 'This is a dummy function!'
    