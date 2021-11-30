import os
from unittest import mock
import pytest
import tempfile
import numpy as np
from os import listdir


from style.constants import MODEL_EXPORT_PATH
from style.predict.servable.base import SklearnBasedClassifierServable


@pytest.fixture(scope="module")
def sklearn_classifier():
    return SklearnBasedClassifierServable.load(path=MODEL_EXPORT_PATH / "small")


class TestSklearnBasedClassifierServable:
    def test_run_inference(self, sklearn_classifier):
        assert isinstance(sklearn_classifier.run_inference("text"), str)

    # def test_run_inference_multiclass():
    # export, load, multiclass, tempfile (library) bu dosyaya export edecem
    # ondan sonra otomatik olarak siliyor zaten ben o dosya var mi kontrolu yapacam
    # bu sayede exportu kontrol etmis olacam

    # def test_inference_multiclass(self, sklearn_classifier):
    #     with open()
    #     sklearn_classifier.model.predict_proba = mock.Mock(return_value = np.array(['deneme']))
    #     assert sklearn_classifier.run_inference_multiclass('text') == "deneme"

    def test_export(self, sklearn_classifier):
        filepath = ""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = tmpdir + "/small"
            sklearn_classifier.export(filepath)
            assert os.path.exists(filepath)
        assert not os.path.exists(filepath)
