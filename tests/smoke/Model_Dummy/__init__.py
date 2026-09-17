# Test-only model package (Phase 5). The directory MUST be named
# ``Model_<Class>``: the official ``models.ModelBase.__init__`` derives
# ``model_class_name`` from the model class's module folder
# (``.../Model_X`` -> ``X``), exactly like the official model packages
# (``models/Model_SAEHD``). The dummy model is test-only and is NOT a
# production architecture (it lives in tests, never in ``models/``).
