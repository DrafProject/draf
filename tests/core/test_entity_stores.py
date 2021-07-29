import pytest

from draf.core import entity_stores


@pytest.fixture
def scenarios() -> entity_stores.Scenarios:
    return entity_stores.Scenarios()


def test_scenario___repr__(scenarios):
    assert scenarios.__repr__() == "<Scenarios object> (empty)"


@pytest.fixture
def dimensions() -> entity_stores.Dimensions:
    return entity_stores.Dimensions()


def test_dimension___init__(dimensions):
    dimensions.__init__()
    assert dimensions._meta == {}


def test_dimension___repr__(dimensions):
    assert dimensions.__repr__() == "<Dimensions object> (empty)"


@pytest.fixture
def entity() -> entity_stores.EntityStore:
    return entity_stores.EntityStore()


def test_entity___repr__(entity):
    assert entity.__repr__() == "<EntityStore object> (empty)"


def test_entity___init__(entity):
    entity.__init__()
    assert entity._changed_since_last_dic_export == False


def test__empty_dims_dic(entity):
    assert entity._empty_dims_dic == {}


@pytest.fixture
def params() -> entity_stores.Params:
    return entity_stores.Params()


def test_params___init__(params):
    params.__init__()
    assert params._meta == {}


def test_params___repr__(params):
    assert params.__repr__() == "<Params object> (empty)"


@pytest.fixture
def ent_vars() -> entity_stores.Vars:
    return entity_stores.Vars()


def test_ent_vars___init__(ent_vars):
    ent_vars.__init__()
    assert ent_vars._meta == {}


def test_ent_vars___repr__(ent_vars):
    assert ent_vars.__repr__() == "<Vars object> (empty)"


def test_ent_vars___getstate__(ent_vars):
    assert ent_vars.__getstate__() == {"_meta": {}}
