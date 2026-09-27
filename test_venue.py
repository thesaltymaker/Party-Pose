import json

from src.types import BoundingBox
from src.venue import ignored_by, load_venue


def test_lamp_box_is_ignored_and_daughter_next_to_it_is_not(tmp_path):
    # Orin: the lamp (body 0.40-0.64, with a false 0.47 head on top) stands right beside a person on the couch.
    f = tmp_path / 'venue.json'
    f.write_text(json.dumps({'ignore': [{'name': 'lamp', 'box': [669, 423, 128, 241]}]}))
    zones = load_venue(str(f))
    assert ignored_by(BoundingBox(670, 422, 123, 240, 0.44), zones).name == 'lamp'
    assert ignored_by(BoundingBox(773, 537, 200, 328, 0.74), zones) is None


def test_person_standing_in_front_of_a_prop_is_not_ignored(tmp_path):
    # A person's box in front of the lamp is much bigger than the lamp's, so the IoU stays low.
    f = tmp_path / 'venue.json'
    f.write_text(json.dumps({'ignore': [{'name': 'lamp', 'box': [669, 423, 128, 241]}]}))
    zones = load_venue(str(f))
    assert ignored_by(BoundingBox(600, 300, 260, 700, 0.8), zones) is None


def test_empty_venue(tmp_path):
    f = tmp_path / 'venue.json'
    f.write_text('{}')
    assert load_venue(str(f)) == []
