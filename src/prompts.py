"""Text side of the classifier: class names, prompt templates and short class descriptions.

The descriptions were written from general knowledge of each scene type (what such a place looks like),
not by looking at dataset images.
"""

# Folder name -> natural-language name.
NATURAL = {
    'Bedroom': 'bedroom',
    'Coast': 'coast',
    'Flower': 'flower',
    'Forest': 'forest',
    'Highway': 'highway',
    'Industrial': 'industrial area',
    'InsideCity': 'city centre',
    'Kitchen': 'kitchen',
    'LivingRoom': 'living room',
    'Mountain': 'mountain',
    'Office': 'office',
    'OpenCountry': 'open countryside',
    'Store': 'store',
    'Street': 'street',
    'Suburb': 'suburban house',
    'TallBuilding': 'tall building',
}

# Templates for prompt ensembling. Most images are grayscale, so several templates say so.
TEMPLATES = [
    'a photo of a {}.',
    'a black and white photo of a {}.',
    'a grayscale photo of a {}.',
    'a photo of the {} scene.',
    'a scene of a {}.',
    'a low resolution photo of a {}.',
    'an old photo of a {}.',
    'a photo taken in a {}.',
]

# Short descriptions of each scene (CuPL-style), used as extra prompts.
DESCRIPTIONS = {
    'Bedroom': ['a bedroom with a bed, pillows and a nightstand.', 'the inside of a bedroom with a large bed.'],
    'Coast': ['a coast with the sea, a beach and the horizon.', 'the shoreline where the ocean meets the land.'],
    'Flower': ['a close-up photo of a flower with petals.', 'a colourful flower in bloom.'],
    'Forest': ['a forest full of trees and dense foliage.', 'tall tree trunks and leaves in the woods.'],
    'Highway': ['a highway with lanes, cars and road markings.', 'a long road or freeway stretching to the horizon.'],
    'Industrial': ['an industrial area with factories, chimneys and smokestacks.', 'a power plant or factory buildings with pipes.'],
    'InsideCity': ['a city centre with buildings, shop fronts and windows.', 'the facades of buildings inside a city.'],
    'Kitchen': ['a kitchen with cabinets, a counter, a stove and a sink.', 'the inside of a kitchen with cupboards.'],
    'LivingRoom': ['a living room with a sofa, armchairs and a coffee table.', 'the inside of a living room with a couch.'],
    'Mountain': ['a mountain landscape with peaks, slopes and rocks.', 'snowy or rocky mountains under the sky.'],
    'Office': ['an office with desks, a computer and chairs.', 'the inside of an office room with a workstation.'],
    'OpenCountry': ['open countryside with fields, grass and a wide sky.', 'a rural landscape with meadows and hills.'],
    'Store': ['the inside of a store with shelves full of products.', 'a shop or supermarket aisle with goods for sale.'],
    'Street': ['a street with cars, sidewalks and buildings on both sides.', 'a road in a town with houses and lamp posts.'],
    'Suburb': ['a suburban house with a front yard and a lawn.', 'a residential house in the suburbs.'],
    'TallBuilding': ['a tall skyscraper seen from below.', 'a high-rise building with many floors of windows.'],
}


def prompt_sets(classes):
    """Each prompt set maps to one list of prompts per class (embeddings are averaged within a class)."""
    raw = {c: [f'a photo of a {c}.'] for c in classes}
    name = {c: [f'a photo of a {NATURAL[c]}.'] for c in classes}
    templates = {c: [t.format(NATURAL[c]) for t in TEMPLATES] for c in classes}
    full = {c: templates[c] + DESCRIPTIONS[c] for c in classes}
    return {'raw': raw, 'name': name, 'templates': templates, 'templates+desc': full}
