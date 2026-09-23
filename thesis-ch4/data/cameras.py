"""Exact normalized OCR grouping, bounded edit-distance merges, explicit uncertainty."""
from collections import defaultdict


def edit_distance(a, b, limit):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i]
        for j, right in enumerate(b, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


class Components:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def root(self, value):
        if self.parent[value] != value:
            self.parent[value] = self.root(self.parent[value])
        return self.parent[value]

    def join(self, a, b):
        ra, rb = self.root(a), self.root(b)
        self.parent[max(ra, rb)] = min(ra, rb)


def group_cameras(rows, cfg):
    o = cfg["ocr"]
    strings = sorted({r["normalized"] for r in rows if r["eligible"]})
    components = Components(strings)
    close = set()
    for i, a in enumerate(strings):
        for b in strings[i + 1:]:
            if abs(len(a) - len(b)) <= o["edit_distance"] and edit_distance(a, b, o["edit_distance"]) <= o["edit_distance"]:
                components.join(a, b)
                close.add((a, b))
    members = defaultdict(list)
    for value in strings:
        members[components.root(value)].append(value)
    # A-B-C chains with A-C outside threshold are cluster-boundary cases, never transitive merges.
    ambiguous = set()
    for group in members.values():
        if any((a, b) not in close for i, a in enumerate(group) for b in group[i + 1:]):
            ambiguous.update(group)
    assignments, groups, uncertain = {}, defaultdict(list), []
    for i, row in enumerate(sorted(rows, key=lambda r: r["video_id"])):
        text = row["normalized"]
        if not row["eligible"] or text in ambiguous:
            group = o["unknown_template"].format(i=i)
            confidence = min(row["confidence"], o["boundary_confidence"])
            uncertain.append(dict(row, camera_group=group,
                                  reason="cluster_boundary" if text in ambiguous else "low_confidence_or_missing_overlay"))
        else:
            # The group ID is actual normalized OCR evidence, never a guessed camera/location.
            group = components.root(text)
            confidence = row["confidence"]
        assignments[row["video_id"]] = (group, confidence)
        groups[group].append(row["video_id"])
    return assignments, dict(groups, uncertain=uncertain)
