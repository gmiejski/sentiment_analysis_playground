def load(review_lines_generator, max_entries=None):
    if max_entries is None:
        max_entries = 9999999999
    data = []
    for review in review_lines_generator:
        if len(data) > max_entries:
            return data
        data.append((review['reviewText'], isPositive(review)))
    return data


def isPositive(review):
    return review['overall'] > 3.0
