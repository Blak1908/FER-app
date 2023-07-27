from collections import Counter

def calculate_mean(lst):
    if not lst:
        return None
    total_sum = sum(lst)
    mean = total_sum / len(lst)
    return mean

def most_common_element(lst):
    counter = Counter(lst)
    most_common_items = counter.most_common()
    most_common_element, count = most_common_items[0]

    return most_common_element