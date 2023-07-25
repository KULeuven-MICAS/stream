def intersections(a, b):
    """Get the intersections of two lists of ranges.
    https://stackoverflow.com/questions/40367461/intersection-of-two-lists-of-ranges-in-python

    Args:
        a (list): The first list.
        b (list): The second list.

    Returns:
        list: The intersections between the two lists.
    """
    ranges = []
    i = j = 0
    while i < len(a) and j < len(b):
        a_left, a_right = a[i]
        b_left, b_right = b[j]

        if a_right < b_right:
            i += 1
        else:
            j += 1

        if a_right >= b_left and b_right >= a_left:
            end_pts = sorted([a_left, a_right, b_left, b_right])
            middle = (end_pts[1], end_pts[2])
            ranges.append(middle)

    ri = 0
    while ri < len(ranges) - 1:
        if ranges[ri][1] == ranges[ri + 1][0]:
            ranges[ri : ri + 2] = [(ranges[ri][0], ranges[ri + 1][1])]

        ri += 1

    return ranges


if __name__ == "__main__":
    a = [(8, 10), (13, 15), (17, 20)]
    b = [(0, 9), (14, 18)]
    print(intersections(a, b))
