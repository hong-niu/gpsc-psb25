import sys
sys.path.append('../')
from algorithms.gdbscan import GDBSCAN, Points
import math

UNCLASSIFIED = -2

class Point:
    def __init__(self, x, y, val1, val2, val3):
        self.x = x
        self.y = y
        self.val1 = val1
        self.val2 = val2
        self.val3 = val3
        self.cluster_id = UNCLASSIFIED

    def __repr__(self):
        return '(x:{}, y:{}, val1:{}, val2:{}, val3:{}, cluster:{})' \
            .format(self.x, self.y, self.val1, self.val2, self.val3, self.cluster_id)

    def get_dict(self):
        return {'long': [self.x],'lat': [self.y], 'x1': [self.val1], 'x2': [self.val2],'y1': [self.val3], 'gdbscan': [self.cluster_id]}


def n_pred(p1, p2):
    return all([math.sqrt((p1.val1 - p2.val1) ** 2 + (p1.val2 - p2.val2) ** 2 + (p1.val3 - p2.val3) ** 2) <= 9,
                math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) <= 12
                ])


def w_card(points):
    return len(points)


def test():
    # p1 = Point(0, 0, 1)
    # p2 = Point(0, 1, 2)
    # p2 = Point(1, 0, 3)
    # p3 = Point(2, 1, 1)
    # p4 = Point(2, 2, 2)
    # p4 = Point(1, 2, 3)

    p1 = Point(0, 0, 1, 1, 1)
    p2 = Point(1, 0, 1, 1, 1)
    p3 = Point(2, 2, 5, 5, 5)
    p4 = Point(1, 2, 5, 5, 5)

    points = [p1, p2, p3, p4]

    clustered = GDBSCAN(Points(points), n_pred, 0, w_card)
    # print()
    # print(clustered)


if __name__ == '__main__':
    test()
