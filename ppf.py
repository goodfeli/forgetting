__author__ = "Ian Goodfellow"

def cloud_to_ppf(x, y, higher_better=True):
    """
    x: A list of x coordinates
    y: A list of y coordinates

    Makes a production possibility frontier, i.e. a curve showing
    how much y is possible for each value of x.

    If higher_better, we assume bigger values of y are better, so we
    make a curve that lies above all points in the cloud.

    Returns ppf_x, ppf_y, which are lists of x and y points defining the ppf
    curve.
    """

    raise NotImplementedError()


